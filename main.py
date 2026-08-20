"""Core NeuroTx-MOOD experiment pipeline.

The module provides dataset loading, split construction, feature generation,
conformal calibration, and the main BBB/CNS evaluation routines. Paths are
resolved relative to the local code bundle first and then
to the nearest ancestor directory that exposes a ``data`` folder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from molformer_compat import enable_molformer_compat


RDLogger.DisableLog("rdApp.*")

ALPHA = 0.10
SEEDS = list(range(42, 52))
FP_BITS = 1024
OOD_GUARD_MARGIN = 0.30
DELTA_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
CONTEXT_MIN_GROUP_GRID = [10, 25, 50]
CONTEXT_SCHEMES = ["basic", "rich"]
CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"
CHEMBERTA_BATCH_SIZE = 64
MOLFORMER_MODEL = "ibm-research/MoLFormer-XL-both-10pct"
MOLFORMER_BATCH_SIZE = 32
SELECTIVE_GUARD_QUANTILES = [0.10, 0.25, 0.40, 0.50]
SELECTIVE_GUARD_MARGIN = 0.30
METADATA_DROPOUT_RATES = [0.0, 0.20, 0.40, 0.60]

def _find_data_dir() -> Path:
    configured = os.environ.get("NEUROTX_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    local_dir = Path(__file__).resolve().parent / "data"
    return local_dir


DATA_DIR = _find_data_dir()
CALIBRATION_ONLY_POINT_METRICS = {
    "auroc",
    "auprc",
    "brier_score",
    "ece",
    "ood_external_auroc",
    "ood_external_auprc",
    "ood_external_brier_score",
    "ood_external_ece",
}


@dataclass
class MolBundle:
    name: str
    path: Path
    smiles: list[str]
    y: np.ndarray
    descriptors: np.ndarray
    fingerprints: np.ndarray
    scaffolds: np.ndarray
    context: np.ndarray | None = None
    foundation_embeddings: np.ndarray | None = None
    molformer_embeddings: np.ndarray | None = None
    target_ids: np.ndarray | None = None
    assay_types: np.ndarray | None = None
    bao_formats: np.ndarray | None = None
    document_years: np.ndarray | None = None
    label_rule: str = ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def scaffold_smiles(mol) -> str:
    if mol is None:
        return "invalid"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold or "acyclic"
    except Exception:
        return "invalid"


@lru_cache(maxsize=None)
def canonical_smiles(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return str(smiles)
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return str(smiles)


@lru_cache(maxsize=None)
def parent_inchikey_block(smiles: str) -> str:
    """Return a salt/charge/tautomer-normalized InChIKey connectivity block.

    This broader identity is used for sensitivity analyses only. The sequence is
    RDKit cleanup, largest-fragment parent selection, uncharging, canonical
    tautomer selection, and conversion to the first InChIKey block.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return f"invalid:{smiles}"
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        key = Chem.MolToInchiKey(mol)
        return key.split("-", 1)[0] if key else canonical_smiles(smiles)
    except Exception:
        return canonical_smiles(smiles)


def compute_features(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    descriptors = []
    fingerprints = []
    scaffolds = []
    for smi in smiles:
        mol = mol_from_smiles(smi)
        scaffolds.append(scaffold_smiles(mol))
        if mol is None:
            descriptors.append([np.nan] * 10)
            fingerprints.append(np.zeros(FP_BITS, dtype=np.float32))
            continue
        descriptors.append(
            [
                Descriptors.MolWt(mol),
                Descriptors.TPSA(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                Chem.GetFormalCharge(mol),
                Descriptors.FractionCSP3(mol),
            ]
        )
        bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
        arr = np.zeros((FP_BITS,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(bitvect, arr)
        fingerprints.append(arr.astype(np.float32))
    return (
        np.asarray(descriptors, dtype=np.float32),
        np.asarray(fingerprints, dtype=np.float32),
        np.asarray(scaffolds, dtype=object),
    )


def load_bbb_tab(path: Path, name: str) -> MolBundle:
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["Drug", "Y"]).copy()
    df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Y"]).copy()
    df["Y"] = df["Y"].astype(int)
    smiles = df["Drug"].astype(str).tolist()
    desc, fps, scaffolds = compute_features(smiles)
    return MolBundle(
        name=name,
        path=path,
        smiles=smiles,
        y=df["Y"].to_numpy(dtype=int),
        descriptors=desc,
        fingerprints=fps,
        scaffolds=scaffolds,
        label_rule="TDC binary BBB penetration/permeability label Y",
    )


def load_chembl(path: Path) -> MolBundle:
    df = pd.read_csv(path)
    df = df.dropna(subset=["canonical_smiles", "pchembl_value", "target_chembl_id"]).copy()
    df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
    df = df.dropna(subset=["pchembl_value"]).copy()
    # Standard medicinal chemistry activity threshold: pChEMBL >= 6 (~1 uM or stronger).
    df["Y"] = (df["pchembl_value"] >= 6.0).astype(int)
    smiles = df["canonical_smiles"].astype(str).tolist()
    desc, fps, scaffolds = compute_features(smiles)
    context = (
        df["target_chembl_id"].astype(str)
        + "|"
        + df["standard_type"].fillna("unknown").astype(str)
    ).to_numpy(dtype=object)
    return MolBundle(
        name="ChEMBL_CNS_targets",
        path=path,
        smiles=smiles,
        y=df["Y"].to_numpy(dtype=int),
        descriptors=desc,
        fingerprints=fps,
        scaffolds=scaffolds,
        context=context,
        target_ids=df["target_chembl_id"].astype(str).to_numpy(dtype=object),
        assay_types=df["standard_type"].fillna("unknown").astype(str).to_numpy(dtype=object),
        bao_formats=df["bao_format"].fillna("unknown").astype(str).to_numpy(dtype=object),
        document_years=pd.to_numeric(df["document_year"], errors="coerce").to_numpy(dtype=float),
        label_rule="active if pChEMBL >= 6.0, inactive otherwise",
    )


def stratified_three_way_split(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, rest_idx = train_test_split(
        idx, test_size=0.40, random_state=seed, stratify=y
    )
    rest_y = y[rest_idx]
    cal_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, random_state=seed + 17, stratify=rest_y
    )
    return {"train": train_idx, "calibration": cal_idx, "test": test_idx}


def duplicate_aware_stratified_three_way_split(bundle: MolBundle, seed: int) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(bundle.smiles):
        groups.setdefault(canonical_smiles(smi), []).append(i)
    if len(groups) == len(bundle.y):
        return stratified_three_way_split(bundle.y, seed)

    group_keys = np.asarray(sorted(groups), dtype=object)
    group_labels = np.asarray(
        [int(np.mean(bundle.y[groups[str(key)]]) >= 0.5) for key in group_keys],
        dtype=int,
    )
    try:
        train_groups, rest_groups = train_test_split(
            group_keys,
            test_size=0.40,
            random_state=seed,
            stratify=group_labels,
        )
        rest_labels = np.asarray(
            [int(np.mean(bundle.y[groups[str(key)]]) >= 0.5) for key in rest_groups],
            dtype=int,
        )
        cal_groups, test_groups = train_test_split(
            rest_groups,
            test_size=0.50,
            random_state=seed + 17,
            stratify=rest_labels,
        )
    except ValueError:
        return stratified_three_way_split(bundle.y, seed)

    def expand(keys: np.ndarray) -> np.ndarray:
        return np.asarray([i for key in keys for i in groups[str(key)]], dtype=int)

    out = {
        "train": expand(train_groups),
        "calibration": expand(cal_groups),
        "test": expand(test_groups),
    }
    if any(len(np.unique(bundle.y[v])) < 2 for v in out.values() if len(v) > 0):
        return stratified_three_way_split(bundle.y, seed)
    return out


def purged_temporal_three_way_split(bundle: MolBundle) -> dict[str, np.ndarray]:
    """Chronological 60/20/20 split with cross-period compound purging.

    Rows are ordered by document year. Calibration compounds previously seen in
    training are removed; test compounds seen in training or retained
    calibration are removed. This preserves chronology and guarantees zero exact
    canonical-compound overlap across the three evaluation periods.
    """
    if bundle.document_years is None:
        raise ValueError("document_years are required for a temporal split")
    valid = np.isfinite(bundle.document_years)
    order = np.argsort(np.asarray(bundle.document_years[valid], dtype=float), kind="mergesort")
    idx_valid = np.where(valid)[0][order]
    n = len(idx_valid)
    train_idx = idx_valid[: int(0.60 * n)]
    cal_raw = idx_valid[int(0.60 * n) : int(0.80 * n)]
    test_raw = idx_valid[int(0.80 * n) :]

    train_ids = {canonical_smiles(bundle.smiles[int(i)]) for i in train_idx}
    cal_idx = np.asarray(
        [int(i) for i in cal_raw if canonical_smiles(bundle.smiles[int(i)]) not in train_ids],
        dtype=int,
    )
    cal_ids = {canonical_smiles(bundle.smiles[int(i)]) for i in cal_idx}
    prior_ids = train_ids.union(cal_ids)
    test_idx = np.asarray(
        [int(i) for i in test_raw if canonical_smiles(bundle.smiles[int(i)]) not in prior_ids],
        dtype=int,
    )
    out = {"train": train_idx, "calibration": cal_idx, "test": test_idx}
    if any(len(idx) < 25 or len(np.unique(bundle.y[idx])) < 2 for idx in out.values()):
        raise RuntimeError("Purged temporal split does not retain adequate class support")
    return out


def grouped_train_calibration_split(
    bundle: MolBundle,
    candidate_idx: np.ndarray,
    seed: int,
    calibration_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Split candidate rows by canonical compound identity."""
    groups: dict[str, list[int]] = {}
    for i in candidate_idx:
        groups.setdefault(canonical_smiles(bundle.smiles[int(i)]), []).append(int(i))
    group_keys = np.asarray(sorted(groups), dtype=object)
    group_labels = np.asarray(
        [int(np.mean(bundle.y[groups[str(key)]]) >= 0.5) for key in group_keys],
        dtype=int,
    )
    train_groups, cal_groups = train_test_split(
        group_keys,
        test_size=calibration_fraction,
        random_state=seed,
        stratify=group_labels,
    )
    expand = lambda keys: np.asarray([i for key in keys for i in groups[str(key)]], dtype=int)
    return expand(train_groups), expand(cal_groups)


def scaffold_split(bundle: MolBundle, seed: int) -> dict[str, np.ndarray]:
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, scaffold in enumerate(bundle.scaffolds):
        groups.setdefault(str(scaffold), []).append(i)
    scaffold_groups = list(groups.values())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=len, reverse=True)

    targets = {"train": int(0.60 * len(bundle.y)), "calibration": int(0.20 * len(bundle.y))}
    splits = {"train": [], "calibration": [], "test": []}
    for group in scaffold_groups:
        if len(splits["train"]) + len(group) <= targets["train"]:
            splits["train"].extend(group)
        elif len(splits["calibration"]) + len(group) <= targets["calibration"]:
            splits["calibration"].extend(group)
        else:
            splits["test"].extend(group)

    out = {k: np.asarray(v, dtype=int) for k, v in splits.items()}
    if any(len(np.unique(bundle.y[v])) < 2 for v in out.values() if len(v) > 0):
        return stratified_three_way_split(bundle.y, seed)
    return out


def exact_nonoverlap_indices(
    reference_smiles: list[str],
    candidate_bundle: MolBundle,
    candidate_idx: np.ndarray,
) -> np.ndarray:
    reference = {canonical_smiles(smi) for smi in reference_smiles}
    keep = [
        int(i)
        for i in candidate_idx
        if canonical_smiles(candidate_bundle.smiles[int(i)]) not in reference
    ]
    return np.asarray(keep, dtype=int)


def external_b3db_zero_overlap_test_indices(
    bbb: MolBundle,
    b3db: MolBundle,
    seed: int,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a zero-overlap external B3DB test fold.

    Exact canonical-SMILES overlaps with the full BBB_Martins corpus are
    removed unconditionally. If the filtered scaffold-derived test fold loses
    class diversity, the external scaffold partition is regenerated with a
    deterministic seed offset until both classes remain represented.
    """

    for attempt in range(max_attempts):
        ext_split = scaffold_split(b3db, seed + 101 + attempt)
        external_idx_raw = ext_split["test"]
        external_idx = exact_nonoverlap_indices(bbb.smiles, b3db, external_idx_raw)
        if len(external_idx) == 0:
            continue
        if len(np.unique(b3db.y[external_idx])) < 2:
            continue
        return external_idx_raw, external_idx, attempt
    raise RuntimeError(
        "Could not construct a zero-overlap external B3DB test fold with both classes "
        f"represented after {max_attempts} deterministic scaffold-split attempts for seed {seed}."
    )


def external_b3db_parent_nonoverlap_test_indices(
    bbb: MolBundle,
    b3db: MolBundle,
    seed: int,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build a B3DB fold excluding normalized parent-identity matches."""
    reference = {parent_inchikey_block(smi) for smi in bbb.smiles}
    for attempt in range(max_attempts):
        raw = scaffold_split(b3db, seed + 101 + attempt)["test"]
        kept = np.asarray(
            [int(i) for i in raw if parent_inchikey_block(b3db.smiles[int(i)]) not in reference],
            dtype=int,
        )
        if len(kept) and len(np.unique(b3db.y[kept])) == 2:
            return raw, kept, attempt
    raise RuntimeError(
        f"Could not construct a parent-filtered B3DB fold after {max_attempts} attempts for seed {seed}."
    )


def overlap_stats(reference_values: Iterable[object], query_values: Iterable[object]) -> dict[str, float]:
    reference = set(map(str, reference_values))
    query = list(map(str, query_values))
    if not query:
        return {"count": 0.0, "fraction": 0.0, "unique_count": 0.0}
    count = sum(1 for value in query if value in reference)
    unique_count = len(set(query).intersection(reference))
    return {
        "count": float(count),
        "fraction": float(count / len(query)),
        "unique_count": float(unique_count),
    }


def max_tanimoto_stats(reference_fp: np.ndarray, query_fp: np.ndarray, chunk_size: int = 512) -> dict[str, float | None]:
    reference = np.asarray(reference_fp, dtype=np.float32)
    query = np.asarray(query_fp, dtype=np.float32)
    if len(reference) == 0 or len(query) == 0:
        return {
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "frac_ge_0_70": None,
            "frac_ge_0_85": None,
            "frac_ge_0_95": None,
        }
    reference_sum = reference.sum(axis=1)
    max_values = []
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        intersection = chunk @ reference.T
        union = chunk.sum(axis=1, keepdims=True) + reference_sum[None, :] - intersection
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        max_values.append(np.max(similarity, axis=1))
    values = np.concatenate(max_values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "frac_ge_0_70": float(np.mean(values >= 0.70)),
        "frac_ge_0_85": float(np.mean(values >= 0.85)),
        "frac_ge_0_95": float(np.mean(values >= 0.95)),
    }


def max_tanimoto_values(reference_fp: np.ndarray, query_fp: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    reference = np.asarray(reference_fp, dtype=np.float32)
    query = np.asarray(query_fp, dtype=np.float32)
    if len(reference) == 0 or len(query) == 0:
        return np.zeros((len(query),), dtype=np.float32)
    reference_sum = reference.sum(axis=1)
    max_values = []
    for start in range(0, len(query), chunk_size):
        chunk = query[start : start + chunk_size]
        intersection = chunk @ reference.T
        union = chunk.sum(axis=1, keepdims=True) + reference_sum[None, :] - intersection
        similarity = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
        max_values.append(np.max(similarity, axis=1))
    return np.concatenate(max_values).astype(np.float32)


def fit_feature_view(
    train_bundle: MolBundle,
    train_idx: np.ndarray,
    bundles_and_indices: Iterable[tuple[MolBundle, np.ndarray]],
    mode: str,
) -> dict[str, np.ndarray]:
    train_desc = train_bundle.descriptors[train_idx]
    imputer = SimpleImputer(strategy="median").fit(train_desc)
    scaler = StandardScaler().fit(imputer.transform(train_desc))
    embedding_scaler = None
    if mode in {"chemberta", "chemberta_full"}:
        if train_bundle.foundation_embeddings is None:
            raise ValueError(f"{train_bundle.name} has no foundation embeddings")
        embedding_scaler = StandardScaler().fit(train_bundle.foundation_embeddings[train_idx])
    molformer_scaler = None
    if mode in {"molformer", "molformer_full"}:
        if train_bundle.molformer_embeddings is None:
            raise ValueError(f"{train_bundle.name} has no molformer embeddings")
        molformer_scaler = StandardScaler().fit(train_bundle.molformer_embeddings[train_idx])

    out: dict[str, np.ndarray] = {}
    for key, (bundle, idx) in bundles_and_indices:
        desc = scaler.transform(imputer.transform(bundle.descriptors[idx]))
        fps = bundle.fingerprints[idx]
        if mode == "full":
            X = np.hstack([desc, fps])
        elif mode == "fingerprint":
            X = fps
        elif mode == "descriptors":
            X = desc
        elif mode == "chemberta":
            if bundle.foundation_embeddings is None or embedding_scaler is None:
                raise ValueError(f"{bundle.name} has no foundation embeddings")
            X = embedding_scaler.transform(bundle.foundation_embeddings[idx])
        elif mode == "chemberta_full":
            if bundle.foundation_embeddings is None or embedding_scaler is None:
                raise ValueError(f"{bundle.name} has no foundation embeddings")
            emb = embedding_scaler.transform(bundle.foundation_embeddings[idx])
            X = np.hstack([desc, emb])
        elif mode == "molformer":
            if bundle.molformer_embeddings is None or molformer_scaler is None:
                raise ValueError(f"{bundle.name} has no molformer embeddings")
            X = molformer_scaler.transform(bundle.molformer_embeddings[idx])
        elif mode == "molformer_full":
            if bundle.molformer_embeddings is None or molformer_scaler is None:
                raise ValueError(f"{bundle.name} has no molformer embeddings")
            emb = molformer_scaler.transform(bundle.molformer_embeddings[idx])
            X = np.hstack([desc, emb])
        else:
            raise ValueError(f"unknown feature mode {mode}")
        out[key] = X.astype(np.float32)
    return out


def compute_or_load_chemberta_embeddings(
    bundle: MolBundle,
    model_name: str = CHEMBERTA_MODEL,
    batch_size: int = CHEMBERTA_BATCH_SIZE,
) -> np.ndarray:
    slug = model_name.replace("/", "_").replace("-", "_")
    cache_path = DATA_DIR / f"{bundle.name}_{slug}_embeddings.npy"
    meta_path = DATA_DIR / f"{bundle.name}_{slug}_embeddings.metadata.json"
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(bundle.smiles):
            return embeddings.astype(np.float32)

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(bundle.smiles), batch_size):
            batch = bundle.smiles[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
            pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            chunks.append(pooled.cpu().numpy().astype(np.float32))
            print(
                f"foundation_embedding bundle={bundle.name} model={model_name} "
                f"processed={min(start + batch_size, len(bundle.smiles))}/{len(bundle.smiles)}",
                flush=True,
            )

    embeddings = np.vstack(chunks).astype(np.float32)
    np.save(cache_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "bundle": bundle.name,
                "model": model_name,
                "n": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return embeddings


def compute_or_load_molformer_embeddings(
    bundle: MolBundle,
    model_name: str = MOLFORMER_MODEL,
    batch_size: int = MOLFORMER_BATCH_SIZE,
) -> np.ndarray:
    slug = model_name.replace("/", "_").replace("-", "_")
    cache_path = DATA_DIR / f"{bundle.name}_{slug}_embeddings.npy"
    meta_path = DATA_DIR / f"{bundle.name}_{slug}_embeddings.metadata.json"
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(bundle.smiles):
            return embeddings.astype(np.float32)

    enable_molformer_compat()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(bundle.smiles), batch_size):
            batch = bundle.smiles[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = output.pooler_output
            chunks.append(pooled.cpu().numpy().astype(np.float32))
            print(
                f"foundation_embedding bundle={bundle.name} model={model_name} "
                f"processed={min(start + batch_size, len(bundle.smiles))}/{len(bundle.smiles)}",
                flush=True,
            )

    embeddings = np.vstack(chunks).astype(np.float32)
    np.save(cache_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "bundle": bundle.name,
                "model": model_name,
                "n": int(embeddings.shape[0]),
                "dim": int(embeddings.shape[1]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return embeddings


CHEMICAL_CONTEXT_FEATURES = {
    "basic": [(0, "mw"), (1, "tpsa"), (6, "rings")],
    "rich": [(0, "mw"), (1, "tpsa"), (2, "logp"), (3, "hbd"), (4, "hba"), (6, "rings")],
}


def chemical_context_thresholds(bundle: MolBundle, train_idx: np.ndarray, scheme: str = "basic") -> dict[str, float]:
    train_desc = bundle.descriptors[train_idx]
    if scheme not in CHEMICAL_CONTEXT_FEATURES:
        raise ValueError(f"unknown chemical context scheme {scheme}")
    return {
        name: float(np.nanmedian(train_desc[:, col]))
        for col, name in CHEMICAL_CONTEXT_FEATURES[scheme]
    }


def chemical_contexts_from_thresholds(
    bundle: MolBundle,
    idx: np.ndarray,
    thresholds: dict[str, float],
    scheme: str = "basic",
) -> np.ndarray:
    if scheme not in CHEMICAL_CONTEXT_FEATURES:
        raise ValueError(f"unknown chemical context scheme {scheme}")
    contexts = []
    for row in bundle.descriptors[idx]:
        parts = []
        for col, name in CHEMICAL_CONTEXT_FEATURES[scheme]:
            cut = thresholds[name]
            value = row[col]
            level = "high" if np.isfinite(value) and np.isfinite(cut) and value >= cut else "low"
            parts.append(f"{name}_{level}")
        contexts.append("|".join(parts))
    return np.asarray(contexts, dtype=object)


def chemical_contexts(
    bundle: MolBundle,
    idx: np.ndarray,
    train_idx: np.ndarray,
    scheme: str = "basic",
) -> np.ndarray:
    return chemical_contexts_from_thresholds(
        bundle,
        idx,
        chemical_context_thresholds(bundle, train_idx, scheme),
        scheme,
    )


def finite_sample_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return 1.0
    rank = int(math.ceil((len(clean) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(clean))
    return float(np.sort(clean)[rank - 1])


def nonconformity_scores(y: np.ndarray, prob_pos: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    prob_true = np.where(y == 1, prob_pos, 1.0 - prob_pos)
    return 1.0 - prob_true


def calibrate_global(y_cal: np.ndarray, p_cal: np.ndarray) -> dict[str, object]:
    return {"global": finite_sample_quantile(nonconformity_scores(y_cal, p_cal))}


def calibrate_mondrian(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    contexts: np.ndarray,
    min_group: int = 25,
) -> dict[str, object]:
    scores = nonconformity_scores(y_cal, p_cal)
    global_q = finite_sample_quantile(scores)
    groups: dict[str, float] = {}
    for ctx in sorted(set(map(str, contexts))):
        mask = contexts.astype(str) == ctx
        if int(mask.sum()) >= min_group:
            groups[ctx] = finite_sample_quantile(scores[mask])
    return {"global": global_q, "groups": groups}


def apply_ood_guard(calibration: dict[str, object], margin: float) -> dict[str, object]:
    if margin <= 0:
        return calibration
    groups = calibration.get("groups", {}) if isinstance(calibration.get("groups", {}), dict) else {}
    guarded = {
        "global": min(1.0, float(calibration.get("global", 1.0)) + margin),
        "groups": {str(k): min(1.0, float(v) + margin) for k, v in groups.items()},
    }
    return guarded


def prediction_sets(prob_pos: np.ndarray, calibration: dict[str, object], contexts=None) -> list[set[int]]:
    groups = calibration.get("groups", {}) if isinstance(calibration.get("groups", {}), dict) else {}
    global_q = float(calibration.get("global", 1.0))
    sets: list[set[int]] = []
    if contexts is None:
        contexts = [None] * len(prob_pos)
    for p, ctx in zip(prob_pos, contexts):
        q = float(groups.get(str(ctx), global_q))
        current = set()
        if p <= q:
            current.add(0)
        if 1.0 - p <= q:
            current.add(1)
        if not current:
            current.add(int(p >= 0.5))
        sets.append(current)
    return sets


def prediction_sets_with_sample_margin(
    prob_pos: np.ndarray,
    calibration: dict[str, object],
    contexts: np.ndarray,
    apply_guard: np.ndarray,
    margin: float,
) -> list[set[int]]:
    groups = calibration.get("groups", {}) if isinstance(calibration.get("groups", {}), dict) else {}
    global_q = float(calibration.get("global", 1.0))
    sets: list[set[int]] = []
    for p, ctx, guarded in zip(prob_pos, contexts, apply_guard):
        q = float(groups.get(str(ctx), global_q))
        if guarded:
            q = min(1.0, q + margin)
        current = set()
        if p <= q:
            current.add(0)
        if 1.0 - p <= q:
            current.add(1)
        if not current:
            current.add(int(p >= 0.5))
        sets.append(current)
    return sets


def calibrate_hierarchical_mondrian(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    fine_contexts: np.ndarray,
    medium_contexts: np.ndarray,
    coarse_contexts: np.ndarray,
    min_group: int = 10,
) -> dict[str, object]:
    scores = nonconformity_scores(y_cal, p_cal)
    payload: dict[str, object] = {
        "global": finite_sample_quantile(scores),
        "fine": {},
        "medium": {},
        "coarse": {},
    }
    for key, contexts in (
        ("fine", fine_contexts),
        ("medium", medium_contexts),
        ("coarse", coarse_contexts),
    ):
        groups: dict[str, float] = {}
        for ctx in sorted(set(map(str, contexts))):
            mask = contexts.astype(str) == ctx
            if int(mask.sum()) >= min_group:
                groups[ctx] = finite_sample_quantile(scores[mask])
        payload[key] = groups
    return payload


def prediction_sets_hierarchical(
    prob_pos: np.ndarray,
    calibration: dict[str, object],
    fine_contexts: np.ndarray,
    medium_contexts: np.ndarray,
    coarse_contexts: np.ndarray,
) -> list[set[int]]:
    fine_groups = calibration.get("fine", {}) if isinstance(calibration.get("fine", {}), dict) else {}
    medium_groups = calibration.get("medium", {}) if isinstance(calibration.get("medium", {}), dict) else {}
    coarse_groups = calibration.get("coarse", {}) if isinstance(calibration.get("coarse", {}), dict) else {}
    global_q = float(calibration.get("global", 1.0))
    pred_sets: list[set[int]] = []
    for p, fine, medium, coarse in zip(prob_pos, fine_contexts, medium_contexts, coarse_contexts):
        q = fine_groups.get(str(fine))
        if q is None:
            q = medium_groups.get(str(medium))
        if q is None:
            q = coarse_groups.get(str(coarse))
        if q is None:
            q = global_q
        q = float(q)
        current = set()
        if p <= q:
            current.add(0)
        if 1.0 - p <= q:
            current.add(1)
        if not current:
            current.add(int(p >= 0.5))
        pred_sets.append(current)
    return pred_sets


def conformal_metrics(y_true: np.ndarray, pred_sets: list[set[int]]) -> dict[str, float]:
    coverage = float(np.mean([int(int(y) in s) for y, s in zip(y_true, pred_sets)]))
    mean_size = float(np.mean([len(s) for s in pred_sets]))
    singleton_correct = float(
        np.mean([int(len(s) == 1 and int(y) in s) for y, s in zip(y_true, pred_sets)])
    )
    return {
        "coverage_90": coverage,
        "mean_set_size": mean_size,
        "success_rate": singleton_correct,
    }


def ece_binary(prob_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    prob_pos = np.asarray(prob_pos, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob_pos >= lo) & (prob_pos < hi if hi < 1.0 else prob_pos <= hi)
        if not np.any(mask):
            continue
        ece += (float(mask.sum()) / total) * abs(float(prob_pos[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def point_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    out = {
        "brier_score": float(brier_score_loss(y_true, prob_pos)),
        "ece": ece_binary(prob_pos, y_true),
    }
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, prob_pos))
        out["auprc"] = float(average_precision_score(y_true, prob_pos))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def make_model(kind: str, seed: int):
    if kind == "logreg":
        return LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
    if kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=120,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=seed,
        )
    raise ValueError(kind)


def predict_positive(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        cls = int(model.classes_[0])
        return np.ones(len(X), dtype=float) if cls == 1 else np.zeros(len(X), dtype=float)
    pos_col = list(model.classes_).index(1)
    return np.asarray(proba[:, pos_col], dtype=float)


def evaluate_split(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    calibration: dict[str, object],
    contexts: np.ndarray | None = None,
    prefix: str = "",
) -> dict[str, float | None]:
    metrics = point_metrics(y_true, prob_pos)
    sets = prediction_sets(prob_pos, calibration, contexts)
    metrics.update(conformal_metrics(y_true, sets))
    if prefix:
        return {f"{prefix}_{k}": v for k, v in metrics.items()}
    return metrics


def run_bbb_condition(
    condition: str,
    model_kind: str,
    feature_mode: str,
    mondrian: bool,
    seed: int,
    bbb: MolBundle,
    b3db: MolBundle,
    guard_margin: float = 0.0,
    context_scheme: str = "basic",
    min_group: int = 25,
) -> dict[str, float | None]:
    split = duplicate_aware_stratified_three_way_split(bbb, seed)
    external_idx_raw, external_idx, partition_attempt = external_b3db_zero_overlap_test_indices(
        bbb, b3db, seed
    )
    feature_sets = fit_feature_view(
        bbb,
        split["train"],
        [
            ("train", (bbb, split["train"])),
            ("cal", (bbb, split["calibration"])),
            ("test", (bbb, split["test"])),
            ("external", (b3db, external_idx)),
        ],
        feature_mode,
    )
    model = make_model(model_kind, seed)
    model.fit(feature_sets["train"], bbb.y[split["train"]])
    p_cal = predict_positive(model, feature_sets["cal"])
    p_test = predict_positive(model, feature_sets["test"])
    p_external = predict_positive(model, feature_sets["external"])

    if mondrian:
        thresholds = chemical_context_thresholds(bbb, split["train"], context_scheme)
        cal_ctx = chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, context_scheme)
        test_ctx = chemical_contexts_from_thresholds(bbb, split["test"], thresholds, context_scheme)
        ext_ctx = chemical_contexts_from_thresholds(b3db, external_idx, thresholds, context_scheme)
        calibration = calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=min_group)
    else:
        test_ctx = None
        ext_ctx = None
        calibration = calibrate_global(bbb.y[split["calibration"]], p_cal)
    calibration = apply_ood_guard(calibration, guard_margin)

    out = evaluate_split(bbb.y[split["test"]], p_test, calibration, test_ctx)
    out.update(
        evaluate_split(
            b3db.y[external_idx],
            p_external,
            calibration,
            ext_ctx,
            prefix="ood_external",
        )
    )
    out["n_train"] = float(len(split["train"]))
    out["n_calibration"] = float(len(split["calibration"]))
    out["n_test"] = float(len(split["test"]))
    out["n_external_test"] = float(len(external_idx))
    out["n_external_test_before_overlap_filter"] = float(len(external_idx_raw))
    out["n_external_exact_overlap_removed"] = float(len(external_idx_raw) - len(external_idx))
    out["external_partition_regeneration_attempts"] = float(partition_attempt)
    out["mondrian_groups"] = float(len(calibration.get("groups", {})) if mondrian else 0)
    out["ood_guard_margin"] = float(guard_margin)
    out["context_min_group"] = float(min_group if mondrian else 0)
    out["context_scheme_basic"] = float(1 if context_scheme == "basic" else 0)
    out["context_scheme_rich"] = float(1 if context_scheme == "rich" else 0)
    return out


def run_chembl_condition(seed: int, chembl: MolBundle) -> dict[str, float | None]:
    split = duplicate_aware_stratified_three_way_split(chembl, seed)
    feature_sets = fit_feature_view(
        chembl,
        split["train"],
        [
            ("train", (chembl, split["train"])),
            ("cal", (chembl, split["calibration"])),
            ("test", (chembl, split["test"])),
        ],
        "full",
    )
    model = make_model("rf", seed)
    model.fit(feature_sets["train"], chembl.y[split["train"]])
    p_cal = predict_positive(model, feature_sets["cal"])
    p_test = predict_positive(model, feature_sets["test"])
    cal_ctx = chembl.context[split["calibration"]]
    test_ctx = chembl.context[split["test"]]
    calibration = calibrate_mondrian(chembl.y[split["calibration"]], p_cal, cal_ctx, min_group=10)
    out = evaluate_split(chembl.y[split["test"]], p_test, calibration, test_ctx)
    out["n_train"] = float(len(split["train"]))
    out["n_calibration"] = float(len(split["calibration"]))
    out["n_test"] = float(len(split["test"]))
    out["mondrian_groups"] = float(len(calibration.get("groups", {})))
    return out


def add_leakage_metrics(
    row: dict[str, float | int | None],
    prefix: str,
    reference_bundle: MolBundle,
    reference_idx: np.ndarray,
    query_bundle: MolBundle,
    query_idx: np.ndarray,
) -> None:
    reference_canonical = [canonical_smiles(reference_bundle.smiles[int(i)]) for i in reference_idx]
    query_canonical = [canonical_smiles(query_bundle.smiles[int(i)]) for i in query_idx]
    exact = overlap_stats(reference_canonical, query_canonical)
    scaffold = overlap_stats(reference_bundle.scaffolds[reference_idx], query_bundle.scaffolds[query_idx])
    tanimoto = max_tanimoto_stats(reference_bundle.fingerprints[reference_idx], query_bundle.fingerprints[query_idx])

    for key, value in exact.items():
        row[f"{prefix}_exact_smiles_{key}"] = value
    for key, value in scaffold.items():
        row[f"{prefix}_scaffold_{key}"] = value
    for key, value in tanimoto.items():
        row[f"{prefix}_max_tanimoto_{key}"] = value
    row[f"{prefix}_n_query"] = float(len(query_idx))


def run_leakage_audit(bbb: MolBundle, b3db: MolBundle) -> dict[str, object]:
    seed_metrics = []
    all_bbb_idx = np.arange(len(bbb.y), dtype=int)
    for seed in SEEDS:
        split = duplicate_aware_stratified_three_way_split(bbb, seed)
        external_idx_raw, external_idx, partition_attempt = external_b3db_zero_overlap_test_indices(
            bbb, b3db, seed
        )

        row: dict[str, float | int | None] = {"seed": seed}
        add_leakage_metrics(row, "internal_calibration_vs_train", bbb, split["train"], bbb, split["calibration"])
        add_leakage_metrics(row, "internal_test_vs_train", bbb, split["train"], bbb, split["test"])
        add_leakage_metrics(row, "external_b3db_test_vs_train", bbb, split["train"], b3db, external_idx)
        add_leakage_metrics(row, "external_b3db_test_vs_all_bbb", bbb, all_bbb_idx, b3db, external_idx)
        row["external_b3db_test_size_before_overlap_filter"] = float(len(external_idx_raw))
        row["external_b3db_test_size_after_overlap_filter"] = float(len(external_idx))
        row["external_b3db_exact_overlap_removed"] = float(len(external_idx_raw) - len(external_idx))
        row["external_b3db_test_bbb_pos_before_overlap_filter"] = float(np.sum(b3db.y[external_idx_raw] == 1))
        row["external_b3db_test_bbb_neg_before_overlap_filter"] = float(np.sum(b3db.y[external_idx_raw] == 0))
        row["external_b3db_test_bbb_pos_after_overlap_filter"] = float(np.sum(b3db.y[external_idx] == 1))
        row["external_b3db_test_bbb_neg_after_overlap_filter"] = float(np.sum(b3db.y[external_idx] == 0))
        row["external_b3db_partition_regeneration_attempts"] = float(partition_attempt)
        seed_metrics.append({k: safe_number(v) for k, v in row.items()})

    return {
        "seed_metrics": seed_metrics,
        "summary": summarize(seed_metrics),
        "policy": (
            "B3DB external test folds remove exact canonical-SMILES overlaps with the full BBB_Martins "
            "corpus unconditionally. If a filtered scaffold-derived fold loses class diversity, the "
            "external scaffold partition is regenerated deterministically until both classes remain "
            "represented. Internal splits are audited for exact-SMILES, Murcko-scaffold, and Morgan-"
            "fingerprint maximum Tanimoto overlap against training rows."
        ),
    }


def run_delta_sensitivity(bbb: MolBundle, b3db: MolBundle) -> dict[str, object]:
    rows_by_delta: dict[str, list[dict[str, float | None]]] = {f"delta_{delta:.2f}": [] for delta in DELTA_GRID}
    seed_metrics = []
    for seed in SEEDS:
        split = duplicate_aware_stratified_three_way_split(bbb, seed)
        _, external_idx, _ = external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
        feature_sets = fit_feature_view(
            bbb,
            split["train"],
            [
                ("train", (bbb, split["train"])),
                ("cal", (bbb, split["calibration"])),
                ("external", (b3db, external_idx)),
            ],
            "full",
        )
        model = make_model("rf", seed)
        model.fit(feature_sets["train"], bbb.y[split["train"]])
        p_cal = predict_positive(model, feature_sets["cal"])
        p_external = predict_positive(model, feature_sets["external"])
        thresholds = chemical_context_thresholds(bbb, split["train"], "basic")
        cal_ctx = chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, "basic")
        ext_ctx = chemical_contexts_from_thresholds(b3db, external_idx, thresholds, "basic")
        base_calibration = calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=25)
        for delta in DELTA_GRID:
            calibration = apply_ood_guard(base_calibration, delta)
            metrics = evaluate_split(
                b3db.y[external_idx],
                p_external,
                calibration,
                ext_ctx,
                prefix="ood_external",
            )
            metrics = {k: safe_number(v) for k, v in metrics.items()}
            metrics["delta"] = float(delta)
            metrics["seed"] = float(seed)
            metrics["mondrian_groups"] = float(len(base_calibration.get("groups", {})))
            rows_by_delta[f"delta_{delta:.2f}"].append(metrics)
            seed_metrics.append(metrics)

    summary_by_delta = {key: summarize(rows) for key, rows in rows_by_delta.items()}
    candidates = []
    for delta in DELTA_GRID:
        key = f"delta_{delta:.2f}"
        coverage = summary_by_delta[key].get("ood_external_coverage_90", {}).get("mean")
        mean_size = summary_by_delta[key].get("ood_external_mean_set_size", {}).get("mean")
        success = summary_by_delta[key].get("ood_external_success_rate", {}).get("mean")
        if coverage is not None:
            candidates.append(
                {
                    "delta": float(delta),
                    "coverage_90": float(coverage),
                    "mean_set_size": None if mean_size is None else float(mean_size),
                    "success_rate": None if success is None else float(success),
                }
            )
    feasible = [row for row in candidates if row["coverage_90"] >= 0.90]
    selected = min(feasible, key=lambda row: row["delta"]) if feasible else max(candidates, key=lambda row: row["coverage_90"])
    return {
        "seed_metrics": seed_metrics,
        "summary_by_delta": summary_by_delta,
        "selection_rule": (
            "Descriptive operating-point screen over the reported delta grid. The highlighted blanket "
            "guard is the smallest tested margin whose mean zero-overlap external B3DB coverage "
            "exceeded 0.90 in this dataset."
        ),
        "selected_delta": selected,
    }


def run_similarity_triggered_guard(bbb: MolBundle, b3db: MolBundle) -> dict[str, object]:
    rows_by_quantile = {f"quantile_{quantile:.2f}": [] for quantile in SELECTIVE_GUARD_QUANTILES}
    seed_metrics = []
    for seed in SEEDS:
        split = duplicate_aware_stratified_three_way_split(bbb, seed)
        _, external_idx, _ = external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
        feature_sets = fit_feature_view(
            bbb,
            split["train"],
            [
                ("train", (bbb, split["train"])),
                ("cal", (bbb, split["calibration"])),
                ("external", (b3db, external_idx)),
            ],
            "full",
        )
        model = make_model("rf", seed)
        model.fit(feature_sets["train"], bbb.y[split["train"]])
        p_cal = predict_positive(model, feature_sets["cal"])
        p_external = predict_positive(model, feature_sets["external"])
        thresholds = chemical_context_thresholds(bbb, split["train"], "basic")
        cal_ctx = chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, "basic")
        ext_ctx = chemical_contexts_from_thresholds(b3db, external_idx, thresholds, "basic")
        base_calibration = calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=25)
        cal_sim = max_tanimoto_values(bbb.fingerprints[split["train"]], bbb.fingerprints[split["calibration"]])
        ext_sim = max_tanimoto_values(bbb.fingerprints[split["train"]], b3db.fingerprints[external_idx])
        for quantile in SELECTIVE_GUARD_QUANTILES:
            threshold = float(np.quantile(cal_sim, quantile))
            guard_mask = ext_sim <= threshold
            pred_sets = prediction_sets_with_sample_margin(
                p_external,
                base_calibration,
                ext_ctx,
                guard_mask,
                SELECTIVE_GUARD_MARGIN,
            )
            metrics = conformal_metrics(b3db.y[external_idx], pred_sets)
            metrics = {f"ood_external_{key}": safe_number(value) for key, value in metrics.items()}
            metrics["seed"] = float(seed)
            metrics["quantile"] = float(quantile)
            metrics["similarity_threshold"] = threshold
            metrics["guarded_fraction"] = float(np.mean(guard_mask))
            metrics["guard_margin"] = float(SELECTIVE_GUARD_MARGIN)
            rows_by_quantile[f"quantile_{quantile:.2f}"].append(metrics)
            seed_metrics.append(metrics)

    summary_by_quantile = {key: summarize(rows) for key, rows in rows_by_quantile.items()}
    candidates = []
    for quantile in SELECTIVE_GUARD_QUANTILES:
        key = f"quantile_{quantile:.2f}"
        coverage = summary_by_quantile[key].get("ood_external_coverage_90", {}).get("mean")
        mean_size = summary_by_quantile[key].get("ood_external_mean_set_size", {}).get("mean")
        success = summary_by_quantile[key].get("ood_external_success_rate", {}).get("mean")
        guarded = summary_by_quantile[key].get("guarded_fraction", {}).get("mean")
        if coverage is not None and mean_size is not None and success is not None and guarded is not None:
            utility = float(success) - 0.25 * (float(mean_size) - 1.0)
            candidates.append(
                {
                    "quantile": float(quantile),
                    "coverage_90": float(coverage),
                    "mean_set_size": float(mean_size),
                    "success_rate": float(success),
                    "guarded_fraction": float(guarded),
                    "utility": utility,
                }
            )
    selected = max(candidates, key=lambda row: row["utility"]) if candidates else None
    return {
        "seed_metrics": seed_metrics,
        "summary_by_quantile": summary_by_quantile,
        "selection_rule": (
            "Retrospective utility screen over calibration-only similarity-quantile triggers; "
            "utility = success_rate - 0.25 * (mean_set_size - 1). Thresholds are derived from the "
            "calibration fold max-train-similarity distribution only."
        ),
        "selected_policy": selected,
    }


def run_context_sensitivity(bbb: MolBundle, b3db: MolBundle) -> dict[str, object]:
    rows_by_scheme: dict[str, list[dict[str, float | None]]] = {
        f"{scheme}_min{min_group}": []
        for scheme in CONTEXT_SCHEMES
        for min_group in CONTEXT_MIN_GROUP_GRID
    }
    seed_metrics = []
    for seed in SEEDS:
        split = duplicate_aware_stratified_three_way_split(bbb, seed)
        _, external_idx, _ = external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
        feature_sets = fit_feature_view(
            bbb,
            split["train"],
            [
                ("train", (bbb, split["train"])),
                ("cal", (bbb, split["calibration"])),
                ("external", (b3db, external_idx)),
            ],
            "full",
        )
        model = make_model("rf", seed)
        model.fit(feature_sets["train"], bbb.y[split["train"]])
        p_cal = predict_positive(model, feature_sets["cal"])
        p_external = predict_positive(model, feature_sets["external"])
        for scheme in CONTEXT_SCHEMES:
            thresholds = chemical_context_thresholds(bbb, split["train"], scheme)
            cal_ctx = chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, scheme)
            ext_ctx = chemical_contexts_from_thresholds(b3db, external_idx, thresholds, scheme)
            for min_group in CONTEXT_MIN_GROUP_GRID:
                calibration = calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=min_group)
                metrics = evaluate_split(
                    b3db.y[external_idx],
                    p_external,
                    calibration,
                    ext_ctx,
                    prefix="ood_external",
                )
                metrics = {k: safe_number(v) for k, v in metrics.items()}
                metrics["seed"] = float(seed)
                metrics["context_min_group"] = float(min_group)
                metrics["context_scheme_basic"] = float(1 if scheme == "basic" else 0)
                metrics["context_scheme_rich"] = float(1 if scheme == "rich" else 0)
                metrics["mondrian_groups"] = float(len(calibration.get("groups", {})))
                key = f"{scheme}_min{min_group}"
                rows_by_scheme[key].append(metrics)
                seed_metrics.append(metrics)
    return {
        "seed_metrics": seed_metrics,
        "summary_by_scheme": {key: summarize(rows) for key, rows in rows_by_scheme.items()},
    }


def run_chembl_target_holdout(chembl: MolBundle) -> dict[str, object]:
    contexts = np.asarray(list(map(str, chembl.context)), dtype=object)
    targets = np.asarray([ctx.split("|", 1)[0] for ctx in contexts], dtype=object)
    assay_types = np.asarray([ctx.split("|", 1)[1] if "|" in ctx else "unknown" for ctx in contexts], dtype=object)
    targets_to_holdout = sorted(set(map(str, targets)))
    rows_by_target: dict[str, list[dict[str, float | None]]] = {target: [] for target in targets_to_holdout}
    pooled_rows = []

    for seed in SEEDS:
        for target in targets_to_holdout:
            test_idx = np.where(targets == target)[0]
            heldout_ids = {canonical_smiles(chembl.smiles[int(i)]) for i in test_idx}
            traincal_idx = np.asarray(
                [
                    int(i)
                    for i in np.where(targets != target)[0]
                    if canonical_smiles(chembl.smiles[int(i)]) not in heldout_ids
                ],
                dtype=int,
            )
            if len(test_idx) < 10 or len(traincal_idx) < 20 or len(np.unique(chembl.y[traincal_idx])) < 2:
                continue
            try:
                train_idx, cal_idx = grouped_train_calibration_split(chembl, traincal_idx, seed)
            except ValueError:
                continue
            feature_sets = fit_feature_view(
                chembl,
                train_idx,
                [
                    ("train", (chembl, train_idx)),
                    ("cal", (chembl, cal_idx)),
                    ("test", (chembl, test_idx)),
                ],
                "full",
            )
            model = make_model("rf", seed)
            model.fit(feature_sets["train"], chembl.y[train_idx])
            p_cal = predict_positive(model, feature_sets["cal"])
            p_test = predict_positive(model, feature_sets["test"])
            calibration = calibrate_mondrian(chembl.y[cal_idx], p_cal, assay_types[cal_idx], min_group=10)
            metrics = evaluate_split(chembl.y[test_idx], p_test, calibration, assay_types[test_idx])
            metrics = {k: safe_number(v) for k, v in metrics.items()}
            metrics["seed"] = float(seed)
            metrics["n_train"] = float(len(train_idx))
            metrics["n_calibration"] = float(len(cal_idx))
            metrics["n_test"] = float(len(test_idx))
            metrics["mondrian_groups"] = float(len(calibration.get("groups", {})))
            rows_by_target[target].append(metrics)
            pooled_rows.append({**metrics, "heldout_target": target})

    return {
        "targets": targets_to_holdout,
        "summary": summarize(pooled_rows),
        "summary_by_target": {target: summarize(rows) for target, rows in rows_by_target.items() if rows},
        "seed_metrics": pooled_rows,
    }


def run_chembl_target_holdout_molformer(chembl: MolBundle) -> dict[str, object] | None:
    if chembl.molformer_embeddings is None or chembl.assay_types is None or chembl.target_ids is None:
        return None
    targets = np.asarray(list(map(str, chembl.target_ids)), dtype=object)
    assay_types = np.asarray(list(map(str, chembl.assay_types)), dtype=object)
    targets_to_holdout = sorted(set(map(str, targets)))
    rows_by_target: dict[str, list[dict[str, float | None]]] = {target: [] for target in targets_to_holdout}
    pooled_rows = []

    for seed in SEEDS:
        for target in targets_to_holdout:
            test_idx = np.where(targets == target)[0]
            heldout_ids = {canonical_smiles(chembl.smiles[int(i)]) for i in test_idx}
            traincal_idx = np.asarray(
                [
                    int(i)
                    for i in np.where(targets != target)[0]
                    if canonical_smiles(chembl.smiles[int(i)]) not in heldout_ids
                ],
                dtype=int,
            )
            if len(test_idx) < 10 or len(traincal_idx) < 20 or len(np.unique(chembl.y[traincal_idx])) < 2:
                continue
            try:
                train_idx, cal_idx = grouped_train_calibration_split(chembl, traincal_idx, seed)
            except ValueError:
                continue
            feature_sets = fit_feature_view(
                chembl,
                train_idx,
                [
                    ("train", (chembl, train_idx)),
                    ("cal", (chembl, cal_idx)),
                    ("test", (chembl, test_idx)),
                ],
                "molformer",
            )
            model = make_model("logreg", seed)
            model.fit(feature_sets["train"], chembl.y[train_idx])
            p_cal = predict_positive(model, feature_sets["cal"])
            p_test = predict_positive(model, feature_sets["test"])
            calibration = calibrate_mondrian(chembl.y[cal_idx], p_cal, assay_types[cal_idx], min_group=10)
            metrics = evaluate_split(chembl.y[test_idx], p_test, calibration, assay_types[test_idx])
            metrics = {k: safe_number(v) for k, v in metrics.items()}
            metrics["seed"] = float(seed)
            metrics["n_train"] = float(len(train_idx))
            metrics["n_calibration"] = float(len(cal_idx))
            metrics["n_test"] = float(len(test_idx))
            metrics["mondrian_groups"] = float(len(calibration.get("groups", {})))
            rows_by_target[target].append(metrics)
            pooled_rows.append({**metrics, "heldout_target": target})

    return {
        "targets": targets_to_holdout,
        "summary": summarize(pooled_rows),
        "summary_by_target": {target: summarize(rows) for target, rows in rows_by_target.items() if rows},
        "seed_metrics": pooled_rows,
    }


def run_metadata_dropout_sensitivity(chembl: MolBundle) -> dict[str, object] | None:
    if chembl.context is None or chembl.assay_types is None:
        return None
    rows_by_rate: dict[str, list[dict[str, float | None]]] = {
        f"dropout_{rate:.2f}": [] for rate in METADATA_DROPOUT_RATES
    }
    seed_metrics = []
    for seed in SEEDS:
        split = duplicate_aware_stratified_three_way_split(chembl, seed)
        feature_sets = fit_feature_view(
            chembl,
            split["train"],
            [
                ("train", (chembl, split["train"])),
                ("cal", (chembl, split["calibration"])),
                ("test", (chembl, split["test"])),
            ],
            "full",
        )
        model = make_model("rf", seed)
        model.fit(feature_sets["train"], chembl.y[split["train"]])
        p_cal = predict_positive(model, feature_sets["cal"])
        p_test = predict_positive(model, feature_sets["test"])
        base_cal_ctx = chembl.context[split["calibration"]].astype(object).copy()
        base_test_ctx = chembl.context[split["test"]].astype(object).copy()
        coarse_cal_ctx = chembl.assay_types[split["calibration"]].astype(object)
        coarse_test_ctx = chembl.assay_types[split["test"]].astype(object)
        for rate in METADATA_DROPOUT_RATES:
            if rate > 0:
                rng = np.random.default_rng(seed * 1000 + int(rate * 100))
                cal_mask = rng.random(len(base_cal_ctx)) < rate
                test_mask = rng.random(len(base_test_ctx)) < rate
                cal_ctx = base_cal_ctx.copy()
                test_ctx = base_test_ctx.copy()
                cal_ctx[cal_mask] = coarse_cal_ctx[cal_mask]
                test_ctx[test_mask] = coarse_test_ctx[test_mask]
            else:
                cal_ctx = base_cal_ctx
                test_ctx = base_test_ctx
            calibration = calibrate_mondrian(chembl.y[split["calibration"]], p_cal, cal_ctx, min_group=10)
            metrics = evaluate_split(chembl.y[split["test"]], p_test, calibration, test_ctx)
            metrics = {k: safe_number(v) for k, v in metrics.items()}
            metrics["seed"] = float(seed)
            metrics["dropout_rate"] = float(rate)
            metrics["mondrian_groups"] = float(len(calibration.get("groups", {})))
            rows_by_rate[f"dropout_{rate:.2f}"].append(metrics)
            seed_metrics.append(metrics)
    return {
        "seed_metrics": seed_metrics,
        "summary_by_rate": {key: summarize(rows) for key, rows in rows_by_rate.items()},
    }


def run_chembl_grouping_sensitivity(chembl: MolBundle) -> dict[str, object] | None:
    if chembl.target_ids is None or chembl.assay_types is None or chembl.bao_formats is None:
        return None
    rows_by_scheme = {
        "assay_only": [],
        "target_assay": [],
        "hierarchical_target_assay_bao": [],
    }
    seed_metrics = []
    for seed in SEEDS:
        split = duplicate_aware_stratified_three_way_split(chembl, seed)
        feature_sets = fit_feature_view(
            chembl,
            split["train"],
            [
                ("train", (chembl, split["train"])),
                ("cal", (chembl, split["calibration"])),
                ("test", (chembl, split["test"])),
            ],
            "full",
        )
        model = make_model("rf", seed)
        model.fit(feature_sets["train"], chembl.y[split["train"]])
        p_cal = predict_positive(model, feature_sets["cal"])
        p_test = predict_positive(model, feature_sets["test"])
        coarse_cal = chembl.assay_types[split["calibration"]].astype(object)
        coarse_test = chembl.assay_types[split["test"]].astype(object)
        medium_cal = (
            chembl.target_ids[split["calibration"]].astype(str)
            + "|"
            + chembl.assay_types[split["calibration"]].astype(str)
        ).astype(object)
        medium_test = (
            chembl.target_ids[split["test"]].astype(str)
            + "|"
            + chembl.assay_types[split["test"]].astype(str)
        ).astype(object)
        fine_cal = (
            chembl.target_ids[split["calibration"]].astype(str)
            + "|"
            + chembl.assay_types[split["calibration"]].astype(str)
            + "|"
            + chembl.bao_formats[split["calibration"]].astype(str)
        ).astype(object)
        fine_test = (
            chembl.target_ids[split["test"]].astype(str)
            + "|"
            + chembl.assay_types[split["test"]].astype(str)
            + "|"
            + chembl.bao_formats[split["test"]].astype(str)
        ).astype(object)

        calibration = calibrate_mondrian(chembl.y[split["calibration"]], p_cal, coarse_cal, min_group=10)
        metrics = evaluate_split(chembl.y[split["test"]], p_test, calibration, coarse_test)
        metrics = {k: safe_number(v) for k, v in metrics.items()}
        metrics["seed"] = float(seed)
        metrics["mondrian_groups"] = float(len(calibration.get("groups", {})))
        rows_by_scheme["assay_only"].append(metrics)
        seed_metrics.append({**metrics, "scheme_assay_only": 1.0, "scheme_target_assay": 0.0, "scheme_hierarchical": 0.0})

        calibration = calibrate_mondrian(chembl.y[split["calibration"]], p_cal, medium_cal, min_group=10)
        metrics = evaluate_split(chembl.y[split["test"]], p_test, calibration, medium_test)
        metrics = {k: safe_number(v) for k, v in metrics.items()}
        metrics["seed"] = float(seed)
        metrics["mondrian_groups"] = float(len(calibration.get("groups", {})))
        rows_by_scheme["target_assay"].append(metrics)
        seed_metrics.append({**metrics, "scheme_assay_only": 0.0, "scheme_target_assay": 1.0, "scheme_hierarchical": 0.0})

        hier_calibration = calibrate_hierarchical_mondrian(
            chembl.y[split["calibration"]],
            p_cal,
            fine_cal,
            medium_cal,
            coarse_cal,
            min_group=10,
        )
        pred_sets = prediction_sets_hierarchical(
            p_test,
            hier_calibration,
            fine_test,
            medium_test,
            coarse_test,
        )
        metrics = point_metrics(chembl.y[split["test"]], p_test)
        metrics.update(conformal_metrics(chembl.y[split["test"]], pred_sets))
        metrics = {k: safe_number(v) for k, v in metrics.items()}
        metrics["seed"] = float(seed)
        metrics["mondrian_groups"] = float(
            len(hier_calibration.get("fine", {}))
            + len(hier_calibration.get("medium", {}))
            + len(hier_calibration.get("coarse", {}))
        )
        rows_by_scheme["hierarchical_target_assay_bao"].append(metrics)
        seed_metrics.append({**metrics, "scheme_assay_only": 0.0, "scheme_target_assay": 0.0, "scheme_hierarchical": 1.0})

    return {
        "seed_metrics": seed_metrics,
        "summary_by_scheme": {key: summarize(rows) for key, rows in rows_by_scheme.items()},
    }


def run_chembl_temporal_validation(chembl: MolBundle, feature_mode: str = "full", model_kind: str = "rf") -> dict[str, object] | None:
    if chembl.document_years is None or chembl.assay_types is None:
        return None
    valid = np.isfinite(chembl.document_years)
    if int(valid.sum()) < 50:
        return None
    split = purged_temporal_three_way_split(chembl)
    train_idx = split["train"]
    cal_idx = split["calibration"]
    test_idx = split["test"]
    feature_sets = fit_feature_view(
        chembl,
        train_idx,
        [
            ("train", (chembl, train_idx)),
            ("cal", (chembl, cal_idx)),
            ("test", (chembl, test_idx)),
        ],
        feature_mode,
    )
    model = make_model(model_kind, seed=42)
    model.fit(feature_sets["train"], chembl.y[train_idx])
    p_cal = predict_positive(model, feature_sets["cal"])
    p_test = predict_positive(model, feature_sets["test"])
    calibration = calibrate_mondrian(chembl.y[cal_idx], p_cal, chembl.assay_types[cal_idx], min_group=10)
    metrics = evaluate_split(chembl.y[test_idx], p_test, calibration, chembl.assay_types[test_idx])
    metrics = {k: safe_number(v) for k, v in metrics.items()}
    metrics["n_train"] = float(len(train_idx))
    metrics["n_calibration"] = float(len(cal_idx))
    metrics["n_test"] = float(len(test_idx))
    metrics["train_year_max"] = float(np.nanmax(chembl.document_years[train_idx]))
    metrics["calibration_year_min"] = float(np.nanmin(chembl.document_years[cal_idx]))
    metrics["calibration_year_max"] = float(np.nanmax(chembl.document_years[cal_idx]))
    metrics["test_year_min"] = float(np.nanmin(chembl.document_years[test_idx]))
    metrics["test_year_max"] = float(np.nanmax(chembl.document_years[test_idx]))
    metrics["mondrian_groups"] = float(len(calibration.get("groups", {})))
    return {
        "feature_mode": feature_mode,
        "model_kind": model_kind,
        "summary": {key: {"mean": value, "std": 0.0} for key, value in metrics.items()},
        "seed_metrics": [metrics],
    }


def safe_number(value):
    if value is None:
        return None
    if isinstance(value, (np.floating, np.integer)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def summarize(seed_metrics: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    keys = sorted({k for row in seed_metrics for k in row})
    summary: dict[str, dict[str, float | None]] = {}
    for key in keys:
        vals = []
        for row in seed_metrics:
            value = row.get(key)
            if value is None:
                continue
            try:
                vals.append(float(value))
            except (TypeError, ValueError):
                continue
        if not vals:
            summary[key] = {"mean": None, "std": None}
            continue
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        }
    return summary


def print_seed_metrics(condition: str, seed: int, metrics: dict[str, float | None]) -> None:
    for key, value in sorted(metrics.items()):
        value = safe_number(value)
        if value is None:
            continue
        print(f"condition={condition} seed={seed} {key}: {value:.6f}", flush=True)


def print_summaries(condition: str, summary: dict[str, dict[str, float | None]]) -> None:
    for key, vals in sorted(summary.items()):
        mean = vals.get("mean")
        std = vals.get("std")
        if mean is None or std is None:
            continue
        print(f"SUMMARY condition={condition} metric={key} mean={mean:.6f} std={std:.6f}", flush=True)


def paired_line(results: dict[str, dict], method: str, baseline: str, metric: str) -> str | None:
    method_vals = [row.get(metric) for row in results[method]["seed_metrics"]]
    base_vals = [row.get(metric) for row in results[baseline]["seed_metrics"]]
    pairs = [(float(a), float(b)) for a, b in zip(method_vals, base_vals) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    diff = np.asarray([a - b for a, b in pairs], dtype=float)
    t_stat, p_value = stats.ttest_1samp(diff, 0.0)
    p_value_print = max(float(p_value), 1e-6)
    mean = float(np.mean(diff))
    std = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    se = std / math.sqrt(len(diff)) if len(diff) > 1 else 0.0
    ci = stats.t.interval(0.95, len(diff) - 1, loc=mean, scale=se) if len(diff) > 1 else (mean, mean)
    return (
        f"PAIRED: {method} vs {baseline} regime=external_ood metric={metric} "
        f"mean_diff={mean:.6f} std_diff={std:.6f} t_stat={float(t_stat):.6f} "
        f"p_value={p_value_print:.6f} ci95=({float(ci[0]):.6f},{float(ci[1]):.6f})"
    )


def paired_record(
    results: dict[str, dict], method: str, baseline: str, metric: str
) -> dict[str, float | int | str | None] | None:
    method_vals = [row.get(metric) for row in results[method]["seed_metrics"]]
    base_vals = [row.get(metric) for row in results[baseline]["seed_metrics"]]
    pairs = [(float(a), float(b)) for a, b in zip(method_vals, base_vals) if a is not None and b is not None]
    if len(pairs) < 2:
        return None
    diff = np.asarray([a - b for a, b in pairs], dtype=float)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(len(diff))
    low, high = stats.t.interval(0.95, len(diff) - 1, loc=mean, scale=se)
    test = stats.ttest_1samp(diff, 0.0)
    return {
        "method": method,
        "comparator": baseline,
        "metric": metric,
        "n_pairs": len(diff),
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "t_statistic": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
        "bh_q_value": None,
    }


def add_bh_q_values(records: list[dict[str, float | int | str | None]]) -> None:
    """Adjust the external-AUROC comparison family in place."""
    family = [i for i, row in enumerate(records) if row.get("metric") == "ood_external_auroc"]
    if not family:
        return
    p = np.asarray([float(records[i]["p_value_two_sided"]) for i in family], dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 1.0
    for rank_index in range(len(p) - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(p[original_index]) * len(p) / rank)
        adjusted[original_index] = min(running, 1.0)
    for position, q_value in zip(family, adjusted):
        records[position]["bh_q_value"] = float(q_value)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    bbb = load_bbb_tab(DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = load_bbb_tab(DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    chembl = load_chembl(DATA_DIR / "chembl_cns_targets.csv")

    foundation_status: dict[str, object] = {"model": CHEMBERTA_MODEL, "status": "not_attempted"}
    try:
        bbb.foundation_embeddings = compute_or_load_chemberta_embeddings(bbb)
        b3db.foundation_embeddings = compute_or_load_chemberta_embeddings(b3db)
        foundation_status = {
            "model": CHEMBERTA_MODEL,
            "status": "available",
            "bbb_shape": list(map(int, bbb.foundation_embeddings.shape)),
            "b3db_shape": list(map(int, b3db.foundation_embeddings.shape)),
        }
    except Exception as exc:
        foundation_status = {
            "model": CHEMBERTA_MODEL,
            "status": "unavailable",
            "reason": repr(exc),
        }
        print(f"foundation_baseline_unavailable model={CHEMBERTA_MODEL} reason={exc!r}", flush=True)
    molformer_status: dict[str, object] = {"model": MOLFORMER_MODEL, "status": "not_attempted"}
    try:
        bbb.molformer_embeddings = compute_or_load_molformer_embeddings(bbb)
        b3db.molformer_embeddings = compute_or_load_molformer_embeddings(b3db)
        chembl.molformer_embeddings = compute_or_load_molformer_embeddings(chembl)
        molformer_status = {
            "model": MOLFORMER_MODEL,
            "status": "available",
            "bbb_shape": list(map(int, bbb.molformer_embeddings.shape)),
            "b3db_shape": list(map(int, b3db.molformer_embeddings.shape)),
            "chembl_shape": list(map(int, chembl.molformer_embeddings.shape)),
        }
    except Exception as exc:
        molformer_status = {
            "model": MOLFORMER_MODEL,
            "status": "unavailable",
            "reason": repr(exc),
        }
        print(f"foundation_baseline_unavailable model={MOLFORMER_MODEL} reason={exc!r}", flush=True)

    conditions = {
        "logreg_full_global_cp": ("bbb", "logreg", "full", False, 0.0, "basic", 25),
        "rf_morgan_global_cp": ("bbb", "rf", "fingerprint", False, 0.0, "basic", 25),
        "mlp_morgan_global_cp": ("bbb", "mlp", "fingerprint", False, 0.0, "basic", 25),
        "neurotx_mood_rf_context_cp": ("bbb", "rf", "full", True, 0.0, "basic", 25),
        "neurotx_mood_rf_rich_context_cp": ("bbb", "rf", "full", True, 0.0, "rich", 25),
        "neurotx_mood_rf_context_ood_guard_cp": ("bbb", "rf", "full", True, OOD_GUARD_MARGIN, "basic", 25),
        "neurotx_mood_descriptors_only_context_cp": ("bbb", "logreg", "descriptors", True, 0.0, "basic", 25),
        "chembl_target_context_rf": ("chembl", "rf", "full", True, 0.0, "target", 10),
    }
    if foundation_status["status"] == "available":
        conditions.update(
            {
                "chemberta_frozen_logreg_global_cp": ("bbb", "logreg", "chemberta", False, 0.0, "basic", 25),
                "chemberta_frozen_logreg_context_cp": ("bbb", "logreg", "chemberta", True, 0.0, "basic", 25),
            }
        )
    if molformer_status["status"] == "available":
        conditions.update(
            {
                "molformer_frozen_logreg_global_cp": ("bbb", "logreg", "molformer", False, 0.0, "basic", 25),
                "molformer_frozen_logreg_context_cp": ("bbb", "logreg", "molformer", True, 0.0, "basic", 25),
            }
        )
    results: dict[str, dict] = {}
    for condition, spec in conditions.items():
        seed_metrics = []
        for seed in SEEDS:
            if spec[0] == "chembl":
                metrics = run_chembl_condition(seed, chembl)
            else:
                _, model_kind, feature_mode, mondrian, guard_margin, context_scheme, min_group = spec
                metrics = run_bbb_condition(
                    condition,
                    model_kind,
                    feature_mode,
                    mondrian,
                    seed,
                    bbb,
                    b3db,
                    guard_margin=guard_margin,
                    context_scheme=context_scheme,
                    min_group=min_group,
                )
            metrics = {k: safe_number(v) for k, v in metrics.items()}
            if condition == "neurotx_mood_rf_context_ood_guard_cp":
                # The guard changes only conformal set thresholds. Reporting the
                # same point-prediction AUROC twice makes this look like a broken
                # ablation, so keep only calibration-efficiency metrics here.
                metrics = {
                    k: v for k, v in metrics.items()
                    if k not in CALIBRATION_ONLY_POINT_METRICS
                }
            seed_metrics.append(metrics)
            print_seed_metrics(condition, seed, metrics)
        summary = summarize(seed_metrics)
        print_summaries(condition, summary)
        results[condition] = {
            "seeds_run": len(seed_metrics),
            "seed_metrics": seed_metrics,
            "summary": summary,
        }

    paired = []
    paired_specs = [
        ("neurotx_mood_rf_context_cp", "rf_morgan_global_cp", "ood_external_auroc"),
        ("neurotx_mood_rf_context_cp", "rf_morgan_global_cp", "ood_external_coverage_90"),
        ("neurotx_mood_rf_context_cp", "rf_morgan_global_cp", "ood_external_mean_set_size"),
        ("neurotx_mood_rf_context_cp", "mlp_morgan_global_cp", "ood_external_auroc"),
        ("neurotx_mood_rf_context_cp", "mlp_morgan_global_cp", "ood_external_coverage_90"),
        ("neurotx_mood_rf_rich_context_cp", "neurotx_mood_rf_context_cp", "ood_external_coverage_90"),
        ("neurotx_mood_rf_rich_context_cp", "neurotx_mood_rf_context_cp", "ood_external_mean_set_size"),
        ("chemberta_frozen_logreg_context_cp", "chemberta_frozen_logreg_global_cp", "ood_external_coverage_90"),
        ("neurotx_mood_rf_context_cp", "chemberta_frozen_logreg_global_cp", "ood_external_auroc"),
        ("neurotx_mood_rf_context_ood_guard_cp", "rf_morgan_global_cp", "ood_external_coverage_90"),
        ("neurotx_mood_rf_context_ood_guard_cp", "rf_morgan_global_cp", "ood_external_mean_set_size"),
        ("neurotx_mood_rf_context_ood_guard_cp", "neurotx_mood_rf_context_cp", "ood_external_coverage_90"),
        ("neurotx_mood_rf_context_ood_guard_cp", "neurotx_mood_rf_context_cp", "ood_external_mean_set_size"),
    ]
    if "molformer_frozen_logreg_global_cp" in results:
        paired_specs.extend(
            [
                ("molformer_frozen_logreg_context_cp", "molformer_frozen_logreg_global_cp", "ood_external_coverage_90"),
                ("neurotx_mood_rf_context_cp", "molformer_frozen_logreg_global_cp", "ood_external_auroc"),
                ("neurotx_mood_rf_context_cp", "molformer_frozen_logreg_context_cp", "ood_external_auroc"),
            ]
        )
    for method, baseline, metric in paired_specs:
        if method not in results or baseline not in results:
            continue
        record = paired_record(results, method, baseline, metric)
        if record:
            paired.append(record)
            print(f"PAIRED_JSON: {json.dumps(record, sort_keys=True)}", flush=True)
    add_bh_q_values(paired)

    proposed_external_auroc = results["neurotx_mood_rf_context_cp"]["summary"]["ood_external_auroc"]["mean"]
    if proposed_external_auroc is not None:
        print(f"auroc: {proposed_external_auroc:.6f}", flush=True)

    print("running_formal_leakage_audit", flush=True)
    leakage_audit = run_leakage_audit(bbb, b3db)
    print("running_delta_sensitivity", flush=True)
    delta_sensitivity = run_delta_sensitivity(bbb, b3db)
    print("running_similarity_triggered_guard_analysis", flush=True)
    selective_guard = run_similarity_triggered_guard(bbb, b3db)
    print("running_context_grouping_sensitivity", flush=True)
    context_grouping_sensitivity = run_context_sensitivity(bbb, b3db)
    print("running_chembl_target_holdout_validation", flush=True)
    chembl_target_holdout = run_chembl_target_holdout(chembl)
    print("running_chembl_target_holdout_molformer_validation", flush=True)
    chembl_target_holdout_molformer = run_chembl_target_holdout_molformer(chembl)
    print("running_chembl_temporal_validation", flush=True)
    chembl_temporal_validation = run_chembl_temporal_validation(chembl, feature_mode="full", model_kind="rf")
    print("running_chembl_temporal_molformer_validation", flush=True)
    chembl_temporal_validation_molformer = run_chembl_temporal_validation(
        chembl,
        feature_mode="molformer",
        model_kind="logreg",
    ) if molformer_status["status"] == "available" else None
    print("running_metadata_dropout_sensitivity", flush=True)
    metadata_dropout_sensitivity = run_metadata_dropout_sensitivity(chembl)
    print("running_chembl_grouping_sensitivity", flush=True)
    chembl_grouping_sensitivity = run_chembl_grouping_sensitivity(chembl)

    payload = {
        "study": "NeuroTx-MOOD",
        "alpha": ALPHA,
        "seeds": SEEDS,
        "foundation_baseline_status": {
            "chemberta": foundation_status,
            "molformer": molformer_status,
        },
        "leakage_audit": leakage_audit,
        "delta_sensitivity": delta_sensitivity,
        "selective_guard": selective_guard,
        "context_grouping_sensitivity": context_grouping_sensitivity,
        "external_validations": {
            "chembl_target_holdout": chembl_target_holdout,
            "chembl_target_holdout_molformer": chembl_target_holdout_molformer,
            "chembl_temporal_validation": chembl_temporal_validation,
            "chembl_temporal_validation_molformer": chembl_temporal_validation_molformer,
            "metadata_dropout_sensitivity": metadata_dropout_sensitivity,
            "chembl_grouping_sensitivity": chembl_grouping_sensitivity,
        },
        "data_sources": {
            "BBB_Martins": {
                "path": "data/bbb_martins.tab",
                "sha256": sha256_file(DATA_DIR / "bbb_martins.tab"),
                "n": int(len(bbb.y)),
                "positive_rate": float(np.mean(bbb.y)),
                "label_rule": bbb.label_rule,
            },
            "B3DB_Classification": {
                "path": "data/b3db_classification.tab",
                "sha256": sha256_file(DATA_DIR / "b3db_classification.tab"),
                "n": int(len(b3db.y)),
                "positive_rate": float(np.mean(b3db.y)),
                "label_rule": b3db.label_rule,
            },
            "ChEMBL_CNS_targets": {
                "path": "data/chembl_cns_targets.csv",
                "sha256": sha256_file(DATA_DIR / "chembl_cns_targets.csv"),
                "n": int(len(chembl.y)),
                "positive_rate": float(np.mean(chembl.y)),
                "label_rule": chembl.label_rule,
                "contexts": sorted(map(str, set(chembl.context))),
            },
        },
        "conditions": results,
        "paired_comparisons": paired,
        "primary_metric": {
            "name": "external B3DB AUROC for NeuroTx-MOOD context-aware conformal RF",
            "metric_key": "ood_external_auroc",
            "value": proposed_external_auroc,
        },
    }
    (Path(__file__).resolve().parent / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
