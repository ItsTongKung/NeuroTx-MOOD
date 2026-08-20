"""Applicability-region and local conformal probes for the BBB study.

These supplementary analyses are designed to address methodological audit questions
about external undercoverage under strong chemical shift without changing the
main NeuroTx-MOOD benchmark.

The script adds two diagnostics:

1. Applicability-region summaries for the default RF + proxy-context conformal
   model, using calibration-derived similarity thresholds and retained-context
   support to define progressively stricter deployment regions.
2. A calibration-only local similarity-aware conformal baseline that replaces
   the proxy-context quantile with a k-nearest-calibration quantile while
   keeping the RF point predictor fixed.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RDLogger.DisableLog("rdApp.*")

ALPHA = 0.10
SEEDS = list(range(42, 52))
FP_BITS = 1024
MIN_GROUP = 25
LOCAL_K = 75
OUTPUT_PATH = Path(__file__).resolve().parent / "applicability_region_probes.json"


def _find_data_dir() -> Path:
    configured = os.environ.get("NEUROTX_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent / "data"


DATA_DIR = _find_data_dir()


@dataclass
class MolBundle:
    name: str
    path: Path
    smiles: list[str]
    y: np.ndarray
    descriptors: np.ndarray
    fingerprints: np.ndarray
    scaffolds: np.ndarray


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return str(smiles)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def scaffold_smiles(mol) -> str:
    if mol is None:
        return "invalid"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return scaffold or "acyclic"
    except Exception:
        return "invalid"


def compute_features(smiles: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    descriptors = []
    fingerprints = []
    scaffolds = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
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
    )


def stratified_three_way_split(y: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, rest_idx = train_test_split(idx, test_size=0.40, random_state=seed, stratify=y)
    cal_idx, test_idx = train_test_split(
        rest_idx,
        test_size=0.50,
        random_state=seed + 17,
        stratify=y[rest_idx],
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

    return {
        "train": expand(train_groups),
        "calibration": expand(cal_groups),
        "test": expand(test_groups),
    }


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

    return {k: np.asarray(v, dtype=int) for k, v in splits.items()}


def exact_nonoverlap_indices(reference_smiles: list[str], candidate_bundle: MolBundle, candidate_idx: np.ndarray) -> np.ndarray:
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


def fit_full_features(
    train_bundle: MolBundle,
    train_idx: np.ndarray,
    targets: list[tuple[str, MolBundle, np.ndarray]],
) -> dict[str, np.ndarray]:
    train_desc = train_bundle.descriptors[train_idx]
    imputer = SimpleImputer(strategy="median").fit(train_desc)
    scaler = StandardScaler().fit(imputer.transform(train_desc))
    out: dict[str, np.ndarray] = {}
    for key, bundle, idx in targets:
        desc = scaler.transform(imputer.transform(bundle.descriptors[idx]))
        fps = bundle.fingerprints[idx]
        out[key] = np.hstack([desc, fps]).astype(np.float32)
    return out


def make_rf(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def predict_positive(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        cls = int(model.classes_[0])
        return np.ones(len(X), dtype=float) if cls == 1 else np.zeros(len(X), dtype=float)
    pos_col = list(model.classes_).index(1)
    return np.asarray(proba[:, pos_col], dtype=float)


def finite_sample_quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return 1.0
    rank = int(math.ceil((len(clean) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(clean))
    return float(np.sort(clean)[rank - 1])


def nonconformity_scores(y: np.ndarray, prob_pos: np.ndarray) -> np.ndarray:
    prob_true = np.where(np.asarray(y, dtype=int) == 1, prob_pos, 1.0 - prob_pos)
    return 1.0 - np.asarray(prob_true, dtype=float)


def calibrate_global(y_cal: np.ndarray, p_cal: np.ndarray) -> dict[str, object]:
    return {"global": finite_sample_quantile(nonconformity_scores(y_cal, p_cal))}


def chemical_context_thresholds(bundle: MolBundle, train_idx: np.ndarray) -> dict[str, float]:
    train_desc = bundle.descriptors[train_idx]
    return {
        "mw": float(np.nanmedian(train_desc[:, 0])),
        "tpsa": float(np.nanmedian(train_desc[:, 1])),
        "rings": float(np.nanmedian(train_desc[:, 6])),
    }


def chemical_contexts_from_thresholds(bundle: MolBundle, idx: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    contexts = []
    for row in bundle.descriptors[idx]:
        parts = []
        for col, name in ((0, "mw"), (1, "tpsa"), (6, "rings")):
            cut = thresholds[name]
            value = row[col]
            level = "high" if np.isfinite(value) and np.isfinite(cut) and value >= cut else "low"
            parts.append(f"{name}_{level}")
        contexts.append("|".join(parts))
    return np.asarray(contexts, dtype=object)


def calibrate_mondrian(y_cal: np.ndarray, p_cal: np.ndarray, contexts: np.ndarray, min_group: int = MIN_GROUP) -> dict[str, object]:
    scores = nonconformity_scores(y_cal, p_cal)
    global_q = finite_sample_quantile(scores)
    groups: dict[str, float] = {}
    for ctx in sorted(set(map(str, contexts))):
        mask = contexts.astype(str) == ctx
        if int(mask.sum()) >= min_group:
            groups[ctx] = finite_sample_quantile(scores[mask])
    return {"global": global_q, "groups": groups}


def prediction_sets(prob_pos: np.ndarray, calibration: dict[str, object], contexts: np.ndarray | None = None) -> list[set[int]]:
    groups = calibration.get("groups", {}) if isinstance(calibration.get("groups", {}), dict) else {}
    global_q = float(calibration.get("global", 1.0))
    if contexts is None:
        contexts = np.asarray([None] * len(prob_pos), dtype=object)
    sets: list[set[int]] = []
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


def prediction_sets_local_similarity(
    prob_pos: np.ndarray,
    cal_scores: np.ndarray,
    sim_matrix: np.ndarray,
    k_neighbors: int = LOCAL_K,
) -> list[set[int]]:
    k = min(int(k_neighbors), int(cal_scores.shape[0]))
    sets: list[set[int]] = []
    for p, sim_row in zip(prob_pos, sim_matrix):
        local_idx = np.argpartition(sim_row, -k)[-k:]
        q = finite_sample_quantile(cal_scores[local_idx])
        current = set()
        if p <= q:
            current.add(0)
        if 1.0 - p <= q:
            current.add(1)
        if not current:
            current.add(int(p >= 0.5))
        sets.append(current)
    return sets


def conformal_metrics(y_true: np.ndarray, pred_sets: list[set[int]]) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    coverage = float(np.mean([int(int(y) in s) for y, s in zip(y_true, pred_sets)]))
    mean_size = float(np.mean([len(s) for s in pred_sets]))
    singleton_correct = float(
        np.mean([int(len(s) == 1 and int(y) in s) for y, s in zip(y_true, pred_sets)])
    )
    singleton_rate = float(np.mean([int(len(s) == 1) for s in pred_sets]))
    return {
        "coverage_90": coverage,
        "mean_set_size": mean_size,
        "singleton_success": singleton_correct,
        "singleton_rate": singleton_rate,
    }


def point_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=int)
    prob_pos = np.asarray(prob_pos, dtype=float)
    out: dict[str, float | None] = {
        "brier_score": float(brier_score_loss(y_true, prob_pos)),
    }
    if len(np.unique(y_true)) >= 2:
        out["auroc"] = float(roc_auc_score(y_true, prob_pos))
        out["auprc"] = float(average_precision_score(y_true, prob_pos))
    else:
        out["auroc"] = None
        out["auprc"] = None
    return out


def tanimoto_matrix(query_fp: np.ndarray, reference_fp: np.ndarray) -> np.ndarray:
    query = np.asarray(query_fp, dtype=np.float32)
    reference = np.asarray(reference_fp, dtype=np.float32)
    if len(query) == 0 or len(reference) == 0:
        return np.zeros((len(query), len(reference)), dtype=np.float32)
    intersection = query @ reference.T
    union = query.sum(axis=1, keepdims=True) + reference.sum(axis=1)[None, :] - intersection
    similarity = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )
    return similarity.astype(np.float32)


def max_tanimoto_values(reference_fp: np.ndarray, query_fp: np.ndarray) -> np.ndarray:
    sims = tanimoto_matrix(query_fp, reference_fp)
    if sims.size == 0:
        return np.zeros((len(query_fp),), dtype=np.float32)
    return np.max(sims, axis=1).astype(np.float32)


def summarize(rows: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    keys = sorted({key for row in rows for key in row if key != "seed"})
    out: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            out[key] = {"mean": None, "std": None}
            continue
        arr = np.asarray(values, dtype=float)
        out[key] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out


def _region_metrics(y_true: np.ndarray, pred_sets: list[set[int]], mask: np.ndarray) -> dict[str, float | None]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {
            "support_fraction": 0.0,
            "n_region": 0.0,
            "coverage_90": None,
            "mean_set_size": None,
            "singleton_success": None,
            "singleton_rate": None,
        }
    subset_sets = [pred_sets[i] for i in np.where(mask)[0]]
    out = conformal_metrics(y_true[mask], subset_sets)
    out["support_fraction"] = float(np.mean(mask))
    out["n_region"] = float(int(mask.sum()))
    return out


def _external_setup(seed: int, bbb: MolBundle, b3db: MolBundle):
    split = duplicate_aware_stratified_three_way_split(bbb, seed)
    _, external_idx, _ = external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
    return split, external_idx


def run_applicability_region_probe(bbb: MolBundle, b3db: MolBundle) -> dict[str, object]:
    region_rows: dict[str, list[dict[str, float | None]]] = {
        "all_external": [],
        "non_fallback_context": [],
        "fallback_context": [],
        "sim_ge_cal_q50": [],
        "sim_ge_cal_q75": [],
        "sim_ge_cal_q90": [],
        "non_fallback_and_sim_ge_cal_q75": [],
        "non_fallback_and_sim_ge_cal_q90": [],
    }
    seed_metrics = []
    for seed in SEEDS:
        print(f"applicability_region seed={seed}", flush=True)
        split, external_idx = _external_setup(seed, bbb, b3db)
        features = fit_full_features(
            bbb,
            split["train"],
            [
                ("train", bbb, split["train"]),
                ("cal", bbb, split["calibration"]),
                ("external", b3db, external_idx),
            ],
        )
        model = make_rf(seed)
        model.fit(features["train"], bbb.y[split["train"]])
        p_cal = predict_positive(model, features["cal"])
        p_external = predict_positive(model, features["external"])

        thresholds = chemical_context_thresholds(bbb, split["train"])
        cal_ctx = chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds)
        ext_ctx = chemical_contexts_from_thresholds(b3db, external_idx, thresholds)
        calibration = calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=MIN_GROUP)
        pred_sets = prediction_sets(p_external, calibration, ext_ctx)

        retained_groups = set(map(str, calibration.get("groups", {}).keys()))
        non_fallback_mask = np.asarray([str(ctx) in retained_groups for ctx in ext_ctx], dtype=bool)

        sim_cal = max_tanimoto_values(bbb.fingerprints[split["train"]], bbb.fingerprints[split["calibration"]])
        sim_external = max_tanimoto_values(bbb.fingerprints[split["train"]], b3db.fingerprints[external_idx])
        q50, q75, q90 = np.quantile(sim_cal, [0.50, 0.75, 0.90])

        regions = {
            "all_external": np.ones(len(external_idx), dtype=bool),
            "non_fallback_context": non_fallback_mask,
            "fallback_context": ~non_fallback_mask,
            "sim_ge_cal_q50": sim_external >= q50,
            "sim_ge_cal_q75": sim_external >= q75,
            "sim_ge_cal_q90": sim_external >= q90,
            "non_fallback_and_sim_ge_cal_q75": non_fallback_mask & (sim_external >= q75),
            "non_fallback_and_sim_ge_cal_q90": non_fallback_mask & (sim_external >= q90),
        }
        for region_name, mask in regions.items():
            metrics = _region_metrics(b3db.y[external_idx], pred_sets, mask)
            metrics.update(
                {
                    "seed": float(seed),
                    "similarity_threshold": {
                        "sim_ge_cal_q50": float(q50),
                        "sim_ge_cal_q75": float(q75),
                        "sim_ge_cal_q90": float(q90),
                    }.get(region_name),
                }
            )
            cleaned = {
                key: (float(value) if isinstance(value, (int, float, np.floating)) and value is not None else value)
                for key, value in metrics.items()
            }
            region_rows[region_name].append(cleaned)
            seed_metrics.append({**cleaned, "region": region_name})
    return {
        "description": (
            "Default NeuroTx-MOOD proxy-context conformal model summarized over "
            "progressively stricter deployment regions defined by retained-context "
            "support and calibration-derived similarity thresholds."
        ),
        "summary_by_region": {key: summarize(rows) for key, rows in region_rows.items()},
        "seed_metrics": seed_metrics,
    }


def run_local_similarity_baseline(bbb: MolBundle, b3db: MolBundle) -> dict[str, object]:
    rows_by_scheme: dict[str, list[dict[str, float | None]]] = {
        "global_full_rf": [],
        "proxy_basic": [],
        "local_similarity_knn75": [],
    }
    seed_metrics = []
    for seed in SEEDS:
        print(f"local_similarity seed={seed}", flush=True)
        split, external_idx = _external_setup(seed, bbb, b3db)
        features = fit_full_features(
            bbb,
            split["train"],
            [
                ("train", bbb, split["train"]),
                ("cal", bbb, split["calibration"]),
                ("external", b3db, external_idx),
            ],
        )
        model = make_rf(seed)
        model.fit(features["train"], bbb.y[split["train"]])
        p_cal = predict_positive(model, features["cal"])
        p_external = predict_positive(model, features["external"])
        base_point = point_metrics(b3db.y[external_idx], p_external)

        global_cal = calibrate_global(bbb.y[split["calibration"]], p_cal)
        global_sets = prediction_sets(p_external, global_cal)
        global_metrics = {**base_point, **conformal_metrics(b3db.y[external_idx], global_sets), "seed": float(seed)}
        rows_by_scheme["global_full_rf"].append(global_metrics)
        seed_metrics.append({**global_metrics, "scheme": "global_full_rf"})

        thresholds = chemical_context_thresholds(bbb, split["train"])
        cal_ctx = chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds)
        ext_ctx = chemical_contexts_from_thresholds(b3db, external_idx, thresholds)
        proxy_cal = calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=MIN_GROUP)
        proxy_sets = prediction_sets(p_external, proxy_cal, ext_ctx)
        proxy_metrics = {
            **base_point,
            **conformal_metrics(b3db.y[external_idx], proxy_sets),
            "seed": float(seed),
            "external_fallback_fraction": float(np.mean([str(ctx) not in proxy_cal["groups"] for ctx in ext_ctx])),
        }
        rows_by_scheme["proxy_basic"].append(proxy_metrics)
        seed_metrics.append({**proxy_metrics, "scheme": "proxy_basic"})

        cal_scores = nonconformity_scores(bbb.y[split["calibration"]], p_cal)
        sim_ext_to_cal = tanimoto_matrix(b3db.fingerprints[external_idx], bbb.fingerprints[split["calibration"]])
        local_sets = prediction_sets_local_similarity(
            p_external,
            cal_scores,
            sim_ext_to_cal,
            k_neighbors=LOCAL_K,
        )
        local_metrics = {
            **base_point,
            **conformal_metrics(b3db.y[external_idx], local_sets),
            "seed": float(seed),
            "k_neighbors": float(LOCAL_K),
        }
        rows_by_scheme["local_similarity_knn75"].append(local_metrics)
        seed_metrics.append({**local_metrics, "scheme": "local_similarity_knn75"})

    return {
        "description": (
            "Calibration-only comparison for a fixed descriptor-plus-fingerprint RF "
            "point predictor: global split conformal, default proxy-context conformal, "
            "and a local similarity-aware conformal baseline using the 75 nearest "
            "calibration molecules per external sample."
        ),
        "summary_by_scheme": {key: summarize(rows) for key, rows in rows_by_scheme.items()},
        "seed_metrics": seed_metrics,
    }


def main() -> None:
    bbb = load_bbb_tab(DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = load_bbb_tab(DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    payload = {
        "generated_from": "applicability_region_probes.py",
        "data_dir": "data",
        "applicability_region_probe": run_applicability_region_probe(bbb, b3db),
        "local_similarity_conformal_baseline": run_local_similarity_baseline(bbb, b3db),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
