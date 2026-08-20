"""B3DB metadata-context calibration experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import main as exp


OUTPUT_PATH = Path(__file__).resolve().parent / "b3db_metadata_probe.json"
OFFICIAL_B3DB_SHA256 = "47a160aea1551423ead2aab70b301fb1646aea03cf151fd614eb37c78fa55b82"


class _Bundle:
    pass


def _canonical_smiles(smiles: str) -> str:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol is not None else str(smiles)
    except Exception:
        return str(smiles)


def _ref_bucket(n_refs: int) -> str:
    if n_refs <= 1:
        return "ref1"
    if n_refs <= 3:
        return "ref2_3"
    return "ref4p"


def _ref_bucket_rich(n_refs: int) -> str:
    if n_refs <= 1:
        return "ref1"
    if n_refs <= 3:
        return "ref2_3"
    if n_refs <= 6:
        return "ref4_6"
    if n_refs <= 10:
        return "ref7_10"
    return "ref11p"


def _load_enriched_b3db() -> _Bundle:
    local_path = exp.DATA_DIR / "b3db_classification.tab"
    official_path = exp.DATA_DIR / "B3DB_official_classification.tsv"
    if not local_path.exists() or not official_path.exists():
        raise FileNotFoundError("Run `python prepare_data.py` before the B3DB metadata probe.")
    official_hash = hashlib.sha256(official_path.read_bytes()).hexdigest()
    if official_hash != OFFICIAL_B3DB_SHA256:
        raise ValueError(
            "B3DB official snapshot checksum mismatch; rerun prepare_data.py to obtain the complete pinned file."
        )
    local = pd.read_csv(local_path, sep="\t")
    official = pd.read_csv(official_path, sep="\t")
    local["canon"] = local["Drug"].map(_canonical_smiles)
    official["canon"] = official["SMILES"].map(_canonical_smiles)
    official["reference_count"] = official["reference"].fillna("").astype(str).map(
        lambda s: len([part for part in s.split("|") if part])
    )
    official["ref_bucket"] = official["reference_count"].map(_ref_bucket)
    official["ref_bucket_rich"] = official["reference_count"].map(_ref_bucket_rich)
    official["has_logbb"] = pd.to_numeric(official["logBB"], errors="coerce").notna().astype(int)
    official["group_clean"] = official["group"].fillna("unknown").astype(str).str.strip().replace({"": "unknown"})
    official["metadata_context"] = (
        official["group_clean"]
        + "|"
        + official["ref_bucket"]
        + "|logbb_"
        + official["has_logbb"].astype(str)
    )
    official["metadata_context_rich"] = (
        official["group_clean"]
        + "|"
        + official["ref_bucket_rich"]
        + "|logbb_"
        + official["has_logbb"].astype(str)
    )
    merged = local.merge(
        official[
            [
                "canon",
                "BBB+/BBB-",
                "comments",
                "group_clean",
                "has_logbb",
                "metadata_context",
                "metadata_context_rich",
                "ref_bucket",
                "ref_bucket_rich",
                "reference",
                "reference_count",
                "threshold",
            ]
        ],
        on="canon",
        how="left",
    )
    merged = merged.dropna(subset=["metadata_context"]).copy()
    merged["Y"] = pd.to_numeric(merged["Y"], errors="coerce").astype(int)
    desc, fps, scaff = exp.compute_features(merged["Drug"].astype(str).tolist())
    bundle = _Bundle()
    bundle.name = "B3DB_Metadata"
    bundle.path = exp.DATA_DIR / "B3DB_official_classification.tsv"
    bundle.smiles = merged["Drug"].astype(str).tolist()
    bundle.y = merged["Y"].to_numpy(dtype=int)
    bundle.descriptors = desc
    bundle.fingerprints = fps
    bundle.scaffolds = scaff
    bundle.context = merged["metadata_context"].astype(str).to_numpy(dtype=object)
    bundle.context_rich = merged["metadata_context_rich"].astype(str).to_numpy(dtype=object)
    bundle.merged = merged
    return bundle


def _fit_full_view(bundle: _Bundle, train_idx: np.ndarray, idx_map: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    imputer = SimpleImputer(strategy="median").fit(bundle.descriptors[train_idx])
    scaler = StandardScaler().fit(imputer.transform(bundle.descriptors[train_idx]))
    out: dict[str, np.ndarray] = {}
    for key, idx in idx_map.items():
        desc = scaler.transform(imputer.transform(bundle.descriptors[idx]))
        out[key] = np.hstack([desc, bundle.fingerprints[idx]]).astype(np.float32)
    return out


def _paired_effect(
    method_rows: list[dict[str, float]],
    baseline_rows: list[dict[str, float]],
    metric: str,
) -> dict[str, float | int]:
    """Summarize a seed-paired method-minus-global contrast with a t interval."""
    diff = np.asarray(
        [float(method[metric]) - float(baseline[metric]) for method, baseline in zip(method_rows, baseline_rows)],
        dtype=float,
    )
    if diff.size == 0:
        raise ValueError(f"No paired values available for metric {metric!r}")
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    se = sd / float(np.sqrt(diff.size))
    if diff.size > 1 and se > 0.0:
        low, high = stats.t.interval(0.95, diff.size - 1, loc=mean, scale=se)
    else:
        low, high = mean, mean
    return {
        "mean_difference": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "sd_difference": sd,
        "n_pairs": int(diff.size),
    }


def main() -> None:
    bundle = _load_enriched_b3db()
    rows_global = []
    rows_metadata = []
    rows_metadata_rich = []
    rows_chemistry = []

    for seed in exp.SEEDS:
        split = exp.scaffold_split(bundle, seed)
        if any(len(np.unique(bundle.y[idx])) < 2 for idx in split.values()):
            split = exp.stratified_three_way_split(bundle.y, seed)
        view = _fit_full_view(
            bundle,
            split["train"],
            {
                "train": split["train"],
                "cal": split["calibration"],
                "test": split["test"],
            },
        )
        model = RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(view["train"], bundle.y[split["train"]])
        p_cal = exp.predict_positive(model, view["cal"])
        p_test = exp.predict_positive(model, view["test"])

        global_cal = exp.calibrate_global(bundle.y[split["calibration"]], p_cal)
        rows_global.append(exp.evaluate_split(bundle.y[split["test"]], p_test, global_cal))

        meta_cal = exp.calibrate_mondrian(
            bundle.y[split["calibration"]],
            p_cal,
            bundle.context[split["calibration"]],
            min_group=25,
        )
        meta_metrics = exp.evaluate_split(
            bundle.y[split["test"]],
            p_test,
            meta_cal,
            bundle.context[split["test"]],
        )
        meta_metrics["mondrian_groups"] = float(len(meta_cal.get("groups", {})))
        rows_metadata.append(meta_metrics)

        rich_meta_cal = exp.calibrate_mondrian(
            bundle.y[split["calibration"]],
            p_cal,
            bundle.context_rich[split["calibration"]],
            min_group=25,
        )
        rich_meta_metrics = exp.evaluate_split(
            bundle.y[split["test"]],
            p_test,
            rich_meta_cal,
            bundle.context_rich[split["test"]],
        )
        rich_meta_metrics["mondrian_groups"] = float(len(rich_meta_cal.get("groups", {})))
        rows_metadata_rich.append(rich_meta_metrics)

        thresholds = exp.chemical_context_thresholds(bundle, split["train"], "basic")
        cal_ctx = exp.chemical_contexts_from_thresholds(bundle, split["calibration"], thresholds, "basic")
        test_ctx = exp.chemical_contexts_from_thresholds(bundle, split["test"], thresholds, "basic")
        chem_cal = exp.calibrate_mondrian(
            bundle.y[split["calibration"]],
            p_cal,
            cal_ctx,
            min_group=25,
        )
        chem_metrics = exp.evaluate_split(bundle.y[split["test"]], p_test, chem_cal, test_ctx)
        chem_metrics["mondrian_groups"] = float(len(chem_cal.get("groups", {})))
        rows_chemistry.append(chem_metrics)

    merged = bundle.merged
    output = {
        "description": "Fixed-score B3DB scaffold-split calibration probe using complete source metadata from the pinned official B3DB snapshot.",
        "generated_from": "b3db_metadata_probe.py",
        "official_b3db_sha256": OFFICIAL_B3DB_SHA256,
        "n_rows": int(len(bundle.y)),
        "n_contexts": int(len(set(bundle.context))),
        "n_contexts_rich": int(len(set(bundle.context_rich))),
        "context_counts_top10": merged["metadata_context"].value_counts().head(10).to_dict(),
        "context_counts_top10_rich": merged["metadata_context_rich"].value_counts().head(10).to_dict(),
        "global_cp": exp.summarize(rows_global),
        "metadata_context_cp": exp.summarize(rows_metadata),
        "metadata_context_rich_cp": exp.summarize(rows_metadata_rich),
        "chemistry_proxy_cp": exp.summarize(rows_chemistry),
        "paired_effects_vs_global_cp": {
            "chemistry_proxy_cp": {
                metric: _paired_effect(rows_chemistry, rows_global, metric)
                for metric in ("coverage_90", "mean_set_size", "success_rate")
            },
            "metadata_context_cp": {
                metric: _paired_effect(rows_metadata, rows_global, metric)
                for metric in ("coverage_90", "mean_set_size", "success_rate")
            },
            "metadata_context_rich_cp": {
                metric: _paired_effect(rows_metadata_rich, rows_global, metric)
                for metric in ("coverage_90", "mean_set_size", "success_rate")
            },
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
