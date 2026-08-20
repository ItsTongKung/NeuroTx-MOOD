"""Compound-controlled leakage, sensitivity, and paired-statistics analyses.

The output contains only public logical file names and numerical
results. It does not serialize local filesystem paths.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

import main as nt


HERE = Path(__file__).resolve().parent


def summarize(rows: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    return nt.summarize(rows)


def bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjustment with monotonicity enforcement."""
    n = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for rank_index in range(n - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(p_values[original_index]) * n / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def paired_result(method_values: list[float], comparator_values: list[float]) -> dict[str, float | int]:
    diff = np.asarray(method_values, dtype=float) - np.asarray(comparator_values, dtype=float)
    mean = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    se = sd / math.sqrt(len(diff))
    low, high = stats.t.interval(0.95, len(diff) - 1, loc=mean, scale=se)
    test = stats.ttest_1samp(diff, 0.0)
    return {
        "n_pairs": int(len(diff)),
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "t_statistic": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
    }


def load_comparison_rows() -> dict[str, list[dict[str, float]]]:
    main_results = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    advanced = json.loads((HERE / "advanced_baselines.json").read_text(encoding="utf-8"))
    chemberta = json.loads((HERE / "fine_tuned_chemberta_baseline.json").read_text(encoding="utf-8"))
    molformer = json.loads((HERE / "fine_tuned_molformer_baseline.json").read_text(encoding="utf-8"))
    chemprop = json.loads((HERE / "chemprop_bbb_baseline.json").read_text(encoding="utf-8"))
    return {
        "NeuroTx-MOOD RF context CP": main_results["conditions"]["neurotx_mood_rf_context_cp"]["seed_metrics"],
        "RF Morgan global CP": main_results["conditions"]["rf_morgan_global_cp"]["seed_metrics"],
        "Chemprop D-MPNN": chemprop["seed_metrics"],
        "fine-tuned ChemBERTa context CP": chemberta["conditions"]["chemberta_finetuned_context_cp"]["seed_metrics"],
        "validation-tuned GNN context CP": advanced["conditions"]["gnn_message_passing_context_cp"]["seed_metrics"],
        "fine-tuned MoLFormer global CP": molformer["conditions"]["molformer_finetuned_global_cp"]["seed_metrics"],
    }


def representation_statistics() -> list[dict[str, float | int | str]]:
    rows = load_comparison_rows()
    proposed = [float(row["ood_external_auroc"]) for row in rows["NeuroTx-MOOD RF context CP"]]
    results: list[dict[str, float | int | str]] = []
    for name, comparator_rows in rows.items():
        if name == "NeuroTx-MOOD RF context CP":
            continue
        comparator = [float(row["ood_external_auroc"]) for row in comparator_rows]
        result: dict[str, float | int | str] = {
            "method": "NeuroTx-MOOD RF context CP",
            "comparator": name,
            "metric": "external B3DB AUROC",
            **paired_result(proposed, comparator),
        }
        results.append(result)
    q_values = bh_adjust([float(row["p_value_two_sided"]) for row in results])
    for row, q_value in zip(results, q_values):
        row["bh_q_value"] = q_value
    return results


def parent_filtered_external_indices(
    bbb: nt.MolBundle,
    b3db: nt.MolBundle,
    seed: int,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray, int]:
    reference = {nt.parent_inchikey_block(smi) for smi in bbb.smiles}
    for attempt in range(max_attempts):
        raw = nt.scaffold_split(b3db, seed + 101 + attempt)["test"]
        kept = np.asarray(
            [int(i) for i in raw if nt.parent_inchikey_block(b3db.smiles[int(i)]) not in reference],
            dtype=int,
        )
        if len(kept) and len(np.unique(b3db.y[kept])) == 2:
            return raw, kept, attempt
    raise RuntimeError(f"No class-valid parent-filtered B3DB fold for seed {seed}")


def run_parent_filtered_condition(
    bbb: nt.MolBundle,
    b3db: nt.MolBundle,
    seed: int,
    feature_mode: str,
    mondrian: bool,
) -> dict[str, float | None]:
    split = nt.duplicate_aware_stratified_three_way_split(bbb, seed)
    raw_idx, external_idx, attempt = parent_filtered_external_indices(bbb, b3db, seed)
    features = nt.fit_feature_view(
        bbb,
        split["train"],
        [
            ("train", (bbb, split["train"])),
            ("cal", (bbb, split["calibration"])),
            ("external", (b3db, external_idx)),
        ],
        feature_mode,
    )
    model = nt.make_model("rf", seed)
    model.fit(features["train"], bbb.y[split["train"]])
    p_cal = nt.predict_positive(model, features["cal"])
    p_external = nt.predict_positive(model, features["external"])
    if mondrian:
        thresholds = nt.chemical_context_thresholds(bbb, split["train"], "basic")
        cal_ctx = nt.chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, "basic")
        ext_ctx = nt.chemical_contexts_from_thresholds(b3db, external_idx, thresholds, "basic")
        calibration = nt.calibrate_mondrian(bbb.y[split["calibration"]], p_cal, cal_ctx, min_group=25)
    else:
        ext_ctx = None
        calibration = nt.calibrate_global(bbb.y[split["calibration"]], p_cal)
    metrics = nt.evaluate_split(b3db.y[external_idx], p_external, calibration, ext_ctx, prefix="ood_external")
    metrics.update(
        {
            "seed": float(seed),
            "n_external_before_parent_filter": float(len(raw_idx)),
            "n_external_after_parent_filter": float(len(external_idx)),
            "n_parent_identity_matches_removed": float(len(raw_idx) - len(external_idx)),
            "partition_regeneration_attempts": float(attempt),
        }
    )
    return {key: nt.safe_number(value) for key, value in metrics.items()}


def compound_overlap_audit(bundle: nt.MolBundle) -> list[dict[str, float]]:
    rows = []
    for seed in nt.SEEDS:
        split = nt.duplicate_aware_stratified_three_way_split(bundle, seed)
        identity = {
            part: {nt.canonical_smiles(bundle.smiles[int(i)]) for i in idx}
            for part, idx in split.items()
        }
        rows.append(
            {
                "seed": float(seed),
                "n_train": float(len(split["train"])),
                "n_calibration": float(len(split["calibration"])),
                "n_test": float(len(split["test"])),
                "train_calibration_overlap": float(len(identity["train"] & identity["calibration"])),
                "train_test_overlap": float(len(identity["train"] & identity["test"])),
                "calibration_test_overlap": float(len(identity["calibration"] & identity["test"])),
            }
        )
    return rows


def temporal_overlap_audit(bundle: nt.MolBundle) -> dict[str, float]:
    split = nt.purged_temporal_three_way_split(bundle)
    identity = {
        part: {nt.canonical_smiles(bundle.smiles[int(i)]) for i in idx}
        for part, idx in split.items()
    }
    return {
        "n_train": float(len(split["train"])),
        "n_calibration": float(len(split["calibration"])),
        "n_test": float(len(split["test"])),
        "train_calibration_overlap": float(len(identity["train"] & identity["calibration"])),
        "train_test_overlap": float(len(identity["train"] & identity["test"])),
        "calibration_test_overlap": float(len(identity["calibration"] & identity["test"])),
    }


def main() -> None:
    print("loading public datasets", flush=True)
    bbb = nt.load_bbb_tab(nt.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = nt.load_bbb_tab(nt.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    chembl = nt.load_chembl(nt.DATA_DIR / "chembl_cns_targets.csv")

    print("running compound-grouped ChEMBL default condition", flush=True)
    chembl_default_rows = [nt.run_chembl_condition(seed, chembl) for seed in nt.SEEDS]
    for seed, row in zip(nt.SEEDS, chembl_default_rows):
        row["seed"] = float(seed)

    print("running parent-standardized B3DB sensitivity", flush=True)
    parent_rows = {
        "NeuroTx-MOOD RF context CP": [
            run_parent_filtered_condition(bbb, b3db, seed, "full", True) for seed in nt.SEEDS
        ],
        "RF Morgan global CP": [
            run_parent_filtered_condition(bbb, b3db, seed, "fingerprint", False) for seed in nt.SEEDS
        ],
    }

    print("running ChEMBL grouping, dropout, target-holdout, and temporal analyses", flush=True)
    payload = {
        "study": "NeuroTx-MOOD compound-controlled analyses",
        "public_data_files": [
            "data/bbb_martins.tab",
            "data/b3db_classification.tab",
            "data/chembl_cns_targets.csv",
        ],
        "identity_definitions": {
            "chembl_random": "RDKit canonical isomeric SMILES; all rows for one compound stay in one split",
            "chembl_temporal": "chronological row split followed by purging compounds observed in earlier periods",
            "b3db_parent_sensitivity": "RDKit cleanup, fragment parent, uncharging, canonical tautomer, first InChIKey block",
        },
        "chembl_compound_grouped_random": {
            "overlap_audit": compound_overlap_audit(chembl),
            "default_target_assay": {
                "seed_metrics": chembl_default_rows,
                "summary": summarize(chembl_default_rows),
            },
            "grouping_sensitivity": nt.run_chembl_grouping_sensitivity(chembl),
            "metadata_dropout_sensitivity": nt.run_metadata_dropout_sensitivity(chembl),
        },
        "chembl_compound_purged_target_holdout": nt.run_chembl_target_holdout(chembl),
        "chembl_compound_purged_temporal": {
            "overlap_audit": temporal_overlap_audit(chembl),
            "results": nt.run_chembl_temporal_validation(chembl, feature_mode="full", model_kind="rf"),
        },
        "b3db_parent_standardized_sensitivity": {
            "conditions": {
                name: {"seed_metrics": rows, "summary": summarize(rows)}
                for name, rows in parent_rows.items()
            },
            "paired_auroc": paired_result(
                [float(row["ood_external_auroc"]) for row in parent_rows["NeuroTx-MOOD RF context CP"]],
                [float(row["ood_external_auroc"]) for row in parent_rows["RF Morgan global CP"]],
            ),
        },
        "paired_external_auroc_comparison_family": representation_statistics(),
        "software": {
            "rdkit": nt.rdBase.rdkitVersion if hasattr(nt, "rdBase") else "2025.09.6",
            "random_forest_trees": 160,
            "seeds": nt.SEEDS,
        },
    }
    output = HERE / "compound_controlled_analyses.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output.name)


if __name__ == "__main__":
    main()
