"""External conformal diagnostics for the NeuroTx-MOOD experiments.

This script summarizes the core external-coverage diagnostics:

1. Dataset curation summaries (raw rows, valid rows, canonical duplicates,
   conflicting duplicate groups, class balance, and scaffold diversity).
2. An extended overlap audit using InChIKey first-block comparisons in
   addition to the exact canonical-SMILES zero-overlap rule used in the main
   benchmark.
3. Fixed-point external BBB diagnostics covering
   - class-wise coverage for BBB- and BBB+,
   - class-conditional conformal baselines,
   - empty-set rate before operational singleton replacement, and
   - coverage across an alpha grid.

The script reuses the bundled NeuroTx-MOOD code paths and writes a compact JSON
artifact that can be used to verify the main class-wise, class-conditional, and
alpha-grid conformal diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats

import applicability_region_probes as arp
import main as exp


OUTPUT_PATH = Path(__file__).resolve().parent / "external_conformal_diagnostics.json"
ALPHA_GRID = [0.20, 0.15, 0.10, 0.05]


def _mean_std(values: list[float | None]) -> dict[str, float | None]:
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return {"mean": None, "std": None}
    arr = np.asarray(cleaned, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
    }


def _canonical_series(smiles_list: list[str]) -> list[str]:
    return [exp.canonical_smiles(smi) for smi in smiles_list]


def _inchikey1(smiles: str) -> str:
    mol = exp.mol_from_smiles(smiles)
    if mol is None:
        return "invalid"
    try:
        return Chem.MolToInchiKey(mol).split("-")[0]
    except Exception:
        return "invalid"


def _dataset_summary() -> dict[str, dict[str, float | int | str]]:
    summaries: dict[str, dict[str, float | int | str]] = {}

    # BBB_Martins
    bbb_path = exp.DATA_DIR / "bbb_martins.tab"
    bbb_raw = pd.read_csv(bbb_path, sep="\t")
    bbb_valid = bbb_raw.dropna(subset=["Drug", "Y"]).copy()
    bbb_valid["Y"] = pd.to_numeric(bbb_valid["Y"], errors="coerce")
    bbb_valid = bbb_valid.dropna(subset=["Y"]).copy()
    bbb_valid["Y"] = bbb_valid["Y"].astype(int)
    bbb_valid["canonical"] = _canonical_series(bbb_valid["Drug"].astype(str).tolist())
    grouped = bbb_valid.groupby("canonical")["Y"]
    summaries["BBB_Martins"] = {
        "source_file": str(bbb_path.name),
        "raw_rows": int(len(bbb_raw)),
        "valid_rows": int(len(bbb_valid)),
        "unique_canonical_smiles": int(bbb_valid["canonical"].nunique()),
        "duplicate_rows": int(len(bbb_valid) - bbb_valid["canonical"].nunique()),
        "conflicting_duplicate_groups": int(grouped.nunique().gt(1).sum()),
        "bbb_negative": int((bbb_valid["Y"] == 0).sum()),
        "bbb_positive": int((bbb_valid["Y"] == 1).sum()),
        "bemis_murcko_scaffolds": int(exp.load_bbb_tab(bbb_path, "BBB_Martins").scaffolds.size and len(set(exp.load_bbb_tab(bbb_path, "BBB_Martins").scaffolds))),
    }

    # B3DB
    b3db_path = exp.DATA_DIR / "b3db_classification.tab"
    b3db_raw = pd.read_csv(b3db_path, sep="\t")
    b3db_valid = b3db_raw.dropna(subset=["Drug", "Y"]).copy()
    b3db_valid["Y"] = pd.to_numeric(b3db_valid["Y"], errors="coerce")
    b3db_valid = b3db_valid.dropna(subset=["Y"]).copy()
    b3db_valid["Y"] = b3db_valid["Y"].astype(int)
    b3db_valid["canonical"] = _canonical_series(b3db_valid["Drug"].astype(str).tolist())
    grouped = b3db_valid.groupby("canonical")["Y"]
    summaries["B3DB"] = {
        "source_file": str(b3db_path.name),
        "raw_rows": int(len(b3db_raw)),
        "valid_rows": int(len(b3db_valid)),
        "unique_canonical_smiles": int(b3db_valid["canonical"].nunique()),
        "duplicate_rows": int(len(b3db_valid) - b3db_valid["canonical"].nunique()),
        "conflicting_duplicate_groups": int(grouped.nunique().gt(1).sum()),
        "bbb_negative": int((b3db_valid["Y"] == 0).sum()),
        "bbb_positive": int((b3db_valid["Y"] == 1).sum()),
        "bemis_murcko_scaffolds": int(exp.load_bbb_tab(b3db_path, "B3DB_Classification").scaffolds.size and len(set(exp.load_bbb_tab(b3db_path, "B3DB_Classification").scaffolds))),
    }

    # ChEMBL
    chembl_path = exp.DATA_DIR / "chembl_cns_targets.csv"
    chembl_raw = pd.read_csv(chembl_path)
    chembl_valid = chembl_raw.dropna(subset=["canonical_smiles", "pchembl_value", "target_chembl_id"]).copy()
    chembl_valid["pchembl_value"] = pd.to_numeric(chembl_valid["pchembl_value"], errors="coerce")
    chembl_valid = chembl_valid.dropna(subset=["pchembl_value"]).copy()
    chembl_valid["Y"] = (chembl_valid["pchembl_value"] >= 6.0).astype(int)
    chembl_valid["canonical"] = _canonical_series(chembl_valid["canonical_smiles"].astype(str).tolist())
    grouped = chembl_valid.groupby("canonical")["Y"]
    chembl_bundle = exp.load_chembl(chembl_path)
    summaries["ChEMBL_CNS_panel"] = {
        "source_file": str(chembl_path.name),
        "raw_rows": int(len(chembl_raw)),
        "valid_rows": int(len(chembl_valid)),
        "unique_canonical_smiles": int(chembl_valid["canonical"].nunique()),
        "duplicate_rows": int(len(chembl_valid) - chembl_valid["canonical"].nunique()),
        "conflicting_duplicate_groups": int(grouped.nunique().gt(1).sum()),
        "inactive_rows": int((chembl_valid["Y"] == 0).sum()),
        "active_rows": int((chembl_valid["Y"] == 1).sum()),
        "bemis_murcko_scaffolds": int(len(set(chembl_bundle.scaffolds))),
        "targets": int(pd.Series(chembl_bundle.target_ids).nunique()),
        "assay_types": int(pd.Series(chembl_bundle.assay_types).nunique()),
    }
    return summaries


def _raw_prediction_sets(prob_pos: np.ndarray, calibration: dict[str, object], contexts=None) -> list[set[int]]:
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
        sets.append(current)
    return sets


def _raw_prediction_sets_class_conditional(
    prob_pos: np.ndarray,
    calibration: dict[str, object],
    contexts=None,
) -> list[set[int]]:
    groups = calibration.get("groups", {}) if isinstance(calibration.get("groups", {}), dict) else {}
    classes = calibration.get("classes", {}) if isinstance(calibration.get("classes", {}), dict) else {}
    global_q = float(calibration.get("global", 1.0))
    if contexts is None:
        contexts = [None] * len(prob_pos)
    sets: list[set[int]] = []
    for p, ctx in zip(prob_pos, contexts):
        context_payload = groups.get(str(ctx), {}) if isinstance(groups.get(str(ctx), {}), dict) else {}
        q0 = float(context_payload.get("0", classes.get("0", global_q)))
        q1 = float(context_payload.get("1", classes.get("1", global_q)))
        current = set()
        if p <= q0:
            current.add(0)
        if 1.0 - p <= q1:
            current.add(1)
        sets.append(current)
    return sets


def _operationalize_prediction_sets(prob_pos: np.ndarray, raw_sets: list[set[int]]) -> list[set[int]]:
    prob_pos = np.asarray(prob_pos, dtype=float)
    processed: list[set[int]] = []
    for p, current in zip(prob_pos, raw_sets):
        if current:
            processed.append(set(current))
        else:
            processed.append({int(p >= 0.5)})
    return processed


def _raw_prediction_sets_local_similarity(
    prob_pos: np.ndarray,
    cal_scores: np.ndarray,
    sim_matrix: np.ndarray,
    alpha: float,
    k_neighbors: int = arp.LOCAL_K,
) -> list[set[int]]:
    k = min(int(k_neighbors), int(cal_scores.shape[0]))
    sets: list[set[int]] = []
    for p, sim_row in zip(prob_pos, sim_matrix):
        local_idx = np.argpartition(sim_row, -k)[-k:]
        q = exp.finite_sample_quantile(cal_scores[local_idx], alpha=alpha)
        current = set()
        if p <= q:
            current.add(0)
        if 1.0 - p <= q:
            current.add(1)
        sets.append(current)
    return sets


def _raw_prediction_sets_local_class_conditional(
    prob_pos: np.ndarray,
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    sim_matrix: np.ndarray,
    alpha: float,
    class_fallback: dict[str, float],
    k_neighbors: int = arp.LOCAL_K,
    min_label_neighbors: int = 10,
) -> list[set[int]]:
    k = min(int(k_neighbors), int(p_cal.shape[0]))
    y_cal = np.asarray(y_cal, dtype=int)
    p_cal = np.asarray(p_cal, dtype=float)
    scores = exp.nonconformity_scores(y_cal, p_cal)
    sets: list[set[int]] = []
    for p, sim_row in zip(prob_pos, sim_matrix):
        local_idx = np.argpartition(sim_row, -k)[-k:]
        q_by_class: dict[str, float] = {}
        for label in (0, 1):
            label_idx = local_idx[y_cal[local_idx] == label]
            if int(label_idx.shape[0]) >= min_label_neighbors:
                q_by_class[str(label)] = exp.finite_sample_quantile(scores[label_idx], alpha=alpha)
            else:
                q_by_class[str(label)] = float(class_fallback[str(label)])
        current = set()
        if p <= q_by_class["0"]:
            current.add(0)
        if 1.0 - p <= q_by_class["1"]:
            current.add(1)
        sets.append(current)
    return sets


def _coverage_for_mask(y_true: np.ndarray, pred_sets: list[set[int]], mask: np.ndarray) -> float | None:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return None
    idx = np.where(mask)[0]
    return float(np.mean([int(int(y_true[i]) in pred_sets[i]) for i in idx]))


def _summarize_rows(rows: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    keys = sorted({k for row in rows for k in row if k != "seed"})
    return {key: _mean_std([row.get(key) for row in rows]) for key in keys}


def _paired_effect(
    method_rows: list[dict[str, float | None]],
    baseline_rows: list[dict[str, float | None]],
    metric: str,
) -> dict[str, float | int]:
    """Summarize a seed-paired method-minus-baseline contrast with a t interval."""
    method_by_seed = {
        int(row["seed"]): float(row[metric])
        for row in method_rows
        if row.get(metric) is not None
    }
    baseline_by_seed = {
        int(row["seed"]): float(row[metric])
        for row in baseline_rows
        if row.get(metric) is not None
    }
    seeds = sorted(set(method_by_seed) & set(baseline_by_seed))
    diff = np.asarray(
        [method_by_seed[seed] - baseline_by_seed[seed] for seed in seeds],
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


def _extended_overlap_audit(bbb: exp.MolBundle, b3db: exp.MolBundle) -> dict[str, object]:
    seed_rows = []
    ref_canonical = [exp.canonical_smiles(smi) for smi in bbb.smiles]
    ref_inchi1 = {_inchikey1(smi) for smi in ref_canonical}
    ref_inchi1.discard("invalid")
    for seed in exp.SEEDS:
        _, external_idx, attempt = exp.external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
        ext_smiles = [b3db.smiles[int(i)] for i in external_idx]
        ext_canonical = [exp.canonical_smiles(smi) for smi in ext_smiles]
        ext_inchi1 = [_inchikey1(smi) for smi in ext_canonical]
        exact_remaining = sum(1 for smi in ext_canonical if smi in set(ref_canonical))
        inchi1_remaining = sum(1 for key in ext_inchi1 if key != "invalid" and key in ref_inchi1)
        seed_rows.append(
            {
                "seed": float(seed),
                "partition_attempt": float(attempt),
                "external_size": float(len(external_idx)),
                "remaining_exact_overlap": float(exact_remaining),
                "remaining_inchikey1_overlap": float(inchi1_remaining),
            }
        )
    return {
        "description": (
            "Extended leakage audit for the zero-overlap external B3DB fold. "
            "The benchmark excludes exact canonical-SMILES overlaps by design; "
            "this extended audit also reports InChIKey first-block overlap counts."
        ),
        "seed_metrics": seed_rows,
        "summary": _summarize_rows(seed_rows),
    }


def _calibrate_global_alpha(y_cal: np.ndarray, p_cal: np.ndarray, alpha: float) -> dict[str, object]:
    scores = exp.nonconformity_scores(y_cal, p_cal)
    return {"global": exp.finite_sample_quantile(scores, alpha=alpha)}


def _calibrate_class_conditional_alpha(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    y_cal = np.asarray(y_cal, dtype=int)
    scores = exp.nonconformity_scores(y_cal, p_cal)
    global_q = exp.finite_sample_quantile(scores, alpha=alpha)
    classes: dict[str, float] = {}
    for label in (0, 1):
        mask = y_cal == label
        classes[str(label)] = (
            exp.finite_sample_quantile(scores[mask], alpha=alpha)
            if int(mask.sum()) > 0
            else global_q
        )
    return {"global": global_q, "classes": classes}


def _calibrate_mondrian_alpha(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    contexts: np.ndarray,
    alpha: float,
    min_group: int = 25,
) -> dict[str, object]:
    scores = exp.nonconformity_scores(y_cal, p_cal)
    global_q = exp.finite_sample_quantile(scores, alpha=alpha)
    groups: dict[str, float] = {}
    for ctx in sorted(set(map(str, contexts))):
        mask = contexts.astype(str) == ctx
        if int(mask.sum()) >= min_group:
            groups[ctx] = exp.finite_sample_quantile(scores[mask], alpha=alpha)
    return {"global": global_q, "groups": groups}


def _calibrate_context_class_conditional_alpha(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    contexts: np.ndarray,
    alpha: float,
    min_class_group: int = 10,
) -> dict[str, object]:
    y_cal = np.asarray(y_cal, dtype=int)
    contexts = np.asarray(contexts, dtype=object).astype(str)
    scores = exp.nonconformity_scores(y_cal, p_cal)
    base = _calibrate_class_conditional_alpha(y_cal, p_cal, alpha=alpha)
    groups: dict[str, dict[str, float]] = {}
    for ctx in sorted(set(map(str, contexts))):
        ctx_payload: dict[str, float] = {}
        for label in (0, 1):
            mask = (contexts == str(ctx)) & (y_cal == label)
            if int(mask.sum()) >= min_class_group:
                ctx_payload[str(label)] = exp.finite_sample_quantile(scores[mask], alpha=alpha)
        if ctx_payload:
            groups[str(ctx)] = ctx_payload
    return {"global": base["global"], "classes": base["classes"], "groups": groups}


def _set_metrics(y_true: np.ndarray, pred_sets: list[set[int]]) -> dict[str, float]:
    return {
        "coverage": float(np.mean([int(int(y) in s) for y, s in zip(y_true, pred_sets)])),
        "mean_set_size": float(np.mean([len(s) for s in pred_sets])),
        "singleton_success": float(
            np.mean([int(len(s) == 1 and int(y) in s) for y, s in zip(y_true, pred_sets)])
        ),
    }


def _fixed_predictor_external_probes(bbb: exp.MolBundle, b3db: exp.MolBundle) -> dict[str, object]:
    scheme_names = [
        "global_full_rf",
        "global_class_conditional",
        "proxy_context",
        "proxy_context_class_conditional",
        "local_similarity_knn75",
        "local_similarity_class_conditional_knn75",
    ]
    metric_rows = {name: [] for name in scheme_names}
    classwise_rows = {name: [] for name in scheme_names}
    alpha_rows = {name: [] for name in scheme_names}
    context_rows = []

    for seed in exp.SEEDS:
        split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
        _, external_idx, _ = exp.external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
        features = exp.fit_feature_view(
            bbb,
            split["train"],
            [
                ("train", (bbb, split["train"])),
                ("cal", (bbb, split["calibration"])),
                ("external", (b3db, external_idx)),
            ],
            "full",
        )
        model = exp.make_model("rf", seed)
        model.fit(features["train"], bbb.y[split["train"]])
        p_cal = exp.predict_positive(model, features["cal"])
        p_external = exp.predict_positive(model, features["external"])
        y_external = b3db.y[external_idx]

        thresholds = exp.chemical_context_thresholds(bbb, split["train"], "basic")
        cal_ctx = exp.chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, "basic")
        ext_ctx = exp.chemical_contexts_from_thresholds(b3db, external_idx, thresholds, "basic")

        global_cal = _calibrate_global_alpha(bbb.y[split["calibration"]], p_cal, alpha=0.10)
        global_cc_cal = _calibrate_class_conditional_alpha(bbb.y[split["calibration"]], p_cal, alpha=0.10)
        proxy_cal = _calibrate_mondrian_alpha(bbb.y[split["calibration"]], p_cal, cal_ctx, alpha=0.10, min_group=25)
        proxy_cc_cal = _calibrate_context_class_conditional_alpha(
            bbb.y[split["calibration"]], p_cal, cal_ctx, alpha=0.10, min_class_group=10
        )
        cal_scores = exp.nonconformity_scores(bbb.y[split["calibration"]], p_cal)
        sim_ext_to_cal = arp.tanimoto_matrix(b3db.fingerprints[external_idx], bbb.fingerprints[split["calibration"]])

        global_raw = _raw_prediction_sets(p_external, global_cal)
        global_cc_raw = _raw_prediction_sets_class_conditional(p_external, global_cc_cal)
        proxy_raw = _raw_prediction_sets(p_external, proxy_cal, ext_ctx)
        proxy_cc_raw = _raw_prediction_sets_class_conditional(p_external, proxy_cc_cal, ext_ctx)
        local_raw = _raw_prediction_sets_local_similarity(p_external, cal_scores, sim_ext_to_cal, alpha=0.10)
        local_cc_raw = _raw_prediction_sets_local_class_conditional(
            p_external,
            bbb.y[split["calibration"]],
            p_cal,
            sim_ext_to_cal,
            alpha=0.10,
            class_fallback=global_cc_cal["classes"],
        )
        global_post = _operationalize_prediction_sets(p_external, global_raw)
        global_cc_post = _operationalize_prediction_sets(p_external, global_cc_raw)
        proxy_post = _operationalize_prediction_sets(p_external, proxy_raw)
        proxy_cc_post = _operationalize_prediction_sets(p_external, proxy_cc_raw)
        local_post = _operationalize_prediction_sets(p_external, local_raw)
        local_cc_post = _operationalize_prediction_sets(p_external, local_cc_raw)

        for scheme_name, raw_sets, post_sets in (
            ("global_full_rf", global_raw, global_post),
            ("global_class_conditional", global_cc_raw, global_cc_post),
            ("proxy_context", proxy_raw, proxy_post),
            ("proxy_context_class_conditional", proxy_cc_raw, proxy_cc_post),
            ("local_similarity_knn75", local_raw, local_post),
            ("local_similarity_class_conditional_knn75", local_cc_raw, local_cc_post),
        ):
            row_metrics = _set_metrics(y_external, post_sets)
            row_metrics.update(
                {
                    "seed": float(seed),
                    "empty_set_rate": float(np.mean([len(s) == 0 for s in raw_sets])),
                }
            )
            metric_rows[scheme_name].append(row_metrics)
            classwise_rows[scheme_name].append(
                {
                    "seed": float(seed),
                    "coverage_bbb_negative": _coverage_for_mask(y_external, post_sets, y_external == 0),
                    "coverage_bbb_positive": _coverage_for_mask(y_external, post_sets, y_external == 1),
                    "empty_set_rate": float(np.mean([len(s) == 0 for s in raw_sets])),
                }
            )

        retained_groups = set(map(str, proxy_cal.get("groups", {}).keys()))
        retained_mask = np.asarray([str(ctx) in retained_groups for ctx in ext_ctx], dtype=bool)
        fallback_mask = ~retained_mask
        context_rows.append(
            {
                "seed": float(seed),
                "retained_support_fraction": float(np.mean(retained_mask)),
                "fallback_support_fraction": float(np.mean(fallback_mask)),
                "retained_context_coverage": _coverage_for_mask(y_external, proxy_post, retained_mask),
                "fallback_context_coverage": _coverage_for_mask(y_external, proxy_post, fallback_mask),
                "retained_context_set_size": float(
                    np.mean([len(proxy_post[i]) for i in np.where(retained_mask)[0]])
                )
                if int(retained_mask.sum()) > 0
                else None,
                "fallback_context_set_size": float(
                    np.mean([len(proxy_post[i]) for i in np.where(fallback_mask)[0]])
                )
                if int(fallback_mask.sum()) > 0
                else None,
            }
        )

        for alpha in ALPHA_GRID:
            global_cal_alpha = _calibrate_global_alpha(bbb.y[split["calibration"]], p_cal, alpha=alpha)
            global_cc_cal_alpha = _calibrate_class_conditional_alpha(
                bbb.y[split["calibration"]], p_cal, alpha=alpha
            )
            proxy_cal_alpha = _calibrate_mondrian_alpha(
                bbb.y[split["calibration"]], p_cal, cal_ctx, alpha=alpha, min_group=25
            )
            proxy_cc_cal_alpha = _calibrate_context_class_conditional_alpha(
                bbb.y[split["calibration"]], p_cal, cal_ctx, alpha=alpha, min_class_group=10
            )
            global_sets_alpha = _operationalize_prediction_sets(
                p_external,
                _raw_prediction_sets(p_external, global_cal_alpha),
            )
            global_cc_sets_alpha = _operationalize_prediction_sets(
                p_external,
                _raw_prediction_sets_class_conditional(p_external, global_cc_cal_alpha),
            )
            proxy_sets_alpha = _operationalize_prediction_sets(
                p_external,
                _raw_prediction_sets(p_external, proxy_cal_alpha, ext_ctx),
            )
            proxy_cc_sets_alpha = _operationalize_prediction_sets(
                p_external,
                _raw_prediction_sets_class_conditional(p_external, proxy_cc_cal_alpha, ext_ctx),
            )
            local_sets_alpha = _operationalize_prediction_sets(
                p_external,
                _raw_prediction_sets_local_similarity(
                p_external, cal_scores, sim_ext_to_cal, alpha=alpha
                ),
            )
            local_cc_sets_alpha = _operationalize_prediction_sets(
                p_external,
                _raw_prediction_sets_local_class_conditional(
                    p_external,
                    bbb.y[split["calibration"]],
                    p_cal,
                    sim_ext_to_cal,
                    alpha=alpha,
                    class_fallback=global_cc_cal_alpha["classes"],
                ),
            )
            for scheme_name, pred_sets in (
                ("global_full_rf", global_sets_alpha),
                ("global_class_conditional", global_cc_sets_alpha),
                ("proxy_context", proxy_sets_alpha),
                ("proxy_context_class_conditional", proxy_cc_sets_alpha),
                ("local_similarity_knn75", local_sets_alpha),
                ("local_similarity_class_conditional_knn75", local_cc_sets_alpha),
            ):
                alpha_rows[scheme_name].append(
                    {
                        "seed": float(seed),
                        "alpha": float(alpha),
                        "nominal_coverage": float(1.0 - alpha),
                        "coverage": float(np.mean([int(int(y) in s) for y, s in zip(y_external, pred_sets)])),
                        "mean_set_size": float(np.mean([len(s) for s in pred_sets])),
                    }
                )

    alpha_summary = {}
    for scheme_name, rows in alpha_rows.items():
        by_alpha = {}
        for alpha in ALPHA_GRID:
            subset = [row for row in rows if abs(float(row["alpha"]) - alpha) < 1e-9]
            by_alpha[str(alpha)] = _summarize_rows(subset)
        alpha_summary[scheme_name] = by_alpha

    paired_effects = {}
    for scheme_name in scheme_names:
        if scheme_name == "global_full_rf":
            continue
        paired_effects[scheme_name] = {
            metric: _paired_effect(
                metric_rows[scheme_name],
                metric_rows["global_full_rf"],
                metric,
            )
            for metric in ("coverage", "mean_set_size", "singleton_success")
        }

    return {
        "description": (
            "Fixed-point external BBB conformal diagnostics: "
            "class-wise coverage, class-conditional conformal baselines, empty-set rate before operational singleton "
            "replacement, context-wise coverage for retained versus fallback "
            "proxy groups, and coverage across an alpha grid."
        ),
        "fixed_point_metrics_by_scheme": {
            key: _summarize_rows(rows) for key, rows in metric_rows.items()
        },
        "paired_effects_vs_global_full_rf": paired_effects,
        "classwise_summary_by_scheme": {
            key: _summarize_rows(rows) for key, rows in classwise_rows.items()
        },
        "context_support_summary": _summarize_rows(context_rows),
        "alpha_grid_summary_by_scheme": alpha_summary,
    }


def main() -> None:
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    payload = {
        "generated_from": "run_external_conformal_diagnostics.py",
        "data_dir": "data",
        "dataset_curation_summary": _dataset_summary(),
        "extended_overlap_audit": _extended_overlap_audit(bbb, b3db),
        "fixed_predictor_external_probes": _fixed_predictor_external_probes(bbb, b3db),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote external conformal diagnostics to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
