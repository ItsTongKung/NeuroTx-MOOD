"""Context and similarity ablation experiments for the NeuroTx-MOOD BBB study.

This script evaluates three diagnostics:

1. A stronger uncertainty-quantification comparator based on a deep ensemble
   of independently seeded MLPs with global split conformal calibration.
2. Alternative context-definition ablations for the BBB conformal layer,
   including descriptor k-means clusters and a deterministic hash control.
3. Similarity-stratified external coverage diagnostics to explain why nominal
   90% coverage fails under the strongest external shift regime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import main as exp


OUTPUT_PATH = Path(__file__).resolve().parent / "context_ablation_similarity_diagnostics.json"
ENSEMBLE_MEMBERS = 3
MIN_GROUP = 25
CONTEXT_GROUPS = 5


def _load_bundles():
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    return bbb, b3db


def _external_indices(seed: int, bbb: exp.MolBundle, b3db: exp.MolBundle) -> tuple[dict[str, np.ndarray], np.ndarray]:
    split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
    _, external_idx, _ = exp.external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
    return split, external_idx


def _ensemble_predict(
    train_X: np.ndarray,
    train_y: np.ndarray,
    pred_X: np.ndarray,
    member_seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    member_preds = []
    for member_seed in member_seeds:
        model = MLPClassifier(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            alpha=1e-4,
            batch_size=128,
            learning_rate_init=1e-3,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=8,
            random_state=member_seed,
        )
        model.fit(train_X, train_y)
        member_preds.append(exp.predict_positive(model, pred_X))
    stacked = np.vstack(member_preds).astype(np.float32)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


def run_deep_ensemble_baseline(bbb: exp.MolBundle, b3db: exp.MolBundle) -> dict[str, object]:
    seed_metrics: list[dict[str, float | None]] = []
    for seed in exp.SEEDS:
        print(f"deep_ensemble seed={seed}", flush=True)
        split, external_idx = _external_indices(seed, bbb, b3db)
        feature_sets = exp.fit_feature_view(
            bbb,
            split["train"],
            [
                ("train", (bbb, split["train"])),
                ("cal", (bbb, split["calibration"])),
                ("test", (bbb, split["test"])),
                ("external", (b3db, external_idx)),
            ],
            "full",
        )
        member_seeds = [seed * 100 + member for member in range(ENSEMBLE_MEMBERS)]
        p_cal, cal_std = _ensemble_predict(
            feature_sets["train"],
            bbb.y[split["train"]],
            feature_sets["cal"],
            member_seeds,
        )
        p_test, test_std = _ensemble_predict(
            feature_sets["train"],
            bbb.y[split["train"]],
            feature_sets["test"],
            member_seeds,
        )
        p_external, ext_std = _ensemble_predict(
            feature_sets["train"],
            bbb.y[split["train"]],
            feature_sets["external"],
            member_seeds,
        )
        calibration = exp.calibrate_global(bbb.y[split["calibration"]], p_cal)
        metrics = exp.evaluate_split(bbb.y[split["test"]], p_test, calibration)
        metrics.update(
            exp.evaluate_split(
                b3db.y[external_idx],
                p_external,
                calibration,
                prefix="ood_external",
            )
        )
        metrics["seed"] = float(seed)
        metrics["n_ensemble_members"] = float(ENSEMBLE_MEMBERS)
        metrics["n_train"] = float(len(split["train"]))
        metrics["n_calibration"] = float(len(split["calibration"]))
        metrics["n_test"] = float(len(split["test"]))
        metrics["n_external_test"] = float(len(external_idx))
        metrics["cal_member_std_mean"] = float(np.mean(cal_std))
        metrics["test_member_std_mean"] = float(np.mean(test_std))
        metrics["ood_external_member_std_mean"] = float(np.mean(ext_std))
        seed_metrics.append({k: exp.safe_number(v) for k, v in metrics.items()})
    return {
        "description": (
            "Three-member deep ensemble of independently seeded MLPs trained on "
            "descriptor-plus-fingerprint features, with probabilities averaged "
            "before global split conformal calibration."
        ),
        "seed_metrics": seed_metrics,
        "summary": exp.summarize(seed_metrics),
    }


def _fit_kmeans_context(
    bundle: exp.MolBundle,
    train_idx: np.ndarray,
    seed: int,
    n_clusters: int = CONTEXT_GROUPS,
):
    train_desc = bundle.descriptors[train_idx]
    imputer = SimpleImputer(strategy="median").fit(train_desc)
    scaler = StandardScaler().fit(imputer.transform(train_desc))
    train_desc_scaled = scaler.transform(imputer.transform(train_desc))
    model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    model.fit(train_desc_scaled)
    return imputer, scaler, model


def _predict_kmeans_context(
    bundle: exp.MolBundle,
    idx: np.ndarray,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    model: KMeans,
) -> np.ndarray:
    desc = scaler.transform(imputer.transform(bundle.descriptors[idx]))
    labels = model.predict(desc)
    return np.asarray([f"kmeans_{int(label)}" for label in labels], dtype=object)


def _hash_contexts(bundle: exp.MolBundle, idx: np.ndarray, n_groups: int = CONTEXT_GROUPS) -> np.ndarray:
    labels = []
    for i in idx:
        smi = exp.canonical_smiles(bundle.smiles[int(i)])
        digest = hashlib.sha1(smi.encode("utf-8")).hexdigest()
        labels.append(f"hash_{int(digest[:8], 16) % n_groups}")
    return np.asarray(labels, dtype=object)


def _fallback_fraction(contexts: np.ndarray, retained_groups: set[str]) -> float:
    if len(contexts) == 0:
        return 0.0
    ctx = np.asarray(list(map(str, contexts)), dtype=object)
    return float(np.mean([value not in retained_groups for value in ctx]))


def run_context_definition_ablation(bbb: exp.MolBundle, b3db: exp.MolBundle) -> dict[str, object]:
    rows_by_scheme: dict[str, list[dict[str, float | None]]] = {
        "global": [],
        "proxy_basic": [],
        "proxy_rich": [],
        "descriptor_kmeans5": [],
        "hash_control5": [],
    }
    seed_metrics = []
    for seed in exp.SEEDS:
        print(f"context_ablation seed={seed}", flush=True)
        split, external_idx = _external_indices(seed, bbb, b3db)
        feature_sets = exp.fit_feature_view(
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
        model.fit(feature_sets["train"], bbb.y[split["train"]])
        p_cal = exp.predict_positive(model, feature_sets["cal"])
        p_external = exp.predict_positive(model, feature_sets["external"])

        global_calibration = exp.calibrate_global(bbb.y[split["calibration"]], p_cal)
        global_metrics = exp.evaluate_split(
            b3db.y[external_idx],
            p_external,
            global_calibration,
            prefix="ood_external",
        )
        global_metrics.update(
            {
                "seed": float(seed),
                "mondrian_groups": 0.0,
                "calibration_fallback_fraction": 0.0,
                "external_fallback_fraction": 0.0,
            }
        )
        rows_by_scheme["global"].append({k: exp.safe_number(v) for k, v in global_metrics.items()})
        seed_metrics.append({**rows_by_scheme["global"][-1], "scheme": "global"})

        scheme_payloads = []

        thresholds_basic = exp.chemical_context_thresholds(bbb, split["train"], "basic")
        scheme_payloads.append(
            (
                "proxy_basic",
                exp.chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds_basic, "basic"),
                exp.chemical_contexts_from_thresholds(b3db, external_idx, thresholds_basic, "basic"),
            )
        )
        thresholds_rich = exp.chemical_context_thresholds(bbb, split["train"], "rich")
        scheme_payloads.append(
            (
                "proxy_rich",
                exp.chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds_rich, "rich"),
                exp.chemical_contexts_from_thresholds(b3db, external_idx, thresholds_rich, "rich"),
            )
        )

        imputer, scaler, kmeans_model = _fit_kmeans_context(bbb, split["train"], seed)
        scheme_payloads.append(
            (
                "descriptor_kmeans5",
                _predict_kmeans_context(bbb, split["calibration"], imputer, scaler, kmeans_model),
                _predict_kmeans_context(b3db, external_idx, imputer, scaler, kmeans_model),
            )
        )
        scheme_payloads.append(
            (
                "hash_control5",
                _hash_contexts(bbb, split["calibration"]),
                _hash_contexts(b3db, external_idx),
            )
        )

        for scheme_name, cal_ctx, ext_ctx in scheme_payloads:
            calibration = exp.calibrate_mondrian(
                bbb.y[split["calibration"]],
                p_cal,
                cal_ctx,
                min_group=MIN_GROUP,
            )
            retained_groups = set(map(str, calibration.get("groups", {}).keys()))
            metrics = exp.evaluate_split(
                b3db.y[external_idx],
                p_external,
                calibration,
                ext_ctx,
                prefix="ood_external",
            )
            metrics["seed"] = float(seed)
            metrics["mondrian_groups"] = float(len(retained_groups))
            metrics["calibration_fallback_fraction"] = _fallback_fraction(cal_ctx, retained_groups)
            metrics["external_fallback_fraction"] = _fallback_fraction(ext_ctx, retained_groups)
            metrics["n_external_test"] = float(len(external_idx))
            metrics = {k: exp.safe_number(v) for k, v in metrics.items()}
            rows_by_scheme[scheme_name].append(metrics)
            seed_metrics.append({**metrics, "scheme": scheme_name})

    return {
        "description": (
            "Calibration-only BBB context-definition ablation with a fixed RF "
            "point predictor. Proxy descriptor bins are compared against "
            "descriptor k-means clusters and a deterministic hash control."
        ),
        "seed_metrics": seed_metrics,
        "summary_by_scheme": {key: exp.summarize(rows) for key, rows in rows_by_scheme.items()},
    }


def run_similarity_undercoverage_probe(bbb: exp.MolBundle, b3db: exp.MolBundle) -> dict[str, object]:
    pooled_rows: list[dict[str, float | int]] = []
    for seed in exp.SEEDS:
        print(f"similarity_probe seed={seed}", flush=True)
        split, external_idx = _external_indices(seed, bbb, b3db)
        feature_sets = exp.fit_feature_view(
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
        model.fit(feature_sets["train"], bbb.y[split["train"]])
        p_cal = exp.predict_positive(model, feature_sets["cal"])
        p_external = exp.predict_positive(model, feature_sets["external"])
        thresholds = exp.chemical_context_thresholds(bbb, split["train"], "basic")
        cal_ctx = exp.chemical_contexts_from_thresholds(bbb, split["calibration"], thresholds, "basic")
        ext_ctx = exp.chemical_contexts_from_thresholds(b3db, external_idx, thresholds, "basic")
        calibration = exp.calibrate_mondrian(
            bbb.y[split["calibration"]],
            p_cal,
            cal_ctx,
            min_group=MIN_GROUP,
        )
        pred_sets = exp.prediction_sets(p_external, calibration, ext_ctx)
        ext_sim = exp.max_tanimoto_values(bbb.fingerprints[split["train"]], b3db.fingerprints[external_idx])
        for idx_local, sim, pred_set, y_true in zip(external_idx, ext_sim, pred_sets, b3db.y[external_idx]):
            pooled_rows.append(
                {
                    "seed": int(seed),
                    "external_index": int(idx_local),
                    "similarity": float(sim),
                    "covered": int(int(y_true) in pred_set),
                    "set_size": int(len(pred_set)),
                    "singleton_correct": int(len(pred_set) == 1 and int(y_true) in pred_set),
                }
            )

    ordered = sorted(pooled_rows, key=lambda row: row["similarity"])
    n = len(ordered)
    quartile_rows = {
        "Q1_lowest_similarity": ordered[: n // 4],
        "Q2": ordered[n // 4 : n // 2],
        "Q3": ordered[n // 2 : (3 * n) // 4],
        "Q4_highest_similarity": ordered[(3 * n) // 4 :],
    }
    summary_by_quartile = {}
    for label, rows in quartile_rows.items():
        sims = np.asarray([row["similarity"] for row in rows], dtype=float)
        summary_by_quartile[label] = {
            "n": int(len(rows)),
            "similarity_min": float(np.min(sims)),
            "similarity_max": float(np.max(sims)),
            "similarity_mean": float(np.mean(sims)),
            "coverage_90": float(np.mean([row["covered"] for row in rows])),
            "mean_set_size": float(np.mean([row["set_size"] for row in rows])),
            "singleton_correct_success": float(np.mean([row["singleton_correct"] for row in rows])),
        }
    return {
        "description": (
            "External B3DB coverage stratified by pooled train-to-test maximum "
            "Morgan Tanimoto similarity quartiles for the default NeuroTx-MOOD "
            "RF context model."
        ),
        "n_pooled_predictions": int(len(pooled_rows)),
        "summary_by_quartile": summary_by_quartile,
    }


def main() -> None:
    bbb, b3db = _load_bundles()
    payload = {
        "generated_from": "run_context_ablation_similarity_diagnostics.py",
        "deep_ensemble_bbb_baseline": run_deep_ensemble_baseline(bbb, b3db),
        "context_definition_ablation": run_context_definition_ablation(bbb, b3db),
        "similarity_undercoverage_probe": run_similarity_undercoverage_probe(bbb, b3db),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
