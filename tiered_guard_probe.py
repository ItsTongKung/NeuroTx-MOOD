"""Seed-wise evaluation of the fixed two-tier novelty guard policy.

The two-tier guard is evaluated as a descriptive deployment policy rather than
as a prospectively selected operating point. This script recomputes that fixed
policy across all BBB seeds and writes mean/std summaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import main as exp


OUTPUT_PATH = Path(__file__).resolve().parent / "tiered_guard_probe.json"
POLICY = {
    "q_low": 0.05,
    "q_high": 0.15,
    "d_mid": 0.20,
    "d_low": 0.30,
}


def run_fixed_tiered_guard(bbb: exp.MolBundle, b3db: exp.MolBundle) -> dict[str, object]:
    seed_metrics: list[dict[str, float]] = []
    for seed in exp.SEEDS:
        print(f"tiered_guard seed={seed}", flush=True)
        split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
        _, external_idx, _ = exp.external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
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
            min_group=25,
        )

        cal_sim = exp.max_tanimoto_values(
            bbb.fingerprints[split["train"]],
            bbb.fingerprints[split["calibration"]],
        )
        ext_sim = exp.max_tanimoto_values(
            bbb.fingerprints[split["train"]],
            b3db.fingerprints[external_idx],
        )
        low_threshold = float(np.quantile(cal_sim, POLICY["q_low"]))
        high_threshold = float(np.quantile(cal_sim, POLICY["q_high"]))
        low_mask = ext_sim <= low_threshold
        mid_mask = (ext_sim <= high_threshold) & (~low_mask)

        groups = calibration.get("groups", {}) if isinstance(calibration.get("groups", {}), dict) else {}
        global_q = float(calibration.get("global", 1.0))
        pred_sets: list[set[int]] = []
        for prob_pos, ctx, guarded_mid, guarded_low in zip(p_external, ext_ctx, mid_mask, low_mask):
            q = float(groups.get(str(ctx), global_q))
            if guarded_low:
                q = min(1.0, q + POLICY["d_low"])
            elif guarded_mid:
                q = min(1.0, q + POLICY["d_mid"])
            current = set()
            if prob_pos <= q:
                current.add(0)
            if 1.0 - prob_pos <= q:
                current.add(1)
            if not current:
                current.add(int(prob_pos >= 0.5))
            pred_sets.append(current)

        metrics = exp.conformal_metrics(b3db.y[external_idx], pred_sets)
        seed_metrics.append(
            {
                "seed": float(seed),
                "coverage_90": float(metrics["coverage_90"]),
                "mean_set_size": float(metrics["mean_set_size"]),
                "success_rate": float(metrics["success_rate"]),
                "guarded_fraction_mid_or_low": float(np.mean(ext_sim <= high_threshold)),
                "guarded_fraction_low": float(np.mean(low_mask)),
                "similarity_threshold_mid_or_low": high_threshold,
                "similarity_threshold_low": low_threshold,
            }
        )

    return {
        "description": (
            "Fixed two-tier novelty guard for the default descriptor-plus-fingerprint RF "
            "proxy-context conformal model, using calibration-derived similarity quantiles "
            "q_low = 0.05 and q_high = 0.15 with additive margins d_mid = 0.20 and d_low = 0.30."
        ),
        "policy": POLICY,
        "seed_metrics": seed_metrics,
        "summary": exp.summarize(seed_metrics),
    }


def main() -> None:
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    payload = run_fixed_tiered_guard(bbb, b3db)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
