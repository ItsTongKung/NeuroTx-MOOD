"""Compute seed-wise and aggregate reliability-curve statistics on B3DB."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MAIN_PATH = ROOT / "main.py"
OUT_PATH = ROOT / "calibration_curve_probe.json"
BINS = np.linspace(0.0, 1.0, 11)


def _load_main_module():
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("neurotx_exp_main", MAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bin_stats(prob_pos: np.ndarray, y_true: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for left, right in zip(BINS[:-1], BINS[1:]):
        if right >= 1.0:
            mask = (prob_pos >= left) & (prob_pos <= right)
        else:
            mask = (prob_pos >= left) & (prob_pos < right)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_left": float(left),
                    "bin_right": float(right),
                    "count": 0.0,
                    "mean_probability": float("nan"),
                    "positive_rate": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "bin_left": float(left),
                "bin_right": float(right),
                "count": float(count),
                "mean_probability": float(np.mean(prob_pos[mask])),
                "positive_rate": float(np.mean(y_true[mask])),
            }
        )
    return rows


def _aggregate_bins(seed_rows: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    aggregated = []
    for idx in range(len(BINS) - 1):
        counts = []
        mean_probs = []
        positive_rates = []
        for rows in seed_rows:
            row = rows[idx]
            if int(row["count"]) <= 0:
                continue
            counts.append(float(row["count"]))
            mean_probs.append(float(row["mean_probability"]))
            positive_rates.append(float(row["positive_rate"]))
        if not counts:
            aggregated.append(
                {
                    "bin_left": float(BINS[idx]),
                    "bin_right": float(BINS[idx + 1]),
                    "count_mean": 0.0,
                    "count_std": 0.0,
                    "mean_probability_mean": float("nan"),
                    "mean_probability_std": float("nan"),
                    "positive_rate_mean": float("nan"),
                    "positive_rate_std": float("nan"),
                }
            )
            continue
        aggregated.append(
            {
                "bin_left": float(BINS[idx]),
                "bin_right": float(BINS[idx + 1]),
                "count_mean": float(np.mean(counts)),
                "count_std": float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0,
                "mean_probability_mean": float(np.mean(mean_probs)),
                "mean_probability_std": float(np.std(mean_probs, ddof=1)) if len(mean_probs) > 1 else 0.0,
                "positive_rate_mean": float(np.mean(positive_rates)),
                "positive_rate_std": float(np.std(positive_rates, ddof=1)) if len(positive_rates) > 1 else 0.0,
            }
        )
    return aggregated


def _collect_external_probabilities(
    exp,
    *,
    model_kind: str,
    feature_mode: str,
) -> dict[str, object]:
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    seed_rows = []
    for seed in exp.SEEDS:
        split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
        ext_split = exp.scaffold_split(b3db, seed + 101)
        external_idx_raw = ext_split["test"]
        external_idx = exp.exact_nonoverlap_indices(bbb.smiles, b3db, external_idx_raw)
        if len(external_idx) == 0 or len(np.unique(b3db.y[external_idx])) < 2:
            external_idx = external_idx_raw
        feature_sets = exp.fit_feature_view(
            bbb,
            split["train"],
            [
                ("train", (bbb, split["train"])),
                ("external", (b3db, external_idx)),
            ],
            feature_mode,
        )
        model = exp.make_model(model_kind, seed)
        model.fit(feature_sets["train"], bbb.y[split["train"]])
        p_external = exp.predict_positive(model, feature_sets["external"])
        seed_rows.append(_bin_stats(p_external, b3db.y[external_idx]))
    return {
        "seed_curves": seed_rows,
        "aggregate_curve": _aggregate_bins(seed_rows),
    }


def main() -> None:
    exp = _load_main_module()
    payload = {
        "source_files": ["main.py"],
        "bin_edges": BINS.tolist(),
        "rf_morgan_global_cp_point_model": _collect_external_probabilities(
            exp, model_kind="rf", feature_mode="fingerprint"
        ),
        "neurotx_mood_point_model": _collect_external_probabilities(
            exp, model_kind="rf", feature_mode="full"
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
