"""Bootstrap intervals for the compound-purged ChEMBL temporal test set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import main as nt


def main() -> None:
    chembl = nt.load_chembl(nt.DATA_DIR / "chembl_cns_targets.csv")
    split = nt.purged_temporal_three_way_split(chembl)
    train_idx = split["train"]
    cal_idx = split["calibration"]
    test_idx = split["test"]
    features = nt.fit_feature_view(
        chembl,
        train_idx,
        [
            ("train", (chembl, train_idx)),
            ("cal", (chembl, cal_idx)),
            ("test", (chembl, test_idx)),
        ],
        "full",
    )
    model = nt.make_model("rf", 42)
    model.fit(features["train"], chembl.y[train_idx])
    p_cal = nt.predict_positive(model, features["cal"])
    p_test = nt.predict_positive(model, features["test"])
    calibration = nt.calibrate_mondrian(
        chembl.y[cal_idx], p_cal, chembl.assay_types[cal_idx], min_group=10
    )
    pred_sets = nt.prediction_sets(p_test, calibration, chembl.assay_types[test_idx])
    point = nt.point_metrics(chembl.y[test_idx], p_test)
    point.update(nt.conformal_metrics(chembl.y[test_idx], pred_sets))

    rng = np.random.default_rng(20260812)
    boot = {key: [] for key in ("auroc", "coverage_90", "mean_set_size", "success_rate")}
    n = len(test_idx)
    for _ in range(2000):
        sample = rng.integers(0, n, size=n)
        y = chembl.y[test_idx][sample]
        if len(np.unique(y)) < 2:
            continue
        probs = p_test[sample]
        sets = [pred_sets[int(i)] for i in sample]
        metrics = nt.point_metrics(y, probs)
        metrics.update(nt.conformal_metrics(y, sets))
        for key in boot:
            boot[key].append(float(metrics[key]))

    intervals = {
        key: {
            "point_estimate": float(point[key]),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "n_bootstrap": int(len(values)),
        }
        for key, values in boot.items()
    }
    payload = {
        "split": "chronological 60/20/20 with prior-period canonical-compound purging",
        "n_train": int(len(train_idx)),
        "n_calibration": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "bootstrap_seed": 20260812,
        "intervals": intervals,
    }
    output = Path(__file__).resolve().parent / "temporal_purged_bootstrap.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output.name)


if __name__ == "__main__":
    main()
