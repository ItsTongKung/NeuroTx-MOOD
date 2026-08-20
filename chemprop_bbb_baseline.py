"""Chemprop D-MPNN baseline on the duplicate-controlled BBB benchmark."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import main as exp


OUTPUT_PATH = Path(__file__).resolve().parent / "chemprop_bbb_baseline.json"
RUN_ROOT = Path(__file__).resolve().parent / "_chemprop_bbb_runs"

CHEMPROP_ARGS = {
    "batch_size": 64,
    "message_hidden_dim": 256,
    "depth": 4,
    "dropout": 0.10,
    "ffn_hidden_dim": 256,
    "ffn_num_layers": 2,
    "epochs": 35,
    "patience": 8,
}


def _chemprop_exe() -> Path:
    exe_name = "chemprop.exe" if os.name == "nt" else "chemprop"
    return Path(sys.executable).resolve().parent / exe_name


def _write_split_csv(path: Path, smiles: list[str], y: np.ndarray, idx: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "smiles": [smiles[int(i)] for i in idx],
            "y": np.asarray(y[idx], dtype=int),
        }
    )
    df.to_csv(path, index=False)


def _run_cmd(args: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.run(
        args,
        capture_output=True,
        env=env,
        check=False,
    )
    stdout = proc.stdout.decode("utf-8", errors="replace") if isinstance(proc.stdout, bytes) else str(proc.stdout or "")
    stderr = proc.stderr.decode("utf-8", errors="replace") if isinstance(proc.stderr, bytes) else str(proc.stderr or "")
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(map(str, args))
        + "\n\nSTDOUT:\n"
        + stdout
        + "\n\nSTDERR:\n"
        + stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(args)}")


def _predict_with_chemprop(model_path: Path, input_csv: Path, output_csv: Path, log_path: Path) -> pd.DataFrame:
    cmd = [
        str(_chemprop_exe()),
        "predict",
        "-q",
        "-q",
        "-i",
        str(input_csv),
        "-o",
        str(output_csv),
        "--smiles-columns",
        "smiles",
        "--model-paths",
        str(model_path),
    ]
    _run_cmd(cmd, log_path)
    df = pd.read_csv(output_csv)
    if "y" not in df.columns:
        raise RuntimeError(f"prediction output missing 'y' column: {output_csv}")
    if df["y"].isna().any():
        raise RuntimeError(f"prediction output contains NaN values: {output_csv}")
    return df


def _train_seed(seed: int, bbb: exp.MolBundle, b3db: exp.MolBundle) -> dict[str, float | None]:
    seed_dir = RUN_ROOT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
    train_fit_idx, val_idx = train_test_split(
        split["train"],
        test_size=0.15,
        random_state=seed,
        stratify=bbb.y[split["train"]],
    )

    ext_split = exp.scaffold_split(b3db, seed + 101)
    external_idx_raw = ext_split["test"]
    external_idx = exp.exact_nonoverlap_indices(bbb.smiles, b3db, external_idx_raw)
    if len(external_idx) == 0 or len(np.unique(b3db.y[external_idx])) < 2:
        external_idx = external_idx_raw

    train_csv = seed_dir / "train.csv"
    val_csv = seed_dir / "val.csv"
    cal_csv = seed_dir / "cal.csv"
    test_csv = seed_dir / "test.csv"
    external_csv = seed_dir / "external.csv"

    _write_split_csv(train_csv, bbb.smiles, bbb.y, train_fit_idx)
    _write_split_csv(val_csv, bbb.smiles, bbb.y, val_idx)
    _write_split_csv(cal_csv, bbb.smiles, bbb.y, split["calibration"])
    _write_split_csv(test_csv, bbb.smiles, bbb.y, split["test"])
    _write_split_csv(external_csv, b3db.smiles, b3db.y, external_idx)

    accelerator = "gpu"
    devices = "1"
    try:
        import torch

        if not torch.cuda.is_available():
            accelerator = "cpu"
            devices = "1"
    except Exception:
        accelerator = "cpu"
        devices = "1"

    output_dir = seed_dir / "chemprop_out"
    train_cmd = [
        str(_chemprop_exe()),
        "train",
        "-q",
        "-q",
        "-i",
        str(train_csv),
        str(val_csv),
        str(cal_csv),
        "-o",
        str(output_dir),
        "--smiles-columns",
        "smiles",
        "--target-columns",
        "y",
        "--task-type",
        "classification",
        "--loss-function",
        "bce",
        "--metrics",
        "roc",
        "prc",
        "accuracy",
        "--tracking-metric",
        "roc",
        "--epochs",
        str(CHEMPROP_ARGS["epochs"]),
        "--patience",
        str(CHEMPROP_ARGS["patience"]),
        "--batch-size",
        str(CHEMPROP_ARGS["batch_size"]),
        "--accelerator",
        accelerator,
        "--devices",
        devices,
        "--class-balance",
        "--message-hidden-dim",
        str(CHEMPROP_ARGS["message_hidden_dim"]),
        "--depth",
        str(CHEMPROP_ARGS["depth"]),
        "--dropout",
        str(CHEMPROP_ARGS["dropout"]),
        "--ffn-hidden-dim",
        str(CHEMPROP_ARGS["ffn_hidden_dim"]),
        "--ffn-num-layers",
        str(CHEMPROP_ARGS["ffn_num_layers"]),
        "--ensemble-size",
        "1",
        "--num-workers",
        "0",
        "--pytorch-seed",
        str(seed),
    ]
    _run_cmd(train_cmd, seed_dir / "train.log")

    model_path = output_dir / "model_0" / "best.pt"
    if not model_path.exists():
        raise RuntimeError(f"missing Chemprop model checkpoint: {model_path}")

    cal_pred = _predict_with_chemprop(model_path, cal_csv, seed_dir / "cal_preds.csv", seed_dir / "predict_cal.log")
    test_pred = _predict_with_chemprop(model_path, test_csv, seed_dir / "test_preds.csv", seed_dir / "predict_test.log")
    ext_pred = _predict_with_chemprop(
        model_path,
        external_csv,
        seed_dir / "external_preds.csv",
        seed_dir / "predict_external.log",
    )

    calibration = exp.calibrate_global(
        bbb.y[split["calibration"]],
        cal_pred["y"].to_numpy(dtype=float),
    )
    out = exp.evaluate_split(
        bbb.y[split["test"]],
        test_pred["y"].to_numpy(dtype=float),
        calibration,
    )
    out.update(
        exp.evaluate_split(
            b3db.y[external_idx],
            ext_pred["y"].to_numpy(dtype=float),
            calibration,
            prefix="ood_external",
        )
    )
    out["n_train"] = float(len(train_fit_idx))
    out["n_calibration"] = float(len(split["calibration"]))
    out["n_test"] = float(len(split["test"]))
    out["n_external_test"] = float(len(external_idx))
    out["n_external_exact_overlap_removed"] = float(len(external_idx_raw) - len(external_idx))
    out["context_min_group"] = 0.0
    out["mondrian_groups"] = 0.0
    out["ood_guard_margin"] = 0.0
    out["chemprop_message_hidden_dim"] = float(CHEMPROP_ARGS["message_hidden_dim"])
    out["chemprop_depth"] = float(CHEMPROP_ARGS["depth"])
    out["chemprop_dropout"] = float(CHEMPROP_ARGS["dropout"])
    out["chemprop_ffn_hidden_dim"] = float(CHEMPROP_ARGS["ffn_hidden_dim"])
    out["chemprop_epochs_max"] = float(CHEMPROP_ARGS["epochs"])
    return out


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB")

    seed_metrics = []
    for seed in exp.SEEDS:
        print(f"chemprop_bbb seed={seed} start", flush=True)
        metrics = _train_seed(seed, bbb, b3db)
        seed_metrics.append(metrics)
        print(
            "chemprop_bbb seed=%s external_auroc=%.4f coverage=%.4f"
            % (
                seed,
                float(metrics["ood_external_auroc"]),
                float(metrics["ood_external_coverage_90"]),
            ),
            flush=True,
        )

    payload = {
        "model": "Chemprop v2 D-MPNN baseline",
        "method": (
            "Supervised Chemprop v2 message-passing neural network trained on "
            "BBB_Martins with duplicate-aware train/cal/test splits and seed-matched "
            "scaffold-derived non-overlap B3DB external evaluation."
        ),
        "hyperparameters": CHEMPROP_ARGS,
        "seed_metrics": seed_metrics,
        "summary": exp.summarize(seed_metrics),
        "status": "completed",
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
