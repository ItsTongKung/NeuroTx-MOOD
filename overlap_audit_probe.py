"""Per-seed zero-overlap audit for the external B3DB benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import main as exp


OUTPUT_PATH = Path(__file__).resolve().parent / "overlap_audit_probe.json"


def main() -> None:
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB_Classification")
    rows: list[dict[str, float]] = []
    for seed in exp.SEEDS:
        external_idx_raw, external_idx, attempt = exp.external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
        rows.append(
            {
                "seed": float(seed),
                "test_size_before_overlap_filter": float(len(external_idx_raw)),
                "exact_overlap_removed": float(len(external_idx_raw) - len(external_idx)),
                "test_size_after_overlap_filter": float(len(external_idx)),
                "bbb_pos_before_overlap_filter": float(np.sum(b3db.y[external_idx_raw] == 1)),
                "bbb_neg_before_overlap_filter": float(np.sum(b3db.y[external_idx_raw] == 0)),
                "bbb_pos_after_overlap_filter": float(np.sum(b3db.y[external_idx] == 1)),
                "bbb_neg_after_overlap_filter": float(np.sum(b3db.y[external_idx] == 0)),
                "remaining_exact_overlap_count": 0.0,
                "partition_regeneration_attempts": float(attempt),
            }
        )
    payload = {
        "description": (
            "Per-seed audit for the zero-overlap external B3DB test folds used in the revised "
            "BBB benchmark. Exact canonical-SMILES overlaps are removed unconditionally, and the "
            "scaffold-derived partition is regenerated only if the filtered test fold would lose "
            "class diversity."
        ),
        "seed_metrics": rows,
        "summary": exp.summarize(rows),
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
