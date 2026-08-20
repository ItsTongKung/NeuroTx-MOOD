"""Export exact split indices and dataset manifests for experiment auditing.

This script materializes the deterministic train/calibration/test indices used
by the main NeuroTx-MOOD experiments so that the archive contains
explicit split files in addition to code and seed definitions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import main as exp


ROOT = Path(__file__).resolve().parent


def to_int_list(values: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(values, dtype=int).tolist()]


def chembl_temporal_indices(chembl: exp.MolBundle) -> dict[str, np.ndarray]:
    return exp.purged_temporal_three_way_split(chembl)


def main() -> None:
    bbb = exp.load_bbb_tab(exp.DATA_DIR / "bbb_martins.tab", "BBB_Martins")
    b3db = exp.load_bbb_tab(exp.DATA_DIR / "b3db_classification.tab", "B3DB")
    chembl = exp.load_chembl(exp.DATA_DIR / "chembl_cns_targets.csv")

    payload: dict[str, object] = {
        "description": (
            "Deterministic split manifests for the NeuroTx-MOOD BMC Bioinformatics submission "
            "package. Exact index files are exported for BBB_Martins, "
            "zero-overlap external B3DB evaluation folds, and ChEMBL random/temporal "
            "splits so that the submission artifact contains explicit rerun inputs."
        ),
        "data_dir": "data",
        "seed_range": exp.SEEDS,
        "bundles": {
            "BBB_Martins": {"n": int(len(bbb.y)), "path": "data/bbb_martins.tab"},
            "B3DB": {"n": int(len(b3db.y)), "path": "data/b3db_classification.tab"},
            "ChEMBL_CNS_targets": {
                "n": int(len(chembl.y)),
                "path": "data/chembl_cns_targets.csv",
            },
        },
        "bbb_main_splits": {},
        "b3db_external_zero_overlap_splits": {},
        "b3db_external_parent_filtered_splits": {},
        "chembl_random_splits": {},
        "chembl_temporal_split": {},
    }

    npz_payload: dict[str, np.ndarray] = {}

    for seed in exp.SEEDS:
        bbb_split = exp.duplicate_aware_stratified_three_way_split(bbb, seed)
        external_idx_raw, external_idx, attempt = exp.external_b3db_zero_overlap_test_indices(bbb, b3db, seed)
        parent_raw, parent_idx, parent_attempt = exp.external_b3db_parent_nonoverlap_test_indices(bbb, b3db, seed)
        b3db_split = exp.scaffold_split(b3db, seed + 101 + attempt)
        chembl_split = exp.duplicate_aware_stratified_three_way_split(chembl, seed)

        payload["bbb_main_splits"][str(seed)] = {
            "train": to_int_list(bbb_split["train"]),
            "calibration": to_int_list(bbb_split["calibration"]),
            "test": to_int_list(bbb_split["test"]),
        }
        payload["b3db_external_zero_overlap_splits"][str(seed)] = {
            "train": to_int_list(b3db_split["train"]),
            "calibration": to_int_list(b3db_split["calibration"]),
            "test_raw": to_int_list(external_idx_raw),
            "test_zero_overlap": to_int_list(external_idx),
            "regeneration_attempts": int(attempt),
            "remaining_exact_overlap_count": int(
                exp.overlap_stats(
                    [exp.canonical_smiles(s) for s in bbb.smiles],
                    [exp.canonical_smiles(b3db.smiles[int(i)]) for i in external_idx],
                )["count"]
            ),
        }
        payload["b3db_external_parent_filtered_splits"][str(seed)] = {
            "test_raw": to_int_list(parent_raw),
            "test_parent_filtered": to_int_list(parent_idx),
            "regeneration_attempts": int(parent_attempt),
            "remaining_parent_overlap_count": int(
                exp.overlap_stats(
                    [exp.parent_inchikey_block(s) for s in bbb.smiles],
                    [exp.parent_inchikey_block(b3db.smiles[int(i)]) for i in parent_idx],
                )["count"]
            ),
        }
        payload["chembl_random_splits"][str(seed)] = {
            "train": to_int_list(chembl_split["train"]),
            "calibration": to_int_list(chembl_split["calibration"]),
            "test": to_int_list(chembl_split["test"]),
        }

        npz_payload[f"bbb_seed_{seed}_train"] = np.asarray(bbb_split["train"], dtype=np.int32)
        npz_payload[f"bbb_seed_{seed}_calibration"] = np.asarray(bbb_split["calibration"], dtype=np.int32)
        npz_payload[f"bbb_seed_{seed}_test"] = np.asarray(bbb_split["test"], dtype=np.int32)
        npz_payload[f"b3db_seed_{seed}_train"] = np.asarray(b3db_split["train"], dtype=np.int32)
        npz_payload[f"b3db_seed_{seed}_calibration"] = np.asarray(b3db_split["calibration"], dtype=np.int32)
        npz_payload[f"b3db_seed_{seed}_test_raw"] = np.asarray(external_idx_raw, dtype=np.int32)
        npz_payload[f"b3db_seed_{seed}_test_zero_overlap"] = np.asarray(external_idx, dtype=np.int32)
        npz_payload[f"b3db_seed_{seed}_test_parent_filtered"] = np.asarray(parent_idx, dtype=np.int32)
        npz_payload[f"chembl_seed_{seed}_train"] = np.asarray(chembl_split["train"], dtype=np.int32)
        npz_payload[f"chembl_seed_{seed}_calibration"] = np.asarray(chembl_split["calibration"], dtype=np.int32)
        npz_payload[f"chembl_seed_{seed}_test"] = np.asarray(chembl_split["test"], dtype=np.int32)

    temporal_split = chembl_temporal_indices(chembl)
    payload["chembl_temporal_split"] = {
        "train": to_int_list(temporal_split["train"]),
        "calibration": to_int_list(temporal_split["calibration"]),
        "test": to_int_list(temporal_split["test"]),
        "train_year_max": float(np.nanmax(chembl.document_years[temporal_split["train"]])),
        "calibration_year_min": float(np.nanmin(chembl.document_years[temporal_split["calibration"]])),
        "calibration_year_max": float(np.nanmax(chembl.document_years[temporal_split["calibration"]])),
        "test_year_min": float(np.nanmin(chembl.document_years[temporal_split["test"]])),
        "test_year_max": float(np.nanmax(chembl.document_years[temporal_split["test"]])),
    }

    npz_payload["chembl_temporal_train"] = np.asarray(temporal_split["train"], dtype=np.int32)
    npz_payload["chembl_temporal_calibration"] = np.asarray(temporal_split["calibration"], dtype=np.int32)
    npz_payload["chembl_temporal_test"] = np.asarray(temporal_split["test"], dtype=np.int32)

    json_path = ROOT / "split_manifests.json"
    npz_path = ROOT / "split_indices.npz"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(npz_path, **npz_payload)
    print(f"wrote {json_path}")
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
