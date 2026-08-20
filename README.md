# NeuroTx-MOOD

Experimental code and results for **NeuroTx-MOOD: Contextual Conformal Calibration for Blood-Brain Barrier Screening under Chemical Shift and Assay-Aware CNS Modeling**.

The archive contains the experiment pipeline, fixed split manifests, machine-readable result files, data checksums, and the processed ChEMBL panel.

## Included experiments

- Leakage-audited BBB_Martins-to-B3DB external evaluation across seeds 42–51.
- Global, proxy-context, local similarity-aware, and class-conditional conformal ablations at fixed point scores.
- Exact-canonical and parent-standardized overlap audits.
- Compound-grouped and compound-purged ChEMBL analyses.
- Paired confidence intervals and Benjamini–Hochberg-adjusted comparisons.

The release does **not** claim formal conformal validity under external chemical shift. External coverage is reported as empirical behavior under shift.

## Verify data and result integrity

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python prepare_data.py
python -m pytest -q
```

`prepare_data.py` downloads the non-redistributed inputs from pinned public URLs, verifies SHA-256 checksums, and validates the deposited ChEMBL panel. Set `NEUROTX_DATA_DIR` only when the four inputs are intentionally stored outside `data/`.

## B3DB metadata experiment

The release pins and verifies the complete official B3DB file and maps all 6167 TDC B3DB classification compounds. `b3db_metadata_probe.json` reports fixed-score coverage of 0.7751 ± 0.0156 for global calibration and 0.8157 ± 0.0101 for richer source-metadata calibration. The external and metadata result files also contain seed-paired effect estimates and 95% confidence intervals relative to global calibration.

## Main entry points

- `main.py`: primary BBB/CNS experiment pipeline and shared utilities.
- `prepare_data.py`: pinned downloads and checksum validation.
- `run_compound_controlled_analyses.py`: compound-controlled ChEMBL analyses, parent-standardized B3DB sensitivity, and paired statistics.
- `run_temporal_purged_bootstrap.py`: temporal-test bootstrap intervals.
- `run_external_conformal_diagnostics.py`: class-wise and alpha-grid external diagnostics.
- `b3db_metadata_probe.py`: fixed-score source-metadata experiment.
- `applicability_region_probes.py`: supported-region and local-similarity analyses.

Machine-readable experiment outputs are stored as top-level JSON files. `data_manifest.json` and `manifests/SHA256SUMS.txt` identify the exact inputs and released artifacts.

## Data and licensing

The exact file hashes, source locators, row counts, and redistribution status are in `data_manifest.json`. BBB_Martins is downloaded rather than redistributed because its TDC page does not specify a dataset license. The processed ChEMBL panel is deposited under CC BY-SA 3.0. B3DB is CC0. See [LICENSES/DATA_LICENSES.md](LICENSES/DATA_LICENSES.md).

Original source code is licensed under Apache-2.0. Dataset and model licenses remain separate and are not changed by the code license.

## Citation

Use `CITATION.cff` for the software release and cite the manuscript when available. A version-specific Zenodo DOI should be used once the GitHub--Zenodo deposit is minted.
