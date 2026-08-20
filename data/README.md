# Data inputs

Run from repository root:

```bash
python prepare_data.py
```

The command downloads and verifies:

- `bbb_martins.tab`
- `b3db_classification.tab`
- `B3DB_official_classification.tsv`

The exact processed `chembl_cns_targets.csv` is included in the release and validated in place. Do not replace any file without updating `data_manifest.json` and documenting the resulting analysis version.

BBB_Martins is deliberately not committed because the TDC page does not specify its redistribution license. B3DB is CC0-1.0. The processed ChEMBL panel is distributed under CC BY-SA 3.0.
