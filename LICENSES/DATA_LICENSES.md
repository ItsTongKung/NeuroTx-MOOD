# Data and model licensing notes

The Apache-2.0 license at repository root applies only to original NeuroTx-MOOD source code. It does not relicense third-party datasets, pretrained models, or their documentation.

| Input | Distribution in this release | Source and terms |
|---|---|---|
| `bbb_martins.tab` | Not committed; downloaded by `prepare_data.py` | Therapeutics Data Commons. The current dataset page states that the dataset license is not specified. |
| `b3db_classification.tab` | Not committed; downloaded by `prepare_data.py` | TDC-formatted B3DB snapshot. Upstream B3DB materials are released under CC0-1.0. |
| `B3DB_official_classification.tsv` | Not committed; pinned download | Official B3DB repository, CC0-1.0. |
| `chembl_cns_targets.csv` | Included | Derived from ChEMBL and distributed under CC BY-SA 3.0 with attribution. |
| Authored JSON results and split manifests | Included | Research outputs released with the repository; cite the software release and manuscript. |

ChEMBL attribution: ChEMBL is an EMBL-EBI resource. The exact processed panel is provided because the original extraction did not record a ChEMBL release identifier; its SHA-256 checksum defines the analysed snapshot.

Pretrained model weights are not redistributed. Users must follow the license and access terms of each model provider.
