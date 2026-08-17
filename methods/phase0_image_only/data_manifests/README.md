# Immutable dataset manifests

`qata.csv`, `mosmed.csv`, and `busi.csv` preserve the existing train,
validation, and test membership. Paths are relative to each manifest. The
loader verifies SHA-256 hashes before returning cases.

Required columns:

`sample_id,source_id,image_path,mask_path,text_path_or_id,split,image_sha256,mask_sha256`

`text_path_or_id` is intentionally empty for Phase 0. No text is generated
from masks.

The three original MosMed split CSVs are snapshotted under `source_splits/`.
No split-generation script or seed was found in the supplied MosMed directory;
that provenance gap must be resolved or explicitly documented before final
benchmark sign-off.

Current manifest counts:

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| QaTa-COV19-v2 | 5716 | 1429 | 2113 |
| MosMedData+ | 1746 | 437 | 546 |
| BUSI | 413 | 104 | 130 |

To reproduce the manifests locally, run `generate.py --datasets-root PATH`.
The dataset root is supplied at runtime and is not embedded in benchmark code.
