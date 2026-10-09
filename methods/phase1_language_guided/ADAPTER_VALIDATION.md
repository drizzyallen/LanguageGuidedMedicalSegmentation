# Adapter validation

Date: 2026-10-09 (supersedes the 2026-09-12 record, which validated the
archived annotation-workbook split).

`validate_adapter.py` was run across every row of both Phase 1 manifests.
Each manifest is byte-identical to the frozen Phase 0 manifest (checked by
sha256 inside the script). All referenced image and mask files exist, every
`sample_id` is unique, and every row resolves to exactly one nonempty report,
joined by image basename (never by directory order).

| Dataset | Train | Validation | Test | Total | Unique reports | Manifest sha256 | Result |
|---|---:|---:|---:|---:|---:|---|---|
| QaTa | 5,716 | 1,429 | 2,113 | 9,258 | 394 | `e12e88a0…` | PASS |
| MosMed | 1,746 | 437 | 546 | 2,729 | 115 | `5f2cbe3a…` | PASS |

Additional policy checks passed:

- validation augmentation is rejected;
- test augmentation is rejected;
- geometric training augmentation without a matching text transform is
  rejected;
- embedding caching without explicit method permission is rejected;
- a permitted cache binds the exact method, encoder fingerprint, sample ID,
  and report text and reuses the cached result.

Report text comes unchanged from the annotation workbooks pinned in
`adapter.py` (`PINNED_SHA256`). No report was generated or edited in this
repository. See `PROVENANCE_CURRENT.md` for the current input hashes.
