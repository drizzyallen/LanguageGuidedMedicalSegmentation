# Adapter validation

Date: 2026-09-12

The shared adapter was run across every row of both Phase 1 manifest copies.
The copies preserve Phase 0 image/mask records while using annotation-defined
split membership. The frozen Phase 0 manifests remain unchanged.
All referenced image and mask files exist, every `sample_id` is unique, and
every row resolves to one unique nonempty report.

| Dataset | Train | Validation | Test | Total | Result |
|---|---:|---:|---:|---:|---|
| QaTa | 5,716 | 1,429 | 2,113 | 9,258 | PASS |
| MosMed | 2,183 | 273 | 273 | 2,729 | PASS |

Additional policy checks passed:

- validation augmentation is rejected;
- test augmentation is rejected;
- geometric training augmentation without a matching text transform is
  rejected;
- embedding caching without explicit method permission is rejected;
- a permitted cache binds the exact method, encoder fingerprint, sample ID,
  and report text and reuses the cached result.

The source hashes are pinned in `PROVENANCE.yaml`.
