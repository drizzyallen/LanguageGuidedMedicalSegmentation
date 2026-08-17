# Frozen split limitations

The Phase 0 benchmark retains the supplied splits exactly, as required. The
splits are image/slice-level rather than uniformly patient-level:

- QaTa-COV19-v2 has 444 `sub-S...` subject IDs represented in both training
  and validation. No defensible cluster is encoded for `covid_N` files, so
  those images are their own analysis units. No detected subject IDs cross
  into the supplied test set.
- MosMedData+ has 70 identified study/case clusters spanning more than one
  split: 64 span train/validation/test, five span train/test, and one spans
  train/validation. `Morozov_study_*` and `Jun_*_case*` files use their
  study/case prefix as `source_id`. No defensible cluster is encoded for
  `bjorke_N`, so those images are their own analysis units.
- BUSI filenames do not provide a defensible patient identifier; each image is
  its own analysis unit.

These overlaps are properties of the supplied frozen splits. Phase 0 does not
repair or re-create them. Test confidence intervals and paired tests must use
`source_id` cluster resampling where a cluster exists and explicitly state the
image-level limitation elsewhere.
