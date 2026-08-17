# Phase 0: frozen image-only benchmarks

This directory stages the frozen U-Net, TriPathLesionNet, and Panoptic FPN
implementations for retraining. No training has been run from this directory.

## Freeze contract

- Upstream source is imported through thin wrappers in `models/`; it is not
  copied or edited.
- `PROVENANCE.yaml` pins the exact upstream repository commit and source files.
- `configs/*.yaml` records the original command-line defaults and model/loss
  settings. These values are descriptive and versioned; wrappers do not tune
  them.
- Dataset membership is supplied by the immutable per-dataset CSV files in
  `data_manifests/`. Dataset paths are relative to those manifests.
- Seeds are fixed to `[1001, 1002, 1003]` for the eventual three-run benchmark.
- No architecture, loss, optimizer, scheduler, augmentation, or checkpoint
  selection behavior has been changed in Phase 0.

## Known gaps before retraining

Use `run.py check-data` for manifest/configuration checks and `run.py train` as
the only supported training entry point. It enforces deterministic algorithms,
explicitly seeds DataLoader workers and samplers, and records configuration and
manifest hashes. After training, `export_predictions.py` exports one float32
probability array, one thresholded PNG, and one metric row per test `sample_id`.
`smoke_checks.py` performs the non-training acceptance suite.

The requested Git ref `methods/phase0_image_only` was not present on `origin`
when this staging area was created; this is a directory in the current unborn
root worktree, not a checked-out branch.
