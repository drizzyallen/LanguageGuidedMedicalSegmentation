# TriPathLesionNet — source record

| Field | Value |
|---|---|
| Reference | The student's own architecture; no publication. Design described in the upstream module docstring. |
| Source file | `TriPathLesionNetmodel/TriPathLesionNet.py` |
| Original environment | Upstream `requirements.txt` in `upstream/Summer2026Research` |
| Local environment | Conda `myenv`: Python 3.10.20, Torch 2.5.1+cu121, RTX 3090. Full package list: `runs/phase0/<method>/<dataset>/<seed>/environment.txt` |
| Source repository | https://github.com/drizzyallen/Summer2026Research, pinned commit `fdeb44037aba52900d852271293b165a697ecf06` (submodule `upstream/Summer2026Research`) |
| License | **None recorded**: the upstream repository has no LICENSE file. It is the student's own code; add a license there before any public release. |
| Text at train / test | No / No |
| Pretrained weights | None (trained from scratch) |

## Frozen settings (`configs/tripath_lesionnet.yaml`)

Image-only network with three outputs (segmentation, lesion boundary, coarse location); base channels 32, dropout 0.10, context dilations 1/2/4/8, GroupNorm (8 groups); loss = Dice (1.0) + boundary (0.20) + location (0.20) + consistency (0.10); AdamW lr 3e-4, weight decay 1e-4; ReduceLROnPlateau (max, patience 5, factor 0.5); 100 epochs, batch 4, early stopping on validation Dice (patience 15); checkpoint = best validation Dice (strictly greater); 384x384 grayscale; same augmentation as U-Net v2; AMP on; threshold: probability >= 0.5 on the segmentation head.

## Modifications (all outside `upstream/`)

No upstream file is changed and no correctness fix was made (`CORRECTNESS_CHANGES.md`).

1. Manifest-based loading: `models/<method>.py` imports the upstream module unchanged and feeds it the frozen manifest rows (`data_manifests/`) instead of hard-coded paths.
2. Determinism (`determinism.py`): seeded Python, NumPy, Torch and CUDA RNGs; deterministic algorithms; seeded DataLoader workers and samplers.
3. Prediction export (`export_predictions.py`): one float32 probability map, one thresholded mask and one metric row per test `sample_id`, ordered as in the manifest.
4. Final comparison: predictions are re-scored on the native mask grid by `evaluation/shared_evaluator.py`. The Phase 0 report keeps the original 384x384 scoring.

## Results

`reports/PHASE0_IMAGE_ONLY_RESULTS.md` (384x384 scoring), `reports/FINAL_FIVE_METHOD_COMPARISON.md` (native grid); per-run records in `runs/phase0/<method>/`.
