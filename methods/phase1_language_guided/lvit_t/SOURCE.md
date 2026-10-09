# LViT-T — source record

| Field | Value |
|---|---|
| Paper | Li Z, Li Y, Li Q, Wang P, Guo D, Lu L, Jin D, Zhang Y, Hong Q. *LViT: Language Meets Vision Transformer in Medical Image Segmentation.* IEEE Transactions on Medical Imaging, 2023. |
| Repository | https://github.com/HUANGLIZI/LViT |
| Pinned commit | `ba90775ead37d6f445f65ce83d1a0be64157f76f` (submodule `upstream/LViT`) |
| License | MIT (Copyright (c) 2022 Zihan Li); preserved in `upstream/LViT/LICENSE` |
| Original environment | Python 3.7, `torch==1.8.0`, `torchvision==0.9.0`, upstream `requirements.txt` (sha256 `49e12e3e…`) |
| Local environment | Conda `phase1_official_lvit`: Python 3.7.16, Torch 1.8.0+cu111, CUDA 11.1, cuDNN 8005, RTX 3090. Full package list: `runs/phase1/lvit/<dataset>/<seed>/environment.txt` |
| Text at train / test | Yes / Yes |

## Native settings preserved

LViT-T model (`nets/LViT.py`, `Config.get_CTranS_config()`), `WeightedDiceBCE(0.5, 0.5)`, Adam, `CosineAnnealingWarmRestarts(T_0=10, T_mult=1, eta_min=1e-4)`, batch size 2, maximum 2,000 epochs, early-stopping patience 50, 224x224 RGB input. Text: frozen BERT-base (`bert-embedding`, 12-layer, 768-d, uncased) token embeddings, first 10 tokens, zero-padded, as in the upstream loader. Embeddings are cached by sample ID and exact report text; the encoder is frozen, so caching does not change the algorithm.

Learning rate 3e-4: upstream `Config.py` sets `learning_rate = 1e-3` with the comment `MoNuSeg: 1e-3, Covid19: 3e-4`. The Covid19 value was used for both QaTa and MosMedData+ (upstream gives no MosMedData+ value).

## Modifications (all outside `upstream/`)

No upstream file is changed (`patches/lvit.patch` is empty by design).

1. Environment: Torch installed before the requirements; pip legacy resolver for the historical pins; official Torch 1.8.0 CUDA 11.1 wheel substituted for the CUDA 10.2 wheel to support RTX 3090 (`environments/OFFICIAL_BUILD_LOG.md`, `COMPATIBILITY_CHANGELOG.md`).
2. Data: shared adapter (`adapter.py`) joins frozen Phase 0 manifest rows to reports by image basename; manifests are byte-identical to Phase 0.
3. Training augmentation disabled (text-safe policy: flips/rotations can contradict laterality and location text).
4. Runner (`main_experiment.py`): seeds 1001-1003, deterministic algorithms, validation mean per-case Dice for checkpoint selection, single test evaluation guarded by `test_started.json`, nonfinite-loss stop that tests the pre-event best checkpoint, explicit process exit after each run.

## Note on `PROVENANCE.yaml`

`PROVENANCE.yaml` is hashed into every run's `run_config.json` and is left unchanged so those hashes still verify. Its `phase1_manifest` hashes and `split_authority` entries describe the archived 2026-09-16 launch. The manifests used by the reported runs are byte-identical to Phase 0: QaTa sha256 `e12e88a0…`, MosMed sha256 `5f2cbe3a…`.

## Results

`reports/PHASE1_LANGUAGE_GUIDED_RESULTS.md`; per-run records in `runs/phase1/lvit/`.
