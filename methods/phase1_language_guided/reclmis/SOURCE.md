# RecLMIS — source record

| Field | Value |
|---|---|
| Paper | Huang X, Li H, Cao M, Chen L, You C, An D. *Cross-Modal Conditioned Reconstruction for Language-guided Medical Image Segmentation.* IEEE Transactions on Medical Imaging, 2024 (arXiv:2404.02845). |
| Repository | https://github.com/ShawnHuang497/RecLMIS |
| Pinned commit | `d3265c8b4d4582a94595fe1c8b1892c618bc9715` (submodule `upstream/RecLMIS`) |
| License | MIT (Copyright (c) 2024 ShashankHuang); preserved in `upstream/RecLMIS/LICENSE` |
| Original environment | Python 3.9, `torch==1.8.0+cu111`, upstream `requirements.txt` (sha256 `86aaa2a3…`) |
| Local environment | Conda `phase1_official_reclmis`: Python 3.9.25, Torch 1.8.0+cu111, RTX 3090. Full package list: `runs/phase1/reclmis/<dataset>/<seed>/environment.txt` |
| Pretrained weights | OpenAI CLIP ViT-B/32 (`assets/reclmis/ViT-B-32.pt`, loaded through the upstream `_PT_NAME` table); author QaTa checkpoint used only for the official-check gate |
| Text at train / test | Yes / Yes |

## Native settings preserved

RecLMIS model and its three auxiliary losses (`loss_ccl`, `loss_text_rec`, `loss_img_rec`) with the upstream `loss_weight` table, plus `WeightedDiceBCE(0.5, 0.5)`; CLIP tokenization with `token_len = 18`; Adam, learning rate 3e-4; 224x224 RGB input. Per-dataset settings come from the upstream config files:

| Dataset | Config | Batch | Max epochs | Patience | Schedule |
|---|---|---|---|---|---|
| QaTa | `Config_covid19.py` | 32 | 2,000 | 100 | `CosineAnnealingWarmRestarts(10, 1, 1e-4)` (`cosineLR`) |
| MosMed | `Config_MosMedPlus.py` | 20 | 400 | 30 | `LambdaLR(max(0.99^epoch, 0.1))` (`exp`) |

The runner asserts on every training step that all three auxiliary losses are present.

## Modifications (all outside `upstream/`)

No upstream file is changed (`patches/reclmis.patch` is empty by design).

1. Environment: CUDA 11.1 Torch wheels from PyTorch's index; `environments/reclmis-compatible.txt` drops the unused, unsatisfiable `torchaudio==2.4.1` pin and adds packages the source imports but the requirements omit (timm, einops, scipy, scikit-learn, thop, Pillow, matplotlib, ml-collections, openai-clip).
2. `reclmis_compat.py`: fairseq 0.10.2 and 0.12.2 failed to build. Without fairseq, an upstream bare `except` silently disables the auxiliary losses. This file supplies only `fairseq.utils.softmax`, copied from fairseq v0.10.2.
3. Evaluation passes a zero mask placeholder: the upstream forward signature takes the mask, which is used only by the training-time reconstruction branch, so no ground truth enters inference.
4. Data, augmentation and runner changes are the same as for LViT-T (`../lvit_t/SOURCE.md`, items 2-4). The nonfinite-loss stop was triggered once (QaTa seed 1003, epoch 97); the upstream loop has no finiteness check, so it would have tested the same pre-event checkpoint.

## Note on `PROVENANCE.yaml`

See `../lvit_t/SOURCE.md`: the file is left unchanged because run configs hash it; the reported runs use the Phase 0 manifests.

## Results

`reports/PHASE1_LANGUAGE_GUIDED_RESULTS.md`; per-run records in `runs/phase1/reclmis/`.
