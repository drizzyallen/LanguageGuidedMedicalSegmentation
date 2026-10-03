# Environment compatibility commands

Run these only after the unchanged official installation attempt has been
recorded in `OFFICIAL_BUILD_LOG.md`.

## LViT-T

Install the official Torch pins first, then install the unchanged upstream
requirements with pip's legacy resolver. This accepts the historical package
set despite newly published `mkl-fft` metadata contradicting the repository's
NumPy pin.

```bash
conda run -n phase1_official_lvit python -m pip install torch==1.8.0 torchvision==0.9.0
conda run -n phase1_official_lvit python -m pip install --use-deprecated=legacy-resolver -r upstream/LViT/requirements.txt
conda run -n phase1_official_lvit python -m pip install --force-reinstall --no-deps torch==1.8.0+cu111 torchvision==0.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
```

The final wheel-build substitution retains Torch/Torchvision 1.8.0/0.9.0 APIs
while adding `sm_86` support required by this machine's RTX 3090 GPUs.

## RecLMIS

Install the official CUDA 11.1 Torch pair, then use the external overlay that
removes only the unused and impossible Torchaudio pin.

```bash
conda run -n phase1_official_reclmis python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 --extra-index-url https://download.pytorch.org/whl/cu111
conda run -n phase1_official_reclmis python -m pip install -r methods/phase1_language_guided/environments/reclmis-compatible.txt
```

## ProLearn

No dependency overlay is currently required.
