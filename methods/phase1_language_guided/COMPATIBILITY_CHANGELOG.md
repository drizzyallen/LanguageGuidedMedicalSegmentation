# Compatibility change log

All entries are line-by-line logical changes relative to the official setup.
No file under `upstream/` is modified.

## LViT-T

1. Installation order only: install the official Torch 1.8.0 and Torchvision
   0.9.0 pins before packages whose setup metadata imports Torch.
2. Resolver behavior: use pip's legacy resolver for the unchanged historical
   requirements because current `mkl-fft` metadata contradicts the official
   NumPy pin.
3. CUDA wheel build: replace the default Torch 1.8.0 CUDA 10.2 wheel with the
   official Torch 1.8.0 CUDA 11.1 wheel because CUDA 10.2 lacks RTX 3090
   `sm_86` kernels.
4. Algorithmic effect: none; framework and Torchvision versions are unchanged.

## RecLMIS

1. Package-index routing only: obtain the official CUDA 11.1 Torch wheels from
   PyTorch's wheel index.
2. Dependency overlay: `environments/reclmis-compatible.txt` reproduces the
   remaining official pins while omitting the unused, incompatible Torchaudio
   pin.
3. Missing dependencies: the overlay adds TIMM, Einops, SciPy, scikit-learn,
   THOP, Pillow, Matplotlib, ml-collections, and OpenAI CLIP because pinned
   upstream source imports them but its requirements file does not list them.
4. Algorithmic effect: none; repository source never imports Torchaudio, and
   all added packages satisfy existing upstream imports.

## ProLearn

1. No dependency compatibility change was required for installation.
2. Writable cache locations will be supplied by Phase 1 launchers rather than
   changing upstream source.
3. Algorithmic effect: none.
4. Gate-only shape probe: upstream `BiomedCLIP.encode_text_feature()` targets a
   nonexistent `CustomTextCLIP.text_encoder` attribute. The external gate calls
   the already-instantiated native `MMUNet.text_encoder` directly to log its
   embedding shape. Segmentation forward behavior is unchanged.

## 2026-09-16: RecLMIS main-training dependency recovery

The native reconstruction attention imports fairseq.utils.softmax lazily. Missing fairseq caused upstream to silently disable auxiliary losses. Both package installation attempts failed to build (0.10.2 packaging and 0.12.2 extension compilation). The external reclmis_compat.py provides only the required helper with the float32 calculation from fairseq v0.10.2. Both dataset smoke checks passed with all native losses asserted. No upstream source changes.
