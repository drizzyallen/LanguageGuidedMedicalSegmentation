# Official environment build record

Date: 2026-09-12

These attempts used each repository unchanged at the commit recorded in
`../PROVENANCE.yaml`. The base Conda environments were created successfully.

## LViT-T

- Environment: `phase1_official_lvit`, Python 3.7.16.
- Official command: `python -m pip install -r requirements.txt`.
- First failure: `entmax==1.0` executes its package metadata by importing
  `torch`, but Torch appears later in the official requirements file and was
  not installed yet (`ModuleNotFoundError: No module named 'torch'`).
- Documented ordering retry: installed the exact `torch==1.8.0` and
  `torchvision==0.9.0` first, then retried the unchanged requirements.
- Retry failure: the official `numpy==1.17.5` pin is unsatisfiable with
  `mkl-fft==1.3.1`, whose current wheel metadata requires NumPy >=1.21.4 and
  <1.22. Pip reported `ResolutionImpossible`.
- Status: official requirements do not resolve on the current package index.
- External compatibility result: the exact upstream package list installed
  with Torch-first ordering and pip's legacy resolver. LViT model and dataset
  imports succeed with Torch 1.8.0 and NumPy 1.17.5. Resolver-reported
  dependency conflicts remain preserved above as compatibility warnings.
- GPU probe failure: PyPI's default Torch 1.8.0 wheel uses CUDA 10.2 and has no
  RTX 3090 `sm_86` kernel. The API-identical official CUDA 11.1 wheel was
  substituted externally and passed a CUDA execution probe on GPU 3.

## RecLMIS

- Environment: `phase1_official_reclmis`, Python 3.9.25.
- Official command: `python -m pip install -r requirements.txt`.
- First failure: the default index does not expose the CUDA-local wheel
  `torch==1.8.0+cu111`.
- Index retry: the exact Torch and Torchvision CUDA 11.1 wheels installed from
  PyTorch's official CUDA 11.1 wheel index.
- Full retry failure: `torchaudio==2.4.1+cu118` requires `torch==2.4.1`, which
  directly conflicts with the repository's `torch==1.8.0+cu111`. Pip reported
  `ResolutionImpossible`. RecLMIS source contains no Torchaudio import.
- Status: the official requirements are internally inconsistent.
- Post-install import probe: after omitting Torchaudio, importing the pinned
  model first failed on `timm`, revealing that several source imports are also
  absent from the official requirements file. They are enumerated in the
  external compatibility overlay and change log.
- External compatibility result: the overlay installed successfully. RecLMIS
  model, dataset, and OpenAI CLIP imports succeed with Torch 1.8.0+cu111 and
  NumPy 1.22.3.

## ProLearn

- Environment: `phase1_official_prolearn`, Python 3.11.15.
- Official command: `python -m pip install -r requirements.txt`.
- Result: success with the unchanged official requirements.
- Verified core versions: Torch 2.5.1+cu124, Torchvision 0.20.1+cu124,
  Transformers 4.43.3, MONAI 1.5.0, and OpenCLIP 2.32.0.
- Warning: `torchmetrics==1.3.0` is yanked but remains downloadable.
- Runtime note: this execution sandbox could not communicate with the NVIDIA
  driver, so GPU execution is not asserted by this environment-build step.
- Import result: training entry point, testing entry point, and ProLearn model
  imports all succeed.
- Gate probe failure: `BiomedCLIP.encode_text_feature()` references a missing
  `CustomTextCLIP.text_encoder` attribute. The external gate bypasses only that
  broken convenience method for shape logging and invokes the native model text
  encoder directly.
