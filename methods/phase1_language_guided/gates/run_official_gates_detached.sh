#!/usr/bin/env bash
set -uo pipefail

ROOT="/data/ramialle/LanguageGuidedMedicalSegmentation"
GATES="$ROOT/methods/phase1_language_guided/gates"
CONDA="/data/ramialle/miniconda3/bin/conda"

export CUDA_VISIBLE_DEVICES="${GATE_GPU:-0}"
cd "$ROOT"

echo "[$(date --iso-8601=seconds)] Starting official QaTa gates on physical GPU $CUDA_VISIBLE_DEVICES"

echo "[$(date --iso-8601=seconds)] Starting LViT"
if "$CONDA" run --no-capture-output -n phase1_official_lvit \
    python -u "$GATES/official_qata_run.py" lvit --device cuda:0; then
  echo "[$(date --iso-8601=seconds)] LViT finished successfully"
else
  status=$?
  echo "[$(date --iso-8601=seconds)] LViT failed with exit code $status"
fi

echo "[$(date --iso-8601=seconds)] Starting ProLearn"
if "$CONDA" run --no-capture-output -n phase1_official_prolearn \
    python -u "$GATES/official_qata_run.py" prolearn --device cuda:0; then
  echo "[$(date --iso-8601=seconds)] ProLearn finished successfully"
else
  status=$?
  echo "[$(date --iso-8601=seconds)] ProLearn failed with exit code $status"
  exit "$status"
fi

echo "[$(date --iso-8601=seconds)] All official QaTa gates completed"
