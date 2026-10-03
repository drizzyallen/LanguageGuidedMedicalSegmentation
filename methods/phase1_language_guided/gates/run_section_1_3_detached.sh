#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/ramialle/LanguageGuidedMedicalSegmentation"
PHASE1="$ROOT/methods/phase1_language_guided"
GATES="$PHASE1/gates"
CONDA="/data/ramialle/miniconda3/bin/conda"

export CUDA_VISIBLE_DEVICES="${GATE_GPU:-0}"
cd "$ROOT"

stage() {
  local name="$1"
  echo "[$(date --iso-8601=seconds)] START $name"
  shift
  "$@"
  echo "[$(date --iso-8601=seconds)] PASS $name"
}

echo "[$(date --iso-8601=seconds)] Starting Section 1.3 on physical GPU $CUDA_VISIBLE_DEVICES"

stage adapter_validation python3 "$PHASE1/validate_adapter.py"
stage lvit_gates_1_to_6 "$CONDA" run --no-capture-output -n phase1_official_lvit \
  python -u "$GATES/lvit_gate.py" --device cuda:0
stage reclmis_gates_1_to_7 "$CONDA" run --no-capture-output -n phase1_official_reclmis \
  python -u "$GATES/reclmis_gate.py" --device cuda:0
stage prolearn_gates_1_to_6 "$CONDA" run --no-capture-output -n phase1_official_prolearn \
  python -u "$GATES/prolearn_gate.py" --device cuda:0
stage lvit_official_check "$CONDA" run --no-capture-output -n phase1_official_lvit \
  python -u "$GATES/official_qata_run.py" lvit --device cuda:0
stage prolearn_official_check "$CONDA" run --no-capture-output -n phase1_official_prolearn \
  python -u "$GATES/official_qata_run.py" prolearn --device cuda:0

echo "[$(date --iso-8601=seconds)] SECTION 1.3 COMPLETE"
