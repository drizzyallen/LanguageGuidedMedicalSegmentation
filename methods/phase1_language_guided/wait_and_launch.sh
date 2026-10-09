#!/usr/bin/env bash
# Wait until a healthy GPU has enough free memory, smoke-check there, then
# start the seed queue. Datasets are launched one at a time so two waiters
# never claim the same free memory.
# Usage: wait_and_launch.sh METHOD NEED_MIB DATASET [DATASET ...]
set -euo pipefail
method="$1"; need_mib="$2"; shift 2
phase1="/data/ramialle/LanguageGuidedMedicalSegmentation/methods/phase1_language_guided"
python_bin="/data/ramialle/miniconda3/envs/phase1_official_${method}/bin/python"
cd /data/ramialle/LanguageGuidedMedicalSegmentation

free_gpu() {
  # Query each healthy GPU separately: failed GPUs make a combined query error out.
  for index in $(nvidia-smi -L 2>/dev/null | sed -n 's/^GPU \([0-9]*\):.*/\1/p'); do
    free=$(nvidia-smi -i "$index" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null || echo 0)
    if [ "${free:-0}" -ge "$need_mib" ]; then echo "$index"; return; fi
  done
}

for dataset in "$@"; do
  echo "$(date -Is) WAIT $method $dataset for >= $need_mib MiB free"
  until gpu=$(free_gpu) && [ -n "$gpu" ]; do sleep 120; done
  echo "$(date -Is) GPU $gpu free; smoke $method $dataset"
  rm -rf "$phase1/smoke_runs_v2/$method/$dataset"
  CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$python_bin" -u "$phase1/main_experiment.py" "$method" --dataset "$dataset" --seed 1001 --smoke
  "$python_bin" "$phase1/launch_main_queues.py" "$method" --dataset "$dataset=$gpu"
  echo "$(date -Is) LAUNCHED $method $dataset GPU $gpu"
  sleep 600  # let the new job's memory show up before choosing the next GPU
done
