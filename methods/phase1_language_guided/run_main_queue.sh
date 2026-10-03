#!/usr/bin/env bash
set -euo pipefail
method="$1"
dataset="$2"
gpu="$3"
phase1="/data/ramialle/LanguageGuidedMedicalSegmentation/methods/phase1_language_guided"
export CUDA_VISIBLE_DEVICES="$gpu"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>"$phase1/${method}_${dataset}_queue.lock"
flock -n 9
python_bin="/data/ramialle/miniconda3/envs/phase1_official_${method}/bin/python"
cd /data/ramialle/LanguageGuidedMedicalSegmentation
for seed in 1001 1002 1003; do
  echo "START $method $dataset $seed GPU $gpu"
  "$python_bin" -u "$phase1/main_experiment.py" "$method" --dataset "$dataset" --seed "$seed"
  echo "COMPLETE $method $dataset $seed"
done
