"""One shared test-set evaluator for all Phase 0 and Phase 1 methods.

Every method saved float32 test probability maps at its own network
resolution (Phase 0: 384x384, Phase 1: 224x224). This evaluator scores all of
them in one common space: each probability map is resampled bilinearly to the
original ground-truth mask grid (QaTa 224x224, MosMed 512x512), binarized with
the method's frozen threshold rule, and compared with the original mask
(pixel > 0). HD95 and ASSD are therefore in native-image pixels for every
method.

Resampling inverts the coordinate mapping that the method's own pipeline used
to build its training targets, so a prediction lands where its target came
from:
  Phase 0 targets: PIL NEAREST (pixel-centre aligned) -> centre-aligned
                   bilinear (cv2.INTER_LINEAR).
  Phase 1 targets: cv2.INTER_NEAREST, which samples source pixel floor(x*s)
                   (pixel-corner aligned) -> corner-aligned bilinear, sampling
                   native pixel X at stored coordinate X*stored/native.
The rule follows from the resize code and was confirmed on validation masks
only (no predictions): a perfect target round-tripped through each pipeline
recovers MosMed Dice 0.933 (Phase 1, corner) vs 0.885 (centre) and 0.967
(Phase 0, centre) vs 0.962 (corner); QaTa is 1.000 for both phases.

Before native scoring, each run is re-scored at its stored resolution with the
same mask resizing it used in training, and must reproduce its stored
per-case metrics; this proves the metric code is the one each run used.

Outputs, per run, under runs/<phase>/<method>/<dataset>/<seed>/:
  per_case_metrics.csv  sample_id,source_id,method,seed,dice,miou,hd95,assd,prediction_path,probability_path
  test_summary.json     native-resolution means plus the reproduction check
  predictions/masks/    native-resolution binary masks (PNG, 0/255)
  predictions/probabilities -> symlink to the run's saved probability maps

Python 3.7 compatible (runs in the phase1_official_lvit environment).
"""
from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "methods" / "phase0_image_only" / "data_manifests"
PHASE1_MANIFESTS = ROOT / "methods" / "phase1_language_guided" / "data_manifests"
DATASETS = ("qata", "mosmed")
SEEDS = (1001, 1002, 1003)
THRESHOLD = 0.5
# phase, source runs root, stored resolution, threshold rule frozen in that
# method's exporter, alignment of the pipeline's target resizing
METHODS = {
    "unet": ("phase0", ROOT / "methods" / "phase0_image_only" / "runs", 384, "gt", "centre"),
    "tripath_lesionnet": ("phase0", ROOT / "methods" / "phase0_image_only" / "runs", 384, "ge", "centre"),
    "panoptic_fpn": ("phase0", ROOT / "methods" / "phase0_image_only" / "runs", 384, "ge", "centre"),
    "lvit": ("phase1", ROOT / "methods" / "phase1_language_guided" / "runs", 224, "ge", "corner"),
    "reclmis": ("phase1", ROOT / "methods" / "phase1_language_guided" / "runs", 224, "ge", "corner"),
}
METRICS = ("dice", "miou", "hd95", "assd")


def metrics(prediction, target):
    """Identical to the Phase 0 and Phase 1 exporters (empty-mask policy included)."""
    prediction, target = prediction.astype(bool), target.astype(bool)
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    denominator = prediction.sum() + target.sum()
    dice = (2 * intersection + 1e-7) / (denominator + 1e-7)
    miou = (intersection + 1e-7) / (union + 1e-7)
    if not prediction.any() and not target.any():
        return dice, miou, 0.0, 0.0
    if not prediction.any() or not target.any():
        diagonal = float(np.hypot(*prediction.shape))
        return dice, miou, diagonal, diagonal
    structure = np.ones((3, 3), dtype=bool)
    ps = prediction ^ binary_erosion(prediction, structure=structure, border_value=0)
    ts = target ^ binary_erosion(target, structure=structure, border_value=0)
    distances = np.concatenate([distance_transform_edt(~ts)[ps], distance_transform_edt(~ps)[ts]])
    return dice, miou, float(np.percentile(distances, 95)), float(distances.mean())


def binarize(probability, rule):
    return probability > THRESHOLD if rule == "gt" else probability >= THRESHOLD


def to_native(probability, height, width, alignment):
    if alignment == "centre":
        return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)
    rows = np.arange(height) * probability.shape[0] / float(height)
    cols = np.arange(width) * probability.shape[1] / float(width)
    grid = np.meshgrid(rows, cols, indexing="ij")
    return map_coordinates(probability, grid, order=1, mode="nearest").astype(np.float32)


def stored_resolution_mask(path, phase, size):
    """The exact ground-truth resizing each phase's exporter used."""
    if phase == "phase0":
        mask = Image.open(str(path)).convert("L").resize((size, size), Image.NEAREST)
        return np.asarray(mask, dtype=np.uint8) > 0
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0


def sha256(path):
    value = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def test_cases(dataset):
    manifest = MANIFESTS / (dataset + ".csv")
    if sha256(manifest) != sha256(PHASE1_MANIFESTS / (dataset + ".csv")):
        raise RuntimeError("Phase 0 and Phase 1 manifests differ for " + dataset)
    with manifest.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    return [(row["sample_id"], row["source_id"], (manifest.parent / row["mask_path"]).resolve()) for row in rows]


def score_case(job):
    sample_id, mask_path, probability_path, phase, size, rule, alignment, output_mask = job
    probability = np.load(str(probability_path), allow_pickle=False)
    if probability.shape != (size, size) or not np.isfinite(probability).all() \
            or probability.min() < 0 or probability.max() > 1:
        raise RuntimeError("Invalid probability map " + str(probability_path))
    stored = metrics(binarize(probability, rule), stored_resolution_mask(mask_path, phase, size))
    target = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    height, width = target.shape
    native_probability = to_native(probability, height, width, alignment)
    prediction = binarize(native_probability, rule)
    Image.fromarray(prediction.astype(np.uint8) * 255).save(str(output_mask))
    return sample_id, stored, metrics(prediction, target)


def evaluate_run(method, dataset, seed, pool):
    phase, runs_root, size, rule, alignment = METHODS[method]
    source = runs_root / method / dataset / ("seed_%d" % seed) / "predictions"
    with (source / "per_case_metrics.csv").open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    cases = test_cases(dataset)
    if [row["sample_id"] for row in saved] != [case[0] for case in cases]:
        raise RuntimeError("Saved test order differs from the frozen manifest: %s/%s/%d" % (method, dataset, seed))
    if [row["source_id"] for row in saved] != [case[1] for case in cases]:
        raise RuntimeError("Saved source IDs differ from the frozen manifest: %s/%s/%d" % (method, dataset, seed))

    output = ROOT / "runs" / phase / method / dataset / str(seed)
    masks = output / "predictions" / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    link = output / "predictions" / "probabilities"
    if not link.is_symlink():
        os.symlink(os.path.relpath(str(source / "probabilities"), str(link.parent)), str(link))

    jobs = [(sample_id, mask_path, source / row["probability_path"], phase, size, rule, alignment,
             masks / (sample_id + ".png"))
            for (sample_id, _, mask_path), row in zip(cases, saved)]
    results = pool.map(score_case, jobs, chunksize=16)

    reproduction = {}
    for index, name in enumerate(METRICS):
        error = max(abs(stored[index] - float(row[name])) for (_, stored, _), row in zip(results, saved))
        reproduction[name] = error
        if error > 1e-6:
            raise RuntimeError("Stored-resolution %s not reproduced for %s/%s/%d (max error %g)"
                               % (name, method, dataset, seed, error))

    rows = []
    for (sample_id, source_id, _), (_, _, native) in zip(cases, results):
        rows.append(dict(sample_id=sample_id, source_id=source_id, method=method, seed=seed,
                         dice=native[0], miou=native[1], hd95=native[2], assd=native[3],
                         prediction_path="predictions/masks/%s.png" % sample_id,
                         probability_path="predictions/probabilities/%s.npy" % sample_id))
    with (output / "per_case_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_case_metrics.csv").open(newline="") as stream:
        reread = list(csv.DictReader(stream))
    means = {name: float(np.mean([float(row[name]) for row in reread])) for name in METRICS}
    stored_means = {name: float(np.mean([float(row[name]) for row in saved])) for name in METRICS}
    summary = dict(phase=phase, method=method, dataset=dataset, seed=seed, test_samples=len(rows),
                   evaluator="evaluation/shared_evaluator.py", evaluator_sha256=sha256(__file__),
                   evaluation_space="native ground-truth mask grid",
                   native_mask_shape=list(cv2.imread(str(cases[0][2]), cv2.IMREAD_GRAYSCALE).shape),
                   stored_probability_shape=[size, size],
                   probability_resize="bilinear, %s-aligned (inverse of the pipeline's target resizing)" % alignment,
                   threshold=THRESHOLD, threshold_rule="probability > 0.5" if rule == "gt" else "probability >= 0.5",
                   source_predictions=str(source.relative_to(ROOT)),
                   stored_resolution_reproduction_max_abs_error=reproduction,
                   stored_resolution_means=stored_means,
                   **{"test_" + name: value for name, value in means.items()})
    with (output / "test_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print("%-18s %-6s %d  native dice %.4f (stored %.4f)  reproduction max err %.1e"
          % (method, dataset, seed, means["dice"], stored_means["dice"], max(reproduction.values())), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    with Pool(args.workers) as pool:
        for method in args.methods:
            for dataset in args.datasets:
                for seed in SEEDS:
                    evaluate_run(method, dataset, seed, pool)


if __name__ == "__main__":
    main()
