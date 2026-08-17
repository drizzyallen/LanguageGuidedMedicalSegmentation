"""Run the prespecified Phase 0 clustered bootstrap and paired tests."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
METHODS = ("unet", "tripath_lesionnet", "panoptic_fpn")
DATASETS = ("qata", "mosmed", "busi")
SEEDS = (1001, 1002, 1003)


def read_metrics(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def load_cell(runs_root, method, dataset):
    runs = [read_metrics(runs_root / method / dataset / f"seed_{seed}" / "predictions" / "per_case_metrics.csv") for seed in SEEDS]
    orders = [[row["sample_id"] for row in run] for run in runs]
    if not all(order == orders[0] for order in orders[1:]):
        raise ValueError(f"Seed sample order mismatch: {method}/{dataset}")
    return runs


def averaged_cases(runs, metric):
    return [{"sample_id": rows[0]["sample_id"], "source_id": rows[0]["source_id"],
             "value": float(np.mean([float(row[metric]) for row in rows]))}
            for rows in zip(*runs)]


def cluster_values(cases):
    grouped = defaultdict(list)
    for case in cases: grouped[case["source_id"]].append(case["value"])
    return [np.asarray(values, dtype=np.float64) for values in grouped.values()]


def bootstrap_mean_ci(cases, count, rng):
    clusters = cluster_values(cases); estimates = np.empty(count)
    for index in range(count):
        chosen = rng.integers(0, len(clusters), len(clusters))
        estimates[index] = np.concatenate([clusters[i] for i in chosen]).mean()
    return np.quantile(estimates, [0.025, 0.975]).tolist(), len(clusters)


def paired_cluster_differences(left, right):
    if [x["sample_id"] for x in left] != [x["sample_id"] for x in right]:
        raise ValueError("Cross-method ordered sample IDs differ")
    grouped = defaultdict(list)
    for a, b in zip(left, right):
        if a["source_id"] != b["source_id"]: raise ValueError("Cross-method source IDs differ")
        grouped[a["source_id"]].append(a["value"] - b["value"])
    return [np.asarray(values, dtype=np.float64) for values in grouped.values()]


def paired_inference(differences, boot_count, perm_count, rng):
    sampled = np.empty(boot_count)
    for index in range(boot_count):
        chosen = rng.integers(0, len(differences), len(differences))
        sampled[index] = np.concatenate([differences[i] for i in chosen]).mean()
    sums = np.asarray([values.sum() for values in differences])
    total_cases = sum(len(values) for values in differences)
    point = np.concatenate(differences).mean()
    observed = abs(point); extreme = 0; remaining = perm_count
    while remaining:
        size = min(remaining, 10000)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        extreme += int(np.sum(np.abs((signs * sums).sum(axis=1) / total_cases) >= observed))
        remaining -= size
    return point, *np.quantile(sampled, [0.025, 0.975]), (extreme + 1) / (perm_count + 1)


def holm(rows):
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["raw_p_value"])
    running = 0.0; adjusted = [0.0] * len(rows); total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * row["raw_p_value"]))
        adjusted[index] = running
    for row, value in zip(rows, adjusted): row["holm_p_value"] = value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--results-root", type=Path, default=ROOT.parents[1] / "results" / "phase0")
    parser.add_argument("--report", type=Path, default=ROOT.parents[1] / "reports" / "PHASE0_IMAGE_ONLY_RESULTS.md")
    args = parser.parse_args(); cfg = yaml.safe_load((ROOT / "configs" / "statistics.yaml").read_text())
    rng = np.random.default_rng(cfg["analysis_seed"]); cells = {}
    for method in METHODS:
        for dataset in DATASETS: cells[method, dataset] = load_cell(args.runs_root, method, dataset)
    absolute = []
    for (method, dataset), runs in cells.items():
        for metric in cfg["metrics"]:
            cases = averaged_cases(runs, metric); ci, n_clusters = bootstrap_mean_ci(cases, cfg["bootstrap_resamples"], rng)
            seed_means = [np.mean([float(row[metric]) for row in run]) for run in runs]
            absolute.append({"method": method, "dataset": dataset, "metric": metric,
                             "three_seed_mean": float(np.mean(seed_means)), "three_seed_sample_sd": float(np.std(seed_means, ddof=1)),
                             "case_averaged_point_estimate": float(np.mean([x["value"] for x in cases])),
                             "ci_95_lower": ci[0], "ci_95_upper": ci[1], "n_cases": len(cases), "n_clusters": n_clusters})
    comparisons = []
    for dataset in DATASETS:
        dataset_rows = []
        for left, right in cfg["pairs"]:
            a, b = averaged_cases(cells[left, dataset], "dice"), averaged_cases(cells[right, dataset], "dice")
            differences = paired_cluster_differences(a, b)
            estimate, low, high, p = paired_inference(differences, cfg["bootstrap_resamples"], cfg["permutation_resamples"], rng)
            dataset_rows.append({"dataset": dataset, "metric": "dice", "method_a": left, "method_b": right,
                                 "difference_a_minus_b": float(estimate), "ci_95_lower": float(low), "ci_95_upper": float(high),
                                 "raw_p_value": float(p), "holm_p_value": 0.0, "n_clusters": len(differences)})
        holm(dataset_rows); comparisons.extend(dataset_rows)
    write_csv(args.results_root / "absolute_ci.csv", absolute); write_csv(args.results_root / "paired_comparisons.csv", comparisons)
    with (args.results_root / "paired_comparisons.json").open("w") as stream: json.dump(comparisons, stream, indent=2)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("# Phase 0 Image-Only Results\n\nGenerated from the frozen three-seed per-case outputs. See `results/phase0/` for complete statistics.\n")
    print(f"Wrote {args.results_root} and {args.report}")


if __name__ == "__main__": main()
