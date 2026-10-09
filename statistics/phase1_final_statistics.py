"""Phase 1 absolute statistics and the final cross-phase paired comparison (plan §1.5).

Inputs are the shared evaluator's native-resolution per-case metrics for all
five methods (runs/<phase>/<method>/<dataset>/<seed>/per_case_metrics.csv).
The statistical procedure is imported unchanged from the Phase 0 analysis
(methods/phase0_image_only/analyze.py): seed-averaged cases, source_id
cluster bootstrap (10,000), two-sided paired sign-permutation test on
cluster sums (100,000), Holm within dataset. Run in the Phase 0 environment.

Writes results/phase1/, results/final_five_method/ and both reports.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods" / "phase0_image_only"))
from analyze import averaged_cases, bootstrap_mean_ci, holm, paired_cluster_differences, paired_inference, write_csv  # noqa: E402

CONFIG = ROOT / "configs" / "statistics_phase1_final.yaml"
PHASE1_RESULTS = ROOT / "results" / "phase1"
FINAL_RESULTS = ROOT / "results" / "final_five_method"
PHASE1_REPORT = ROOT / "reports" / "PHASE1_LANGUAGE_GUIDED_RESULTS.md"
FINAL_REPORT = ROOT / "reports" / "FINAL_FIVE_METHOD_COMPARISON.md"
DATASET_LABEL = {"qata": "QaTa-COV19-v2", "mosmed": "MosMedData+", "busi": "BUSI"}
MANIFESTS = ROOT / "methods" / "phase0_image_only" / "data_manifests"
METRIC_LABEL = {"dice": "Dice", "miou": "mIoU", "hd95": "HD95", "assd": "ASSD"}
TEXT = {"phase0": ("No", "No"), "phase1": ("Yes", "Yes")}
SOURCES = {
    "phase0": ("Summer2026Research", "upstream/Summer2026Research"),
    "lvit": ("LViT", "upstream/LViT"),
    "reclmis": ("RecLMIS", "upstream/RecLMIS"),
}
ORIGINAL_RUNS = {"phase0": ROOT / "methods" / "phase0_image_only" / "runs",
                 "phase1": ROOT / "methods" / "phase1_language_guided" / "runs"}


def submodule_commit(path):
    return subprocess.run(["git", "-C", str(ROOT / path), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def load_runs(method, phase, dataset, seeds):
    runs = []
    for seed in seeds:
        folder = ROOT / "runs" / phase / method / dataset / str(seed)
        with (folder / "per_case_metrics.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary = json.loads((folder / "test_summary.json").read_text())
        for name in ("dice", "miou", "hd95", "assd"):
            if np.mean([float(row[name]) for row in rows]) != summary["test_" + name]:
                raise RuntimeError("Per-case mean differs from test_summary.json: %s" % folder)
        runs.append(rows)
    orders = [[row["sample_id"] for row in run] for run in runs]
    if any(order != orders[0] for order in orders[1:]):
        raise RuntimeError("Seed sample order mismatch: %s/%s" % (method, dataset))
    return runs


def stored_resolution_dice(method, phase, dataset, seeds):
    means = []
    for seed in seeds:
        path = ORIGINAL_RUNS[phase] / method / dataset / ("seed_%d" % seed) / "predictions" / "per_case_metrics.csv"
        with path.open(newline="") as stream:
            means.append(np.mean([float(row["dice"]) for row in csv.DictReader(stream)]))
    return float(np.mean(means)), float(np.std(means, ddof=1))


def fmt(value, metric):
    return ("%.4f" if metric in ("dice", "miou") else "%.2f") % value


def fmt_p(value):
    return "%.5f" % value


def main():
    cfg = yaml.safe_load(CONFIG.read_text())
    rng = np.random.default_rng(cfg["analysis_seed"])
    methods = cfg["methods"]
    label = {m["key"]: m["label"] for m in methods}
    phase_of = {m["key"]: m["phase"] for m in methods}
    seeds, datasets, metric_names = cfg["seeds"], cfg["datasets"], cfg["metrics"]
    boot, perm = cfg["bootstrap_resamples"], cfg["permutation_resamples"]

    cells = {(m, d): load_runs(m, phase_of[m], d, seeds) for m in label for d in datasets}

    # Absolute statistics for every method (Phase 1 subset written separately).
    absolute = []
    for method in label:
        for dataset in datasets:
            runs = cells[method, dataset]
            for metric in metric_names:
                cases = averaged_cases(runs, metric)
                ci, n_clusters = bootstrap_mean_ci(cases, boot, rng)
                seed_means = [float(np.mean([float(row[metric]) for row in run])) for run in runs]
                absolute.append(dict(
                    phase=phase_of[method], method=method, dataset=dataset, metric=metric,
                    seed_1001_mean=seed_means[0], seed_1002_mean=seed_means[1], seed_1003_mean=seed_means[2],
                    three_seed_mean=float(np.mean(seed_means)), three_seed_sample_sd=float(np.std(seed_means, ddof=1)),
                    case_averaged_point_estimate=float(np.mean([c["value"] for c in cases])),
                    ci_95_lower=ci[0], ci_95_upper=ci[1], n_cases=len(cases), n_clusters=n_clusters,
                    evaluation="shared evaluator, native mask grid"))
    write_csv(PHASE1_RESULTS / "absolute_ci.csv", [r for r in absolute if r["phase"] == "phase1"])
    write_csv(FINAL_RESULTS / "absolute_ci.csv", absolute)

    # Paired comparisons: all C(5,2) pairs, every metric; Holm across the 10 Dice tests per dataset.
    pairs = list(itertools.combinations([m["key"] for m in methods], 2))
    comparisons = []
    for dataset in datasets:
        for metric in metric_names:
            block = []
            for left, right in pairs:
                a = averaged_cases(cells[left, dataset], metric)
                b = averaged_cases(cells[right, dataset], metric)
                differences = paired_cluster_differences(a, b)
                estimate, low, high, p = paired_inference(differences, boot, perm, rng)
                block.append(dict(dataset=dataset, metric=metric,
                                  inference="confirmatory" if metric == cfg["primary_metric"] else "exploratory",
                                  method_a=left, method_b=right, difference_a_minus_b=float(estimate),
                                  ci_95_lower=float(low), ci_95_upper=float(high), raw_p_value=float(p),
                                  holm_p_value=None, n_cases=len(a), n_clusters=len(differences)))
            if metric == cfg["primary_metric"]:
                holm(block)
                if len(block) != 10:
                    raise RuntimeError("Expected 10 Dice comparisons per dataset")
            comparisons.extend(block)
    write_csv(FINAL_RESULTS / "paired_comparisons.csv",
              [{**r, "holm_p_value": "" if r["holm_p_value"] is None else r["holm_p_value"]} for r in comparisons])
    (FINAL_RESULTS / "paired_comparisons.json").write_text(json.dumps(dict(
        description="Paired seed-averaged case-level differences (method A minus method B), cluster bootstrap "
                    "95% CI, two-sided paired sign-permutation p-values. Dice is confirmatory with Holm "
                    "adjustment across the 10 Dice comparisons within each dataset; mIoU, HD95 and ASSD "
                    "p-values are exploratory and unadjusted (holm_p_value null).",
        config=str(CONFIG.relative_to(ROOT)), comparisons=comparisons), indent=2) + "\n")

    # Significance matrix: Dice difference row minus column, CI and Holm p.
    keys = [m["key"] for m in methods]
    lookup = {(r["dataset"], r["method_a"], r["method_b"]): r for r in comparisons if r["metric"] == "dice"}
    matrix = []
    for dataset in datasets:
        for row in keys:
            entry = {"dataset": dataset, "method": row}
            for col in keys:
                if row == col:
                    entry[col] = "-"
                    continue
                r = lookup.get((dataset, row, col))
                sign = 1.0
                if r is None:
                    r, sign = lookup[(dataset, col, row)], -1.0
                low, high = sorted((sign * r["ci_95_lower"], sign * r["ci_95_upper"]))
                star = "*" if r["holm_p_value"] < 0.05 else ""
                entry[col] = "%+.4f [%+.4f, %+.4f] p_holm=%s%s" % (
                    sign * r["difference_a_minus_b"], low, high, fmt_p(r["holm_p_value"]), star)
            matrix.append(entry)
    write_csv(FINAL_RESULTS / "significance_matrix.csv", matrix)

    # BUSI (Phase 0 methods only; no Phase 1 counterpart). Computed last so the
    # QaTa/MosMed random streams above, and every value derived from them, are unchanged.
    busi_keys = [m for m in keys if phase_of[m] == "phase0"]
    for method in busi_keys:
        runs = cells[method, "busi"] = load_runs(method, "phase0", "busi", seeds)
        for metric in metric_names:
            cases = averaged_cases(runs, metric)
            ci, n_clusters = bootstrap_mean_ci(cases, boot, rng)
            seed_means = [float(np.mean([float(row[metric]) for row in run])) for run in runs]
            absolute.append(dict(
                phase="phase0", method=method, dataset="busi", metric=metric,
                seed_1001_mean=seed_means[0], seed_1002_mean=seed_means[1], seed_1003_mean=seed_means[2],
                three_seed_mean=float(np.mean(seed_means)), three_seed_sample_sd=float(np.std(seed_means, ddof=1)),
                case_averaged_point_estimate=float(np.mean([c["value"] for c in cases])),
                ci_95_lower=ci[0], ci_95_upper=ci[1], n_cases=len(cases), n_clusters=n_clusters,
                evaluation="shared evaluator, native mask grid"))
    write_csv(FINAL_RESULTS / "absolute_ci.csv", absolute)

    # Final required summary table.
    complexity = {m: json.loads((FINAL_RESULTS / "model_complexity" / (m + ".json")).read_text()) for m in keys}
    commits = {name: submodule_commit(path) for name, path in SOURCES.values()}
    stat = {(r["method"], r["dataset"], r["metric"]): r for r in absolute}
    summary = []
    for dataset in list(datasets) + ["busi"]:
        for method in (keys if dataset != "busi" else busi_keys):
            phase = phase_of[method]
            source = SOURCES["phase0" if phase == "phase0" else method]
            dice = stat[method, dataset, "dice"]
            c = complexity[method]
            params = "%.2fM" % (c["parameters"] / 1e6)
            if c["trainable_parameters"] != c["parameters"]:
                params += " (%.2fM trainable)" % (c["trainable_parameters"] / 1e6)
            summary.append({
                "Dataset": DATASET_LABEL[dataset], "Phase": phase[-1], "Method": label[method],
                "Text at Train": TEXT[phase][0], "Text at Test": TEXT[phase][1],
                "Dice 1001": "%.4f" % dice["seed_1001_mean"], "Dice 1002": "%.4f" % dice["seed_1002_mean"],
                "Dice 1003": "%.4f" % dice["seed_1003_mean"],
                "Dice Mean": "%.4f" % dice["three_seed_mean"], "Dice SD": "%.4f" % dice["three_seed_sample_sd"],
                "Dice 95% CI": "[%.4f, %.4f]" % (dice["ci_95_lower"], dice["ci_95_upper"]),
                "mIoU": "%.4f" % stat[method, dataset, "miou"]["three_seed_mean"],
                "HD95": "%.2f" % stat[method, dataset, "hd95"]["three_seed_mean"],
                "ASSD": "%.2f" % stat[method, dataset, "assd"]["three_seed_mean"],
                "Parameters": params, "FLOPs": "%.2fG" % (c["flops"] / 1e9),
                "Source Commit": "%s@%s" % (source[0], commits[source[0]][:12]),
                "Status": "Valid (3/3 seeds)"})
    write_csv(FINAL_RESULTS / "summary_table.csv", summary)

    # Descriptive case-level view of each paired Dice difference (not a hypothesis test).
    case_level = []
    for dataset in datasets:
        for left, right in pairs:
            a = averaged_cases(cells[left, dataset], "dice")
            b = averaged_cases(cells[right, dataset], "dice")
            if [x["sample_id"] for x in a] != [x["sample_id"] for x in b]:
                raise RuntimeError("Sample IDs differ")
            d = np.asarray([x["value"] - y["value"] for x, y in zip(a, b)])
            order = np.argsort(d)
            trimmed = np.sort(d)[int(0.05 * len(d)):len(d) - int(0.05 * len(d))]
            case_level.append(dict(
                dataset=dataset, method_a=left, method_b=right, n_cases=len(d),
                mean_difference=float(d.mean()), median_difference=float(np.median(d)),
                q1_difference=float(np.percentile(d, 25)), q3_difference=float(np.percentile(d, 75)),
                trimmed_mean_difference_5pct=float(trimmed.mean()),
                pct_cases_a_better=float(100 * np.mean(d > 0.001)), pct_cases_b_better=float(100 * np.mean(d < -0.001)),
                pct_cases_within_0_001=float(100 * np.mean(np.abs(d) <= 0.001)),
                largest_a_better="; ".join("%s %+.3f" % (a[i]["sample_id"], d[i]) for i in order[::-1][:3]),
                largest_b_better="; ".join("%s %+.3f" % (a[i]["sample_id"], d[i]) for i in order[:3])))
    write_csv(FINAL_RESULTS / "case_level_paired_summary.csv", case_level)

    stability = []
    for dataset in list(datasets) + ["busi"]:
        for method in (keys if dataset != "busi" else busi_keys):
            r = stat[method, dataset, "dice"]
            values = [r["seed_1001_mean"], r["seed_1002_mean"], r["seed_1003_mean"]]
            stability.append(dict(dataset=dataset, method=method, dice_1001=values[0], dice_1002=values[1],
                                  dice_1003=values[2], mean=r["three_seed_mean"], sample_sd=r["three_seed_sample_sd"],
                                  range=max(values) - min(values)))
    write_csv(FINAL_RESULTS / "seed_stability.csv", stability)

    stored = {(m, d): stored_resolution_dice(m, phase_of[m], d, seeds) for m in keys for d in datasets}
    write_reports(cfg, label, phase_of, keys, datasets, stat, comparisons, summary, stored, complexity, commits, cells,
                  case_level)
    print("Wrote", PHASE1_RESULTS, FINAL_RESULTS, PHASE1_REPORT, FINAL_REPORT)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def metric_cell(r, metric):
    return "%s ± %s [%s, %s]" % (fmt(r["three_seed_mean"], metric), fmt(r["three_seed_sample_sd"], metric),
                                 fmt(r["ci_95_lower"], metric), fmt(r["ci_95_upper"], metric))


def findings(dataset, keys, label, stat, comparisons):
    ranked = sorted(keys, key=lambda m: -stat[m, dataset, "dice"]["three_seed_mean"])
    dice = [r for r in comparisons if r["dataset"] == dataset and r["metric"] == "dice"]
    better, ties = [], []
    for r in dice:
        a, b, d = label[r["method_a"]], label[r["method_b"]], r["difference_a_minus_b"]
        if r["holm_p_value"] < 0.05:
            better.append("%s > %s (%+.4f)" % ((a, b, d) if d > 0 else (b, a, -d)))
        else:
            ties.append("%s vs %s" % (a, b))
    text = "Mean Dice ranking: %s. " % " > ".join(
        "%s (%.4f)" % (label[m], stat[m, dataset, "dice"]["three_seed_mean"]) for m in ranked)
    text += "Holm-significant differences (alpha 0.05): %s. " % ("; ".join(better) if better else "none")
    text += "Not detectably different: %s." % ("; ".join(ties) if ties else "none")
    return text


def split_overlap(dataset):
    with (MANIFESTS / (dataset + ".csv")).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    source = {s: {r["source_id"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    test = [r for r in rows if r["split"] == "test"]
    return dict(train_val=len(source["train"] & source["val"]), train_test=len(source["train"] & source["test"]),
                val_test=len(source["val"] & source["test"]), test_sources=len(source["test"]), test_images=len(test),
                test_images_from_train_source=sum(r["source_id"] in source["train"] for r in test))


CAVEATS = None


def caveats():
    q, m = split_overlap("qata"), split_overlap("mosmed")
    return ("## Interpretation caveats\n\n"
            "- **MosMedData+ patient/scan overlap.** The frozen Phase 0 MosMed split is slice-level. %d of the %d "
            "test CT studies also have slices in training and %d in validation; %d of %d test images (%.0f%%) come "
            "from a study with training slices. All five methods share this split, so the paired comparison is "
            "like-for-like, but absolute MosMed scores describe new slices from mostly seen patients and are likely "
            "optimistic for unseen patients. The archived annotation-workbook split also overlaps (67%% of its test "
            "images).\n"
            "- **QaTa-COV19-v2.** No subject appears in both training and test. %d subjects appear in both training "
            "and validation, which can make checkpoint selection slightly optimistic but does not touch test data.\n"
            "- **Report content.** Every QaTa and MosMed report states the lesion count and lung location (for "
            "example \"Bilateral pulmonary infection, two infected areas, ...\"). LViT-T and RecLMIS receive this text "
            "at test time; the image-only methods do not. The reports come unchanged from the published "
            "QaTa/MosMed text annotations; this repository cannot verify whether they were written from the images "
            "or derived from the masks. Comparisons show what each method achieves with its native inputs; they do "
            "not by themselves show that language improves segmentation.\n"
            "- **Evaluation geometry.** The native-grid alignment rule for Phase 1 was investigated after a "
            "MosMed test-set drop was observed. It was fixed from the resize code, confirmed on validation masks "
            "only, applied identically to every seed, and both the stored-resolution and native-grid values are "
            "reported.\n" % (m["train_test"], m["test_sources"], m["val_test"], m["test_images_from_train_source"],
                              m["test_images"], 100.0 * m["test_images_from_train_source"] / m["test_images"],
                              q["train_val"]))


def write_reports(cfg, label, phase_of, keys, datasets, stat, comparisons, summary, stored, complexity, commits, cells,
                  case_level):
    phase1 = [m for m in keys if phase_of[m] == "phase1"]
    n = {d: (len(cells[keys[0], d][0]), next(r for r in comparisons if r["dataset"] == d)["n_clusters"]) for d in datasets}
    manifests = {d: hashlib.sha256((ROOT / "methods" / "phase1_language_guided" / "data_manifests" / (d + ".csv"))
                                   .read_bytes()).hexdigest() for d in datasets}
    stops = []
    for m in phase1:
        for d in datasets:
            for s in cfg["seeds"]:
                f = ORIGINAL_RUNS["phase1"] / m / d / ("seed_%d" % s) / "nonfinite_stop.json"
                if f.is_file():
                    e = json.loads(f.read_text())
                    stops.append("%s %s seed %d: nonfinite loss at epoch %d (step %d); best validation checkpoint "
                                 "epoch %d (val Dice %.4f) was tested." % (label[m], DATASET_LABEL[d], s, e["epoch"],
                                                                           e["step"], e["best_epoch"], e["best_val_dice"]))
    per_seed = []
    for m in phase1:
        for d in datasets:
            for s in cfg["seeds"]:
                res = json.loads((ORIGINAL_RUNS["phase1"] / m / d / ("seed_%d" % s) / "result.json").read_text())
                summ = json.loads((ROOT / "runs" / "phase1" / m / d / str(s) / "test_summary.json").read_text())
                per_seed.append([label[m], DATASET_LABEL[d], s, res["checkpoint_epoch"], "%.4f" % res["best_val_dice"],
                                 "%.4f" % summ["test_dice"], "%.4f" % summ["test_miou"], "%.2f" % summ["test_hd95"],
                                 "%.2f" % summ["test_assd"], res.get("training_stop", "")])

    p1 = []
    p1.append("# Phase 1 Language-Guided Results\n")
    p1.append("Generated by `statistics/phase1_final_statistics.py` from the shared evaluator's per-case outputs. "
              "Do not edit tables by hand; rerun the script.\n")
    p1.append("## Scope\n")
    p1.append("Phase 1 reproduces **LViT-T** and **RecLMIS** on QaTa-COV19-v2 and MosMedData+ with seeds 1001, 1002 "
              "and 1003 (12 runs). The research advisor removed ProLearn from the study on 2026-10-03; it was not run "
              "in the main experiment. Phase 0 methods were not retrained, modified or extended.\n")
    p1.append("## Data and protocol\n")
    p1.append(table(["Dataset", "Train / val / test", "Test clusters", "Manifest sha256"],
                    [[DATASET_LABEL[d], "%s / %s / {:,}".format(n[d][0]) % (("5,716", "1,429") if d == "qata" else ("1,746", "437")),
                      n[d][1], "`%s`" % manifests[d][:16]] for d in datasets]) + "\n")
    p1.append("- The Phase 1 manifests are byte-identical copies of the frozen Phase 0 manifests (plan §1.4 rule 2), "
              "so all five methods share the same train, validation and test samples.\n"
              "- Every image has its paired report: the main comparison uses 100% of the available reports.\n"
              "- Each method keeps its native text encoder and fusion: LViT-T uses frozen BERT-base token "
              "embeddings (first 10 tokens); RecLMIS uses its CLIP text encoder with 18 tokens.\n"
              "- Both methods require text at training and at test time.\n"
              "- Native architectures, losses, optimizers, schedules, batch sizes and patience were kept. Training "
              "augmentation is disabled by the text-safe adapter policy (flips and rotations can contradict "
              "laterality and location text).\n"
              "- Checkpoint selection uses validation mean per-case Dice at threshold 0.5. Each selected checkpoint "
              "was evaluated once on the test set; a `test_started.json` marker blocks repeat evaluation.\n"
              "- If a training loss is nonfinite, training stops and the best pre-event validation checkpoint is "
              "tested. The upstream loops have no finiteness check, so NaN parameters could never raise "
              "validation Dice and early stopping would select the same checkpoint.\n")
    p1.append("## Reproduction gates\n")
    p1.append("All seven gates from plan §1.3 pass for both methods: data (32/32/32), shape, overfit (training Dice "
              "LViT-T 0.9026, RecLMIS 0.9296), language-module gradient, validation inference (1,429 cases), "
              "text use (correct, null and shuffled text give different logits) and official check (LViT-T "
              "official-style QaTa run, best validation Dice 0.8067; RecLMIS strict load of the author checkpoint). "
              "Details: `methods/phase1_language_guided/gates/STATUS.md`.\n")
    p1.append("## Evaluation\n")
    p1.append("One shared evaluator (`evaluation/shared_evaluator.py`) scores every Phase 0 and Phase 1 run. Saved "
              "probability maps are resampled bilinearly to the original mask grid (QaTa 224x224, MosMed 512x512), "
              "thresholded at 0.5, and compared with the original masks, so HD95 and ASSD are in native pixels for "
              "every method. Resampling inverts each pipeline's own target-resizing geometry (Phase 1: "
              "corner-aligned `cv2.INTER_NEAREST`). The rule was fixed from the resize code and confirmed on "
              "validation masks without predictions. Each run is first re-scored at its stored resolution and "
              "reproduces its saved per-case metrics (maximum error below 1e-13).\n")
    p1.append("Statistics follow Phase 0 exactly: per-case metrics are averaged across the three seeds; absolute "
              "95%% CIs use 10,000 bootstrap resamples of `source_id` clusters (images where no cluster exists). "
              "QaTa: %d cases in %d clusters; MosMed: %d cases in %d clusters.\n" % (n["qata"] + n["mosmed"]))
    p1.append("## Results\n")
    p1.append("Values are three-seed mean ± sample SD, followed by the absolute cluster-bootstrap 95% CI. HD95 and "
              "ASSD are in native pixels (lower is better).\n")
    rows = []
    for d in datasets:
        for m in phase1:
            rows.append([DATASET_LABEL[d], label[m]] + [metric_cell(stat[m, d, x], x) for x in cfg["metrics"]])
    p1.append(table(["Dataset", "Method", "Dice", "mIoU", "HD95", "ASSD"], rows) + "\n")
    p1.append("Machine-readable values, including each seed's mean: `results/phase1/absolute_ci.csv`.\n")
    p1.append("### Per-seed runs\n")
    p1.append(table(["Method", "Dataset", "Seed", "Best epoch", "Val Dice", "Test Dice", "mIoU", "HD95", "ASSD",
                     "Training stop"], per_seed) + "\n")
    if stops:
        p1.append("Training events: " + " ".join(stops) + "\n")
    p1.append("### Sensitivity: stored-resolution scoring\n")
    p1.append("Each run's own exporter scored predictions at the network resolution against a resized mask. Those "
              "values differ from the native-grid values above only through resolution:\n")
    p1.append(table(["Dataset", "Method", "Dice, stored 224x224", "Dice, native grid"],
                    [[DATASET_LABEL[d], label[m], "%.4f ± %.4f" % stored[m, d],
                      "%.4f ± %.4f" % (stat[m, d, "dice"]["three_seed_mean"], stat[m, d, "dice"]["three_seed_sample_sd"])]
                     for d in datasets for m in phase1]) + "\n")
    p1.append("### Post-hoc text-use check on the final checkpoints\n")
    p1.append("Audit check, separate from the historical 1.3 gate: each selected checkpoint was run on the first 32 "
              "validation cases (no test data) with the correct report, null text (LViT-T: zero embeddings; "
              "RecLMIS: empty string) and a shuffled report from another case. "
              "Source: `evaluation/text_use_check.py`, `results/phase1/text_use_check/`.\n")
    rows = []
    for m in phase1:
        for item in json.loads((ROOT / "results" / "phase1" / "text_use_check" / (m + ".json")).read_text())["results"]:
            rows.append([label[m], DATASET_LABEL[item["dataset"]], item["seed"], "%.4f" % item["dice"]["correct"],
                         "%.4f" % item["dice"]["null"], "%.4f" % item["dice"]["shuffled"],
                         "%.3f" % item["shuffled"]["max_abs_probability_change"],
                         "No" if item["null"]["identical_logits"] or item["shuffled"]["identical_logits"] else "Yes"])
    p1.append(table(["Method", "Dataset", "Seed", "Val Dice, correct", "Null", "Shuffled", "Max |dp| shuffled",
                     "Text changes output"], rows) + "\n")
    p1.append("Both methods use their text input in every run. On QaTa, null or shuffled reports lower validation "
              "Dice substantially; on MosMed the effect is small for LViT-T and moderate for RecLMIS.\n")
    p1.append(caveats())
    p1.append("## Run records\n")
    p1.append("Every run has `runs/phase1/<method>/<dataset>/<seed>/` with `resolved_config.yaml`, `environment.txt`, "
              "`train_log.csv`, `best_checkpoint.txt`, `test_summary.json`, `per_case_metrics.csv` and "
              "`predictions/` (native binary masks plus the saved float32 probability maps). Training-time "
              "records, checkpoints and logs are in `methods/phase1_language_guided/runs/`.\n")
    p1.append("## Deviations and limitations\n")
    p1.append("- ProLearn was removed from the study by the research advisor and was not run.\n"
              "- An earlier launch (2026-09-16) used annotation-workbook splits that differed from Phase 0. Those "
              "runs are archived in `methods/phase1_language_guided/archive/xlsx_split_20260916/` and are not used.\n"
              "- RecLMIS needs fairseq's float32 softmax, which would not build; `reclmis_compat.py` supplies that "
              "one function, matching fairseq 0.10.2. Upstream source is unchanged.\n"
              "- Training augmentation was disabled for both methods (text-safe adapter policy).\n"
              "- Phase 1 networks run at 224x224 (their native input). MosMed masks are 512x512, so Phase 1 MosMed "
              "results include a resolution cost that the 384x384 Phase 0 methods pay less of (see sensitivity "
              "table).\n"
              "- Some QaTa subjects and MosMed studies cross the frozen splits "
              "(`methods/phase0_image_only/DATA_LIMITATIONS.md`); CIs resample by `source_id` cluster.\n")
    p1.append("## Phase 1 exit criteria\n")
    p1.append(table(["Criterion", "Status"], [
        ["LViT-T and RecLMIS pass all reproduction gates", "Met"],
        ["ProLearn", "Not applicable: removed by the research advisor"],
        ["Each method completes three seeds on QaTa and MosMed", "Met (12/12 runs)"],
        ["Saved per-case predictions and absolute 95% CIs", "Met"],
        ["Final paired comparison: all Dice pairs, paired 95% CIs, raw and Holm p-values",
         "Met for the 5-method study (10 pairs per dataset); see FINAL_FIVE_METHOD_COMPARISON.md"],
        ["Phase 0 and Phase 1 results in separate reports", "Met"],
        ["Final report combines them only after both phases are complete", "Met"],
    ]) + "\n")
    PHASE1_REPORT.write_text("\n".join(p1))

    f = []
    f.append("# Final Cross-Phase Comparison\n")
    f.append("Generated by `statistics/phase1_final_statistics.py`. The original plan named this file "
              "`FINAL_SIX_METHOD_COMPARISON.md`; the research advisor removed ProLearn, so the study has **five** "
              "methods and **10** pairwise comparisons per dataset (the plan's 15 assumed six).\n")
    f.append("Phase 0 (`reports/PHASE0_IMAGE_ONLY_RESULTS.md`) and Phase 1 "
             "(`reports/PHASE1_LANGUAGE_GUIDED_RESULTS.md`) were completed and reported separately before this "
             "combined report.\n")
    f.append("## Common evaluation\n")
    f.append("All five methods use the same frozen QaTa and MosMed manifests, the same ordered test sample IDs, seeds "
             "1001-1003, validation-Dice checkpoint selection, threshold 0.5 and one shared evaluator on the native "
             "mask grid (`evaluation/shared_evaluator.py`). Because Phase 0 originally scored at 384x384, its "
             "native-grid values here differ slightly from the Phase 0 report; the Phase 0 report is unchanged.\n")
    f.append("## Summary table\n")
    header = list(summary[0])
    f.append(table(header, [[r[h] for h in header] for r in summary]) + "\n")
    f.append("mIoU, HD95 and ASSD are three-seed means (HD95/ASSD in native pixels). Parameters and FLOPs are "
             "counted with thop at each method's test input, batch of one. FLOPs = 2 x thop MACs; thop does not "
             "count functional attention products, so LViT-T and RecLMIS FLOPs are lower bounds. LViT-T excludes "
             "its frozen external BERT-base encoder; RecLMIS includes its frozen CLIP model (trainable count in "
             "parentheses). Measured LViT-T FLOPs (54.16G) match the LViT paper (54.1G); measured parameters "
             "(39.93M) exceed the paper's 29.7M for the instantiated official configuration. All absolute "
             "statistics: `results/final_five_method/absolute_ci.csv`; this table: "
             "`results/final_five_method/summary_table.csv`.\n")
    f.append("## Paired Dice comparisons (confirmatory)\n")
    f.append("Seed-averaged case-level differences (A minus B), 10,000 paired cluster-bootstrap resamples for the "
             "95% CI, two-sided paired sign-permutation test with 100,000 permutations on `source_id` clusters, "
             "Holm adjustment across the 10 Dice comparisons within each dataset.\n")
    for d in datasets:
        f.append("### %s (%d cases, %d clusters)\n" % (DATASET_LABEL[d], n[d][0], n[d][1]))
        f.append(table(["A vs B", "Dice difference", "95% CI", "Raw p", "Holm p"],
                       [["%s vs %s" % (label[r["method_a"]], label[r["method_b"]]), "%+.4f" % r["difference_a_minus_b"],
                         "[%+.4f, %+.4f]" % (r["ci_95_lower"], r["ci_95_upper"]), fmt_p(r["raw_p_value"]),
                         fmt_p(r["holm_p_value"])]
                        for r in comparisons if r["dataset"] == d and r["metric"] == "dice"]) + "\n")
        f.append(findings(d, keys, label, stat, comparisons) + "\n")
    f.append("The smallest attainable permutation p-value is 1/100,001 = 0.00001.\n")
    f.append("## Secondary metrics (exploratory)\n")
    f.append("Paired differences and 95% CIs for mIoU (higher is better), HD95 and ASSD (native pixels, lower is "
             "better). P-values are unadjusted and exploratory.\n")
    for d in datasets:
        f.append("### %s\n" % DATASET_LABEL[d])
        rows = []
        for a, b in itertools.combinations(keys, 2):
            row = ["%s vs %s" % (label[a], label[b])]
            for x in ("miou", "hd95", "assd"):
                r = next(c for c in comparisons if c["dataset"] == d and c["metric"] == x
                         and c["method_a"] == a and c["method_b"] == b)
                row.append("%s [%s, %s] p=%s" % ((("%+.4f" if x == "miou" else "%+.2f") % r["difference_a_minus_b"]),
                                                  ("%+.4f" if x == "miou" else "%+.2f") % r["ci_95_lower"],
                                                  ("%+.4f" if x == "miou" else "%+.2f") % r["ci_95_upper"],
                                                  fmt_p(r["raw_p_value"])))
            rows.append(row)
        f.append(table(["A vs B", "mIoU difference", "HD95 difference", "ASSD difference"], rows) + "\n")
    f.append("## Case-level paired behaviour (descriptive)\n")
    f.append("Seed-averaged per-case Dice differences (A minus B). \"A better\" / \"B better\" count cases whose "
             "difference exceeds 0.001 in that direction. The 5% trimmed mean drops the most extreme 5% of cases at "
             "each end; a mean close to it means the difference is not driven by a few outliers. Descriptive only; "
             "full rows, including the largest individual cases, are in "
             "`results/final_five_method/case_level_paired_summary.csv`.\n")
    for d in datasets:
        f.append("### %s\n" % DATASET_LABEL[d])
        f.append(table(["A vs B", "Mean", "Median [IQR]", "5% trimmed mean", "A better", "B better"],
                       [["%s vs %s" % (label[r["method_a"]], label[r["method_b"]]), "%+.4f" % r["mean_difference"],
                         "%+.4f [%+.4f, %+.4f]" % (r["median_difference"], r["q1_difference"], r["q3_difference"]),
                         "%+.4f" % r["trimmed_mean_difference_5pct"], "%.0f%%" % r["pct_cases_a_better"],
                         "%.0f%%" % r["pct_cases_b_better"]] for r in case_level if r["dataset"] == d]) + "\n")
    f.append("## Seed stability\n")
    f.append(table(["Dataset", "Method", "Dice 1001", "Dice 1002", "Dice 1003", "Sample SD", "Range"],
                   [[DATASET_LABEL[d], label[m]] + ["%.4f" % stat[m, d, "dice"]["seed_%d_mean" % s] for s in cfg["seeds"]]
                    + ["%.4f" % stat[m, d, "dice"]["three_seed_sample_sd"],
                       "%.4f" % (max(stat[m, d, "dice"]["seed_%d_mean" % s] for s in cfg["seeds"])
                                 - min(stat[m, d, "dice"]["seed_%d_mean" % s] for s in cfg["seeds"]))]
                    for d in list(datasets) + ["busi"] for m in keys if (m, d, "dice") in stat]) + "\n")
    f.append(caveats())
    f.append("## Files\n")
    f.append("- `results/final_five_method/paired_comparisons.csv` and `.json`: all 80 paired comparisons "
             "(10 pairs x 4 metrics x 2 datasets).\n"
             "- `results/final_five_method/significance_matrix.csv`: Dice difference (row minus column), 95% CI and "
             "Holm p for every pair; `*` marks Holm p < 0.05.\n"
             "- `results/final_five_method/absolute_ci.csv` (all methods, including BUSI), `summary_table.csv`, "
             "`case_level_paired_summary.csv`, `seed_stability.csv`, `model_complexity/`.\n"
             "- `results/phase1/absolute_ci.csv`.\n")
    f.append("## Limitations\n")
    f.append("- Five methods instead of six: ProLearn was removed by the research advisor.\n"
             "- Phase 0 methods use 384x384 grayscale input; Phase 1 methods use their native 224x224 RGB input. "
             "Each method is compared as designed, on the same native-grid evaluation.\n"
             "- Phase 1 methods use text at test time; Phase 0 methods do not.\n"
             "- QaTa `covid_N` images and MosMed `bjorke_N` images have no defensible patient ID and are their own "
             "resampling units; some subjects and studies cross the frozen splits "
             "(`methods/phase0_image_only/DATA_LIMITATIONS.md`).\n")
    FINAL_REPORT.write_text("\n".join(f))


if __name__ == "__main__":
    main()
