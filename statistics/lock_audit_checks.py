"""Independent pre-lock audit checks for Phase 0 and Phase 1 (read-only).

Re-verifies the 39 required runs from the lowest-level evidence and recomputes
the statistics with separate code (vectorized cluster bootstrap and sign
permutation, different random seed) instead of trusting stored summaries.
Nothing under runs/, results/phase0, results/phase1 or results/final_five_method
is modified. Run in the Phase 0 environment (myenv).

Writes results/audit/lock_audit_checks.json.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "methods" / "phase0_image_only" / "data_manifests"
OUT = ROOT / "results" / "audit"
SEEDS = (1001, 1002, 1003)
PHASE0 = ("unet", "tripath_lesionnet", "panoptic_fpn")
PHASE1 = ("lvit", "reclmis")
FIVE = PHASE0 + PHASE1
ARTIFACTS = ("resolved_config.yaml", "environment.txt", "train_log.csv", "best_checkpoint.txt",
             "test_summary.json", "per_case_metrics.csv", "predictions")
RNG = np.random.default_rng(97531)  # deliberately different from the analysis seed


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def manifest(dataset):
    return read_csv(MANIFESTS / (dataset + ".csv"))


def required_runs():
    for method in PHASE0:
        for dataset in ("qata", "mosmed", "busi"):
            for seed in SEEDS:
                yield "phase0", method, dataset, seed
    for method in PHASE1:
        for dataset in ("qata", "mosmed"):
            for seed in SEEDS:
                yield "phase1", method, dataset, seed


def source_dir(phase, method, dataset, seed):
    base = "phase0_image_only" if phase == "phase0" else "phase1_language_guided"
    return ROOT / "methods" / base / "runs" / method / dataset / ("seed_%d" % seed)


# ---------------------------------------------------------------- run matrix
def audit_run(phase, method, dataset, seed, test_rows):
    run = ROOT / "runs" / phase / method / dataset / str(seed)
    src = source_dir(phase, method, dataset, seed)
    issues = []
    present = {a: (run / a).exists() for a in ARTIFACTS}
    if not all(present.values()):
        issues.append("missing artifacts: " + ", ".join(a for a, ok in present.items() if not ok))
    if phase == "phase0":
        ckpt = src / "upstream" / dataset / "best.pt"
        trained = (src / "upstream" / dataset / "metrics.json").is_file()
    else:
        ckpt = src / "best_checkpoint.pt"
        result = json.loads((src / "result.json").read_text())
        trained = result.get("status") == "complete"
    if not ckpt.is_file():
        issues.append("selected checkpoint missing")
    record = dict(line.split(": ", 1) for line in (run / "best_checkpoint.txt").read_text().splitlines())
    if phase == "phase1" and record["sha256"] != result["checkpoint_sha256"]:
        issues.append("checkpoint hash differs from result.json")
    summary = json.loads((run / "test_summary.json").read_text())
    log = read_csv(run / "train_log.csv")
    best_logged = max(log, key=lambda r: float(r["val_dice"]))
    if int(best_logged["epoch"]) != int(record["epoch"]):
        issues.append("selected epoch %s != best validation epoch %s" % (record["epoch"], best_logged["epoch"]))
    epochs = [int(r["epoch"]) for r in log]
    if epochs != list(range(1, len(epochs) + 1)):
        issues.append("training log is not a complete epoch sequence")

    rows = read_csv(run / "per_case_metrics.csv")
    ids = [r["sample_id"] for r in rows]
    expected = [r["sample_id"] for r in test_rows]
    sources = {r["sample_id"]: r["source_id"] for r in test_rows}
    if ids != expected:
        issues.append("per-case IDs/order differ from the manifest test split")
    if len(set(ids)) != len(ids):
        issues.append("duplicate per-case IDs")
    if any(r["method"] != method or int(r["seed"]) != seed for r in rows):
        issues.append("wrong method/seed column")
    if any(r["source_id"] != sources.get(r["sample_id"]) for r in rows):
        issues.append("source_id differs from manifest")
    values = {k: np.array([float(r[k]) for r in rows]) for k in ("dice", "miou", "hd95", "assd")}
    nonfinite = int(sum((~np.isfinite(v)).sum() for v in values.values()))
    out_of_range = int(((values["dice"] < 0) | (values["dice"] > 1)).sum() + ((values["miou"] < 0) | (values["miou"] > 1)).sum()
                       + (values["hd95"] < 0).sum() + (values["assd"] < 0).sum())
    # For binary masks IoU = Dice / (2 - Dice) exactly (up to the 1e-7 smoothing).
    relation = float(np.max(np.abs(values["miou"] - values["dice"] / (2 - values["dice"]))))
    if nonfinite or out_of_range:
        issues.append("invalid metric values")
    if relation > 1e-4:
        issues.append("mIoU inconsistent with Dice")
    mean_error = max(abs(float(values[k].mean()) - summary["test_" + k]) for k in values)
    if mean_error > 1e-9:
        issues.append("per-case means differ from test_summary.json")
    masks = run / "predictions" / "masks"
    missing_pred = sum(not (masks / (i + ".png")).is_file() for i in ids)
    probs = run / "predictions" / "probabilities"
    missing_prob = sum(not (probs / (i + ".npy")).is_file() for i in ids)
    extra_pred = len({p.stem for p in masks.glob("*.png")} - set(ids))
    if missing_pred or missing_prob or extra_pred:
        issues.append("prediction files incomplete")
    return dict(phase=phase, method=method, dataset=dataset, seed=seed, run_path=str(run.relative_to(ROOT)),
                source_path=str(src.relative_to(ROOT)), run_found=src.is_dir(), training_completed=trained,
                checkpoint_present=ckpt.is_file(), selected_epoch=int(record["epoch"]),
                best_val_epoch=int(best_logged["epoch"]), logged_epochs=len(log),
                checkpoint_sha256=record["sha256"], test_n_expected=len(expected), test_n=len(rows),
                missing_predictions=missing_pred, missing_probabilities=missing_prob, extra_predictions=extra_pred,
                nonfinite_values=nonfinite, out_of_range_values=out_of_range, miou_dice_relation_max_error=relation,
                summary_mean_max_error=mean_error, artifacts=present,
                note_files=sorted(p.name for p in run.glob("*_note.txt")),
                dice_vector_sha=hashlib.sha256(values["dice"].tobytes()).hexdigest()[:16],
                mean_dice=float(values["dice"].mean()), valid=not issues, issues=issues)


# ---------------------------------------------------------------- data checks
def dataset_checks(dataset):
    rows = manifest(dataset)
    split = defaultdict(list)
    for r in rows:
        split[r["split"]].append(r)
    src = {k: {r["source_id"] for r in v} for k, v in split.items()}
    img = {k: {r["image_sha256"] for r in v} for k, v in split.items()}
    msk = {k: {r["mask_sha256"] for r in v} for k, v in split.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    test = split["test"]
    return dict(
        counts={k: len(v) for k, v in split.items()}, unique_sources={k: len(v) for k, v in src.items()},
        duplicate_sample_ids=len(rows) - len({r["sample_id"] for r in rows}),
        missing_images=sum(not (MANIFESTS / r["image_path"]).is_file() for r in rows),
        missing_masks=sum(not (MANIFESTS / r["mask_path"]).is_file() for r in rows),
        source_overlap={"%s-%s" % p: len(src[p[0]] & src[p[1]]) for p in pairs},
        identical_image_hash_across_splits={"%s-%s" % p: len(img[p[0]] & img[p[1]]) for p in pairs},
        identical_mask_hash_across_splits={"%s-%s" % p: len(msk[p[0]] & msk[p[1]]) for p in pairs},
        test_images_from_train_source=sum(r["source_id"] in src["train"] for r in test),
        split_id_sha256={k: hashlib.sha256("\n".join(sorted(r["sample_id"] for r in v)).encode()).hexdigest()[:16]
                         for k, v in split.items()},
        manifest_sha256=sha256(MANIFESTS / (dataset + ".csv")))


def hash_spot_check(dataset, n=200):
    rows = manifest(dataset)
    chosen = RNG.choice(len(rows), size=min(n, len(rows)), replace=False)
    bad = 0
    for i in chosen:
        r = rows[i]
        bad += sha256(MANIFESTS / r["image_path"]) != r["image_sha256"]
        bad += sha256(MANIFESTS / r["mask_path"]) != r["mask_sha256"]
    return dict(files_checked=2 * len(chosen), mismatches=int(bad))


# ---------------------------------------------------------------- statistics
def seed_averaged(per_seed_rows, metric):
    ids = [r["sample_id"] for r in per_seed_rows[0]]
    for rows in per_seed_rows[1:]:
        if [r["sample_id"] for r in rows] != ids:
            raise RuntimeError("seed order mismatch")
    value = np.mean([[float(r[metric]) for r in rows] for rows in per_seed_rows], axis=0)
    return ids, [r["source_id"] for r in per_seed_rows[0]], value


def cluster_arrays(sources, values):
    order = {}
    for s in sources:
        order.setdefault(s, len(order))
    index = np.array([order[s] for s in sources])
    sums = np.bincount(index, weights=values)
    counts = np.bincount(index)
    return sums, counts


def boot_ci(sums, counts, n=10000):
    picks = RNG.integers(0, len(sums), size=(n, len(sums)))
    means = sums[picks].sum(1) / counts[picks].sum(1)
    return np.quantile(means, [0.025, 0.975])


def perm_p(sums, total, n=100000):
    observed = abs(sums.sum() / total)
    extreme = 0
    for start in range(0, n, 20000):
        signs = RNG.choice((-1.0, 1.0), size=(min(20000, n - start), len(sums)))
        extreme += int((np.abs(signs @ sums / total) >= observed - 1e-15).sum())
    return (extreme + 1) / (n + 1)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def p_agrees(stored, recomputed):
    # Monte Carlo agreement: within 4 binomial standard errors (floor 0.002), or both at the 1/(n+1) floor region.
    se = math.sqrt(max(stored * (1 - stored), 1e-12) / 100000)
    return abs(stored - recomputed) <= max(4 * se, 0.002)


def verify_pairs(stored_rows, loader, family_size):
    checks, by_dataset = [], defaultdict(list)
    for r in stored_rows:
        a = loader(r["method_a"], r["dataset"], r["metric"])
        b = loader(r["method_b"], r["dataset"], r["metric"])
        if a[0] != b[0] or a[1] != b[1]:
            checks.append(dict(r, pairing_ok=False))
            continue
        d = a[2] - b[2]
        sums, counts = cluster_arrays(a[1], d)
        low, high = boot_ci(sums, counts)
        p = perm_p(sums, len(d))
        stored_holm = r.get("holm_p_value")
        item = dict(dataset=r["dataset"], metric=r["metric"], method_a=r["method_a"], method_b=r["method_b"],
                    stored_difference=float(r["difference_a_minus_b"]), recomputed_difference=float(d.mean()),
                    stored_ci=[float(r["ci_95_lower"]), float(r["ci_95_upper"])], recomputed_ci=[float(low), float(high)],
                    stored_raw_p=float(r["raw_p_value"]), recomputed_raw_p=float(p),
                    stored_holm_p=None if stored_holm in (None, "") else float(stored_holm),
                    n_cases=len(d), n_clusters=len(sums), pairing_ok=True)
        item["difference_exact"] = abs(item["stored_difference"] - item["recomputed_difference"]) < 1e-12
        # Monte Carlo agreement: endpoints within 5% of the interval width (or 0.003 absolute).
        width = item["stored_ci"][1] - item["stored_ci"][0]
        item["ci_endpoint_gap_fraction_of_width"] = max(abs(x - y) for x, y in zip(item["stored_ci"], item["recomputed_ci"])) / width
        item["ci_agrees"] = max(abs(x - y) for x, y in zip(item["stored_ci"], item["recomputed_ci"])) < max(0.003, 0.05 * width)
        item["p_agrees"] = p_agrees(item["stored_raw_p"], p)
        item["same_ci_excludes_zero"] = ((item["stored_ci"][0] > 0) or (item["stored_ci"][1] < 0)) == ((low > 0) or (high < 0))
        checks.append(item)
        if r["metric"] == "dice":
            by_dataset[r["dataset"]].append(item)
    for dataset, items in by_dataset.items():
        if len(items) != family_size:
            raise RuntimeError("Holm family size %d != %d for %s" % (len(items), family_size, dataset))
        recomputed = holm(np.array([x["stored_raw_p"] for x in items]))
        for x, h in zip(items, recomputed):
            x["holm_family_size"] = len(items)
            x["holm_from_stored_raw_exact"] = abs(h - x["stored_holm_p"]) < 1e-12
            x["recomputed_significant"] = bool(holm(np.array([y["recomputed_raw_p"] for y in items]))[items.index(x)] < 0.05)
            x["stored_significant"] = x["stored_holm_p"] < 0.05
    return checks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    tests = {d: [r for r in manifest(d) if r["split"] == "test"] for d in ("qata", "mosmed", "busi")}

    runs = [audit_run(*key, tests[key[2]]) for key in required_runs()]
    report["runs"] = runs
    report["run_totals"] = dict(expected=39, found=sum(r["run_found"] for r in runs),
                                valid=sum(r["valid"] for r in runs), invalid=sum(not r["valid"] for r in runs),
                                missing=39 - sum(r["run_found"] for r in runs))

    # Seed and cross-method duplication.
    dup = []
    groups = defaultdict(list)
    for r in runs:
        groups[r["method"], r["dataset"]].append(r)
    for (m, d), items in groups.items():
        dup.append(dict(method=m, dataset=d,
                        unique_checkpoints=len({x["checkpoint_sha256"] for x in items}),
                        unique_dice_vectors=len({x["dice_vector_sha"] for x in items}),
                        selected_epochs=[x["selected_epoch"] for x in items],
                        mean_dice=[round(x["mean_dice"], 4) for x in items]))
    vectors = defaultdict(list)
    for r in runs:
        vectors[r["dataset"], r["dice_vector_sha"]].append("%s/%s" % (r["method"], r["seed"]))
    report["seed_duplication"] = dup
    report["identical_dice_vectors_across_runs"] = [v for v in vectors.values() if len(v) > 1]

    # Mask-level spot check: same case, different runs, identical prediction files?
    identical_masks = 0
    compared = 0
    for d in ("qata", "mosmed"):
        for sid in [r["sample_id"] for r in tests[d][:25]]:
            hashes = defaultdict(list)
            for r in runs:
                if r["dataset"] == d:
                    hashes[sha256(ROOT / r["run_path"] / "predictions" / "masks" / (sid + ".png"))].append(r["method"])
            compared += 1
            identical_masks += sum(len(set(v)) > 1 for v in hashes.values())
    report["prediction_mask_spot_check"] = dict(
        cases_compared=compared, cases_with_identical_masks_across_methods=identical_masks,
        note="identical files across methods occur only where predictions are empty (all-background PNGs are byte-identical)")

    # Configuration and environment drift across seeds.
    drift = []
    for (m, d), items in groups.items():
        configs = [yaml.safe_load((ROOT / x["run_path"] / "resolved_config.yaml").read_text()) for x in items]
        envs = {sha256(ROOT / x["run_path"] / "environment.txt") for x in items}
        flat = [flatten(c) for c in configs]
        keys = set().union(*flat)
        differing = sorted(k for k in keys if len({json.dumps(f.get(k), sort_keys=True) for f in flat}) > 1)
        unexpected = [k for k in differing if not re.search(r"(^|\.)seed($|\.)|physical_gpu|output_dir|output_directory|"
                                                           r"run_metadata\.config$|run_metadata\.manifest$|nonfinite_stop|"
                                                           r"training_stop|run_config\.input_sha256\..*/runs/", k)]
        drift.append(dict(method=m, dataset=d, differing_fields=differing, unexpected_differences=unexpected,
                          identical_environment=len(envs) == 1))
    report["config_drift"] = drift
    p1_runner = {json.loads((source_dir("phase1", r["method"], r["dataset"], r["seed"]) / "run_config.json").read_text())["runner_sha256"]
                 for r in runs if r["phase"] == "phase1"}
    p0_config = defaultdict(set)
    for r in runs:
        if r["phase"] == "phase0":
            meta = json.loads((source_dir(*[r[k] for k in ("phase", "method", "dataset", "seed")]) / "run_metadata.json").read_text())
            p0_config[r["method"]].add((meta["config_sha256"], meta["upstream_commit"]))
    report["version_consistency"] = dict(phase1_runner_hashes=sorted(p1_runner),
                                         phase0_config_and_commit_per_method={k: sorted(v) for k, v in p0_config.items()})

    report["datasets"] = {d: dataset_checks(d) for d in ("qata", "mosmed", "busi")}
    report["hash_spot_check"] = {d: hash_spot_check(d) for d in ("qata", "mosmed", "busi")}

    # Statistics: Phase 0 (stored 384x384 per-case) and the final five-method set (native grid).
    def phase0_loader(method, dataset, metric):
        rows = [read_csv(source_dir("phase0", method, dataset, s) / "predictions" / "per_case_metrics.csv") for s in SEEDS]
        return seed_averaged(rows, metric)

    def final_loader(method, dataset, metric):
        phase = "phase0" if method in PHASE0 else "phase1"
        rows = [read_csv(ROOT / "runs" / phase / method / dataset / str(s) / "per_case_metrics.csv") for s in SEEDS]
        return seed_averaged(rows, metric)

    report["phase0_statistics"] = verify_pairs(read_csv(ROOT / "results" / "phase0" / "paired_comparisons.csv"),
                                               phase0_loader, 3)
    report["final_statistics"] = verify_pairs(read_csv(ROOT / "results" / "final_five_method" / "paired_comparisons.csv"),
                                              final_loader, 10)

    absolute = []
    for path, loader in ((ROOT / "results" / "phase0" / "absolute_ci.csv", phase0_loader),
                         (ROOT / "results" / "final_five_method" / "absolute_ci.csv", final_loader)):
        for r in read_csv(path):
            ids, sources, values = loader(r["method"], r["dataset"], r["metric"])
            phase = "phase0" if r["method"] in PHASE0 else "phase1"
            if loader is phase0_loader:
                seed_means = [np.mean([float(x[r["metric"]]) for x in read_csv(source_dir("phase0", r["method"], r["dataset"], s)
                                                                              / "predictions" / "per_case_metrics.csv")]) for s in SEEDS]
            else:
                seed_means = [np.mean([float(x[r["metric"]]) for x in read_csv(ROOT / "runs" / phase / r["method"] / r["dataset"]
                                                                              / str(s) / "per_case_metrics.csv")]) for s in SEEDS]
            sums, counts = cluster_arrays(sources, values)
            low, high = boot_ci(sums, counts)
            scale = 0.003 if r["metric"] in ("dice", "miou") else 0.6
            absolute.append(dict(file=str(path.relative_to(ROOT)), method=r["method"], dataset=r["dataset"], metric=r["metric"],
                                 mean_exact=abs(np.mean(seed_means) - float(r["three_seed_mean"])) < 1e-12,
                                 sd_exact=abs(np.std(seed_means, ddof=1) - float(r["three_seed_sample_sd"])) < 1e-12,
                                 stored_ci=[float(r["ci_95_lower"]), float(r["ci_95_upper"])], recomputed_ci=[low, high],
                                 ci_agrees=max(abs(float(r["ci_95_lower"]) - low), abs(float(r["ci_95_upper"]) - high)) < scale,
                                 n_clusters=len(sums), stored_n_clusters=int(r["n_clusters"])))
    report["absolute_ci_checks"] = absolute

    # Phase 0 report table versus its result file (the report was written by hand).
    text = (ROOT / "reports" / "PHASE0_IMAGE_ONLY_RESULTS.md").read_text()
    names = {"U-Net v2": "unet", "TriPathLesionNet": "tripath_lesionnet", "Panoptic FPN": "panoptic_fpn"}
    datasets = {"QaTa": "qata", "MosMed": "mosmed", "BUSI": "busi"}
    stored = {(r["method"], r["dataset"]): r for r in read_csv(ROOT / "results" / "phase0" / "absolute_ci.csv") if r["metric"] == "dice"}
    mismatches = []
    for line in text.splitlines():
        m = re.match(r"\| (QaTa|MosMed|BUSI) \| (U-Net v2|TriPathLesionNet|Panoptic FPN) \| ([\d.]+) ± ([\d.]+) \| \[([\d.]+), ([\d.]+)\] \|", line)
        if m:
            r = stored[names[m.group(2)], datasets[m.group(1)]]
            for got, key in zip(m.groups()[2:], ("three_seed_mean", "three_seed_sample_sd", "ci_95_lower", "ci_95_upper")):
                if abs(float(got) - float(r[key])) > 0.00005 + 1e-9:
                    mismatches.append((m.group(1), m.group(2), key, got, r[key]))
    pairs = {(r["dataset"], r["method_a"], r["method_b"]): r for r in read_csv(ROOT / "results" / "phase0" / "paired_comparisons.csv")}
    short = {"U-Net v2": "unet", "TriPath": "tripath_lesionnet", "Panoptic": "panoptic_fpn"}
    for line in text.splitlines():
        m = re.match(r"\| (QaTa|MosMed|BUSI) \| (\S+(?: v2)?) vs (\S+) \| (-?[\d.]+) \| \[(-?[\d.]+), (-?[\d.]+)\] \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            r = pairs[datasets[m.group(1)], short[m.group(2)], short[m.group(3)]]
            for got, key in zip(m.groups()[3:], ("difference_a_minus_b", "ci_95_lower", "ci_95_upper", "raw_p_value", "holm_p_value")):
                if abs(float(got) - float(r[key])) > 0.000051:
                    mismatches.append((m.group(1), m.group(2) + " vs " + m.group(3), key, got, r[key]))
    report["phase0_report_vs_files"] = dict(rows_parsed=len(re.findall(r"^\| (QaTa|MosMed|BUSI) \|", text, re.M)),
                                            mismatches=mismatches)

    summary_csv = read_csv(ROOT / "results" / "final_five_method" / "summary_table.csv")
    final_text = (ROOT / "reports" / "FINAL_FIVE_METHOD_COMPARISON.md").read_text()
    missing_rows = [r["Method"] + "/" + r["Dataset"] for r in summary_csv
                    if "| " + " | ".join(r.values()) + " |" not in final_text]
    report["final_report_vs_summary_csv"] = dict(rows=len(summary_csv), rows_not_found_verbatim=missing_rows)

    (OUT / "lock_audit_checks.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    t = report["run_totals"]
    print("runs", t)
    print("invalid runs:", [(r["method"], r["dataset"], r["seed"], r["issues"]) for r in runs if not r["valid"]])
    for name in ("phase0_statistics", "final_statistics"):
        items = report[name]
        print(name, "rows", len(items),
              "| pairing", all(x["pairing_ok"] for x in items),
              "| diff exact", all(x.get("difference_exact") for x in items),
              "| CI agree", sum(x.get("ci_agrees", False) for x in items),
              "| p agree", sum(x.get("p_agrees", False) for x in items),
              "| CI-excludes-zero agree", all(x.get("same_ci_excludes_zero") for x in items),
              "| Holm exact", all(x.get("holm_from_stored_raw_exact", True) for x in items),
              "| significance agrees", all(x.get("recomputed_significant") == x.get("stored_significant")
                                          for x in items if x["metric"] == "dice"))
    a = report["absolute_ci_checks"]
    print("absolute rows", len(a), "mean exact", all(x["mean_exact"] for x in a), "sd exact", all(x["sd_exact"] for x in a),
          "ci agree", sum(x["ci_agrees"] for x in a), "clusters match", all(x["n_clusters"] == x["stored_n_clusters"] for x in a))
    print("seed duplication:", [(x["method"], x["dataset"], x["unique_checkpoints"], x["unique_dice_vectors"]) for x in dup
                                if x["unique_checkpoints"] < 3 or x["unique_dice_vectors"] < 3])
    print("identical dice vectors across runs:", report["identical_dice_vectors_across_runs"])
    print("mask spot check:", report["prediction_mask_spot_check"])
    print("config drift (unexpected):", [(x["method"], x["dataset"], x["unexpected_differences"]) for x in drift if x["unexpected_differences"]])
    print("environment identical across seeds:", all(x["identical_environment"] for x in drift))
    print("versions:", report["version_consistency"])
    print("hash spot check:", report["hash_spot_check"])
    print("phase0 report vs files:", report["phase0_report_vs_files"])
    print("final report vs summary csv:", report["final_report_vs_summary_csv"])


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flatten(v, prefix + str(k) + "."))
        return out
    return {prefix[:-1]: value}


if __name__ == "__main__":
    main()
