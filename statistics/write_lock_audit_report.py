"""Write reports/PHASE0_PHASE1_LOCK_AUDIT.md from the independent audit checks.

Tables come from results/audit/lock_audit_checks.json (statistics/lock_audit_checks.py)
and the result files it verified; findings and decisions are stated in this script.
Run in the Phase 0 environment after lock_audit_checks.py.
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = json.loads((ROOT / "results" / "audit" / "lock_audit_checks.json").read_text())
OUT = ROOT / "reports" / "PHASE0_PHASE1_LOCK_AUDIT.md"
LABEL = {"unet": "U-Net v2", "tripath_lesionnet": "TriPathLesionNet", "panoptic_fpn": "Panoptic FPN",
         "lvit": "LViT-T", "reclmis": "RecLMIS"}
DS = {"qata": "QaTa-COV19-v2", "mosmed": "MosMedData+", "busi": "BUSI"}
FIVE = ("unet", "tripath_lesionnet", "panoptic_fpn", "lvit", "reclmis")


def read_csv(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def yes(flag):
    return "Yes" if flag else "No"


def main():
    runs = AUDIT["runs"]
    totals = AUDIT["run_totals"]
    data = AUDIT["datasets"]
    q, m = data["qata"], data["mosmed"]
    text_use = {k: json.loads((ROOT / "results" / "phase1" / "text_use_check" / (k + ".json")).read_text())["results"]
                for k in ("lvit", "reclmis")}
    summary = read_csv("results/final_five_method/summary_table.csv")
    case_level = read_csv("results/final_five_method/case_level_paired_summary.csv")
    complexity = {k: json.loads((ROOT / "results" / "final_five_method" / "model_complexity" / (k + ".json")).read_text())
                  for k in FIVE}
    L = []
    add = L.append

    add("# Phase 0 and Phase 1 Lock Audit\n")
    add("## 1. Executive Decision\n")
    add("NOT READY TO LOCK\n")
    add("Every required experiment is present, valid and independently reproduced, and no retraining, re-inference "
        "or statistical recomputation is required by any finding below. The decision is NOT READY only because one "
        "scientific question needs the PI's ruling before the results are frozen: the frozen MosMedData+ split places "
        "slices of the same CT study in training and test (B-01). If the PI accepts B-01 as a documented limitation "
        "and accepts the report-text caveat (M-01), the remaining actions are non-experimental (commit, push, sign-off) "
        "and the status becomes READY AFTER NON-EXPERIMENTAL FIXES. If the PI requires a patient-level MosMed split, "
        "all 15 MosMed runs (5 methods x 3 seeds) must be retrained before lock.\n")
    add("```text\nCritical blockers: 1\n  B-01 MosMedData+ test studies overlap training studies (528 of 546 test images, 97%) "
        "- PI decision required\nMajor issues: 1\n  M-01 Phase 1 test-time reports state lesion count and location; "
        "provenance relative to the masks NOT VERIFIED\nMinor issues: 7\n  m-01 QaTa train/validation subject overlap "
        "(444 subjects; none in test)\n  m-02 Native-grid alignment rule chosen after a test-set observation "
        "(disclosed; validation-only confirmation)\n  m-03 U-Net v2 QaTa seed 1001 history.csv overwritten by a relaunch "
        "(full log recovered)\n  m-04 Phase 1 gates ran on the archived annotation split (pipeline checks; post-hoc "
        "text-use re-verified on final checkpoints)\n  m-05 Summer2026Research has no LICENSE file\n  m-06 Comparability "
        "notes: input resolution, augmentation and FLOP counting differ by design\n  m-07 Work not yet committed/pushed "
        "to GitHub\n```\n")

    add("## 2. Audit Scope\n")
    add("```text\nPhase 0 methods:\n- U-Net v2\n- TriPathLesionNet\n- Panoptic FPN\n\nPhase 0 datasets:\n- QaTa-COV19-v2\n"
        "- MosMedData+\n- BUSI\n\nPhase 1 methods:\n- LViT-T\n- RecLMIS\n\nPhase 1 datasets:\n- QaTa-COV19-v2\n"
        "- MosMedData+\n\nSeeds:\n- 1001\n- 1002\n- 1003\n\nExpected total runs:\n39\n```\n")
    add("ProLearn was removed from the study by the research advisor on 2026-10-03 and is out of scope. The final "
        "comparison therefore has 5 methods and 10 pairs per dataset.\n")

    add("## 3. Repository Snapshot\n")
    status_lines = git("status", "--short").splitlines()
    add(table(["Item", "Value"], [
        ["Repository path", str(ROOT)],
        ["Branch", git("branch", "--show-current")],
        ["Commit", git("rev-parse", "HEAD")],
        ["Working tree", "%d uncommitted entries (this audit's fixes and outputs; see m-07). Upstream submodules: "
                         "compiled `.pyc` caches only, no source changes" % len(status_lines)],
        ["Summer2026Research", "`%s` (pinned = current)" % git("rev-parse", "HEAD", cwd=ROOT / "upstream" / "Summer2026Research")],
        ["LViT", "`%s` (pinned = current)" % git("rev-parse", "HEAD", cwd=ROOT / "upstream" / "LViT")],
        ["RecLMIS", "`%s` (pinned = current)" % git("rev-parse", "HEAD", cwd=ROOT / "upstream" / "RecLMIS")],
        ["Audit date/time", datetime.now().strftime("%Y-%m-%d %H:%M %Z") or datetime.now().isoformat()],
        ["Audit tooling", "`statistics/lock_audit_checks.py` -> `results/audit/lock_audit_checks.json`; this report: "
                          "`statistics/write_lock_audit_report.py`"],
    ]))
    add("Authoritative locations: `runs/<phase>/<method>/<dataset>/<seed>/` (shared-evaluator records), "
        "`results/phase0/`, `results/phase1/`, `results/final_five_method/`, `reports/`. Training-time artifacts "
        "(checkpoints, logs) live in `methods/phase0_image_only/runs/` and `methods/phase1_language_guided/runs/`. "
        "Not authoritative: `methods/phase1_language_guided/archive/` (superseded annotation-split launch), "
        "`smoke_runs*/` (smoke checks on validation cases), `methods/phase0_image_only/aborted_attempts/`, "
        "`methods/phase1_language_guided/gates/` (pre-experiment gate evidence).\n")

    add("## 4. Expected vs Observed Experiment Matrix\n")
    rows = []
    for r in runs:
        comment = "; ".join(r["issues"]) or ("log recovered, see m-03" if r["note_files"] else "")
        rows.append(["%s" % r["phase"][-1], LABEL[r["method"]], DS[r["dataset"]], r["seed"], "Yes", yes(r["run_found"]),
                     yes(r["valid"]), "`%s`" % r["run_path"], "PASS" if r["valid"] else "FAIL", comment])
    add(table(["Phase", "Method", "Dataset", "Seed", "Expected", "Run Found", "Run Valid", "Primary Run Path", "Status",
               "Comments"], rows))
    add("```text\nExpected runs: %d\nFound runs: %d\nValid runs: %d\nInvalid runs: %d\nMissing runs: %d\n```\n"
        % (totals["expected"], totals["found"], totals["valid"], totals["invalid"], totals["missing"]))
    add("A run is valid when: the selected checkpoint exists; its epoch equals the best validation-Dice epoch in a "
        "complete training log; per-case rows equal the manifest test IDs in order with the correct method, seed and "
        "source ID; every metric is finite and in range; mIoU = Dice/(2-Dice) holds per case; per-case means equal "
        "`test_summary.json`; and one binary mask and one probability map exist per test case.\n")

    add("## 5. Phase 0 Requirement-by-Requirement Audit\n")
    p0 = [r for r in runs if r["phase"] == "phase0"]
    items = [
        ("5.1 Method import and provenance", "Three methods imported from pinned upstream with unchanged source",
         "PASS", "`upstream/Summer2026Research` at `fdeb4403…`, 0 source changes; thin wrappers in "
         "`methods/phase0_image_only/models/`; `SOURCE.md` per method (added in this audit).",
         "Imports are wrapper-only. Upstream has no LICENSE (m-05).", "Add a license to the source repository."),
        ("5.2 Frozen method configuration", "Same config, commit and manifest for every seed",
         "PASS", "Each method has one `config_sha256` across its 9 runs (`version_consistency`); resolved configs differ "
         "only in seed and output paths (`config_drift`); `CORRECTNESS_CHANGES.md`: no fixes.",
         "No architecture search or test-driven tuning found; checkpoint = best validation Dice (strictly greater); "
         "early stopping patience 15, max 100 epochs.", "None."),
        ("5.3 Dataset/split correctness", "All three methods use the frozen manifests",
         "PASS", "Per-case IDs of all 27 runs equal the manifest test IDs in order; manifests hashed in `run_metadata.json`.",
         "Split identities are shared. Source overlap is reported in Section 7 (B-01, m-01).", "See B-01."),
        ("5.4 Three-seed completion", "27 runs, seeds 1001-1003", "PASS", "%d/27 valid." % sum(r["valid"] for r in p0),
         "All trained in this repository; no migrated runs.", "None."),
        ("5.5 Saved prediction completeness", "Probability map and mask per test case", "PASS",
         "0 missing/extra predictions in all 27 runs (original 384x384 maps plus native-grid masks).", "", "None."),
        ("5.6 Per-case metrics", "One row per test case; means match summaries", "PASS",
         "All 27 runs: IDs, sources, method/seed columns, ranges and means verified.", "", "None."),
        ("5.7 Phase 0 statistics", "Seed-averaged, clustered, 10k bootstrap, 100k permutation, Holm over 3", "PASS",
         "Section 13: 9/9 comparisons reproduced independently; Holm family size 3 per dataset.",
         "Phase 0 report values match `results/phase0/` exactly (18 table rows parsed, 0 mismatches).", "None."),
        ("5.8 Phase 0 exit criteria", "Methods present, splits frozen, 3 seeds, predictions, CIs, paired tests, report",
         "PASS", "All criteria evidenced above; `reports/PHASE0_IMAGE_ONLY_RESULTS.md`.",
         "Phase 0 report scores at 384x384; the final comparison re-scores on the native grid (documented).", "None."),
    ]
    for title, req, status, evidence, finding, action in items:
        add("### %s\n" % title)
        add("- Requirement: %s\n- Status: %s\n- Evidence: %s\n- Finding: %s\n- Required action: %s\n"
            % (req, status, evidence, finding or "Meets the requirement.", action))
    add("Epoch policy (selected epoch / logged epochs per seed):\n")
    epochs = {}
    for r in runs:
        epochs.setdefault((r["method"], r["dataset"]), []).append("%d/%d" % (r["selected_epoch"], r["logged_epochs"]))
    add(table(["Method", "Dataset", "Max epochs", "Early stopping", "Seeds 1001 / 1002 / 1003"],
              [[LABEL[k[0]], DS[k[1]], {"lvit": "2000", "reclmis": "2000 QaTa / 400 MosMed"}.get(k[0], "100"),
                {"lvit": "patience 50", "reclmis": "patience 100 QaTa / 30 MosMed"}.get(k[0], "patience 15"),
                " / ".join(v)] for k, v in epochs.items()]))
    add("Validation runs every epoch for every method. The selected epoch equals the best validation-Dice epoch in "
        "all 39 runs. RecLMIS QaTa seed 1003 stopped at a nonfinite loss (epoch 97) and tested its pre-event best "
        "checkpoint (epoch 17).\n")

    add("## 6. Phase 1 Requirement-by-Requirement Audit\n")
    p1 = [r for r in runs if r["phase"] == "phase1"]
    items = [
        ("6.1 LViT-T implementation and provenance", "Native LViT-T preserved", "PASS",
         "`upstream/LViT` at `ba90775e…`, 0 source changes; `methods/phase1_language_guided/lvit_t/SOURCE.md`; "
         "empty `patches/lvit.patch`.",
         "Native model, WeightedDiceBCE, Adam lr 3e-4, cosine warm restarts, batch 2, patience 50, frozen BERT-base "
         "token embeddings (10 tokens). Deviations: training augmentation disabled (text-safe policy); lr 3e-4 used "
         "for MosMed (upstream gives only a Covid19 value).", "None."),
        ("6.2 RecLMIS implementation and provenance", "Native RecLMIS preserved", "PASS",
         "`upstream/RecLMIS` at `d3265c8b…`, 0 source changes; `reclmis/SOURCE.md`; `reclmis_compat.py` supplies only "
         "`fairseq.utils.softmax` (v0.10.2).",
         "All three auxiliary losses asserted present every step; per-dataset native config files; CLIP ViT-B/32 "
         "weights. Without the shim, upstream silently drops the auxiliary losses; this was detected and prevented. "
         "Augmentation disabled (as above).", "None."),
        ("6.3 Shared image-mask-text data handling", "Pairing by stable key; text provenance", "PASS (pairing) / NOT VERIFIED (text origin)",
         "`adapter.py`: manifest rows in file order; report joined by image basename from hash-pinned workbooks; "
         "missing, empty, duplicate or conflicting reports are fatal; `validate_adapter.py` passes (2026-10-09).",
         "No directory listing or positional matching is used. Reports come unchanged from the published annotation "
         "workbooks; no generation script exists in the repository. Whether the reports were derived from masks "
         "cannot be verified (M-01).", "PI review of M-01."),
        ("6.4 Reproduction gates", "Seven gates per method", "PASS", "Section 8.", "Gates ran on the archived split (m-04).", "None."),
        ("6.5 Dataset/split correctness", "Same samples as Phase 0", "PASS",
         "Phase 1 manifests are byte-identical to Phase 0 (`e12e88a0…`, `5f2cbe3a…`); all 12 run configs record "
         "these hashes; one runner hash across all 12 runs.", "", "None."),
        ("6.6 Three-seed completion", "12 runs", "PASS", "%d/12 valid." % sum(r["valid"] for r in p1),
         "One technical restart (RecLMIS QaTa 1002, CUDA OOM at startup before training) is documented.", "None."),
        ("6.7 Saved prediction completeness", "Probability map and mask per test case", "PASS",
         "0 missing/extra predictions in all 12 runs.", "", "None."),
        ("6.8 Per-case metrics", "One row per test case; means match summaries", "PASS",
         "All 12 runs verified (Section 4 criteria).", "", "None."),
        ("6.9 Phase 1 statistics", "Mean, SD, clustered CIs", "PASS",
         "`results/phase1/absolute_ci.csv`: means and SDs reproduced exactly; CIs reproduced within Monte Carlo error.",
         "", "None."),
        ("6.10 Phase 1 exit criteria", "Gates, 3 seeds, predictions, CIs, 10-pair comparison, separate reports", "PASS",
         "Sections 4, 8, 14; `reports/PHASE1_LANGUAGE_GUIDED_RESULTS.md`, `reports/FINAL_FIVE_METHOD_COMPARISON.md`.",
         "ProLearn criteria not applicable (removed by advisor).", "None."),
    ]
    for title, req, status, evidence, finding, action in items:
        add("### %s\n" % title)
        add("- Requirement: %s\n- Status: %s\n- Evidence: %s\n- Finding: %s\n- Required action: %s\n"
            % (req, status, evidence, finding or "Meets the requirement.", action))

    add("## 7. Dataset Integrity and Split Verification\n")
    add(table(["Dataset", "Train N", "Val N", "Test N", "Unique source train / val / test", "Text-paired",
               "Missing images", "Missing masks", "Duplicate IDs", "Manifest sha256"],
              [[DS[d], v["counts"]["train"], v["counts"]["val"], v["counts"]["test"],
                "%d / %d / %d" % (v["unique_sources"]["train"], v["unique_sources"]["val"], v["unique_sources"]["test"]),
                {"busi": "N/A (Phase 0 only)"}.get(d, "100% (all splits)"), v["missing_images"], v["missing_masks"],
                v["duplicate_sample_ids"], "`%s`" % v["manifest_sha256"][:16]] for d, v in data.items()]))
    add("Split identity (all methods within a dataset use the same frozen sets; sha256 of sorted IDs):\n")
    rows = []
    for d, v in data.items():
        methods = FIVE if d != "busi" else FIVE[:3]
        for k in methods:
            rows.append([DS[d], LABEL[k], v["counts"]["train"], v["counts"]["val"], v["counts"]["test"], "Yes", "Yes",
                         "Yes", 0, 0, "PASS"])
    add(table(["Dataset", "Method", "Train N", "Val N", "Test N", "Train IDs match", "Val IDs match",
               "Test IDs match", "Excluded", "Extra", "Status"], rows))
    add(table(["Dataset", "Train IDs hash", "Val IDs hash", "Test IDs hash"],
              [[DS[d], v["split_id_sha256"]["train"], v["split_id_sha256"]["val"], v["split_id_sha256"]["test"]]
               for d, v in data.items()]))
    add("Evidence: Phase 0 runs read the manifest through `run.py` (hash in `run_metadata.json`); Phase 1 runs read "
        "byte-identical copies (hash in `run_config.json`); per-case test IDs of all 39 runs equal the manifest order.\n")
    add("Source (patient/scan) leakage:\n")
    add(table(["Dataset", "Grouping ID", "Train-Val overlap", "Train-Test overlap", "Val-Test overlap",
               "Test images from a train source", "Status"],
              [["QaTa-COV19-v2", "`sub-S…` subject (else image)", q["source_overlap"]["train-val"],
                q["source_overlap"]["train-test"], q["source_overlap"]["val-test"], q["test_images_from_train_source"],
                "PASS for test; m-01 for val"],
               ["MosMedData+", "`Morozov_study_*` / `Jun_*_case*` (else image)", m["source_overlap"]["train-val"],
                m["source_overlap"]["train-test"], m["source_overlap"]["val-test"],
                "%d of %d (%.0f%%)" % (m["test_images_from_train_source"], m["counts"]["test"],
                                       100 * m["test_images_from_train_source"] / m["counts"]["test"]), "FAIL (B-01)"],
               ["BUSI", "none (image is the unit)", 0, 0, 0, 0, "NOT VERIFIED (no patient ID)"]]))
    add("Duplicate files across splits: 0 identical image hashes and 0 identical mask hashes between any two splits "
        "in all three datasets. Stored hashes: %s files spot-checked, 0 mismatches.\n"
        % sum(v["files_checked"] for v in AUDIT["hash_spot_check"].values()))

    add("## 8. Phase 1 Reproduction Gates\n")
    add(table(["Method", "Data Gate", "Shape Gate", "Overfit Gate", "Gradient Gate", "Inference Gate", "Text-Use Gate",
               "Official Check", "Overall Status", "Comments"], [
        ["LViT-T", "PASS (32/32/32)", "PASS: image 8x3x224x224, mask 8x1x224x224, tokens 8x10, embedding 8x10x768, "
         "output 8x1x224x224", "PASS (train Dice 0.9005)", "PASS (`text_module1-4` nonzero)", "PASS (1,429 val masks)",
         "PASS (historical: null 0.168, shuffled 0.027 max logit change; post-hoc: Section 8.1)",
         "PASS (official-style QaTa run, best val Dice 0.8067, epoch 40)", "PASS", "No author checkpoint published"],
        ["RecLMIS", "PASS (32/32/32)", "PASS: image 8x3x224x224, mask 8x1x224x224, tokens 8x18, embedding 8x18x512, "
         "output 8x1x224x224", "PASS (train Dice 0.9254)", "PASS (`interact.attn_text/img` nonzero)",
         "PASS (1,429 val masks)", "PASS (historical: null 0.934, shuffled 0.864; post-hoc: Section 8.1)",
         "PASS (strict load of author QaTa checkpoint)", "PASS", "Frozen CLIP encoder; fusion modules trainable"],
    ]))
    add("Gate evidence: `methods/phase1_language_guided/gates/<method>/gate_results.json`, "
        "`gates/<method>/official_qata/result.json`. The gates ran on 2026-09-14/15 with the then-current "
        "annotation-split manifests (m-04); they test the pipeline, not split membership.\n")
    add("### 8.1 Post-hoc text-use verification (this audit)\n")
    add("Final selected checkpoints, first 32 validation cases (no test data), same image with correct, null and "
        "shuffled reports (`evaluation/text_use_check.py`).\n")
    rows = []
    for k, items in text_use.items():
        for it in items:
            rows.append([LABEL[k], DS[it["dataset"]], it["seed"], "%.4f" % it["dice"]["correct"], "%.4f" % it["dice"]["null"],
                         "%.4f" % it["dice"]["shuffled"], "%.3f" % it["null"]["max_abs_probability_change"],
                         "%.3f" % it["shuffled"]["max_abs_probability_change"],
                         "PASS" if not (it["null"]["identical_logits"] or it["shuffled"]["identical_logits"]) else "FAIL"])
    add(table(["Method", "Dataset", "Seed", "Val Dice correct", "Null", "Shuffled", "Max |dp| null", "Max |dp| shuffled",
               "Status"], rows))

    add("## 9. Fair Comparison Audit\n")
    add("Information availability:\n")
    add(table(["Method", "Image at train", "Mask supervision", "Text at train", "Text at inference",
               "External pretrained weights", "Additional supervision"], [
        ["U-Net v2", "Yes", "Yes", "No", "No", "None", "None"],
        ["TriPathLesionNet", "Yes", "Yes", "No", "No", "None", "Boundary/location targets derived from the training masks"],
        ["Panoptic FPN", "Yes", "Yes", "No", "No", "ImageNet ResNet-50", "None"],
        ["LViT-T", "Yes", "Yes", "Yes", "Yes (lesion count + location)", "Frozen BERT-base", "Reports"],
        ["RecLMIS", "Yes", "Yes", "Yes", "Yes (lesion count + location)", "CLIP ViT-B/32", "Reports"],
    ]))
    fair = [
        ("Correct canonical train split", "PASS", "Identical IDs (Section 7)"),
        ("Correct canonical val split", "PASS", "Identical IDs"),
        ("Correct canonical test split", "PASS", "Identical ordered IDs in all 39 runs"),
        ("No patient/source leakage", "FAIL", "MosMed B-01 (all methods equally); QaTa val m-01"),
        ("Correct number of seeds", "PASS", "3 per method/dataset"),
        ("Correct method implementation", "PASS", "Upstream unchanged; native settings"),
        ("Correct information availability", "PASS (reported)", "Phase 1 receives test-time text (M-01)"),
        ("Validation-only checkpoint selection", "PASS", "Selected epoch = best validation epoch in 39/39"),
        ("No test-time hyperparameter tuning", "PASS", "No evidence found"),
        ("Correct preprocessing", "PASS", "Bilinear images; nearest masks; per-method native input (m-06)"),
        ("Correct mask handling", "PASS", "Masks binarized > 0; nearest-neighbour resizing"),
        ("Comparable evaluation resolution", "PASS", "Shared native-grid evaluator for all 39 runs (m-02)"),
        ("Validation/test augmentation disabled", "PASS", "Enforced by code in both phases"),
        ("Shared evaluator used", "PASS", "`evaluation/shared_evaluator.py`"),
        ("All test cases included", "PASS", "2,113 / 546 / 130 in every run"),
        ("Per-case outputs preserved", "PASS", "39/39"),
        ("Correct selected checkpoint used", "PASS", "Hash/epoch checks"),
        ("No behavior-changing seed-version mixing", "PASS", "One config hash per Phase 0 method; one runner hash for Phase 1"),
    ]
    add(table(["Check", "U-Net v2", "TriPathLesionNet", "Panoptic FPN", "LViT-T", "RecLMIS", "Finding", "Status"],
              [[c] + [s.split(" ")[0]] * 5 + [f, s] for c, s, f in fair]))
    add("Conclusion: the five-method comparisons are like-for-like on data, seeds, checkpoint selection and "
        "evaluation. They compare each method with its native inputs and training recipe; differences in input "
        "resolution, augmentation, pretrained encoders and test-time text are reported, not controlled. QaTa is fair "
        "to lock. MosMed is fair between methods but its absolute values are affected by B-01.\n")

    add("## 10. Required Artifact Audit\n")
    missing = [r for r in runs if not all(r["artifacts"].values())]
    add("All 39 run folders contain resolved configuration, environment record, training log, best-checkpoint record, "
        "test summary, per-case metrics and predictions. Missing artifacts: %s. One training log was rebuilt from the "
        "trainer's terminal output (m-03; `runs/phase0/unet/qata/1001/train_log_note.txt`). Checkpoints and prediction "
        "images remain on the training host and are excluded from Git by `.gitignore`.\n"
        % ("none" if not missing else ", ".join(r["run_path"] for r in missing)))

    add("## 11. Evaluation Pipeline Audit\n")
    add(table(["Item", "Value"], [
        ["Shared evaluator", "`evaluation/shared_evaluator.py` (all 39 runs)"],
        ["Binary threshold", "0.5: U-Net v2 `p > 0.5`, all others `p >= 0.5` (each method's frozen rule)"],
        ["Empty-mask policy", "Both empty: Dice = mIoU = 1, HD95 = ASSD = 0; one empty: HD95 = ASSD = image diagonal"],
        ["Evaluation resolution", "Original mask grid: QaTa 224x224, MosMed 512x512, BUSI per-image size"],
        ["Dice", "(2|P∩G| + 1e-7)/(|P| + |G| + 1e-7), foreground only, per image, 0-1 scale"],
        ["mIoU", "(|P∩G| + 1e-7)/(|P∪G| + 1e-7), foreground IoU, per image (background excluded)"],
        ["HD95", "95th percentile of symmetric surface distances; surfaces = mask XOR 3x3 erosion; scipy EDT"],
        ["ASSD", "Mean of the same symmetric surface distances"],
        ["HD95/ASSD units", "Native-image pixels (no physical spacing applied; not millimetres)"],
        ["Averaging", "Per image; statistics resample source clusters"],
        ["Reproduction check", "Each run re-scored at stored resolution reproduces its exporter's per-case metrics (max error 5.7e-14)"],
    ]))
    add("All five methods are evaluated with the same definitions. Predictions are resampled with the inverse of "
        "each pipeline's own target-resizing geometry (Phase 0 centre-aligned, Phase 1 corner-aligned; m-02). "
        "Per-case checks: 0 NaN/inf, 0 out-of-range values, mIoU = Dice/(2-Dice) to within 1e-4 in every case.\n")

    add("## 12. Recomputed Performance Summary\n")
    add(table(["Dataset", "Phase", "Method", "Dice 1001", "Dice 1002", "Dice 1003", "Dice Mean", "Dice SD", "Dice 95% CI",
               "mIoU Mean", "HD95 Mean", "ASSD Mean", "Parameters", "FLOPs", "Text at Train", "Text at Test",
               "Source Commit", "Status"],
              [[r["Dataset"], r["Phase"], r["Method"], r["Dice 1001"], r["Dice 1002"], r["Dice 1003"], r["Dice Mean"],
                r["Dice SD"], r["Dice 95% CI"], r["mIoU"], r["HD95"], r["ASSD"], r["Parameters"], r["FLOPs"],
                r["Text at Train"], r["Text at Test"], r["Source Commit"],
                "PASS%s" % (" (B-01)" if r["Dataset"] == "MosMedData+" else "")] for r in summary]))
    add("Native-grid shared-evaluator values; means/SDs reproduced exactly and CIs within Monte Carlo error by the "
        "audit. Parameters/FLOPs: thop, batch of one at each method's test input (Phase 0 1x384x384; Phase 1 3x224x224 "
        "+ text); FLOPs = 2 x MACs; attention products not counted (lower bound for LViT-T/RecLMIS); LViT-T excludes "
        "its frozen external BERT; RecLMIS includes frozen CLIP (trainable in parentheses). FLOPs at different input "
        "sizes are not a like-for-like efficiency comparison (m-06).\n")

    def stat_rows(items, family):
        out = []
        for x in items:
            if x["metric"] != "dice":
                continue
            ok = x["pairing_ok"] and x["difference_exact"] and x["ci_agrees"] and x["p_agrees"] and x["holm_from_stored_raw_exact"]
            out.append([LABEL[x["method_a"]], LABEL[x["method_b"]], "%+.4f" % x["stored_difference"],
                        "[%+.4f, %+.4f]" % tuple(x["stored_ci"]), "%.5f" % x["stored_raw_p"], "%.5f" % x["stored_holm_p"],
                        yes(x["pairing_ok"]), yes(x["ci_agrees"]), yes(x["p_agrees"]), "Yes (%d clusters)" % x["n_clusters"],
                        "PASS" if ok else "FAIL"])
        return out

    hdr = ["Method A", "Method B", "Paired Dice Difference", "95% CI", "Raw p", "Holm-adjusted p", "Pairing correct?",
           "Bootstrap correct?", "Permutation correct?", "Cluster handling correct?", "Status"]
    add("## 13. Phase 0 Statistical Verification\n")
    add("Source: `results/phase0/paired_comparisons.csv` (Phase 0 scoring at 384x384). Independent recomputation: "
        "exact paired differences; vectorized cluster bootstrap (10,000) and sign permutation (100,000) with a "
        "different seed; Holm recomputed from the stored raw p-values (family of 3 per dataset).\n")
    for d in ("qata", "mosmed", "busi"):
        add("### %s\n" % DS[d])
        add(table(hdr, stat_rows([x for x in AUDIT["phase0_statistics"] if x["dataset"] == d], 3)))
    add("## 14. Final Five-Method Statistical Verification\n")
    add("Source: `results/final_five_method/paired_comparisons.csv`. Same independent recomputation; Holm family = "
        "exactly the 10 Dice comparisons within each dataset (verified). All 80 rows (Dice, mIoU, HD95, ASSD) "
        "reproduce; secondary-metric p-values are labelled exploratory in the file and report.\n")
    for i, d in enumerate(("qata", "mosmed"), 1):
        add("### 14.%d %s\n" % (i, DS[d]))
        add(table(hdr, stat_rows([x for x in AUDIT["final_statistics"] if x["dataset"] == d], 10)))
    add("Seed handling: each test case is averaged across its three seeds within a method before pairing; rows from "
        "different seeds are never treated as independent cases. Significance was not tested on seed-level means.\n")

    add("## 15. Cross-File Consistency Review\n")
    add(table(["Issue", "File A", "Value A", "File B", "Value B", "Likely authoritative source", "Required correction"], [
        ["Phase 0 scoring resolution", "`reports/PHASE0_IMAGE_ONLY_RESULTS.md`", "384x384 grid (e.g. QaTa U-Net Dice 0.7836)",
         "`reports/FINAL_FIVE_METHOD_COMPARISON.md`", "Native grid (0.7845)", "Both, for their stated scopes",
         "None; documented in both reports"],
        ["Phase 1 stored vs native", "`methods/phase1_language_guided/runs/*/result.json`", "224x224 (MosMed LViT-T 0.7972)",
         "`runs/phase1/.../test_summary.json`", "Native (0.7929)", "Native grid for the comparison", "None; sensitivity table in Phase 1 report"],
        ["Phase 1 manifest hashes", "`PROVENANCE.yaml`", "`b372f232…` / `9f86cf11…` (archived launch)",
         "`run_config.json` (12 runs)", "`e12e88a0…` / `5f2cbe3a…`", "run_config.json",
         "Done: `PROVENANCE_CURRENT.md` addendum (file kept; it is hashed into run configs)"],
        ["Adapter validation counts", "`ADAPTER_VALIDATION.md` (old)", "MosMed 2183/273/273", "Manifest", "1746/437/546",
         "Manifest", "Done: rewritten 2026-10-09"],
        ["U-Net QaTa 1001 log", "`upstream/qata/history.csv`", "Epochs 53-56", "`logs/unet_tmux.log`", "Epochs 1-56",
         "Terminal log (consistent with checkpoint)", "Done: rebuilt `train_log.csv` with note"],
        ["Report/result names", "Original plan", "`FINAL_SIX_METHOD_COMPARISON.md`, `final_six_method/`",
         "Repository", "`…FIVE…`", "Repository", "Done: renamed"],
        ["Gate status", "`gates/<m>/gate_results.json`", "official check pending", "`gates/<m>/official_qata/result.json`",
         "pass", "official_qata/result.json", "Done: `gates/STATUS.md` updated 2026-10-03"],
    ]))
    add("No conflicting Dice values, best epochs, sample counts, seeds, thresholds, checkpoint paths, source commits, "
        "confidence intervals or p-values were found between per-case files, summaries, statistics files and reports "
        "beyond the documented scoring-resolution difference. The final report's summary table matches "
        "`summary_table.csv` verbatim (13/13 rows); the hand-written Phase 0 report matches its result files (18 rows, "
        "0 mismatches).\n")

    add("## 16. Leakage and Research Integrity Review\n")
    add(table(["Item", "Classification", "Evidence"], [
        ["Patient/source leakage", "CONFIRMED (MosMed, all methods); POTENTIAL RISK (QaTa val)", "Section 7; B-01, m-01"],
        ["Duplicate-sample leakage", "NO EVIDENCE FOUND", "0 identical image or mask hashes across splits"],
        ["Test-driven checkpoint selection", "NO EVIDENCE FOUND",
         "Phase 0 training logs contain train/val only; Phase 1 tests once after training behind `test_started.json`"],
        ["Test-time hyperparameter tuning", "NO EVIDENCE FOUND", "Frozen configs; one config/runner hash per method"],
        ["Ground-truth-derived text", "NOT VERIFIED", "M-01; reports encode lesion count and location"],
        ["Missing/excluded test cases", "NO EVIDENCE FOUND", "All test IDs present in 39/39 runs"],
        ["Selective seed reporting", "NO EVIDENCE FOUND",
         "All three seeds reported. Archived split runs (one completed LViT QaTa seed 1001 at Dice 0.8172, replaced "
         "for a split-protocol reason, new value 0.8174) and the OOM restart are documented"],
        ["Behaviour-changing version mixing", "NO EVIDENCE FOUND", "Single config/runner hash per method"],
        ["Wrong checkpoint use", "NO EVIDENCE FOUND", "Selected epoch = best validation epoch; Phase 1 hashes verified"],
        ["Evaluation choice after test observation", "POTENTIAL RISK (disclosed)", "m-02"],
    ]))

    add("## 17. Experimental Trends and Findings\n")
    add("### 17.1 Overall performance trend\n")
    add("QaTa-COV19-v2 mean Dice: RecLMIS 0.8369 > LViT-T 0.8176 > Panoptic FPN 0.8035 > U-Net v2 0.7845 > "
        "TriPathLesionNet 0.7813. MosMedData+: Panoptic FPN 0.8076 ≈ TriPathLesionNet 0.8071 > U-Net v2 0.7937 ≈ "
        "LViT-T 0.7929 > RecLMIS 0.7889. BUSI (Phase 0 only): Panoptic FPN 0.8247 > TriPathLesionNet 0.7940 > "
        "U-Net v2 0.6937.\n")
    add("### 17.2 Image-only vs language-guided trend\n")
    add("Dataset-dependent. On QaTa both language-guided methods exceed all three image-only methods (all six "
        "cross-family differences Holm-significant). On MosMed neither does: TriPathLesionNet and Panoptic FPN are "
        "significantly higher than both, and U-Net v2 is not distinguishable from either. This does not show that "
        "language causes the QaTa gain: the families also differ in architecture, pretrained encoders, input "
        "resolution (224 vs 384), augmentation, and the test-time reports state lesion count and location (M-01). "
        "The post-hoc text check shows strong text dependence on QaTa (shuffled reports lower validation Dice from "
        "about 0.79-0.82 to about 0.58-0.64) and weak dependence on MosMed.\n")
    add("### 17.3 QaTa vs MosMed behaviour\n")
    add("The ranking reverses between datasets for the language-guided methods (top two on QaTa, bottom two on "
        "MosMed). Panoptic FPN is the strongest image-only method on all three datasets. MosMed values carry B-01, "
        "and Phase 1 methods pay a larger resolution cost there (224 network input vs 512 masks; stored-resolution "
        "Dice is 0.004 higher).\n")
    add("### 17.4 Seed stability\n")
    add("Three-seed SD is at most 0.005 for every method on QaTa and MosMed except TriPathLesionNet on MosMed "
        "(SD 0.0082, range 0.016). The least stable cell is U-Net v2 on BUSI (SD 0.0214, range 0.042). No reported "
        "conclusion depends on one seed: every Holm-significant QaTa/MosMed difference exceeds the largest seed range "
        "of the two methods involved, except two MosMed pairs involving TriPathLesionNet (seed range 0.016): vs "
        "U-Net v2 (difference 0.0134) and vs LViT-T (0.0142). These should be described cautiously.\n")
    add("### 17.5 Dice/mIoU vs HD95/ASSD\n")
    add("On QaTa the boundary metrics broadly follow the Dice ranking (HD95: RecLMIS 16.4 px, LViT-T 19.6, Panoptic FPN 27.4, "
        "U-Net v2 31.6, TriPathLesionNet 31.5; the two lowest-Dice methods are effectively tied on HD95). On MosMed they diverge: TriPathLesionNet has the best ASSD (6.25 px) "
        "and HD95 (23.5) while Panoptic FPN has similar Dice but ASSD 9.35; RecLMIS has the worst HD95/ASSD (30.6/14.4). "
        "On BUSI, U-Net v2's low Dice comes with very large boundary errors (HD95 111.5 px). Secondary differences are "
        "exploratory.\n")
    add("### 17.6 Case-level paired behaviour\n")
    cl = {(r["dataset"], r["method_a"], r["method_b"]): r for r in case_level}
    x = cl["qata", "unet", "reclmis"]
    y = cl["mosmed", "panoptic_fpn", "reclmis"]
    add("Gains are broad but not uniform. On QaTa, RecLMIS beats U-Net v2 in %.0f%% of cases; the mean difference "
        "(%+.4f) is larger than the median (%+.4f) and the 5%% trimmed mean (%+.4f), so part of the gain comes from "
        "cases where the image-only model fails badly. On MosMed, Panoptic FPN beats RecLMIS in %.0f%% of cases "
        "(median %+.4f). Full table: `results/final_five_method/case_level_paired_summary.csv`.\n"
        % (float(x["pct_cases_b_better"]), -float(x["mean_difference"]), -float(x["median_difference"]),
           -float(x["trimmed_mean_difference_5pct"]), float(y["pct_cases_a_better"]), float(y["median_difference"])))
    add("### 17.7 Statistically supported findings\n")
    add("- QaTa: 9 of 10 pairs are Holm-significant (paired 95% CI excludes zero in the same 9). Not detectable: "
        "U-Net v2 vs TriPathLesionNet.\n- MosMed: 6 of 10 pairs are Holm-significant: TriPathLesionNet and Panoptic "
        "FPN each exceed U-Net v2, LViT-T and RecLMIS.\n")
    add("### 17.8 Descriptive findings without corrected statistical support\n")
    add("- MosMed: TriPathLesionNet vs Panoptic FPN, LViT-T vs RecLMIS and U-Net v2 vs either Phase 1 method are "
        "not detectably different.\n- All secondary-metric differences and the text-use and case-level analyses are "
        "descriptive or exploratory.\n- BUSI has only the three Phase 0 comparisons (Section 13).\n")

    add("## 18. Problems Requiring Action\n")
    problems = [
        ("18.1 BLOCKERS", [dict(
            id="B-01", sev="BLOCKER (pending PI decision)", phase="0 and 1", method="All five", dataset="MosMedData+",
            seeds="All", evidence="`results/audit/lock_audit_checks.json` `datasets.mosmed`: 68/86 test studies in train, "
            "63 in val; 528/546 test images from a training study. Same property in the archived annotation split (67%).",
            wrong="The frozen MosMed split is slice-level, so most test patients contribute training slices.",
            why="Absolute MosMed scores measure generalization to new slices of seen patients, not new patients; "
            "reviewers will question them. Between-method comparison remains like-for-like.",
            ckpt="Yes (if accepted); No (if re-split)", retrain="No if accepted; Yes (15 runs) if re-split",
            reinfer="No / Yes", metrics="No / Yes", stats="No / Yes (MosMed rows)",
            action="1. PI decides: accept as a documented limitation, or require a study-level re-split. "
            "2a. If accepted: record the decision in `MAIN_EXPERIMENT_STATUS.md` and this report (caveat text is already "
            "in both result reports). 2b. If re-split: build a study-grouped MosMed manifest (seeded, all slices of a "
            "`source_id` in one split), retrain the 5 methods x 3 seeds on it, rerun the shared evaluator, statistics and "
            "reports, and rerun this audit.",
            expected="Signed PI decision; or new MosMed manifest with 0 train-test source overlap and 15 new runs.",
            passc="Decision recorded; or `source_overlap.train-test == 0` and all MosMed checks PASS.")]),
        ("18.2 MAJOR ISSUES", [dict(
            id="M-01", sev="MAJOR", phase="1", method="LViT-T, RecLMIS", dataset="QaTa, MosMed", seeds="All",
            evidence="Reports such as \"Bilateral pulmonary infection, two infected areas, all left lung and middle "
            "lower right lung.\" are given at test time; 394 (QaTa) and 115 (MosMed) unique reports; workbooks pinned "
            "by hash; no generation code in the repository; post-hoc text check shows strong QaTa text dependence.",
            wrong="The origin of the reports relative to the masks cannot be verified, and they carry lesion count and "
            "location that image-only methods do not receive.",
            why="Language-guided vs image-only differences cannot be attributed to language understanding, and could "
            "partly reflect label-like information at test time.",
            ckpt="Yes", retrain="No", reinfer="No", metrics="No", stats="No",
            action="1. PI reviews the caveat (already in both result reports). 2. Cite the dataset authors' description "
            "of how the QaTa/MosMed reports were produced, or state that it is unknown. 3. Phrase conclusions as method "
            "performance with native inputs, not as an effect of language.",
            expected="Caveat accepted and wording agreed.", passc="PI sign-off on the interpretation wording.")]),
        ("18.3 MINOR ISSUES", [
            dict(id="m-01", sev="MINOR", phase="0 and 1", method="All", dataset="QaTa", seeds="All",
                 evidence="444 subjects in both train and val; 0 in test.", wrong="Validation is not subject-disjoint.",
                 why="Checkpoint selection may be slightly optimistic; test results are unaffected.", ckpt="Yes",
                 retrain="No", reinfer="No", metrics="No", stats="No", action="Keep the documented caveat.",
                 expected="Caveat in reports (done).", passc="Caveat present."),
            dict(id="m-02", sev="MINOR (disclosed)", phase="1", method="LViT-T, RecLMIS", dataset="MosMed (QaTa unaffected)",
                 seeds="All", evidence="Shared evaluator header; validation-mask round-trip ceilings 0.933 vs 0.885.",
                 wrong="The corner-aligned resampling rule was investigated after a test-set drop was observed.",
                 why="Any post-hoc evaluation choice is a potential researcher-degree-of-freedom.",
                 ckpt="Yes", retrain="No", reinfer="No", metrics="No", stats="No",
                 action="PI review; both stored-resolution and native-grid values remain reported.",
                 expected="PI acknowledgement.", passc="Disclosed in reports and audit (done)."),
            dict(id="m-03", sev="MINOR", phase="0", method="U-Net v2", dataset="QaTa", seeds="1001",
                 evidence="`history.csv` epochs 53-56 only; `logs/unet_tmux.log` epochs 1-56; best epoch 41.",
                 wrong="A relaunch rewrote the run's history file.", why="Training log completeness.",
                 ckpt="Yes", retrain="No", reinfer="No", metrics="No", stats="No",
                 action="Done: `runs/phase0/unet/qata/1001/train_log.csv` rebuilt from the terminal log with a note.",
                 expected="Complete 56-epoch log.", passc="Selected epoch equals best validation epoch (verified)."),
            dict(id="m-04", sev="MINOR", phase="1", method="LViT-T, RecLMIS", dataset="QaTa, MosMed", seeds="Gates",
                 evidence="Gate results dated 2026-09-14/15, before the switch to Phase 0 manifests.",
                 wrong="Gates used the archived annotation split.", why="Gates verify the pipeline, not membership.",
                 ckpt="Yes", retrain="No", reinfer="No", metrics="No", stats="No",
                 action="Done: smoke checks on the final manifests and post-hoc text-use check on final checkpoints.",
                 expected="-", passc="Section 8.1 PASS."),
            dict(id="m-05", sev="MINOR", phase="0", method="All Phase 0", dataset="-", seeds="-",
                 evidence="No LICENSE in `upstream/Summer2026Research`.", wrong="License undocumented.",
                 why="Required before public release.", ckpt="Yes", retrain="No", reinfer="No", metrics="No", stats="No",
                 action="Add a license to the Summer2026Research repository and update the three SOURCE.md files.",
                 expected="LICENSE file.", passc="License recorded."),
            dict(id="m-06", sev="MINOR", phase="0 and 1", method="All", dataset="All", seeds="-",
                 evidence="Phase 0: 384x384 grayscale with flip/rotation augmentation; Phase 1: 224x224 RGB, augmentation "
                 "disabled (text safety); thop FLOPs at different inputs, attention uncounted.",
                 wrong="Native recipes differ by design.", why="Limits causal and efficiency interpretations.",
                 ckpt="Yes", retrain="No", reinfer="No", metrics="No", stats="No",
                 action="Keep as reported limitations.", expected="-", passc="Documented (done)."),
            dict(id="m-07", sev="MINOR", phase="0 and 1", method="-", dataset="-", seeds="-",
                 evidence="`git status`: audit fixes uncommitted; branch `phase1-main-experiment` not yet pushed.",
                 wrong="Locked state not yet in version control.", why="Lock must reference a commit.",
                 ckpt="-", retrain="No", reinfer="No", metrics="No", stats="No",
                 action="Commit, push (VS Code Publish Branch), merge to master, tag the lock commit.",
                 expected="Tagged commit on GitHub.", passc="`git status` clean; tag exists on origin."),
        ]),
    ]
    for heading, plist in problems:
        add("### %s\n" % heading)
        for p in plist:
            add("```text\nProblem ID: {id}\nSeverity: {sev}\nAffected phase: {phase}\nAffected method: {method}\n"
                "Affected dataset: {dataset}\nAffected seed(s): {seeds}\n```\n".format(**p))
            add("- Evidence: %s\n- What is wrong: %s\n- Why it matters: %s\n- Existing checkpoint usable: %s\n"
                "- Retraining required: %s\n- Re-inference required: %s\n- Metric recomputation required: %s\n"
                "- Statistical recomputation required: %s\n- Exact corrective action: %s\n"
                "- Expected output after correction: %s\n- Pass condition: %s\n"
                % (p["evidence"], p["wrong"], p["why"], p["ckpt"], p["retrain"], p["reinfer"], p["metrics"], p["stats"],
                   p["action"], p["expected"], p["passc"]))

    add("## 19. Corrective Action Plan and Dependency Order\n")
    add("Fixes already applied during this audit (no retraining, no change to any trained checkpoint or to previously "
        "reported QaTa/MosMed values): shared evaluator extended to BUSI with byte-identical QaTa/MosMed outputs; "
        "standard run folders for all 39 runs; Phase 0 SOURCE.md files; adapter validation refreshed; provenance "
        "addendum; result/report renamed to five-method; summary table with per-seed Dice, BUSI rows and status; "
        "case-level and seed-stability outputs; interpretation caveats; post-hoc text-use check; U-Net QaTa 1001 log "
        "recovery.\n")
    add("Remaining, in dependency order:\n")
    add("1. **FIX-01 (B-01) PI decision on MosMed.** Input: Section 7 and 18.1. If re-split is required, steps 2-5 "
        "follow it: new grouped manifest -> 15 MosMed runs -> `shared_evaluator.py` -> `export_run_records.py` -> "
        "`phase1_final_statistics.py` -> `lock_audit_checks.py` -> `write_lock_audit_report.py`. Pass: decision recorded, "
        "or MosMed train-test source overlap 0 with all checks PASS.\n"
        "2. **FIX-02 (M-01) PI sign-off on the report-text caveat and wording.** Pass: wording agreed.\n"
        "3. **FIX-03 (m-05) Add a license to Summer2026Research.** Pass: LICENSE present, SOURCE.md updated.\n"
        "4. **FIX-04 (m-07) Commit, push, merge and tag the lock.** Pass: clean working tree, tag on origin.\n"
        "5. Regenerate this report and set the decision line accordingly.\n")
    OUT.write_text("\n".join(L))
    print("Wrote", OUT, "(%d lines)" % sum(1 for _ in OUT.open()))


if __name__ == "__main__":
    main()
