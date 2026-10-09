# Phase 0 and Phase 1 Lock Audit

## 1. Executive Decision

NOT READY TO LOCK

Every required experiment is present, valid and independently reproduced, and no retraining, re-inference or statistical recomputation is required by any finding below. The decision is NOT READY only because one scientific question needs the PI's ruling before the results are frozen: the frozen MosMedData+ split places slices of the same CT study in training and test (B-01). If the PI accepts B-01 as a documented limitation and accepts the report-text caveat (M-01), the remaining actions are non-experimental (commit, push, sign-off) and the status becomes READY AFTER NON-EXPERIMENTAL FIXES. If the PI requires a patient-level MosMed split, all 15 MosMed runs (5 methods x 3 seeds) must be retrained before lock.

```text
Critical blockers: 1
  B-01 MosMedData+ test studies overlap training studies (528 of 546 test images, 97%) - PI decision required
Major issues: 1
  M-01 Phase 1 test-time reports state lesion count and location; provenance relative to the masks NOT VERIFIED
Minor issues: 7
  m-01 QaTa train/validation subject overlap (444 subjects; none in test)
  m-02 Native-grid alignment rule chosen after a test-set observation (disclosed; validation-only confirmation)
  m-03 U-Net v2 QaTa seed 1001 history.csv overwritten by a relaunch (full log recovered)
  m-04 Phase 1 gates ran on the archived annotation split (pipeline checks; post-hoc text-use re-verified on final checkpoints)
  m-05 Summer2026Research has no LICENSE file
  m-06 Comparability notes: input resolution, augmentation and FLOP counting differ by design
  m-07 Work not yet committed/pushed to GitHub
```

## 2. Audit Scope

```text
Phase 0 methods:
- U-Net v2
- TriPathLesionNet
- Panoptic FPN

Phase 0 datasets:
- QaTa-COV19-v2
- MosMedData+
- BUSI

Phase 1 methods:
- LViT-T
- RecLMIS

Phase 1 datasets:
- QaTa-COV19-v2
- MosMedData+

Seeds:
- 1001
- 1002
- 1003

Expected total runs:
39
```

ProLearn was removed from the study by the research advisor on 2026-10-03 and is out of scope. The final comparison therefore has 5 methods and 10 pairs per dataset.

## 3. Repository Snapshot

| Item | Value |
|---|---|
| Repository path | /data/ramialle/LanguageGuidedMedicalSegmentation |
| Branch | phase1-main-experiment |
| Commit | f4c7b4671af0173435770df8e04aa244d5ae5fa6 |
| Working tree | 67 uncommitted entries (this audit's fixes and outputs; see m-07). Upstream submodules: compiled `.pyc` caches only, no source changes |
| Summer2026Research | `fdeb44037aba52900d852271293b165a697ecf06` (pinned = current) |
| LViT | `ba90775ead37d6f445f65ce83d1a0be64157f76f` (pinned = current) |
| RecLMIS | `d3265c8b4d4582a94595fe1c8b1892c618bc9715` (pinned = current) |
| Audit date/time | 2026-10-09 01:53  |
| Audit tooling | `statistics/lock_audit_checks.py` -> `results/audit/lock_audit_checks.json`; this report: `statistics/write_lock_audit_report.py` |

Authoritative locations: `runs/<phase>/<method>/<dataset>/<seed>/` (shared-evaluator records), `results/phase0/`, `results/phase1/`, `results/final_five_method/`, `reports/`. Training-time artifacts (checkpoints, logs) live in `methods/phase0_image_only/runs/` and `methods/phase1_language_guided/runs/`. Not authoritative: `methods/phase1_language_guided/archive/` (superseded annotation-split launch), `smoke_runs*/` (smoke checks on validation cases), `methods/phase0_image_only/aborted_attempts/`, `methods/phase1_language_guided/gates/` (pre-experiment gate evidence).

## 4. Expected vs Observed Experiment Matrix

| Phase | Method | Dataset | Seed | Expected | Run Found | Run Valid | Primary Run Path | Status | Comments |
|---|---|---|---|---|---|---|---|---|---|
| 0 | U-Net v2 | QaTa-COV19-v2 | 1001 | Yes | Yes | Yes | `runs/phase0/unet/qata/1001` | PASS | log recovered, see m-03 |
| 0 | U-Net v2 | QaTa-COV19-v2 | 1002 | Yes | Yes | Yes | `runs/phase0/unet/qata/1002` | PASS |  |
| 0 | U-Net v2 | QaTa-COV19-v2 | 1003 | Yes | Yes | Yes | `runs/phase0/unet/qata/1003` | PASS |  |
| 0 | U-Net v2 | MosMedData+ | 1001 | Yes | Yes | Yes | `runs/phase0/unet/mosmed/1001` | PASS |  |
| 0 | U-Net v2 | MosMedData+ | 1002 | Yes | Yes | Yes | `runs/phase0/unet/mosmed/1002` | PASS |  |
| 0 | U-Net v2 | MosMedData+ | 1003 | Yes | Yes | Yes | `runs/phase0/unet/mosmed/1003` | PASS |  |
| 0 | U-Net v2 | BUSI | 1001 | Yes | Yes | Yes | `runs/phase0/unet/busi/1001` | PASS |  |
| 0 | U-Net v2 | BUSI | 1002 | Yes | Yes | Yes | `runs/phase0/unet/busi/1002` | PASS |  |
| 0 | U-Net v2 | BUSI | 1003 | Yes | Yes | Yes | `runs/phase0/unet/busi/1003` | PASS |  |
| 0 | TriPathLesionNet | QaTa-COV19-v2 | 1001 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/qata/1001` | PASS |  |
| 0 | TriPathLesionNet | QaTa-COV19-v2 | 1002 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/qata/1002` | PASS |  |
| 0 | TriPathLesionNet | QaTa-COV19-v2 | 1003 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/qata/1003` | PASS |  |
| 0 | TriPathLesionNet | MosMedData+ | 1001 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/mosmed/1001` | PASS |  |
| 0 | TriPathLesionNet | MosMedData+ | 1002 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/mosmed/1002` | PASS |  |
| 0 | TriPathLesionNet | MosMedData+ | 1003 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/mosmed/1003` | PASS |  |
| 0 | TriPathLesionNet | BUSI | 1001 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/busi/1001` | PASS |  |
| 0 | TriPathLesionNet | BUSI | 1002 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/busi/1002` | PASS |  |
| 0 | TriPathLesionNet | BUSI | 1003 | Yes | Yes | Yes | `runs/phase0/tripath_lesionnet/busi/1003` | PASS |  |
| 0 | Panoptic FPN | QaTa-COV19-v2 | 1001 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/qata/1001` | PASS |  |
| 0 | Panoptic FPN | QaTa-COV19-v2 | 1002 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/qata/1002` | PASS |  |
| 0 | Panoptic FPN | QaTa-COV19-v2 | 1003 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/qata/1003` | PASS |  |
| 0 | Panoptic FPN | MosMedData+ | 1001 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/mosmed/1001` | PASS |  |
| 0 | Panoptic FPN | MosMedData+ | 1002 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/mosmed/1002` | PASS |  |
| 0 | Panoptic FPN | MosMedData+ | 1003 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/mosmed/1003` | PASS |  |
| 0 | Panoptic FPN | BUSI | 1001 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/busi/1001` | PASS |  |
| 0 | Panoptic FPN | BUSI | 1002 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/busi/1002` | PASS |  |
| 0 | Panoptic FPN | BUSI | 1003 | Yes | Yes | Yes | `runs/phase0/panoptic_fpn/busi/1003` | PASS |  |
| 1 | LViT-T | QaTa-COV19-v2 | 1001 | Yes | Yes | Yes | `runs/phase1/lvit/qata/1001` | PASS |  |
| 1 | LViT-T | QaTa-COV19-v2 | 1002 | Yes | Yes | Yes | `runs/phase1/lvit/qata/1002` | PASS |  |
| 1 | LViT-T | QaTa-COV19-v2 | 1003 | Yes | Yes | Yes | `runs/phase1/lvit/qata/1003` | PASS |  |
| 1 | LViT-T | MosMedData+ | 1001 | Yes | Yes | Yes | `runs/phase1/lvit/mosmed/1001` | PASS |  |
| 1 | LViT-T | MosMedData+ | 1002 | Yes | Yes | Yes | `runs/phase1/lvit/mosmed/1002` | PASS |  |
| 1 | LViT-T | MosMedData+ | 1003 | Yes | Yes | Yes | `runs/phase1/lvit/mosmed/1003` | PASS |  |
| 1 | RecLMIS | QaTa-COV19-v2 | 1001 | Yes | Yes | Yes | `runs/phase1/reclmis/qata/1001` | PASS |  |
| 1 | RecLMIS | QaTa-COV19-v2 | 1002 | Yes | Yes | Yes | `runs/phase1/reclmis/qata/1002` | PASS |  |
| 1 | RecLMIS | QaTa-COV19-v2 | 1003 | Yes | Yes | Yes | `runs/phase1/reclmis/qata/1003` | PASS |  |
| 1 | RecLMIS | MosMedData+ | 1001 | Yes | Yes | Yes | `runs/phase1/reclmis/mosmed/1001` | PASS |  |
| 1 | RecLMIS | MosMedData+ | 1002 | Yes | Yes | Yes | `runs/phase1/reclmis/mosmed/1002` | PASS |  |
| 1 | RecLMIS | MosMedData+ | 1003 | Yes | Yes | Yes | `runs/phase1/reclmis/mosmed/1003` | PASS |  |

```text
Expected runs: 39
Found runs: 39
Valid runs: 39
Invalid runs: 0
Missing runs: 0
```

A run is valid when: the selected checkpoint exists; its epoch equals the best validation-Dice epoch in a complete training log; per-case rows equal the manifest test IDs in order with the correct method, seed and source ID; every metric is finite and in range; mIoU = Dice/(2-Dice) holds per case; per-case means equal `test_summary.json`; and one binary mask and one probability map exist per test case.

## 5. Phase 0 Requirement-by-Requirement Audit

### 5.1 Method import and provenance

- Requirement: Three methods imported from pinned upstream with unchanged source
- Status: PASS
- Evidence: `upstream/Summer2026Research` at `fdeb4403…`, 0 source changes; thin wrappers in `methods/phase0_image_only/models/`; `SOURCE.md` per method (added in this audit).
- Finding: Imports are wrapper-only. Upstream has no LICENSE (m-05).
- Required action: Add a license to the source repository.

### 5.2 Frozen method configuration

- Requirement: Same config, commit and manifest for every seed
- Status: PASS
- Evidence: Each method has one `config_sha256` across its 9 runs (`version_consistency`); resolved configs differ only in seed and output paths (`config_drift`); `CORRECTNESS_CHANGES.md`: no fixes.
- Finding: No architecture search or test-driven tuning found; checkpoint = best validation Dice (strictly greater); early stopping patience 15, max 100 epochs.
- Required action: None.

### 5.3 Dataset/split correctness

- Requirement: All three methods use the frozen manifests
- Status: PASS
- Evidence: Per-case IDs of all 27 runs equal the manifest test IDs in order; manifests hashed in `run_metadata.json`.
- Finding: Split identities are shared. Source overlap is reported in Section 7 (B-01, m-01).
- Required action: See B-01.

### 5.4 Three-seed completion

- Requirement: 27 runs, seeds 1001-1003
- Status: PASS
- Evidence: 27/27 valid.
- Finding: All trained in this repository; no migrated runs.
- Required action: None.

### 5.5 Saved prediction completeness

- Requirement: Probability map and mask per test case
- Status: PASS
- Evidence: 0 missing/extra predictions in all 27 runs (original 384x384 maps plus native-grid masks).
- Finding: Meets the requirement.
- Required action: None.

### 5.6 Per-case metrics

- Requirement: One row per test case; means match summaries
- Status: PASS
- Evidence: All 27 runs: IDs, sources, method/seed columns, ranges and means verified.
- Finding: Meets the requirement.
- Required action: None.

### 5.7 Phase 0 statistics

- Requirement: Seed-averaged, clustered, 10k bootstrap, 100k permutation, Holm over 3
- Status: PASS
- Evidence: Section 13: 9/9 comparisons reproduced independently; Holm family size 3 per dataset.
- Finding: Phase 0 report values match `results/phase0/` exactly (18 table rows parsed, 0 mismatches).
- Required action: None.

### 5.8 Phase 0 exit criteria

- Requirement: Methods present, splits frozen, 3 seeds, predictions, CIs, paired tests, report
- Status: PASS
- Evidence: All criteria evidenced above; `reports/PHASE0_IMAGE_ONLY_RESULTS.md`.
- Finding: Phase 0 report scores at 384x384; the final comparison re-scores on the native grid (documented).
- Required action: None.

Epoch policy (selected epoch / logged epochs per seed):

| Method | Dataset | Max epochs | Early stopping | Seeds 1001 / 1002 / 1003 |
|---|---|---|---|---|
| U-Net v2 | QaTa-COV19-v2 | 100 | patience 15 | 41/56 / 58/73 / 47/62 |
| U-Net v2 | MosMedData+ | 100 | patience 15 | 98/100 / 91/100 / 89/100 |
| U-Net v2 | BUSI | 100 | patience 15 | 51/66 / 46/61 / 51/66 |
| TriPathLesionNet | QaTa-COV19-v2 | 100 | patience 15 | 48/63 / 55/70 / 54/69 |
| TriPathLesionNet | MosMedData+ | 100 | patience 15 | 89/100 / 49/64 / 80/95 |
| TriPathLesionNet | BUSI | 100 | patience 15 | 63/78 / 45/60 / 45/60 |
| Panoptic FPN | QaTa-COV19-v2 | 100 | patience 15 | 28/43 / 18/33 / 17/32 |
| Panoptic FPN | MosMedData+ | 100 | patience 15 | 48/63 / 65/80 / 62/77 |
| Panoptic FPN | BUSI | 100 | patience 15 | 15/30 / 34/49 / 26/41 |
| LViT-T | QaTa-COV19-v2 | 2000 | patience 50 | 29/80 / 50/101 / 10/61 |
| LViT-T | MosMedData+ | 2000 | patience 50 | 149/200 / 218/269 / 248/299 |
| RecLMIS | QaTa-COV19-v2 | 2000 QaTa / 400 MosMed | patience 100 QaTa / 30 MosMed | 18/119 / 20/121 / 17/96 |
| RecLMIS | MosMedData+ | 2000 QaTa / 400 MosMed | patience 100 QaTa / 30 MosMed | 35/66 / 70/101 / 48/79 |

Validation runs every epoch for every method. The selected epoch equals the best validation-Dice epoch in all 39 runs. RecLMIS QaTa seed 1003 stopped at a nonfinite loss (epoch 97) and tested its pre-event best checkpoint (epoch 17).

## 6. Phase 1 Requirement-by-Requirement Audit

### 6.1 LViT-T implementation and provenance

- Requirement: Native LViT-T preserved
- Status: PASS
- Evidence: `upstream/LViT` at `ba90775e…`, 0 source changes; `methods/phase1_language_guided/lvit_t/SOURCE.md`; empty `patches/lvit.patch`.
- Finding: Native model, WeightedDiceBCE, Adam lr 3e-4, cosine warm restarts, batch 2, patience 50, frozen BERT-base token embeddings (10 tokens). Deviations: training augmentation disabled (text-safe policy); lr 3e-4 used for MosMed (upstream gives only a Covid19 value).
- Required action: None.

### 6.2 RecLMIS implementation and provenance

- Requirement: Native RecLMIS preserved
- Status: PASS
- Evidence: `upstream/RecLMIS` at `d3265c8b…`, 0 source changes; `reclmis/SOURCE.md`; `reclmis_compat.py` supplies only `fairseq.utils.softmax` (v0.10.2).
- Finding: All three auxiliary losses asserted present every step; per-dataset native config files; CLIP ViT-B/32 weights. Without the shim, upstream silently drops the auxiliary losses; this was detected and prevented. Augmentation disabled (as above).
- Required action: None.

### 6.3 Shared image-mask-text data handling

- Requirement: Pairing by stable key; text provenance
- Status: PASS (pairing) / NOT VERIFIED (text origin)
- Evidence: `adapter.py`: manifest rows in file order; report joined by image basename from hash-pinned workbooks; missing, empty, duplicate or conflicting reports are fatal; `validate_adapter.py` passes (2026-10-09).
- Finding: No directory listing or positional matching is used. Reports come unchanged from the published annotation workbooks; no generation script exists in the repository. Whether the reports were derived from masks cannot be verified (M-01).
- Required action: PI review of M-01.

### 6.4 Reproduction gates

- Requirement: Seven gates per method
- Status: PASS
- Evidence: Section 8.
- Finding: Gates ran on the archived split (m-04).
- Required action: None.

### 6.5 Dataset/split correctness

- Requirement: Same samples as Phase 0
- Status: PASS
- Evidence: Phase 1 manifests are byte-identical to Phase 0 (`e12e88a0…`, `5f2cbe3a…`); all 12 run configs record these hashes; one runner hash across all 12 runs.
- Finding: Meets the requirement.
- Required action: None.

### 6.6 Three-seed completion

- Requirement: 12 runs
- Status: PASS
- Evidence: 12/12 valid.
- Finding: One technical restart (RecLMIS QaTa 1002, CUDA OOM at startup before training) is documented.
- Required action: None.

### 6.7 Saved prediction completeness

- Requirement: Probability map and mask per test case
- Status: PASS
- Evidence: 0 missing/extra predictions in all 12 runs.
- Finding: Meets the requirement.
- Required action: None.

### 6.8 Per-case metrics

- Requirement: One row per test case; means match summaries
- Status: PASS
- Evidence: All 12 runs verified (Section 4 criteria).
- Finding: Meets the requirement.
- Required action: None.

### 6.9 Phase 1 statistics

- Requirement: Mean, SD, clustered CIs
- Status: PASS
- Evidence: `results/phase1/absolute_ci.csv`: means and SDs reproduced exactly; CIs reproduced within Monte Carlo error.
- Finding: Meets the requirement.
- Required action: None.

### 6.10 Phase 1 exit criteria

- Requirement: Gates, 3 seeds, predictions, CIs, 10-pair comparison, separate reports
- Status: PASS
- Evidence: Sections 4, 8, 14; `reports/PHASE1_LANGUAGE_GUIDED_RESULTS.md`, `reports/FINAL_FIVE_METHOD_COMPARISON.md`.
- Finding: ProLearn criteria not applicable (removed by advisor).
- Required action: None.

## 7. Dataset Integrity and Split Verification

| Dataset | Train N | Val N | Test N | Unique source train / val / test | Text-paired | Missing images | Missing masks | Duplicate IDs | Manifest sha256 |
|---|---|---|---|---|---|---|---|---|---|
| QaTa-COV19-v2 | 5716 | 1429 | 2113 | 3840 / 1211 / 474 | 100% (all splits) | 0 | 0 | 0 | `e12e88a0b570ad14` |
| MosMedData+ | 1746 | 437 | 546 | 138 / 78 / 86 | 100% (all splits) | 0 | 0 | 0 | `5f2cbe3a7e1cf91c` |
| BUSI | 413 | 104 | 130 | 413 / 104 / 130 | N/A (Phase 0 only) | 0 | 0 | 0 | `b4063cc1cdd08230` |

Split identity (all methods within a dataset use the same frozen sets; sha256 of sorted IDs):

| Dataset | Method | Train N | Val N | Test N | Train IDs match | Val IDs match | Test IDs match | Excluded | Extra | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| QaTa-COV19-v2 | U-Net v2 | 5716 | 1429 | 2113 | Yes | Yes | Yes | 0 | 0 | PASS |
| QaTa-COV19-v2 | TriPathLesionNet | 5716 | 1429 | 2113 | Yes | Yes | Yes | 0 | 0 | PASS |
| QaTa-COV19-v2 | Panoptic FPN | 5716 | 1429 | 2113 | Yes | Yes | Yes | 0 | 0 | PASS |
| QaTa-COV19-v2 | LViT-T | 5716 | 1429 | 2113 | Yes | Yes | Yes | 0 | 0 | PASS |
| QaTa-COV19-v2 | RecLMIS | 5716 | 1429 | 2113 | Yes | Yes | Yes | 0 | 0 | PASS |
| MosMedData+ | U-Net v2 | 1746 | 437 | 546 | Yes | Yes | Yes | 0 | 0 | PASS |
| MosMedData+ | TriPathLesionNet | 1746 | 437 | 546 | Yes | Yes | Yes | 0 | 0 | PASS |
| MosMedData+ | Panoptic FPN | 1746 | 437 | 546 | Yes | Yes | Yes | 0 | 0 | PASS |
| MosMedData+ | LViT-T | 1746 | 437 | 546 | Yes | Yes | Yes | 0 | 0 | PASS |
| MosMedData+ | RecLMIS | 1746 | 437 | 546 | Yes | Yes | Yes | 0 | 0 | PASS |
| BUSI | U-Net v2 | 413 | 104 | 130 | Yes | Yes | Yes | 0 | 0 | PASS |
| BUSI | TriPathLesionNet | 413 | 104 | 130 | Yes | Yes | Yes | 0 | 0 | PASS |
| BUSI | Panoptic FPN | 413 | 104 | 130 | Yes | Yes | Yes | 0 | 0 | PASS |

| Dataset | Train IDs hash | Val IDs hash | Test IDs hash |
|---|---|---|---|
| QaTa-COV19-v2 | 26e05c48d5cbb05d | f64ff2d35a96c819 | 3fba2d8f4ee8e206 |
| MosMedData+ | d24342ce76892eb8 | 6a3da241ea65788e | 1165120ec3db5458 |
| BUSI | e275999d1fe4a293 | 09060fa6c3d9f76b | 4b6da3717ae2dc97 |

Evidence: Phase 0 runs read the manifest through `run.py` (hash in `run_metadata.json`); Phase 1 runs read byte-identical copies (hash in `run_config.json`); per-case test IDs of all 39 runs equal the manifest order.

Source (patient/scan) leakage:

| Dataset | Grouping ID | Train-Val overlap | Train-Test overlap | Val-Test overlap | Test images from a train source | Status |
|---|---|---|---|---|---|---|
| QaTa-COV19-v2 | `sub-S…` subject (else image) | 444 | 0 | 0 | 0 | PASS for test; m-01 for val |
| MosMedData+ | `Morozov_study_*` / `Jun_*_case*` (else image) | 64 | 68 | 63 | 528 of 546 (97%) | FAIL (B-01) |
| BUSI | none (image is the unit) | 0 | 0 | 0 | 0 | NOT VERIFIED (no patient ID) |

Duplicate files across splits: 0 identical image hashes and 0 identical mask hashes between any two splits in all three datasets. Stored hashes: 1200 files spot-checked, 0 mismatches.

## 8. Phase 1 Reproduction Gates

| Method | Data Gate | Shape Gate | Overfit Gate | Gradient Gate | Inference Gate | Text-Use Gate | Official Check | Overall Status | Comments |
|---|---|---|---|---|---|---|---|---|---|
| LViT-T | PASS (32/32/32) | PASS: image 8x3x224x224, mask 8x1x224x224, tokens 8x10, embedding 8x10x768, output 8x1x224x224 | PASS (train Dice 0.9005) | PASS (`text_module1-4` nonzero) | PASS (1,429 val masks) | PASS (historical: null 0.168, shuffled 0.027 max logit change; post-hoc: Section 8.1) | PASS (official-style QaTa run, best val Dice 0.8067, epoch 40) | PASS | No author checkpoint published |
| RecLMIS | PASS (32/32/32) | PASS: image 8x3x224x224, mask 8x1x224x224, tokens 8x18, embedding 8x18x512, output 8x1x224x224 | PASS (train Dice 0.9254) | PASS (`interact.attn_text/img` nonzero) | PASS (1,429 val masks) | PASS (historical: null 0.934, shuffled 0.864; post-hoc: Section 8.1) | PASS (strict load of author QaTa checkpoint) | PASS | Frozen CLIP encoder; fusion modules trainable |

Gate evidence: `methods/phase1_language_guided/gates/<method>/gate_results.json`, `gates/<method>/official_qata/result.json`. The gates ran on 2026-09-14/15 with the then-current annotation-split manifests (m-04); they test the pipeline, not split membership.

### 8.1 Post-hoc text-use verification (this audit)

Final selected checkpoints, first 32 validation cases (no test data), same image with correct, null and shuffled reports (`evaluation/text_use_check.py`).

| Method | Dataset | Seed | Val Dice correct | Null | Shuffled | Max |dp| null | Max |dp| shuffled | Status |
|---|---|---|---|---|---|---|---|---|
| LViT-T | QaTa-COV19-v2 | 1001 | 0.7895 | 0.6954 | 0.6190 | 0.999 | 1.000 | PASS |
| LViT-T | QaTa-COV19-v2 | 1002 | 0.8075 | 0.7491 | 0.6103 | 1.000 | 1.000 | PASS |
| LViT-T | QaTa-COV19-v2 | 1003 | 0.7937 | 0.5331 | 0.5990 | 0.998 | 0.998 | PASS |
| LViT-T | MosMedData+ | 1001 | 0.8013 | 0.8026 | 0.7974 | 0.984 | 0.993 | PASS |
| LViT-T | MosMedData+ | 1002 | 0.8180 | 0.8144 | 0.8136 | 0.923 | 0.982 | PASS |
| LViT-T | MosMedData+ | 1003 | 0.8190 | 0.8211 | 0.8091 | 0.998 | 1.000 | PASS |
| RecLMIS | QaTa-COV19-v2 | 1001 | 0.8163 | 0.5934 | 0.5782 | 0.980 | 0.993 | PASS |
| RecLMIS | QaTa-COV19-v2 | 1002 | 0.7973 | 0.5671 | 0.6399 | 0.996 | 0.991 | PASS |
| RecLMIS | QaTa-COV19-v2 | 1003 | 0.8130 | 0.7111 | 0.6370 | 0.994 | 0.994 | PASS |
| RecLMIS | MosMedData+ | 1001 | 0.8148 | 0.7593 | 0.8145 | 0.992 | 0.926 | PASS |
| RecLMIS | MosMedData+ | 1002 | 0.8179 | 0.8026 | 0.8207 | 0.991 | 0.792 | PASS |
| RecLMIS | MosMedData+ | 1003 | 0.7924 | 0.7549 | 0.8111 | 0.982 | 0.915 | PASS |

## 9. Fair Comparison Audit

Information availability:

| Method | Image at train | Mask supervision | Text at train | Text at inference | External pretrained weights | Additional supervision |
|---|---|---|---|---|---|---|
| U-Net v2 | Yes | Yes | No | No | None | None |
| TriPathLesionNet | Yes | Yes | No | No | None | Boundary/location targets derived from the training masks |
| Panoptic FPN | Yes | Yes | No | No | ImageNet ResNet-50 | None |
| LViT-T | Yes | Yes | Yes | Yes (lesion count + location) | Frozen BERT-base | Reports |
| RecLMIS | Yes | Yes | Yes | Yes (lesion count + location) | CLIP ViT-B/32 | Reports |

| Check | U-Net v2 | TriPathLesionNet | Panoptic FPN | LViT-T | RecLMIS | Finding | Status |
|---|---|---|---|---|---|---|---|
| Correct canonical train split | PASS | PASS | PASS | PASS | PASS | Identical IDs (Section 7) | PASS |
| Correct canonical val split | PASS | PASS | PASS | PASS | PASS | Identical IDs | PASS |
| Correct canonical test split | PASS | PASS | PASS | PASS | PASS | Identical ordered IDs in all 39 runs | PASS |
| No patient/source leakage | FAIL | FAIL | FAIL | FAIL | FAIL | MosMed B-01 (all methods equally); QaTa val m-01 | FAIL |
| Correct number of seeds | PASS | PASS | PASS | PASS | PASS | 3 per method/dataset | PASS |
| Correct method implementation | PASS | PASS | PASS | PASS | PASS | Upstream unchanged; native settings | PASS |
| Correct information availability | PASS | PASS | PASS | PASS | PASS | Phase 1 receives test-time text (M-01) | PASS (reported) |
| Validation-only checkpoint selection | PASS | PASS | PASS | PASS | PASS | Selected epoch = best validation epoch in 39/39 | PASS |
| No test-time hyperparameter tuning | PASS | PASS | PASS | PASS | PASS | No evidence found | PASS |
| Correct preprocessing | PASS | PASS | PASS | PASS | PASS | Bilinear images; nearest masks; per-method native input (m-06) | PASS |
| Correct mask handling | PASS | PASS | PASS | PASS | PASS | Masks binarized > 0; nearest-neighbour resizing | PASS |
| Comparable evaluation resolution | PASS | PASS | PASS | PASS | PASS | Shared native-grid evaluator for all 39 runs (m-02) | PASS |
| Validation/test augmentation disabled | PASS | PASS | PASS | PASS | PASS | Enforced by code in both phases | PASS |
| Shared evaluator used | PASS | PASS | PASS | PASS | PASS | `evaluation/shared_evaluator.py` | PASS |
| All test cases included | PASS | PASS | PASS | PASS | PASS | 2,113 / 546 / 130 in every run | PASS |
| Per-case outputs preserved | PASS | PASS | PASS | PASS | PASS | 39/39 | PASS |
| Correct selected checkpoint used | PASS | PASS | PASS | PASS | PASS | Hash/epoch checks | PASS |
| No behavior-changing seed-version mixing | PASS | PASS | PASS | PASS | PASS | One config hash per Phase 0 method; one runner hash for Phase 1 | PASS |

Conclusion: the five-method comparisons are like-for-like on data, seeds, checkpoint selection and evaluation. They compare each method with its native inputs and training recipe; differences in input resolution, augmentation, pretrained encoders and test-time text are reported, not controlled. QaTa is fair to lock. MosMed is fair between methods but its absolute values are affected by B-01.

## 10. Required Artifact Audit

All 39 run folders contain resolved configuration, environment record, training log, best-checkpoint record, test summary, per-case metrics and predictions. Missing artifacts: none. One training log was rebuilt from the trainer's terminal output (m-03; `runs/phase0/unet/qata/1001/train_log_note.txt`). Checkpoints and prediction images remain on the training host and are excluded from Git by `.gitignore`.

## 11. Evaluation Pipeline Audit

| Item | Value |
|---|---|
| Shared evaluator | `evaluation/shared_evaluator.py` (all 39 runs) |
| Binary threshold | 0.5: U-Net v2 `p > 0.5`, all others `p >= 0.5` (each method's frozen rule) |
| Empty-mask policy | Both empty: Dice = mIoU = 1, HD95 = ASSD = 0; one empty: HD95 = ASSD = image diagonal |
| Evaluation resolution | Original mask grid: QaTa 224x224, MosMed 512x512, BUSI per-image size |
| Dice | (2|P∩G| + 1e-7)/(|P| + |G| + 1e-7), foreground only, per image, 0-1 scale |
| mIoU | (|P∩G| + 1e-7)/(|P∪G| + 1e-7), foreground IoU, per image (background excluded) |
| HD95 | 95th percentile of symmetric surface distances; surfaces = mask XOR 3x3 erosion; scipy EDT |
| ASSD | Mean of the same symmetric surface distances |
| HD95/ASSD units | Native-image pixels (no physical spacing applied; not millimetres) |
| Averaging | Per image; statistics resample source clusters |
| Reproduction check | Each run re-scored at stored resolution reproduces its exporter's per-case metrics (max error 5.7e-14) |

All five methods are evaluated with the same definitions. Predictions are resampled with the inverse of each pipeline's own target-resizing geometry (Phase 0 centre-aligned, Phase 1 corner-aligned; m-02). Per-case checks: 0 NaN/inf, 0 out-of-range values, mIoU = Dice/(2-Dice) to within 1e-4 in every case.

## 12. Recomputed Performance Summary

| Dataset | Phase | Method | Dice 1001 | Dice 1002 | Dice 1003 | Dice Mean | Dice SD | Dice 95% CI | mIoU Mean | HD95 Mean | ASSD Mean | Parameters | FLOPs | Text at Train | Text at Test | Source Commit | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| QaTa-COV19-v2 | 0 | U-Net v2 | 0.7878 | 0.7846 | 0.7810 | 0.7845 | 0.0034 | [0.7662, 0.8006] | 0.6866 | 31.59 | 9.45 | 31.04M | 245.98G | No | No | Summer2026Research@fdeb44037aba | PASS |
| QaTa-COV19-v2 | 0 | TriPathLesionNet | 0.7798 | 0.7788 | 0.7854 | 0.7813 | 0.0036 | [0.7633, 0.7978] | 0.6811 | 31.49 | 8.69 | 1.15M | 6.61G | No | No | Summer2026Research@fdeb44037aba | PASS |
| QaTa-COV19-v2 | 0 | Panoptic FPN | 0.8036 | 0.8014 | 0.8055 | 0.8035 | 0.0021 | [0.7865, 0.8185] | 0.7096 | 27.38 | 7.41 | 28.62M | 52.09G | No | No | Summer2026Research@fdeb44037aba | PASS |
| QaTa-COV19-v2 | 1 | LViT-T | 0.8174 | 0.8159 | 0.8194 | 0.8176 | 0.0018 | [0.8015, 0.8315] | 0.7275 | 19.61 | 6.70 | 39.93M | 54.16G | Yes | Yes | LViT@ba90775ead37 | PASS |
| QaTa-COV19-v2 | 1 | RecLMIS | 0.8358 | 0.8339 | 0.8411 | 0.8369 | 0.0038 | [0.8232, 0.8495] | 0.7503 | 16.36 | 4.95 | 220.72M (69.44M trainable) | 48.27G | Yes | Yes | RecLMIS@d3265c8b4d45 | PASS |
| MosMedData+ | 0 | U-Net v2 | 0.7927 | 0.7945 | 0.7939 | 0.7937 | 0.0009 | [0.7627, 0.8163] | 0.6830 | 29.38 | 10.43 | 31.04M | 245.98G | No | No | Summer2026Research@fdeb44037aba | PASS (B-01) |
| MosMedData+ | 0 | TriPathLesionNet | 0.8158 | 0.7996 | 0.8057 | 0.8071 | 0.0082 | [0.7747, 0.8286] | 0.6990 | 23.47 | 6.25 | 1.15M | 6.61G | No | No | Summer2026Research@fdeb44037aba | PASS (B-01) |
| MosMedData+ | 0 | Panoptic FPN | 0.8090 | 0.8048 | 0.8089 | 0.8076 | 0.0024 | [0.7748, 0.8299] | 0.7007 | 24.46 | 9.35 | 28.62M | 52.09G | No | No | Summer2026Research@fdeb44037aba | PASS (B-01) |
| MosMedData+ | 1 | LViT-T | 0.7930 | 0.7931 | 0.7924 | 0.7929 | 0.0004 | [0.7576, 0.8171] | 0.6834 | 25.92 | 11.50 | 39.93M | 54.16G | Yes | Yes | LViT@ba90775ead37 | PASS (B-01) |
| MosMedData+ | 1 | RecLMIS | 0.7856 | 0.7947 | 0.7865 | 0.7889 | 0.0050 | [0.7511, 0.8146] | 0.6799 | 30.56 | 14.39 | 220.72M (69.44M trainable) | 48.27G | Yes | Yes | RecLMIS@d3265c8b4d45 | PASS (B-01) |
| BUSI | 0 | U-Net v2 | 0.6701 | 0.7117 | 0.6995 | 0.6937 | 0.0214 | [0.6431, 0.7420] | 0.5989 | 111.50 | 53.01 | 31.04M | 245.98G | No | No | Summer2026Research@fdeb44037aba | PASS |
| BUSI | 0 | TriPathLesionNet | 0.7951 | 0.7916 | 0.7954 | 0.7940 | 0.0021 | [0.7523, 0.8311] | 0.7036 | 55.73 | 17.88 | 1.15M | 6.61G | No | No | Summer2026Research@fdeb44037aba | PASS |
| BUSI | 0 | Panoptic FPN | 0.8201 | 0.8263 | 0.8277 | 0.8247 | 0.0040 | [0.7858, 0.8591] | 0.7431 | 42.06 | 15.15 | 28.62M | 52.09G | No | No | Summer2026Research@fdeb44037aba | PASS |

Native-grid shared-evaluator values; means/SDs reproduced exactly and CIs within Monte Carlo error by the audit. Parameters/FLOPs: thop, batch of one at each method's test input (Phase 0 1x384x384; Phase 1 3x224x224 + text); FLOPs = 2 x MACs; attention products not counted (lower bound for LViT-T/RecLMIS); LViT-T excludes its frozen external BERT; RecLMIS includes frozen CLIP (trainable in parentheses). FLOPs at different input sizes are not a like-for-like efficiency comparison (m-06).

## 13. Phase 0 Statistical Verification

Source: `results/phase0/paired_comparisons.csv` (Phase 0 scoring at 384x384). Independent recomputation: exact paired differences; vectorized cluster bootstrap (10,000) and sign permutation (100,000) with a different seed; Holm recomputed from the stored raw p-values (family of 3 per dataset).

### QaTa-COV19-v2

| Method A | Method B | Paired Dice Difference | 95% CI | Raw p | Holm-adjusted p | Pairing correct? | Bootstrap correct? | Permutation correct? | Cluster handling correct? | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net v2 | TriPathLesionNet | +0.0031 | [-0.0011, +0.0074] | 0.15806 | 0.15806 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| U-Net v2 | Panoptic FPN | -0.0189 | [-0.0229, -0.0152] | 0.00001 | 0.00003 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| TriPathLesionNet | Panoptic FPN | -0.0220 | [-0.0261, -0.0182] | 0.00001 | 0.00003 | Yes | Yes | Yes | Yes (474 clusters) | PASS |

### MosMedData+

| Method A | Method B | Paired Dice Difference | 95% CI | Raw p | Holm-adjusted p | Pairing correct? | Bootstrap correct? | Permutation correct? | Cluster handling correct? | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net v2 | TriPathLesionNet | -0.0131 | [-0.0208, -0.0051] | 0.00638 | 0.01776 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| U-Net v2 | Panoptic FPN | -0.0137 | [-0.0216, -0.0051] | 0.00592 | 0.01776 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| TriPathLesionNet | Panoptic FPN | -0.0007 | [-0.0062, +0.0053] | 0.81778 | 0.81778 | Yes | Yes | Yes | Yes (86 clusters) | PASS |

### BUSI

| Method A | Method B | Paired Dice Difference | 95% CI | Raw p | Holm-adjusted p | Pairing correct? | Bootstrap correct? | Permutation correct? | Cluster handling correct? | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net v2 | TriPathLesionNet | -0.1004 | [-0.1368, -0.0659] | 0.00001 | 0.00003 | Yes | Yes | Yes | Yes (130 clusters) | PASS |
| U-Net v2 | Panoptic FPN | -0.1313 | [-0.1675, -0.0966] | 0.00001 | 0.00003 | Yes | Yes | Yes | Yes (130 clusters) | PASS |
| TriPathLesionNet | Panoptic FPN | -0.0309 | [-0.0520, -0.0118] | 0.00152 | 0.00152 | Yes | Yes | Yes | Yes (130 clusters) | PASS |

## 14. Final Five-Method Statistical Verification

Source: `results/final_five_method/paired_comparisons.csv`. Same independent recomputation; Holm family = exactly the 10 Dice comparisons within each dataset (verified). All 80 rows (Dice, mIoU, HD95, ASSD) reproduce; secondary-metric p-values are labelled exploratory in the file and report.

### 14.1 QaTa-COV19-v2

| Method A | Method B | Paired Dice Difference | 95% CI | Raw p | Holm-adjusted p | Pairing correct? | Bootstrap correct? | Permutation correct? | Cluster handling correct? | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net v2 | TriPathLesionNet | +0.0031 | [-0.0012, +0.0074] | 0.15652 | 0.15652 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| U-Net v2 | Panoptic FPN | -0.0190 | [-0.0230, -0.0152] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| U-Net v2 | LViT-T | -0.0331 | [-0.0397, -0.0268] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| U-Net v2 | RecLMIS | -0.0525 | [-0.0593, -0.0459] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| TriPathLesionNet | Panoptic FPN | -0.0222 | [-0.0262, -0.0183] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| TriPathLesionNet | LViT-T | -0.0362 | [-0.0429, -0.0297] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| TriPathLesionNet | RecLMIS | -0.0556 | [-0.0626, -0.0493] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| Panoptic FPN | LViT-T | -0.0141 | [-0.0201, -0.0081] | 0.00002 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| Panoptic FPN | RecLMIS | -0.0334 | [-0.0399, -0.0277] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |
| LViT-T | RecLMIS | -0.0194 | [-0.0233, -0.0158] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (474 clusters) | PASS |

### 14.2 MosMedData+

| Method A | Method B | Paired Dice Difference | 95% CI | Raw p | Holm-adjusted p | Pairing correct? | Bootstrap correct? | Permutation correct? | Cluster handling correct? | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| U-Net v2 | TriPathLesionNet | -0.0134 | [-0.0210, -0.0052] | 0.00576 | 0.03456 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| U-Net v2 | Panoptic FPN | -0.0139 | [-0.0217, -0.0054] | 0.00609 | 0.03456 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| U-Net v2 | LViT-T | +0.0008 | [-0.0095, +0.0130] | 0.88240 | 1.00000 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| U-Net v2 | RecLMIS | +0.0048 | [-0.0048, +0.0167] | 0.35826 | 1.00000 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| TriPathLesionNet | Panoptic FPN | -0.0005 | [-0.0060, +0.0055] | 0.85633 | 1.00000 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| TriPathLesionNet | LViT-T | +0.0142 | [+0.0067, +0.0233] | 0.00037 | 0.00259 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| TriPathLesionNet | RecLMIS | +0.0181 | [+0.0107, +0.0278] | 0.00003 | 0.00024 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| Panoptic FPN | LViT-T | +0.0147 | [+0.0088, +0.0217] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| Panoptic FPN | RecLMIS | +0.0186 | [+0.0117, +0.0274] | 0.00001 | 0.00010 | Yes | Yes | Yes | Yes (86 clusters) | PASS |
| LViT-T | RecLMIS | +0.0039 | [-0.0022, +0.0113] | 0.25567 | 1.00000 | Yes | Yes | Yes | Yes (86 clusters) | PASS |

Seed handling: each test case is averaged across its three seeds within a method before pairing; rows from different seeds are never treated as independent cases. Significance was not tested on seed-level means.

## 15. Cross-File Consistency Review

| Issue | File A | Value A | File B | Value B | Likely authoritative source | Required correction |
|---|---|---|---|---|---|---|
| Phase 0 scoring resolution | `reports/PHASE0_IMAGE_ONLY_RESULTS.md` | 384x384 grid (e.g. QaTa U-Net Dice 0.7836) | `reports/FINAL_FIVE_METHOD_COMPARISON.md` | Native grid (0.7845) | Both, for their stated scopes | None; documented in both reports |
| Phase 1 stored vs native | `methods/phase1_language_guided/runs/*/result.json` | 224x224 (MosMed LViT-T 0.7972) | `runs/phase1/.../test_summary.json` | Native (0.7929) | Native grid for the comparison | None; sensitivity table in Phase 1 report |
| Phase 1 manifest hashes | `PROVENANCE.yaml` | `b372f232…` / `9f86cf11…` (archived launch) | `run_config.json` (12 runs) | `e12e88a0…` / `5f2cbe3a…` | run_config.json | Done: `PROVENANCE_CURRENT.md` addendum (file kept; it is hashed into run configs) |
| Adapter validation counts | `ADAPTER_VALIDATION.md` (old) | MosMed 2183/273/273 | Manifest | 1746/437/546 | Manifest | Done: rewritten 2026-10-09 |
| U-Net QaTa 1001 log | `upstream/qata/history.csv` | Epochs 53-56 | `logs/unet_tmux.log` | Epochs 1-56 | Terminal log (consistent with checkpoint) | Done: rebuilt `train_log.csv` with note |
| Report/result names | Original plan | `FINAL_SIX_METHOD_COMPARISON.md`, `final_six_method/` | Repository | `…FIVE…` | Repository | Done: renamed |
| Gate status | `gates/<m>/gate_results.json` | official check pending | `gates/<m>/official_qata/result.json` | pass | official_qata/result.json | Done: `gates/STATUS.md` updated 2026-10-03 |

No conflicting Dice values, best epochs, sample counts, seeds, thresholds, checkpoint paths, source commits, confidence intervals or p-values were found between per-case files, summaries, statistics files and reports beyond the documented scoring-resolution difference. The final report's summary table matches `summary_table.csv` verbatim (13/13 rows); the hand-written Phase 0 report matches its result files (18 rows, 0 mismatches).

## 16. Leakage and Research Integrity Review

| Item | Classification | Evidence |
|---|---|---|
| Patient/source leakage | CONFIRMED (MosMed, all methods); POTENTIAL RISK (QaTa val) | Section 7; B-01, m-01 |
| Duplicate-sample leakage | NO EVIDENCE FOUND | 0 identical image or mask hashes across splits |
| Test-driven checkpoint selection | NO EVIDENCE FOUND | Phase 0 training logs contain train/val only; Phase 1 tests once after training behind `test_started.json` |
| Test-time hyperparameter tuning | NO EVIDENCE FOUND | Frozen configs; one config/runner hash per method |
| Ground-truth-derived text | NOT VERIFIED | M-01; reports encode lesion count and location |
| Missing/excluded test cases | NO EVIDENCE FOUND | All test IDs present in 39/39 runs |
| Selective seed reporting | NO EVIDENCE FOUND | All three seeds reported. Archived split runs (one completed LViT QaTa seed 1001 at Dice 0.8172, replaced for a split-protocol reason, new value 0.8174) and the OOM restart are documented |
| Behaviour-changing version mixing | NO EVIDENCE FOUND | Single config/runner hash per method |
| Wrong checkpoint use | NO EVIDENCE FOUND | Selected epoch = best validation epoch; Phase 1 hashes verified |
| Evaluation choice after test observation | POTENTIAL RISK (disclosed) | m-02 |

## 17. Experimental Trends and Findings

### 17.1 Overall performance trend

QaTa-COV19-v2 mean Dice: RecLMIS 0.8369 > LViT-T 0.8176 > Panoptic FPN 0.8035 > U-Net v2 0.7845 > TriPathLesionNet 0.7813. MosMedData+: Panoptic FPN 0.8076 ≈ TriPathLesionNet 0.8071 > U-Net v2 0.7937 ≈ LViT-T 0.7929 > RecLMIS 0.7889. BUSI (Phase 0 only): Panoptic FPN 0.8247 > TriPathLesionNet 0.7940 > U-Net v2 0.6937.

### 17.2 Image-only vs language-guided trend

Dataset-dependent. On QaTa both language-guided methods exceed all three image-only methods (all six cross-family differences Holm-significant). On MosMed neither does: TriPathLesionNet and Panoptic FPN are significantly higher than both, and U-Net v2 is not distinguishable from either. This does not show that language causes the QaTa gain: the families also differ in architecture, pretrained encoders, input resolution (224 vs 384), augmentation, and the test-time reports state lesion count and location (M-01). The post-hoc text check shows strong text dependence on QaTa (shuffled reports lower validation Dice from about 0.79-0.82 to about 0.58-0.64) and weak dependence on MosMed.

### 17.3 QaTa vs MosMed behaviour

The ranking reverses between datasets for the language-guided methods (top two on QaTa, bottom two on MosMed). Panoptic FPN is the strongest image-only method on all three datasets. MosMed values carry B-01, and Phase 1 methods pay a larger resolution cost there (224 network input vs 512 masks; stored-resolution Dice is 0.004 higher).

### 17.4 Seed stability

Three-seed SD is at most 0.005 for every method on QaTa and MosMed except TriPathLesionNet on MosMed (SD 0.0082, range 0.016). The least stable cell is U-Net v2 on BUSI (SD 0.0214, range 0.042). No reported conclusion depends on one seed: every Holm-significant QaTa/MosMed difference exceeds the largest seed range of the two methods involved, except two MosMed pairs involving TriPathLesionNet (seed range 0.016): vs U-Net v2 (difference 0.0134) and vs LViT-T (0.0142). These should be described cautiously.

### 17.5 Dice/mIoU vs HD95/ASSD

On QaTa the boundary metrics broadly follow the Dice ranking (HD95: RecLMIS 16.4 px, LViT-T 19.6, Panoptic FPN 27.4, U-Net v2 31.6, TriPathLesionNet 31.5; the two lowest-Dice methods are effectively tied on HD95). On MosMed they diverge: TriPathLesionNet has the best ASSD (6.25 px) and HD95 (23.5) while Panoptic FPN has similar Dice but ASSD 9.35; RecLMIS has the worst HD95/ASSD (30.6/14.4). On BUSI, U-Net v2's low Dice comes with very large boundary errors (HD95 111.5 px). Secondary differences are exploratory.

### 17.6 Case-level paired behaviour

Gains are broad but not uniform. On QaTa, RecLMIS beats U-Net v2 in 68% of cases; the mean difference (+0.0525) is larger than the median (+0.0178) and the 5% trimmed mean (+0.0446), so part of the gain comes from cases where the image-only model fails badly. On MosMed, Panoptic FPN beats RecLMIS in 68% of cases (median +0.0109). Full table: `results/final_five_method/case_level_paired_summary.csv`.

### 17.7 Statistically supported findings

- QaTa: 9 of 10 pairs are Holm-significant (paired 95% CI excludes zero in the same 9). Not detectable: U-Net v2 vs TriPathLesionNet.
- MosMed: 6 of 10 pairs are Holm-significant: TriPathLesionNet and Panoptic FPN each exceed U-Net v2, LViT-T and RecLMIS.

### 17.8 Descriptive findings without corrected statistical support

- MosMed: TriPathLesionNet vs Panoptic FPN, LViT-T vs RecLMIS and U-Net v2 vs either Phase 1 method are not detectably different.
- All secondary-metric differences and the text-use and case-level analyses are descriptive or exploratory.
- BUSI has only the three Phase 0 comparisons (Section 13).

## 18. Problems Requiring Action

### 18.1 BLOCKERS

```text
Problem ID: B-01
Severity: BLOCKER (pending PI decision)
Affected phase: 0 and 1
Affected method: All five
Affected dataset: MosMedData+
Affected seed(s): All
```

- Evidence: `results/audit/lock_audit_checks.json` `datasets.mosmed`: 68/86 test studies in train, 63 in val; 528/546 test images from a training study. Same property in the archived annotation split (67%).
- What is wrong: The frozen MosMed split is slice-level, so most test patients contribute training slices.
- Why it matters: Absolute MosMed scores measure generalization to new slices of seen patients, not new patients; reviewers will question them. Between-method comparison remains like-for-like.
- Existing checkpoint usable: Yes (if accepted); No (if re-split)
- Retraining required: No if accepted; Yes (15 runs) if re-split
- Re-inference required: No / Yes
- Metric recomputation required: No / Yes
- Statistical recomputation required: No / Yes (MosMed rows)
- Exact corrective action: 1. PI decides: accept as a documented limitation, or require a study-level re-split. 2a. If accepted: record the decision in `MAIN_EXPERIMENT_STATUS.md` and this report (caveat text is already in both result reports). 2b. If re-split: build a study-grouped MosMed manifest (seeded, all slices of a `source_id` in one split), retrain the 5 methods x 3 seeds on it, rerun the shared evaluator, statistics and reports, and rerun this audit.
- Expected output after correction: Signed PI decision; or new MosMed manifest with 0 train-test source overlap and 15 new runs.
- Pass condition: Decision recorded; or `source_overlap.train-test == 0` and all MosMed checks PASS.

### 18.2 MAJOR ISSUES

```text
Problem ID: M-01
Severity: MAJOR
Affected phase: 1
Affected method: LViT-T, RecLMIS
Affected dataset: QaTa, MosMed
Affected seed(s): All
```

- Evidence: Reports such as "Bilateral pulmonary infection, two infected areas, all left lung and middle lower right lung." are given at test time; 394 (QaTa) and 115 (MosMed) unique reports; workbooks pinned by hash; no generation code in the repository; post-hoc text check shows strong QaTa text dependence.
- What is wrong: The origin of the reports relative to the masks cannot be verified, and they carry lesion count and location that image-only methods do not receive.
- Why it matters: Language-guided vs image-only differences cannot be attributed to language understanding, and could partly reflect label-like information at test time.
- Existing checkpoint usable: Yes
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: 1. PI reviews the caveat (already in both result reports). 2. Cite the dataset authors' description of how the QaTa/MosMed reports were produced, or state that it is unknown. 3. Phrase conclusions as method performance with native inputs, not as an effect of language.
- Expected output after correction: Caveat accepted and wording agreed.
- Pass condition: PI sign-off on the interpretation wording.

### 18.3 MINOR ISSUES

```text
Problem ID: m-01
Severity: MINOR
Affected phase: 0 and 1
Affected method: All
Affected dataset: QaTa
Affected seed(s): All
```

- Evidence: 444 subjects in both train and val; 0 in test.
- What is wrong: Validation is not subject-disjoint.
- Why it matters: Checkpoint selection may be slightly optimistic; test results are unaffected.
- Existing checkpoint usable: Yes
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: Keep the documented caveat.
- Expected output after correction: Caveat in reports (done).
- Pass condition: Caveat present.

```text
Problem ID: m-02
Severity: MINOR (disclosed)
Affected phase: 1
Affected method: LViT-T, RecLMIS
Affected dataset: MosMed (QaTa unaffected)
Affected seed(s): All
```

- Evidence: Shared evaluator header; validation-mask round-trip ceilings 0.933 vs 0.885.
- What is wrong: The corner-aligned resampling rule was investigated after a test-set drop was observed.
- Why it matters: Any post-hoc evaluation choice is a potential researcher-degree-of-freedom.
- Existing checkpoint usable: Yes
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: PI review; both stored-resolution and native-grid values remain reported.
- Expected output after correction: PI acknowledgement.
- Pass condition: Disclosed in reports and audit (done).

```text
Problem ID: m-03
Severity: MINOR
Affected phase: 0
Affected method: U-Net v2
Affected dataset: QaTa
Affected seed(s): 1001
```

- Evidence: `history.csv` epochs 53-56 only; `logs/unet_tmux.log` epochs 1-56; best epoch 41.
- What is wrong: A relaunch rewrote the run's history file.
- Why it matters: Training log completeness.
- Existing checkpoint usable: Yes
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: Done: `runs/phase0/unet/qata/1001/train_log.csv` rebuilt from the terminal log with a note.
- Expected output after correction: Complete 56-epoch log.
- Pass condition: Selected epoch equals best validation epoch (verified).

```text
Problem ID: m-04
Severity: MINOR
Affected phase: 1
Affected method: LViT-T, RecLMIS
Affected dataset: QaTa, MosMed
Affected seed(s): Gates
```

- Evidence: Gate results dated 2026-09-14/15, before the switch to Phase 0 manifests.
- What is wrong: Gates used the archived annotation split.
- Why it matters: Gates verify the pipeline, not membership.
- Existing checkpoint usable: Yes
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: Done: smoke checks on the final manifests and post-hoc text-use check on final checkpoints.
- Expected output after correction: -
- Pass condition: Section 8.1 PASS.

```text
Problem ID: m-05
Severity: MINOR
Affected phase: 0
Affected method: All Phase 0
Affected dataset: -
Affected seed(s): -
```

- Evidence: No LICENSE in `upstream/Summer2026Research`.
- What is wrong: License undocumented.
- Why it matters: Required before public release.
- Existing checkpoint usable: Yes
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: Add a license to the Summer2026Research repository and update the three SOURCE.md files.
- Expected output after correction: LICENSE file.
- Pass condition: License recorded.

```text
Problem ID: m-06
Severity: MINOR
Affected phase: 0 and 1
Affected method: All
Affected dataset: All
Affected seed(s): -
```

- Evidence: Phase 0: 384x384 grayscale with flip/rotation augmentation; Phase 1: 224x224 RGB, augmentation disabled (text safety); thop FLOPs at different inputs, attention uncounted.
- What is wrong: Native recipes differ by design.
- Why it matters: Limits causal and efficiency interpretations.
- Existing checkpoint usable: Yes
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: Keep as reported limitations.
- Expected output after correction: -
- Pass condition: Documented (done).

```text
Problem ID: m-07
Severity: MINOR
Affected phase: 0 and 1
Affected method: -
Affected dataset: -
Affected seed(s): -
```

- Evidence: `git status`: audit fixes uncommitted; branch `phase1-main-experiment` not yet pushed.
- What is wrong: Locked state not yet in version control.
- Why it matters: Lock must reference a commit.
- Existing checkpoint usable: -
- Retraining required: No
- Re-inference required: No
- Metric recomputation required: No
- Statistical recomputation required: No
- Exact corrective action: Commit, push (VS Code Publish Branch), merge to master, tag the lock commit.
- Expected output after correction: Tagged commit on GitHub.
- Pass condition: `git status` clean; tag exists on origin.

## 19. Corrective Action Plan and Dependency Order

Fixes already applied during this audit (no retraining, no change to any trained checkpoint or to previously reported QaTa/MosMed values): shared evaluator extended to BUSI with byte-identical QaTa/MosMed outputs; standard run folders for all 39 runs; Phase 0 SOURCE.md files; adapter validation refreshed; provenance addendum; result/report renamed to five-method; summary table with per-seed Dice, BUSI rows and status; case-level and seed-stability outputs; interpretation caveats; post-hoc text-use check; U-Net QaTa 1001 log recovery.

Remaining, in dependency order:

1. **FIX-01 (B-01) PI decision on MosMed.** Input: Section 7 and 18.1. If re-split is required, steps 2-5 follow it: new grouped manifest -> 15 MosMed runs -> `shared_evaluator.py` -> `export_run_records.py` -> `phase1_final_statistics.py` -> `lock_audit_checks.py` -> `write_lock_audit_report.py`. Pass: decision recorded, or MosMed train-test source overlap 0 with all checks PASS.
2. **FIX-02 (M-01) PI sign-off on the report-text caveat and wording.** Pass: wording agreed.
3. **FIX-03 (m-05) Add a license to Summer2026Research.** Pass: LICENSE present, SOURCE.md updated.
4. **FIX-04 (m-07) Commit, push, merge and tag the lock.** Pass: clean working tree, tag on origin.
5. Regenerate this report and set the decision line accordingly.
