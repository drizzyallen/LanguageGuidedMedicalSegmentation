# Section 1.4 main three-seed experiment — status

## 2026-10-03 relaunch on the frozen Phase 0 splits

**Scope.** Per the research advisor, ProLearn is skipped and will not be run.
Phase 1 is LViT-T and RecLMIS only, so the final comparison has 5 methods
(10 pairwise Dice tests, Holm-corrected across 10 per dataset).

**Splits.** Plan §1.4 rule 2 requires the same frozen train/val/test samples
as Phase 0. The Phase 1 manifests are now byte-identical copies of the Phase 0
manifests (`build_authoritative_manifests.py`); every image has a paired
report (100% paired).

| Dataset | train / val / test | sha256 |
|---|---|---|
| QaTa | 5716 / 1429 / 2113 | e12e88a0b570ad143cd6517f35e8d20de0e65e53cbeb46c2c1d4928e3a614a4a |
| MosMed | 1746 / 437 / 546 | 5f2cbe3a7e1cf91cb5dec1697c1d82c65bd8c46860186dfa67f584c5c6d2973e |

The earlier annotation-workbook split (MosMed 2183/273/273; QaTa with 1,128
train/val swaps) and its runs (LViT seed 1001 complete on both datasets;
RecLMIS seed 1001 failed on both) are archived in
`archive/xlsx_split_20260916/` and are not used for any result. The 1.3 gates
test pipeline correctness, not split membership; fresh smoke checks on the new
manifests replace the data check.

**Runner changes (all 12 runs use the same runner hash).**
1. Nonfinite training loss: training stops at the event, the event is written
   to `nonfinite_stop.json`, and the pre-event best validation checkpoint is
   tested; `result.json` records `training_stop`. Upstream loops have no
   finiteness check, so a NaN loss makes every parameter NaN and validation
   Dice can never exceed the earlier best; early stopping then tests that same
   checkpoint. (Archived RecLMIS QaTa seed 1001 hit this at epoch 72; weights
   at epoch 71 were finite and `clip.logit_scale` was at its fixed value of 100.)
2. Explicit `os._exit(0)` after the run: the LViT queues previously never left
   seed 1001 after writing `result.json` (likely mxnet 1.4 teardown stall).
3. Resume no longer refuses solely because the physical GPU index changed.

**Queues** (detached sessions; survive SSH/VPN logout). Updated 2026-10-06 13:50.

| Method | QaTa | MosMed |
|---|---|---|
| LViT-T | 1001 done; 1002 running on GPU 3; 1003 running on GPU 0 (launched 2026-10-06 13:56) | GPU 3; 1001, 1002 done, 1003 running |
| RecLMIS | 1001, 1002, 1003 done | 1001, 1002, 1003 done |

LViT QaTa seed 1003 was started early as a standalone process on GPU 0
(`lvit_qata_seed1003.log`, `.pid`) to run in parallel with seed 1002. Same
runner, manifests and seed; only the physical GPU differs. When the GPU 3 queue
reaches seed 1003 it either finds `result.json` ("Already complete") or fails to
acquire the run lock and exits; both are expected and cannot cause a second run
or a second test evaluation.

RecLMIS QaTa seed 1003 hit a NaN training loss at epoch 97, step 98
(`nonfinite_stop.json`). Its best validation checkpoint was epoch 17 (val Dice
0.8194); 80 epochs without improvement is under the native patience of 100, so
the upstream loop would also have tested epoch 17. Under the stop rule above,
epoch 17 was tested once.

All completed exports were verified: test order equals the manifest, one
probability map and one mask per case, no nonfinite metrics, and the
per-case mean equals `result.json`.

RecLMIS QaTa seed 1002 first failed with CUDA OOM at startup (2026-10-05
03:54): the waiter placed RecLMIS MosMed on GPU 2 in the gap between QaTa
seeds 1001 and 1002. No training had occurred (only `run_config.json`
existed); the queue was relaunched unchanged on GPU 0.

**Hardware.** 4 of 8 GPUs fell off the PCIe bus ~2026-09-29 (Xid 79) and need
an admin. GPUs 0–2 are fully used by another user; GPU 3 is shared.
RecLMIS uses its native batch size (32 QaTa / 20 MosMed) and does not fit
beside the other jobs on GPU 3.

**Monitoring.** `*_queue.log`, `reclmis_waiter.log`,
`runs/<method>/<dataset>/seed_<seed>/progress.json`.
