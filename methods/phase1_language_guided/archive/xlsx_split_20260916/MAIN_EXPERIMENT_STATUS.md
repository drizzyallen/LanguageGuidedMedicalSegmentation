# Section 1.4 recovery — 2026-09-16

## Revised dataset and scope — 2026-10-02

The restored annotation-defined Phase 1 manifests are the common dataset for
the planned three-method comparison. They preserve Phase 0 image/mask records
and attach the full available reports. Counts are QaTa: 5,716/1,429/2,113 and
MosMed: 2,183/273/273 (train/validation/test). Their hashes match those
recorded in the previous run configs. LViT-T and RecLMIS are supported by the
current main runner. ProLearn remains held: its full-training-report prototype
bank is unresolved and it has not been added to the main runner. Do not launch
or claim a completed three-method comparison until that blocker is resolved.

Recovered the local saved chat “Assess three-seed experiment” (01a0aadb-3cbe-7943-a2a0-32cdd40dbff9). User authorized safe launch; Phase 1 XLSX governs the Phase 0 image/mask pool.

| Method | QaTa GPU / queue PID | MosMed GPU / queue PID | Seeds | State |
|---|---|---|---|---|
| LViT | 0 / 1229537 | 3 / 1229538 | 1001, 1002, 1003 | Detached queues launched; seed 1001 workers verified alive |
| RecLMIS | 1 / 1231593 | 4 / 1231594 | 1001, 1002, 1003 | Detached queues launched; seed 1001 workers verified alive |
| ProLearn | — | — | 1001, 1002, 1003 | HELD: full-training-report PSA initialization unresolved |

GPU 2 belongs to another job and is excluded. Queues are independent of browser, SSH and VPN lifetimes; they stop on errors. Files are under runs/<method>/<dataset>/seed_<seed>; queue logs and PIDs are in this directory. Reinvoking a queue is protected by flock. Training resumes from saved epoch checkpoints; a test_started marker prevents silent repeat test evaluation. Select by validation mean per-case Dice. Export float32 probabilities, PNG masks and Dice/mIoU/HD95/ASSD per case. LViT and RecLMIS require text at inference; ProLearn does not.

Both LViT smoke checks passed in the interrupted chat. Both RecLMIS smoke checks passed after recovery, including backward, optimizer step, strict checkpoint reload and exports from two validation cases (no held-out test cases). RecLMIS now asserts all three native auxiliary losses remain active. Missing fairseq previously triggered an upstream bare except and silently disabled these losses. Installation attempts for fairseq 0.10.2 and 0.12.2 failed to build. reclmis_compat.py supplies only the required float32 softmax API, matching fairseq v0.10.2, and is hashed in run provenance. Upstream source is unchanged.

ProLearn PSA.fit requires per-case six-position pseudolabels, missing locally. The paper describes a separately trained attention-filtering model and HDBSCAN surrogate-label extraction; these stages are not implemented in the local source. The official repository has an open unanswered issue about this input: https://github.com/ShuchangYe-bib/ProLearn/issues/1 . Supplied QaTa prototype banks do not establish contribution from 100% of our training reports, and cannot validate MosMed full-report fitting. Next work: obtain or reconstruct and validate the missing label-extraction procedure, then fit a fresh training-only bank for each dataset/seed and add ProLearn to the main runner. Do not claim the full 18-run launch is complete.

The old chat lock file exists. Its presence alone does not prove another browser tab is open; the app-level lock has not been modified.
