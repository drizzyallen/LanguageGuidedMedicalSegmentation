# Reproduction gate status

| Method | Data | Shape | Overfit | Gradient | Inference | Text use | Official check |
|---|---|---|---|---|---|---|---|
| LViT-T | PASS | PASS | PASS (0.9026) | PASS | PASS (1,429) | PASS | PASS — official-style QaTa run, best val Dice 0.8067 (epoch 40), 2026-09-15 |
| RecLMIS | PASS | PASS | PASS (0.9296) | PASS | PASS (1,429) | PASS | PASS — strict author checkpoint load |
| ProLearn | PASS | PASS | PASS (0.9168) | PASS | PASS (1,429) | N/A: text-free inference | PASS — official-style QaTa run, best val Dice 0.7823; method skipped per advisor (2026-10-03) |

All gates pass for LViT-T and RecLMIS. The official-check outcomes are in
`gates/<method>/official_qata/result.json`; `gate_results.json` predates them.
