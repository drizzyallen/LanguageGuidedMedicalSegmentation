# Reproduction gate status

| Method | Data | Shape | Overfit | Gradient | Inference | Text use | Official check |
|---|---|---|---|---|---|---|---|
| LViT-T | PASS | PASS | PASS (0.9026) | PASS | PASS (1,429) | PASS | PENDING official-style QaTa run |
| RecLMIS | PASS | PASS | PASS (0.9296) | PASS | PASS (1,429) | PASS | PASS, strict author checkpoint load |
| ProLearn | PASS | PASS | PASS (0.9168) | PASS | PASS (1,429) | N/A: text-free inference | PENDING official-style QaTa run |

No three-seed run is authorized until both pending official checks pass.

