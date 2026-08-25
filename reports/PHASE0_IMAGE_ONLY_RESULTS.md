# Phase 0 Image-Only Results

## Completion

Phase 0 is complete for U-Net v2, TriPathLesionNet, and Panoptic FPN on
QaTa-COV19-v2, MosMedData+, and BUSI. All 27 method-dataset-seed runs (seeds
1001, 1002, and 1003) have best checkpoints, ordered per-case metrics, float32
probability maps, binary masks, and test summaries. Recalculated per-case means
match the summaries. The frozen upstream commit is
`fdeb44037aba52900d852271293b165a697ecf06`.

## Procedure

Results use the three-seed mean and sample SD. Absolute 95% CIs use 10,000
cluster-bootstrap resamples after averaging each case across seeds. Paired Dice
tests use case-level, seed-averaged differences, 10,000 paired bootstrap
resamples, and 100,000 two-sided sign permutations. P-values are Holm-adjusted
across the three comparisons within each dataset.

QaTa has 2,113 cases in 474 analysis clusters; MosMed has 546 cases in 86
clusters. BUSI lacks defensible patient IDs, so its 130 images are the analysis
units. Supplied QaTa and MosMed source clusters cross some frozen splits; see
`methods/phase0_image_only/DATA_LIMITATIONS.md`.

## Dice results

Values are mean ± sample SD, followed by absolute bootstrap 95% CI.

| Dataset | Method | Dice | 95% CI |
|---|---|---:|---:|
| QaTa | U-Net v2 | 0.7836 ± 0.0034 | [0.7653, 0.7997] |
| QaTa | TriPathLesionNet | 0.7805 ± 0.0035 | [0.7626, 0.7969] |
| QaTa | Panoptic FPN | 0.8025 ± 0.0021 | [0.7858, 0.8174] |
| MosMed | U-Net v2 | 0.7932 ± 0.0012 | [0.7621, 0.8158] |
| MosMed | TriPathLesionNet | 0.8063 ± 0.0086 | [0.7746, 0.8281] |
| MosMed | Panoptic FPN | 0.8070 ± 0.0028 | [0.7726, 0.8299] |
| BUSI | U-Net v2 | 0.6934 ± 0.0214 | [0.6428, 0.7417] |
| BUSI | TriPathLesionNet | 0.7938 ± 0.0022 | [0.7525, 0.8312] |
| BUSI | Panoptic FPN | 0.8247 ± 0.0040 | [0.7864, 0.8592] |

Complete mean, sample SD, point estimate, and absolute 95% CI values for Dice,
mIoU, HD95, and ASSD are reported in `results/phase0/absolute_ci.csv`.

## Paired Dice comparisons

Differences are method A minus method B.

| Dataset | A vs B | Difference | 95% CI | Raw p | Holm p |
|---|---|---:|---:|---:|---:|
| QaTa | U-Net v2 vs TriPath | 0.0031 | [-0.0011, 0.0074] | 0.15806 | 0.15806 |
| QaTa | U-Net v2 vs Panoptic | -0.0189 | [-0.0229, -0.0152] | 0.00001 | 0.00003 |
| QaTa | TriPath vs Panoptic | -0.0220 | [-0.0261, -0.0182] | 0.00001 | 0.00003 |
| MosMed | U-Net v2 vs TriPath | -0.0131 | [-0.0208, -0.0051] | 0.00638 | 0.01776 |
| MosMed | U-Net v2 vs Panoptic | -0.0137 | [-0.0216, -0.0051] | 0.00592 | 0.01776 |
| MosMed | TriPath vs Panoptic | -0.0007 | [-0.0062, 0.0053] | 0.81778 | 0.81778 |
| BUSI | U-Net v2 vs TriPath | -0.1004 | [-0.1368, -0.0659] | 0.00001 | 0.00003 |
| BUSI | U-Net v2 vs Panoptic | -0.1313 | [-0.1675, -0.0966] | 0.00001 | 0.00003 |
| BUSI | TriPath vs Panoptic | -0.0309 | [-0.0520, -0.0118] | 0.00152 | 0.00152 |

Panoptic FPN has the highest mean Dice on all datasets. It outperforms both
alternatives on QaTa and BUSI after Holm adjustment. On MosMed, Panoptic FPN
and TriPathLesionNet are not detectably different, while both outperform U-Net
v2. U-Net v2 and TriPathLesionNet are not detectably different on QaTa.

Machine-readable paired results are in `results/phase0/paired_comparisons.csv`
and `results/phase0/paired_comparisons.json`.
