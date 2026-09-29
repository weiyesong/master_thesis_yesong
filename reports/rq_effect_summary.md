# Paired RQ2 and RQ3 Effect Summary

Generated: 2026-08-27T23:32:30.407256+00:00

Inputs are limited to `reports/thesis_master_results.csv` and the validated prediction/manifests referenced by that master table. Prediction artifacts were used to confirm pairing compatibility; the reported effects are arithmetic differences between already-validated metrics. No model, training loop, or inference pipeline was executed.

All deltas are absolute differences on the recorded 0–1 metric scale: comparison minus baseline. Positive Accuracy, Macro-F1, or mIoU deltas favor the comparison; negative NLL, Brier, or ECE-15 deltas favor the comparison. These signs are descriptive and are not significance tests.

## RQ2 — Full fine-tuning versus frozen adaptation

RQ2 contains 24 independently trained matched-seed contrasts and eight paired three-seed summaries. Aggregate effect standard deviations are sample standard deviations of the three seedwise deltas—not differences or combinations of the two arm standard deviations.

| Task | Dataset | Model | Seeds | Δ Accuracy | Δ Macro-F1 | Δ mIoU | Δ NLL | Δ Brier | Δ ECE-15 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| classification | EuroSAT | DOFA | 42;43;44 | -0.0185 ± 0.0058 | -0.0186 ± 0.0058 | — | +0.0526 ± 0.0239 | +0.0273 ± 0.0093 | +0.0061 ± 0.0061 |
| classification | EuroSAT | PANOPTICON | 42;43;44 | -0.0206 ± 0.0071 | -0.0219 ± 0.0079 | — | +0.0622 ± 0.0344 | +0.0301 ± 0.0129 | +0.0028 ± 0.0043 |
| classification | TreeSatAI | DOFA | 42;43;44 | +0.0102 ± 0.0028 | +0.0207 ± 0.0154 | — | -0.0090 ± 0.0023 | -0.0023 ± 0.0008 | +0.0017 ± 0.0038 |
| classification | TreeSatAI | PANOPTICON | 42;43;44 | +0.0137 ± 0.0289 | +0.0089 ± 0.0261 | — | -0.0065 ± 0.0031 | -0.0022 ± 0.0012 | -0.0019 ± 0.0027 |
| segmentation | CloudSEN12 | DOFA | 42;43;44 | — | — | +0.0524 ± 0.0064 | +0.0064 ± 0.0263 | -0.0297 ± 0.0021 | +0.0237 ± 0.0219 |
| segmentation | CloudSEN12 | PANOPTICON | 42;43;44 | — | — | +0.0181 ± 0.0042 | -0.0256 ± 0.0232 | -0.0147 ± 0.0047 | +0.0005 ± 0.0081 |
| segmentation | SpaceNet7 | DOFA | 42;43;44 | — | — | +0.0095 ± 0.0008 | +0.1577 ± 0.1040 | +0.0033 ± 0.0052 | +0.0105 ± 0.0093 |
| segmentation | SpaceNet7 | PANOPTICON | 42;43;44 | — | — | +0.0030 ± 0.0054 | +0.0770 ± 0.0849 | +0.0049 ± 0.0039 | +0.0041 ± 0.0105 |

## RQ3 — UQ method versus corresponding deterministic baseline

The CSV contains 36 seed-level comparisons, eight paired three-seed summaries, 16 explicitly aggregate Ensemble-versus-member-mean comparisons, and 12 N/A declarations. The compact table below uses a three-seed paired summary where one was predeclared; otherwise MC Dropout is the primary seed42 pair. Deep Ensembles remain single aggregate results and are never pooled with member seeds.

| Task | Dataset | Model | Adaptation | UQ method | Comparison basis | Δ Accuracy | Δ Macro-F1 | Δ mIoU | Δ NLL | Δ Brier | Δ ECE-15 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| classification | EuroSAT | DOFA | frozen | temperature_scaling | paired seeds 42/43/44, mean ± sample std | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | — | +0.0000 ± 0.0000 | -0.0000 ± 0.0000 | -0.0000 ± 0.0004 |
| classification | EuroSAT | DOFA | full_finetune | temperature_scaling | paired seeds 42/43/44, mean ± sample std | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | — | -0.0028 ± 0.0050 | -0.0006 ± 0.0012 | -0.0038 ± 0.0048 |
| classification | EuroSAT | PANOPTICON | frozen | temperature_scaling | paired seeds 42/43/44, mean ± sample std | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | — | -0.0010 ± 0.0008 | -0.0002 ± 0.0002 | -0.0010 ± 0.0010 |
| classification | EuroSAT | PANOPTICON | full_finetune | temperature_scaling | paired seeds 42/43/44, mean ± sample std | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | — | -0.0010 ± 0.0045 | -0.0001 ± 0.0005 | -0.0013 ± 0.0059 |
| classification | EuroSAT | DOFA | frozen | mc_dropout | paired seeds 42/43/44, mean ± sample std | +0.0001 ± 0.0018 | +0.0002 ± 0.0019 | — | +0.0001 ± 0.0078 | +0.0005 ± 0.0031 | +0.0031 ± 0.0021 |
| classification | EuroSAT | DOFA | full_finetune | mc_dropout | paired seed 42 | -0.0122 | -0.0132 | — | +0.0420 | +0.0196 | +0.0030 |
| classification | EuroSAT | PANOPTICON | frozen | mc_dropout | paired seed 42 | -0.0007 | -0.0009 | — | -0.0028 | -0.0011 | -0.0030 |
| classification | EuroSAT | PANOPTICON | full_finetune | mc_dropout | paired seed 42 | -0.0099 | -0.0119 | — | +0.0235 | +0.0160 | +0.0011 |
| classification | TreeSatAI | DOFA | frozen | mc_dropout | paired seed 42 | +0.0000 | -0.0033 | — | -0.0004 | +0.0002 | -0.0011 |
| classification | TreeSatAI | DOFA | full_finetune | mc_dropout | paired seed 42 | +0.0175 | -0.0089 | — | -0.0020 | -0.0011 | +0.0014 |
| classification | TreeSatAI | PANOPTICON | frozen | mc_dropout | paired seed 42 | -0.0015 | -0.0020 | — | +0.0002 | +0.0001 | -0.0004 |
| classification | TreeSatAI | PANOPTICON | full_finetune | mc_dropout | paired seeds 42/43/44, mean ± sample std | +0.0165 ± 0.0223 | +0.0064 ± 0.0076 | — | -0.0030 ± 0.0041 | -0.0009 ± 0.0013 | -0.0000 ± 0.0010 |
| segmentation | CloudSEN12 | DOFA | frozen | mc_dropout | paired seed 42 | — | — | +0.0001 | +0.0023 | +0.0017 | +0.0138 |
| segmentation | CloudSEN12 | DOFA | full_finetune | mc_dropout | paired seed 42 | — | — | +0.0026 | -0.0390 | -0.0059 | -0.0126 |
| segmentation | CloudSEN12 | PANOPTICON | frozen | mc_dropout | paired seeds 42/43/44, mean ± sample std | — | — | -0.0054 ± 0.0050 | -0.0161 ± 0.0220 | -0.0019 ± 0.0013 | -0.0110 ± 0.0183 |
| segmentation | CloudSEN12 | PANOPTICON | full_finetune | mc_dropout | paired seed 42 | — | — | -0.0210 | +0.0006 | +0.0103 | -0.0075 |
| segmentation | SpaceNet7 | DOFA | frozen | mc_dropout | paired seed 42 | — | — | -0.0025 | -0.0047 | -0.0004 | +0.0009 |
| segmentation | SpaceNet7 | DOFA | full_finetune | mc_dropout | paired seeds 42/43/44, mean ± sample std | — | — | -0.0078 ± 0.0023 | -0.1845 ± 0.1381 | -0.0123 ± 0.0072 | -0.0153 ± 0.0176 |
| segmentation | SpaceNet7 | PANOPTICON | frozen | mc_dropout | paired seed 42 | — | — | +0.0005 | -0.0372 | -0.0047 | -0.0023 |
| segmentation | SpaceNet7 | PANOPTICON | full_finetune | mc_dropout | paired seed 42 | — | — | -0.0077 | -0.0655 | -0.0134 | -0.0087 |
| classification | EuroSAT | DOFA | frozen | deep_ensemble | M=3 ensemble vs exact member mean | +0.0004 | +0.0004 | — | -0.0024 | -0.0013 | +0.0005 |
| classification | EuroSAT | DOFA | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | +0.0178 | +0.0185 | — | -0.0430 | -0.0228 | +0.0162 |
| classification | EuroSAT | PANOPTICON | frozen | deep_ensemble | M=3 ensemble vs exact member mean | -0.0006 | -0.0007 | — | -0.0027 | -0.0007 | +0.0002 |
| classification | EuroSAT | PANOPTICON | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | +0.0138 | +0.0145 | — | -0.0400 | -0.0194 | +0.0125 |
| classification | TreeSatAI | DOFA | frozen | deep_ensemble | M=3 ensemble vs exact member mean | +0.0038 | +0.0009 | — | -0.0021 | -0.0005 | -0.0004 |
| classification | TreeSatAI | DOFA | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | +0.0072 | -0.0062 | — | -0.0093 | -0.0030 | +0.0045 |
| classification | TreeSatAI | PANOPTICON | frozen | deep_ensemble | M=3 ensemble vs exact member mean | +0.0012 | -0.0034 | — | -0.0023 | -0.0006 | -0.0007 |
| classification | TreeSatAI | PANOPTICON | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | +0.0140 | +0.0045 | — | -0.0088 | -0.0028 | +0.0021 |
| segmentation | CloudSEN12 | DOFA | frozen | deep_ensemble | M=3 ensemble vs exact member mean | — | — | +0.0163 | -0.0486 | -0.0181 | -0.0231 |
| segmentation | CloudSEN12 | DOFA | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | — | — | +0.0146 | -0.0814 | -0.0202 | -0.0235 |
| segmentation | CloudSEN12 | PANOPTICON | frozen | deep_ensemble | M=3 ensemble vs exact member mean | — | — | +0.0129 | -0.0479 | -0.0134 | -0.0166 |
| segmentation | CloudSEN12 | PANOPTICON | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | — | — | +0.0178 | -0.0526 | -0.0167 | -0.0219 |
| segmentation | SpaceNet7 | DOFA | frozen | deep_ensemble | M=3 ensemble vs exact member mean | — | — | -0.0065 | -0.0713 | -0.0092 | -0.0087 |
| segmentation | SpaceNet7 | DOFA | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | — | — | -0.0110 | -0.1489 | -0.0121 | -0.0097 |
| segmentation | SpaceNet7 | PANOPTICON | frozen | deep_ensemble | M=3 ensemble vs exact member mean | — | — | -0.0100 | -0.1067 | -0.0154 | -0.0104 |
| segmentation | SpaceNet7 | PANOPTICON | full_finetune | deep_ensemble | M=3 ensemble vs exact member mean | — | — | -0.0084 | -0.1715 | -0.0198 | -0.0170 |

### Intentionally not applicable

| Task | Dataset | Model | Adaptation | Method | Reason |
|---|---|---|---|---|---|
| classification | TreeSatAI | DOFA | frozen | temperature_scaling | N/A — no independent calibration split under the completed benchmark protocol |
| classification | TreeSatAI | DOFA | full_finetune | temperature_scaling | N/A — no independent calibration split under the completed benchmark protocol |
| classification | TreeSatAI | PANOPTICON | frozen | temperature_scaling | N/A — no independent calibration split under the completed benchmark protocol |
| classification | TreeSatAI | PANOPTICON | full_finetune | temperature_scaling | N/A — no independent calibration split under the completed benchmark protocol |
| segmentation | CloudSEN12 | DOFA | frozen | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |
| segmentation | CloudSEN12 | DOFA | full_finetune | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |
| segmentation | CloudSEN12 | PANOPTICON | frozen | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |
| segmentation | CloudSEN12 | PANOPTICON | full_finetune | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |
| segmentation | SpaceNet7 | DOFA | frozen | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |
| segmentation | SpaceNet7 | DOFA | full_finetune | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |
| segmentation | SpaceNet7 | PANOPTICON | frozen | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |
| segmentation | SpaceNet7 | PANOPTICON | full_finetune | temperature_scaling | N/A — Temperature Scaling excluded from the frozen segmentation protocol |

## Interpretation boundaries

- RQ2 and MC Dropout seed matches are design-paired independently trained models; only Temperature Scaling is a same-checkpoint post-hoc comparison.
- Deep Ensemble effects compare one probability-mean ensemble with the mean of its exact seeds 42/43/44 members. They have no seedwise effect standard deviation.
- Three-seed summaries are descriptive (n=3); no confidence intervals, hypothesis tests, or causal claims are made.
- TreeSatAI Accuracy is strict multilabel exact-match accuracy, and its calibration quantities follow binary-label semantics; they are not directly pooled with EuroSAT multiclass values.
- ECE means ECE-15 throughout. No sample-level bootstrap, reliability-bin, error-overlap, or uncertainty-diversity analysis was inferred from aggregate metrics.
- TreeSatAI and segmentation Temperature Scaling remain N/A under the frozen final protocol; historical diagnostic artifacts are not reintroduced.

Machine-readable tables:

- `reports/rq2_adaptation_effects.csv`
- `reports/rq3_uq_effects.csv`
