# Thesis Evidence Matrix — A–C Final Experiments

Generated: 2026-08-27T18:33:54.509888+00:00

This document is the unified, time-independent evidence entry point for RQ1–RQ3. Report age does not determine authority: frozen protocol decisions control inclusion; validated per-run artifacts control numbers; later derived summaries are reusable only where they agree with those authorities. No training, model loading, checkpoint inference, or experiment-artifact mutation was performed.

## Evidence precedence and protocol resolution

1. `pre_uq_protocol_freeze.md`, `final_training_protocol.md`, global conventions, and final dataset protocols control eligibility and interpretation.
2. Immutable/promoted C1 and C2 runs control deterministic seed-level evidence.
3. C3 controls valid EuroSAT Temperature Scaling and Deep Ensembles; C5 controls MC Dropout.
4. Derived C6 tables are supporting summaries, not higher authority than the frozen protocol.

**TreeSatAI Temperature Scaling resolution:** the validation-fitted C3 artifacts exist and remain immutable historical/diagnostic evidence, but the later final pre-UQ freeze explicitly excludes them from thesis comparative results because no independent calibration split exists. The four TreeSatAI method cells are therefore `N/A`, not failed or missing. This intentionally overrides the stale `COMPLETE` entries in the C6 classification summary without modifying C3/C6 artifacts.

Segmentation Temperature Scaling is likewise `N/A` by the frozen protocol. SpaceNet7 classwise-ECE values for ensembles come from validated ensemble manifests because the C3 summary CSV omitted those columns.

## Master table coverage

| Record type | Rows | Meaning |
|---|---:|---|
| INDIVIDUAL_SEED | 84 | One independently trained deterministic/MC model or one valid EuroSAT temperature applied to that seed |
| MEAN_STD | 24 | Arithmetic mean ± sample std across the explicitly listed seeds |
| ENSEMBLE | 16 | One separate three-member probability-mean result; never pooled with member statistics |
| NOT_APPLICABLE | 12 | Method intentionally excluded by frozen protocol; contains no synthetic metric values |

Machine-readable source: `reports/thesis_master_results.csv`. Blank timing fields mean unavailable/not recorded unless the status explicitly says `N/A`; they are never interpreted as zero.

## RQ1 — Calibration after downstream adaptation

### Deterministic classification baselines (three-seed mean ± sample std)

| Dataset | Model | Adaptation | Accuracy | Macro-F1 | NLL | Brier | ECE-15 |
|---|---|---|---:|---:|---:|---:|---:|
| EuroSAT | DOFA | frozen | 0.9834 ± 0.0006 | 0.9823 ± 0.0007 | 0.0544 ± 0.0039 | 0.0260 ± 0.0015 | 0.0050 ± 0.0019 |
| EuroSAT | DOFA | full_finetune | 0.9649 ± 0.0054 | 0.9637 ± 0.0055 | 0.1069 ± 0.0203 | 0.0533 ± 0.0082 | 0.0111 ± 0.0049 |
| EuroSAT | PANOPTICON | frozen | 0.9833 ± 0.0004 | 0.9826 ± 0.0005 | 0.0505 ± 0.0036 | 0.0256 ± 0.0009 | 0.0066 ± 0.0026 |
| EuroSAT | PANOPTICON | full_finetune | 0.9627 ± 0.0073 | 0.9607 ± 0.0082 | 0.1126 ± 0.0374 | 0.0558 ± 0.0136 | 0.0095 ± 0.0050 |
| TreeSatAI | DOFA | frozen | 0.2567 ± 0.0015 | 0.2232 ± 0.0037 | 0.2558 ± 0.0007 | 0.0740 ± 0.0002 | 0.0108 ± 0.0009 |
| TreeSatAI | DOFA | full_finetune | 0.2668 ± 0.0028 | 0.2438 ± 0.0124 | 0.2469 ± 0.0030 | 0.0717 ± 0.0010 | 0.0126 ± 0.0039 |
| TreeSatAI | PANOPTICON | frozen | 0.2378 ± 0.0029 | 0.2108 ± 0.0055 | 0.2570 ± 0.0021 | 0.0754 ± 0.0006 | 0.0095 ± 0.0009 |
| TreeSatAI | PANOPTICON | full_finetune | 0.2515 ± 0.0313 | 0.2198 ± 0.0244 | 0.2505 ± 0.0045 | 0.0732 ± 0.0016 | 0.0076 ± 0.0036 |

TreeSatAI is multilabel: Accuracy is strict exact match, and NLL/Brier/ECE operate over binary label decisions. Its values must not be treated as multiclass-simplex metrics.

### Deterministic segmentation baselines (three-seed mean ± sample std)

| Dataset | Model | Adaptation | mIoU | Pixel accuracy | NLL | Brier | ECE-15 | Per-class IoU |
|---|---|---|---:|---:|---:|---:|---:|---|
| CloudSEN12 | DOFA | frozen | 0.6044 ± 0.0037 | 0.8339 ± 0.0021 | 0.4733 ± 0.0112 | 0.2419 ± 0.0033 | 0.0400 ± 0.0179 | clear 0.8184 ± 0.0034; thick 0.7463 ± 0.0049; thin 0.3634 ± 0.0091; shadow 0.4897 ± 0.0016 |
| CloudSEN12 | DOFA | full_finetune | 0.6568 ± 0.0067 | 0.8625 ± 0.0047 | 0.4796 ± 0.0151 | 0.2122 ± 0.0048 | 0.0637 ± 0.0044 | clear 0.8494 ± 0.0059; thick 0.7901 ± 0.0060; thin 0.4366 ± 0.0049; shadow 0.5511 ± 0.0112 |
| CloudSEN12 | PANOPTICON | frozen | 0.6544 ± 0.0005 | 0.8603 ± 0.0010 | 0.4137 ± 0.0214 | 0.2061 ± 0.0040 | 0.0416 ± 0.0073 | clear 0.8507 ± 0.0016; thick 0.7826 ± 0.0026; thin 0.4521 ± 0.0007; shadow 0.5321 ± 0.0051 |
| CloudSEN12 | PANOPTICON | full_finetune | 0.6725 ± 0.0037 | 0.8705 ± 0.0019 | 0.3881 ± 0.0150 | 0.1914 ± 0.0036 | 0.0422 ± 0.0038 | clear 0.8608 ± 0.0021; thick 0.8000 ± 0.0037; thin 0.4651 ± 0.0087; shadow 0.5640 ± 0.0025 |
| SpaceNet7 | DOFA | frozen | 0.4883 ± 0.0009 | 0.9247 ± 0.0021 | 0.3123 ± 0.0274 | 0.1307 ± 0.0032 | 0.0391 ± 0.0030 | background 0.9244 ± 0.0021; building 0.0522 ± 0.0037 |
| SpaceNet7 | DOFA | full_finetune | 0.4978 ± 0.0016 | 0.9232 ± 0.0023 | 0.4701 ± 0.0788 | 0.1341 ± 0.0039 | 0.0496 ± 0.0064 | background 0.9227 ± 0.0024; building 0.0729 ± 0.0056 |
| SpaceNet7 | PANOPTICON | frozen | 0.5013 ± 0.0015 | 0.9186 ± 0.0001 | 0.3549 ± 0.0009 | 0.1363 ± 0.0007 | 0.0413 ± 0.0002 | background 0.9180 ± 0.0000; building 0.0846 ± 0.0030 |
| SpaceNet7 | PANOPTICON | full_finetune | 0.5043 ± 0.0049 | 0.9164 ± 0.0037 | 0.4318 ± 0.0854 | 0.1412 ± 0.0047 | 0.0454 ± 0.0102 | background 0.9156 ± 0.0038; building 0.0930 ± 0.0113 |

Cross-task raw calibration differences are descriptive only: classification uses minimum validation NLL, segmentation uses maximum validation mIoU, and their output units/label semantics differ.

## RQ2 — Frozen versus full fine-tuning

Deltas below are full fine-tuning minus frozen adaptation using deterministic three-seed means. Positive performance delta is favorable; negative NLL/Brier/ECE delta is favorable.

| Task | Dataset | Model | Δ performance | Δ NLL | Δ Brier | Δ ECE-15 |
|---|---|---|---:|---:|---:|---:|
| classification | EuroSAT | DOFA | -0.0185 | +0.0526 | +0.0273 | +0.0061 |
| classification | EuroSAT | PANOPTICON | -0.0206 | +0.0622 | +0.0301 | +0.0028 |
| classification | TreeSatAI | DOFA | +0.0102 | -0.0090 | -0.0023 | +0.0017 |
| classification | TreeSatAI | PANOPTICON | +0.0137 | -0.0065 | -0.0022 | -0.0019 |
| segmentation | CloudSEN12 | DOFA | +0.0524 | +0.0064 | -0.0297 | +0.0237 |
| segmentation | CloudSEN12 | PANOPTICON | +0.0181 | -0.0256 | -0.0147 | +0.0005 |
| segmentation | SpaceNet7 | DOFA | +0.0095 | +0.1577 | +0.0033 | +0.0105 |
| segmentation | SpaceNet7 | PANOPTICON | +0.0030 | +0.0770 | +0.0049 | +0.0041 |

These are descriptive paired-cell contrasts, not causal estimates. Architecture-specific preprocessing history and validation-selected checkpoint behavior remain relevant caveats.

## RQ3 — UQ/calibration methods, predictive performance, and cost

Comparison basis: valid Temperature Scaling uses three-seed means against deterministic three-seed means; replicated MC cells use their predeclared three-seed summaries; other MC cells use seed-42 paired comparisons; ensembles remain single M=3 results compared descriptively with deterministic member means.

| Task | Dataset | Model | Adaptation | Method | Basis | Performance | Δ performance | Δ NLL | Δ Brier | Δ ECE-15 | Training cost | Inference proxy |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| classification | EuroSAT | DOFA | frozen | temperature_scaling | 3-seed mean | 0.9834 | +0.0000 | +0.0000 | -0.0000 | -0.0000 | 1090.9s (REUSED_DETERMINISTIC_CHECKPOINT: MEAN_AND_SAMPLE_STD_OF_MEMBER_TRAINING_SECONDS) | scalar fit: mean 23.0 optimizer iterations across seeds; 1 probability transform |
| classification | EuroSAT | DOFA | frozen | mc_dropout | 3-seed robustness subset | 0.9835 | +0.0001 | +0.0001 | +0.0005 | +0.0031 | 1090.5s (MEAN_AND_SAMPLE_STD_OF_MC_DROPOUT_SEED_TRAINING_SECONDS) | T=30 stochastic probability passes |
| classification | EuroSAT | DOFA | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.9838 | +0.0004 | -0.0024 | -0.0013 | +0.0005 | 3272.8s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| classification | EuroSAT | DOFA | full_finetune | temperature_scaling | 3-seed mean | 0.9649 | +0.0000 | -0.0028 | -0.0006 | -0.0038 | 3170.2s (REUSED_DETERMINISTIC_CHECKPOINT: MEAN_AND_SAMPLE_STD_OF_MEMBER_TRAINING_SECONDS) | scalar fit: mean 17.3 optimizer iterations across seeds; 1 probability transform |
| classification | EuroSAT | DOFA | full_finetune | mc_dropout | paired seed42 | 0.9587 | -0.0122 | +0.0420 | +0.0196 | +0.0030 | 2655.2s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| classification | EuroSAT | DOFA | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.9827 | +0.0178 | -0.0430 | -0.0228 | +0.0162 | 9510.6s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| classification | EuroSAT | PANOPTICON | frozen | temperature_scaling | 3-seed mean | 0.9833 | +0.0000 | -0.0010 | -0.0002 | -0.0010 | 1926.1s (REUSED_DETERMINISTIC_CHECKPOINT: MEAN_AND_SAMPLE_STD_OF_MEMBER_TRAINING_SECONDS) | scalar fit: mean 22.3 optimizer iterations across seeds; 1 probability transform |
| classification | EuroSAT | PANOPTICON | frozen | mc_dropout | paired seed42 | 0.9831 | -0.0007 | -0.0028 | -0.0011 | -0.0030 | 2021.4s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| classification | EuroSAT | PANOPTICON | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.9827 | -0.0006 | -0.0027 | -0.0007 | +0.0002 | 5778.3s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| classification | EuroSAT | PANOPTICON | full_finetune | temperature_scaling | 3-seed mean | 0.9627 | +0.0000 | -0.0010 | -0.0001 | -0.0013 | 4400.0s (REUSED_DETERMINISTIC_CHECKPOINT: MEAN_AND_SAMPLE_STD_OF_MEMBER_TRAINING_SECONDS) | scalar fit: mean 19.3 optimizer iterations across seeds; 1 probability transform |
| classification | EuroSAT | PANOPTICON | full_finetune | mc_dropout | paired seed42 | 0.9580 | -0.0099 | +0.0235 | +0.0160 | +0.0011 | 4513.1s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| classification | EuroSAT | PANOPTICON | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.9764 | +0.0138 | -0.0400 | -0.0194 | +0.0125 | 13200.0s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| classification | TreeSatAI | DOFA | frozen | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| classification | TreeSatAI | DOFA | frozen | mc_dropout | paired seed42 | 0.2550 | +0.0000 | -0.0004 | +0.0002 | -0.0011 | 1346.6s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| classification | TreeSatAI | DOFA | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.2605 | +0.0038 | -0.0021 | -0.0005 | -0.0004 | 2665.6s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| classification | TreeSatAI | DOFA | full_finetune | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| classification | TreeSatAI | DOFA | full_finetune | mc_dropout | paired seed42 | 0.2825 | +0.0175 | -0.0020 | -0.0011 | +0.0014 | 835.8s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| classification | TreeSatAI | DOFA | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.2740 | +0.0072 | -0.0093 | -0.0030 | +0.0045 | 2666.0s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| classification | TreeSatAI | PANOPTICON | frozen | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| classification | TreeSatAI | PANOPTICON | frozen | mc_dropout | paired seed42 | 0.2380 | -0.0015 | +0.0002 | +0.0001 | -0.0004 | 944.7s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| classification | TreeSatAI | PANOPTICON | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.2390 | +0.0012 | -0.0023 | -0.0006 | -0.0007 | 3459.5s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| classification | TreeSatAI | PANOPTICON | full_finetune | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| classification | TreeSatAI | PANOPTICON | full_finetune | mc_dropout | 3-seed robustness subset | 0.2680 | +0.0165 | -0.0030 | -0.0009 | -0.0000 | 1855.6s (MEAN_AND_SAMPLE_STD_OF_MC_DROPOUT_SEED_TRAINING_SECONDS) | T=30 stochastic probability passes |
| classification | TreeSatAI | PANOPTICON | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.2655 | +0.0140 | -0.0088 | -0.0028 | +0.0021 | 5634.0s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | CloudSEN12 | DOFA | frozen | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | CloudSEN12 | DOFA | frozen | mc_dropout | paired seed42 | 0.6005 | +0.0001 | +0.0023 | +0.0017 | +0.0138 | 3783.5s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| segmentation | CloudSEN12 | DOFA | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.6207 | +0.0163 | -0.0486 | -0.0181 | -0.0231 | 10508.6s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | CloudSEN12 | DOFA | full_finetune | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | CloudSEN12 | DOFA | full_finetune | mc_dropout | paired seed42 | 0.6588 | +0.0026 | -0.0390 | -0.0059 | -0.0126 | 5766.8s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| segmentation | CloudSEN12 | DOFA | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.6714 | +0.0146 | -0.0814 | -0.0202 | -0.0235 | 17570.8s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | CloudSEN12 | PANOPTICON | frozen | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | CloudSEN12 | PANOPTICON | frozen | mc_dropout | 3-seed robustness subset | 0.6490 | -0.0054 | -0.0161 | -0.0019 | -0.0110 | 3432.0s (MEAN_AND_SAMPLE_STD_OF_MC_DROPOUT_SEED_TRAINING_SECONDS) | T=30 stochastic probability passes |
| segmentation | CloudSEN12 | PANOPTICON | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.6673 | +0.0129 | -0.0479 | -0.0134 | -0.0166 | 11968.1s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | CloudSEN12 | PANOPTICON | full_finetune | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | CloudSEN12 | PANOPTICON | full_finetune | mc_dropout | paired seed42 | 0.6494 | -0.0210 | +0.0006 | +0.0103 | -0.0075 | 8884.5s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| segmentation | CloudSEN12 | PANOPTICON | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.6903 | +0.0178 | -0.0526 | -0.0167 | -0.0219 | 29132.2s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | SpaceNet7 | DOFA | frozen | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | SpaceNet7 | DOFA | frozen | mc_dropout | paired seed42 | 0.4854 | -0.0025 | -0.0047 | -0.0004 | +0.0009 | 2211.0s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| segmentation | SpaceNet7 | DOFA | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.4818 | -0.0065 | -0.0713 | -0.0092 | -0.0087 | 8235.7s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | SpaceNet7 | DOFA | full_finetune | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | SpaceNet7 | DOFA | full_finetune | mc_dropout | 3-seed robustness subset | 0.4900 | -0.0078 | -0.1845 | -0.0123 | -0.0153 | 2357.9s (MEAN_AND_SAMPLE_STD_OF_MC_DROPOUT_SEED_TRAINING_SECONDS) | T=30 stochastic probability passes |
| segmentation | SpaceNet7 | DOFA | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.4868 | -0.0110 | -0.1489 | -0.0121 | -0.0097 | 10243.8s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | SpaceNet7 | PANOPTICON | frozen | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | SpaceNet7 | PANOPTICON | frozen | mc_dropout | paired seed42 | 0.5007 | +0.0005 | -0.0372 | -0.0047 | -0.0023 | 3815.7s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| segmentation | SpaceNet7 | PANOPTICON | frozen | deep_ensemble | single M=3 ensemble vs member mean | 0.4913 | -0.0100 | -0.1067 | -0.0154 | -0.0104 | 13433.4s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |
| segmentation | SpaceNet7 | PANOPTICON | full_finetune | temperature_scaling | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| segmentation | SpaceNet7 | PANOPTICON | full_finetune | mc_dropout | paired seed42 | 0.5013 | -0.0077 | -0.0655 | -0.0134 | -0.0087 | 5093.4s (MEASURED_CUMULATIVE_MC_DROPOUT_TRAINING_WALL_SECONDS) | T=30 stochastic probability passes |
| segmentation | SpaceNet7 | PANOPTICON | full_finetune | deep_ensemble | single M=3 ensemble vs member mean | 0.4959 | -0.0084 | -0.1715 | -0.0198 | -0.0170 | 18137.9s (DERIVED_SUM_OF_THREE_MEMBER_TRAINING_SECONDS) | M=3 member probability evaluations |

No method is chosen or rejected from test performance. MC Dropout is downstream-head/decoder dropout with T=30; Deep Ensemble averages probabilities across M=3 independently trained seeds. Expected predictive entropy and MI-style disagreement are descriptive quantities/proxies, not literal aleatoric/epistemic truth.

### Cost evidence availability

| Method/result | Training timing | Inference timing | Valid compute proxy |
|---|---|---|---|
| Deterministic classification | 24/24 measured cumulative training seconds | 24/24 measured single-pass test seconds | 1 pass |
| Deterministic segmentation | 24/24 measured cumulative training seconds | 24/24 measured single-pass test seconds | 1 pass |
| EuroSAT Temperature Scaling | Reuses deterministic checkpoint; scalar-fit time not recorded | Apply time not recorded | Optimizer iterations retained per seed |
| MC Dropout | 24/24 measured cumulative MC-training seconds | 0/24; not recorded | T=30 stochastic passes |
| Deep Ensemble | Derived sum of three member-training seconds | Derived serial member-sum available for all 16 ensembles; aggregation overhead not recorded | M=3 members |

`training_seconds` never substitutes process-local `runtime_seconds`, which can be misleading for resumed runs. Missing timings are `UNAVAILABLE_NOT_RECORDED`, never zero.

## SpaceNet7 required calibration evidence

The CSV retains overall ECE, foreground/building ECE, one-vs-rest background/building ECE, foreground NLL/Brier, boundary ECE, and boundary foreground ECE. Boundary-free per-image quantities remain undefined in their source archives and are never converted to zero. Low building IoU is a validated result, not an exclusion criterion.

## Known evidence inconsistencies and preserved caveats

- TreeSatAI validation-fitted TS artifacts/report: historical diagnostic only; final comparative entry is N/A under the later freeze.
- C6 `final_thesis_results.md` and classification table: numerically reusable except TreeSatAI TS `COMPLETE` entries, which this matrix overrides.
- C3 segmentation summary CSV omitted SpaceNet7 classwise ECE; validated ensemble manifests supply the already-computed values.
- Earlier C1/C2 audit snapshots and execution logs remain indexed but are superseded by the final 18/18 and 24/24 promotion audits.
- TreeSatAI uses the validated static T=1 artifact and is multilabel; it must not be described as genuinely multi-temporal here.
- CloudSEN12 DOFA full-finetune seed44 has a one-pixel float32 CPU/GPU tie warning but remains promoted.
- EuroSAT retains a backend-specific preprocessing-history caveat; cross-model comparisons must not conceal it.

## Unified report registry

Every current file under `reports/` is indexed below regardless of generation time. `mtime_utc` is discovery metadata only and never determines precedence. `numeric_use=YES` means the file directly contributed values after protocol filtering; all other reports remain available for provenance, validation, caveats, or historical analysis.

| Path | mtime UTC | Registry role | Authority | Numeric use | Inclusion note |
|---|---|---|---|---|---|
| `reports/c1_classification_deterministic_completion.md` | 2026-08-16T06:47:16.592005+00:00 | SUPPORTING_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Supports final artifacts/protocol interpretation |
| `reports/c1_classification_run_audits/20260815T231201Z/c1_run_audit.csv` | 2026-08-15T23:12:01.776308+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260815T231201Z/c1_run_audit.json` | 2026-08-15T23:12:01.776308+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260815T231201Z/c1_run_audit.md` | 2026-08-15T23:12:01.780308+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T003651Z/c1_run_audit.csv` | 2026-08-16T00:36:51.939252+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T003651Z/c1_run_audit.json` | 2026-08-16T00:36:51.939252+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T003651Z/c1_run_audit.md` | 2026-08-16T00:36:51.939252+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T004644Z/c1_run_audit.csv` | 2026-08-16T00:46:44.482348+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T004644Z/c1_run_audit.json` | 2026-08-16T00:46:44.482348+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T004644Z/c1_run_audit.md` | 2026-08-16T00:46:44.482348+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T012138Z/c1_run_audit.csv` | 2026-08-16T01:21:38.826923+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T012138Z/c1_run_audit.json` | 2026-08-16T01:21:38.826923+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T012138Z/c1_run_audit.md` | 2026-08-16T01:21:38.826923+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T013535Z/c1_run_audit.csv` | 2026-08-16T01:35:35.737305+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T013535Z/c1_run_audit.json` | 2026-08-16T01:35:35.737305+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T013535Z/c1_run_audit.md` | 2026-08-16T01:35:35.737305+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T014800Z/c1_run_audit.csv` | 2026-08-16T01:48:00.308140+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T014800Z/c1_run_audit.json` | 2026-08-16T01:48:00.308140+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T014800Z/c1_run_audit.md` | 2026-08-16T01:48:00.308140+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T020849Z/c1_run_audit.csv` | 2026-08-16T02:08:49.775910+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T020849Z/c1_run_audit.json` | 2026-08-16T02:08:49.775910+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T020849Z/c1_run_audit.md` | 2026-08-16T02:08:49.775910+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T023151Z/c1_run_audit.csv` | 2026-08-16T02:31:51.032629+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T023151Z/c1_run_audit.json` | 2026-08-16T02:31:51.032629+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T023151Z/c1_run_audit.md` | 2026-08-16T02:31:51.032629+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T034554Z/c1_run_audit.csv` | 2026-08-16T03:45:54.109852+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T034554Z/c1_run_audit.json` | 2026-08-16T03:45:54.109852+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T034554Z/c1_run_audit.md` | 2026-08-16T03:45:54.109852+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T050250Z/c1_run_audit.csv` | 2026-08-16T05:02:50.724502+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T050250Z/c1_run_audit.json` | 2026-08-16T05:02:50.724502+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T050250Z/c1_run_audit.md` | 2026-08-16T05:02:50.724502+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T051904Z/c1_run_audit.csv` | 2026-08-16T05:19:04.327524+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T051904Z/c1_run_audit.json` | 2026-08-16T05:19:04.327524+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T051904Z/c1_run_audit.md` | 2026-08-16T05:19:04.327524+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T053338Z/c1_run_audit.csv` | 2026-08-16T05:33:38.138580+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T053338Z/c1_run_audit.json` | 2026-08-16T05:33:38.138580+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T053338Z/c1_run_audit.md` | 2026-08-16T05:33:38.138580+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T060644Z/c1_run_audit.csv` | 2026-08-16T06:06:44.747615+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T060644Z/c1_run_audit.json` | 2026-08-16T06:06:44.747615+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T060644Z/c1_run_audit.md` | 2026-08-16T06:06:44.747615+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T063728Z/c1_run_audit.csv` | 2026-08-16T06:37:28.056007+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T063728Z/c1_run_audit.json` | 2026-08-16T06:37:28.056007+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T063728Z/c1_run_audit.md` | 2026-08-16T06:37:28.056007+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; later C1 audit controls |
| `reports/c1_classification_run_audits/20260816T063820Z/c1_run_audit.csv` | 2026-08-16T06:38:20.716013+00:00 | FINAL_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Latest 18/18 promotion audit |
| `reports/c1_classification_run_audits/20260816T063820Z/c1_run_audit.json` | 2026-08-16T06:38:20.712013+00:00 | FINAL_NUMERIC_SOURCE | AUTHORITATIVE | YES | Used directly after frozen-protocol filters |
| `reports/c1_classification_run_audits/20260816T063820Z/c1_run_audit.md` | 2026-08-16T06:38:20.716013+00:00 | FINAL_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Latest 18/18 promotion audit |
| `reports/c2_execution_logs/20260823T211724Z_gpu0_cloud_full_smoke.log` | 2026-08-23T21:18:05.975419+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211724Z_gpu1_space_full_smoke.log` | 2026-08-23T21:17:57.087407+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211724Z_smoke_wave1_master.log` | 2026-08-23T21:18:07.495421+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211820Z_gpu0_cloud_frozen_smoke.log` | 2026-08-23T21:18:58.023488+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211820Z_gpu1_space_frozen_smoke.log` | 2026-08-23T21:18:49.407476+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211820Z_smoke_wave2_master.log` | 2026-08-23T21:18:59.591490+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211931Z_gpu0_cloud_full_seed42.log` | 2026-08-23T21:19:37.375539+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211931Z_gpu1_space_full_seed42.log` | 2026-08-23T21:19:37.075539+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T211931Z_seed42_two_gpu_master.log` | 2026-08-23T21:19:31.287531+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T214925Z_smoke_wave1_gpu0_cloud_frozen.log` | 2026-08-23T21:50:17.963142+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T214925Z_smoke_wave1_gpu1_space_frozen.log` | 2026-08-23T21:50:17.963142+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T215154Z_gpu0_resume_cloudsen12_dofa_frozen_seed42.log` | 2026-08-23T21:52:00.367084+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T215154Z_gpu1_resume_spacenet7_dofa_frozen_seed42.log` | 2026-08-23T21:52:00.231084+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T215606Z_gpu0_resume_cloudsen12_dofa_frozen_seed42.log` | 2026-08-23T22:46:26.024071+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T215606Z_gpu1_resume_spacenet7_dofa_frozen_seed42.log` | 2026-08-23T22:33:08.323047+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T223410Z_gpu1_spacenet7_full_seed42.log` | 2026-08-24T01:28:04.973825+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260823T224707Z_gpu0_cloudsen12_full_seed42.log` | 2026-08-24T03:14:08.703941+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T012906Z_gpu1_spacenet7_panopticon_frozen_seed42.log` | 2026-08-24T02:45:39.860017+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T024633Z_gpu1_cloudsen12_panopticon_frozen_seed42.log` | 2026-08-24T04:12:12.819014+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T041625Z_gpu0_cloudsen12_full_seed43.log` | 2026-08-24T08:42:45.903816+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T041625Z_gpu1_spacenet7_full_seed43.log` | 2026-08-24T06:33:01.456475+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T063301Z_gpu1_cloudsen12_frozen_seed43.log` | 2026-08-24T08:45:49.583824+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T084245Z_gpu0_spacenet7_frozen_seed43.log` | 2026-08-24T10:50:26.349365+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T084549Z_gpu1_cloudsen12_full_seed44.log` | 2026-08-24T13:10:21.894351+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T105026Z_gpu0_spacenet7_full_seed44.log` | 2026-08-24T13:51:13.280168+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T131021Z_gpu1_spacenet7_frozen_seed44.log` | 2026-08-24T15:25:54.420427+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/20260824T135113Z_gpu0_cloudsen12_frozen_seed44.log` | 2026-08-24T16:04:59.410000+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/seed42_cloud_frozen_gpu0_20260823.log` | 2026-08-23T21:27:49.472161+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/seed42_frozen_two_gpu_20260823.log` | 2026-08-23T21:25:50.996014+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/seed42_gate_20260817T1629Z.log` | 2026-08-17T16:40:42.949753+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/seed42_space_frozen_gpu1_20260823.log` | 2026-08-23T21:28:13.572191+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/smoke_cloud_frozen_gpu0_20260823.log` | 2026-08-23T21:22:51.791789+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/smoke_cloud_full_gpu0_20260823.log` | 2026-08-23T21:23:44.215856+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/smoke_space_frozen_gpu1_20260823.log` | 2026-08-23T21:22:43.059778+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_logs/smoke_space_full_gpu1_20260823.log` | 2026-08-23T21:23:33.343842+00:00 | EXECUTION_LOG | SUPPORTING_ONLY | NO | Operational history; not numeric authority |
| `reports/c2_execution_state.md` | 2026-08-20T14:15:55.133458+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/c2_segmentation_run_audits/20260824T041440Z/audit.csv` | 2026-08-24T04:14:40.723158+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T041440Z/audit.json` | 2026-08-24T04:14:40.723158+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T041440Z/audit.md` | 2026-08-24T04:14:40.723158+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T041519Z/audit.csv` | 2026-08-24T04:15:19.627196+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T041519Z/audit.json` | 2026-08-24T04:15:19.627196+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T041519Z/audit.md` | 2026-08-24T04:15:19.627196+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T161119Z/audit.csv` | 2026-08-24T16:11:19.938253+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T161119Z/audit.json` | 2026-08-24T16:11:19.938253+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T161119Z/audit.md` | 2026-08-24T16:11:19.938253+00:00 | AUDIT_SNAPSHOT | SUPERSEDED | NO | Retained chronology; final C2 audit controls |
| `reports/c2_segmentation_run_audits/20260824T162250Z/audit.csv` | 2026-08-24T16:22:50.594728+00:00 | FINAL_NUMERIC_SOURCE | AUTHORITATIVE | YES | Used directly after frozen-protocol filters |
| `reports/c2_segmentation_run_audits/20260824T162250Z/audit.json` | 2026-08-24T16:22:50.594728+00:00 | FINAL_NUMERIC_SOURCE | AUTHORITATIVE | YES | Used directly after frozen-protocol filters |
| `reports/c2_segmentation_run_audits/20260824T162250Z/audit.md` | 2026-08-24T16:22:50.594728+00:00 | FINAL_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Final 24/24 promotion audit |
| `reports/c3_classification_results.csv` | 2026-08-24T21:30:50.309762+00:00 | FINAL_NUMERIC_SOURCE | AUTHORITATIVE | YES | Used directly after frozen-protocol filters |
| `reports/c3_segmentation_ensemble_results.csv` | 2026-08-24T16:57:36.093147+00:00 | FINAL_NUMERIC_SOURCE | AUTHORITATIVE | YES | Used directly after frozen-protocol filters |
| `reports/c3_temperature_scaling_and_ensembles.md` | 2026-08-24T21:30:50.309762+00:00 | SUPPORTING_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Supports final artifacts/protocol interpretation |
| `reports/c3_treesatai_temperature_scaling.md` | 2026-08-24T21:30:50.309762+00:00 | HISTORICAL_DIAGNOSTIC | HISTORICAL_ONLY | NO | Validation-fitted TreeSatAI TS excluded by later freeze |
| `reports/c3_treesatai_temperature_scaling_results.csv` | 2026-08-24T21:30:48.721757+00:00 | HISTORICAL_DIAGNOSTIC | HISTORICAL_ONLY | NO | Validation-fitted TreeSatAI TS excluded by later freeze |
| `reports/checkpoint_selection_sensitivity.md` | 2026-08-25T22:13:23.959269+00:00 | SUPPORTING_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Supports final artifacts/protocol interpretation |
| `reports/classification_gap_analysis.md` | 2026-08-09T01:04:34.252489+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/cloudsen12_protocol.md` | 2026-08-10T01:02:14.785368+00:00 | SUPPORTING_REPORT | SUPPORTING_ONLY | NO | Indexed for future analysis; not used as final numeric authority |
| `reports/code_audit.md` | 2026-08-09T13:23:58.795271+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/dataset_manifests/cloudsen12_actual_manifest.csv` | 2026-08-10T00:54:44.942992+00:00 | DATASET_PROVENANCE | AUTHORITATIVE_SUPPORT | NO | Downloaded benchmark manifest evidence |
| `reports/dataset_manifests/cloudsen12_actual_manifest_summary.json` | 2026-08-10T00:54:44.942992+00:00 | DATASET_PROVENANCE | AUTHORITATIVE_SUPPORT | NO | Downloaded benchmark manifest evidence |
| `reports/dataset_manifests/spacenet7_actual_manifest.csv` | 2026-08-10T00:54:47.126984+00:00 | DATASET_PROVENANCE | AUTHORITATIVE_SUPPORT | NO | Downloaded benchmark manifest evidence |
| `reports/dataset_manifests/spacenet7_actual_manifest_summary.json` | 2026-08-10T00:54:47.130984+00:00 | DATASET_PROVENANCE | AUTHORITATIVE_SUPPORT | NO | Downloaded benchmark manifest evidence |
| `reports/dataset_manifests/treesatai_actual_manifest.csv` | 2026-08-10T00:54:38.135019+00:00 | DATASET_PROVENANCE | AUTHORITATIVE_SUPPORT | NO | Downloaded benchmark manifest evidence |
| `reports/dataset_manifests/treesatai_actual_manifest_summary.json` | 2026-08-10T00:54:38.135019+00:00 | DATASET_PROVENANCE | AUTHORITATIVE_SUPPORT | NO | Downloaded benchmark manifest evidence |
| `reports/dofa_eurosat_ensemble_results.csv` | 2026-08-09T16:51:14.490374+00:00 | SUPPORTING_REPORT | SUPPORTING_ONLY | NO | Indexed for future analysis; not used as final numeric authority |
| `reports/dofa_eurosat_final_manifest.json` | 2026-08-10T00:35:59.182989+00:00 | FINAL_NUMERIC_SOURCE | AUTHORITATIVE | YES | Used directly after frozen-protocol filters |
| `reports/dofa_eurosat_frozen_artifacts.md` | 2026-08-10T00:59:27.873940+00:00 | SUPPORTING_REPORT | SUPPORTING_ONLY | NO | Indexed for future analysis; not used as final numeric authority |
| `reports/existing_runs.csv` | 2026-08-09T00:40:45.825800+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/experiment_registry.csv` | 2026-08-09T01:04:34.248489+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/final_dataset_protocols.md` | 2026-08-10T18:48:17.241618+00:00 | FROZEN_PROTOCOL | AUTHORITATIVE | NO | Controls inclusion, selection, or interpretation |
| `reports/final_thesis_results.md` | 2026-08-26T21:41:04.820199+00:00 | DERIVED_SUMMARY | REQUIRES_OVERRIDE | NO | Reusable except stale TreeSatAI TS COMPLETE entries |
| `reports/final_thesis_tables/classification_results.csv` | 2026-08-26T21:38:32.292033+00:00 | DERIVED_SUMMARY | REQUIRES_OVERRIDE | NO | Reusable except stale TreeSatAI TS COMPLETE entries |
| `reports/final_thesis_tables/mc_dropout_robustness.csv` | 2026-08-26T21:38:32.296034+00:00 | DERIVED_SUMMARY | SUPPORTING_ONLY | NO | Reusable C6 supporting output; no conflicting TreeSatAI TS result |
| `reports/final_thesis_tables/method_applicability.csv` | 2026-08-26T21:38:32.296034+00:00 | DERIVED_SUMMARY | REQUIRES_OVERRIDE | NO | Reusable except stale TreeSatAI TS COMPLETE entries |
| `reports/final_thesis_tables/package_manifest.json` | 2026-08-26T21:41:04.824199+00:00 | DERIVED_SUMMARY | SUPPORTING_ONLY | NO | Reusable C6 supporting output; no conflicting TreeSatAI TS result |
| `reports/final_thesis_tables/segmentation_results.csv` | 2026-08-26T21:38:32.292033+00:00 | DERIVED_SUMMARY | SUPPORTING_ONLY | NO | Reusable C6 supporting output; no conflicting TreeSatAI TS result |
| `reports/final_thesis_tables/source_provenance.csv` | 2026-08-26T21:41:04.824199+00:00 | DERIVED_SUMMARY | SUPPORTING_ONLY | NO | Reusable C6 supporting output; no conflicting TreeSatAI TS result |
| `reports/final_training_protocol.md` | 2026-08-15T19:32:45.497898+00:00 | FROZEN_PROTOCOL | AUTHORITATIVE | NO | Controls inclusion, selection, or interpretation |
| `reports/full_16_cell_preflight.md` | 2026-08-12T19:26:41.025099+00:00 | SUPPORTING_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Supports final artifacts/protocol interpretation |
| `reports/global_experiment_conventions.md` | 2026-08-10T01:03:32.413111+00:00 | FROZEN_PROTOCOL | AUTHORITATIVE | NO | Controls inclusion, selection, or interpretation |
| `reports/mc_dropout_pilot_validation.md` | 2026-08-25T22:48:06.188185+00:00 | FROZEN_PROTOCOL | AUTHORITATIVE | NO | Controls inclusion, selection, or interpretation |
| `reports/mc_dropout_protocol.md` | 2026-08-25T22:48:06.188185+00:00 | FROZEN_PROTOCOL | AUTHORITATIVE | NO | Controls inclusion, selection, or interpretation |
| `reports/mc_dropout_results.csv` | 2026-08-26T17:14:28.734471+00:00 | FINAL_NUMERIC_SOURCE | AUTHORITATIVE | YES | Used directly after frozen-protocol filters |
| `reports/mc_dropout_robustness.md` | 2026-08-26T17:14:28.738471+00:00 | SUPPORTING_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Supports final artifacts/protocol interpretation |
| `reports/mc_dropout_segmentation_research_subset_ids.json` | 2026-08-25T22:43:42.884098+00:00 | SUPPORTING_REPORT | SUPPORTING_ONLY | NO | Indexed for future analysis; not used as final numeric authority |
| `reports/mc_dropout_summary.md` | 2026-08-26T17:14:28.734471+00:00 | SUPPORTING_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Supports final artifacts/protocol interpretation |
| `reports/panopticon_real_model_validation.md` | 2026-08-09T22:05:20.545743+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/panopticon_validation.md` | 2026-08-10T01:02:46.273263+00:00 | SUPPORTING_VALIDATION | AUTHORITATIVE_SUPPORT | NO | Supports final artifacts/protocol interpretation |
| `reports/pipeline_refactor_plan.md` | 2026-08-09T12:25:25.546741+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/pre_uq_protocol_freeze.md` | 2026-08-25T22:30:27.944180+00:00 | FROZEN_PROTOCOL | AUTHORITATIVE | NO | Controls inclusion, selection, or interpretation |
| `reports/so2sat_protocol.md` | 2026-08-09T22:25:41.890024+00:00 | HISTORICAL_OR_OUT_OF_SCOPE | HISTORICAL_ONLY | NO | Retained but excluded from final numeric authority |
| `reports/spacenet7_protocol.md` | 2026-08-10T01:02:14.789369+00:00 | SUPPORTING_REPORT | SUPPORTING_ONLY | NO | Indexed for future analysis; not used as final numeric authority |
| `reports/treesatai_protocol.md` | 2026-08-10T01:02:14.785368+00:00 | SUPPORTING_REPORT | SUPPORTING_ONLY | NO | Indexed for future analysis; not used as final numeric authority |
| `reports/thesis_master_results.csv` | current build | MASTER_OUTPUT | DERIVED_CURRENT | NO | Current unified evidence output |
| `reports/thesis_evidence_matrix.md` | current build | MASTER_OUTPUT | DERIVED_CURRENT | NO | Current unified evidence output |
