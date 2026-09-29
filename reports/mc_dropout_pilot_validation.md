# MC Dropout Pilot Validation

Created: 2026-08-25T22:48:06.191566+00:00

No deterministic experiment was modified. Two new one-epoch, one-training-batch pilots were run, and all MC convergence analysis used one fixed official-validation batch. Test data were not evaluated or inspected.

The classification batch is a deterministic loader-prefix technical subset and contains one EuroSAT class; its absolute task metrics are not thesis results. It is used only to test stochastic-mode behavior and nested-pass convergence. The segmentation batch is likewise a technical subset. Final MC results must use the complete official evaluation split.

## Classification pilot

Cell: EuroSAT / DOFA / frozen / seed 42. Checkpoint: `/workspace/results/c4_mc_dropout_pilots/classification/eurosat_dofa_frozen/runs/20260825T224415910126Z_c4_mcd_eurosat_dofa_frozen_pilot_seed42_3a6950b6/best.pt`.

| T | Accuracy | NLL | Brier | ECE-15 | Pred. entropy | Expected entropy | MI | Variance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.171875 | 1.995749 | 0.840498 | 0.031735 | 2.201481 | 2.193448 | 0.008033 | 0.000190 |
| 20 | 0.187500 | 1.993870 | 0.839962 | 0.017400 | 2.202436 | 2.193982 | 0.008454 | 0.000201 |
| 30 | 0.171875 | 1.998976 | 0.841378 | 0.023015 | 2.202655 | 2.194136 | 0.008519 | 0.000201 |
| 50 | 0.171875 | 1.999668 | 0.841415 | 0.020949 | 2.202866 | 2.194285 | 0.008581 | 0.000203 |

T=30 vs T=50: probability MAE 0.001180, maximum difference 0.008349; stable = **TRUE**.

## Segmentation pilot

Cell: CloudSEN12 / Panopticon / frozen / seed 42. Checkpoint: `/workspace/results/c4_mc_dropout_pilots/segmentation/cloudsen12_panopticon_frozen/runs/20260825T224422405548Z_c4_mcd_cloudsen12_panopticon_frozen_pilot_seed42_cfc11827/best.pt`.

| T | mIoU | NLL | Brier | ECE-15 | Pred. entropy | Expected entropy | Disagreement/MI | Variance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.199114 | 1.308005 | 0.707367 | 0.062704 | 1.339515 | 1.334664 | 0.004851 | 0.000643 |
| 20 | 0.196016 | 1.308119 | 0.707887 | 0.058320 | 1.339614 | 1.334627 | 0.004988 | 0.000660 |
| 30 | 0.197489 | 1.307941 | 0.707586 | 0.060747 | 1.339955 | 1.334885 | 0.005070 | 0.000668 |
| 50 | 0.198272 | 1.307610 | 0.707244 | 0.062342 | 1.339703 | 1.334695 | 0.005008 | 0.000657 |

T=30 vs T=50: probability MAE 0.001989, maximum difference 0.030893; stable = **TRUE**.

## Validation checks

| Check | Classification | Segmentation |
|---|---|---|
| `aggregate_map_schema_valid` | — | PASS |
| `backbone_stochasticity_disabled` | PASS | PASS |
| `batchnorm_eval_and_buffers_unchanged` | PASS | PASS |
| `checkpoint_file_unchanged` | PASS | PASS |
| `checkpoint_selection_is_validation_miou_max` | — | PASS |
| `checkpoint_selection_is_validation_nll_min` | PASS | — |
| `deterministic_repeated_outputs_equal` | PASS | PASS |
| `dropout_active_during_mc_inference` | PASS | PASS |
| `dropout_active_during_training` | PASS | PASS |
| `dropout_probability_is_0_10` | PASS | PASS |
| `export_validation_passed` | PASS | — |
| `no_gradients_during_mc_inference` | PASS | PASS |
| `one_designated_dropout` | PASS | — |
| `one_designated_dropout2d` | — | PASS |
| `research_subset_stochastic_schema_valid` | — | PASS |
| `same_dropout_seed_reproduces_output` | PASS | PASS |
| `stochastic_outputs_nonidentical` | PASS | PASS |
| `stochastic_shape_n_t_c` | PASS | — |
| `t30_stable_relative_to_t50` | PASS | PASS |
| `test_not_evaluated_by_pilot_training` | — | PASS |
| `training_gradients_valid` | PASS | PASS |

The checks cover training-time activation, inference-time activation, finite/nonzero gradients from the training run audits, deterministic repeated outputs, same-seed reproducibility, non-identical stochastic outputs, BatchNorm evaluation/buffer preservation, disabled backbone stochasticity, checkpoint immutability, pass-count convergence, and storage shapes.

## READY_FOR_FULL_MCD = YES

This readiness decision authorizes the frozen full MC Dropout matrix only if it is YES; it does not launch that matrix.
