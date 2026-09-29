# C1 deterministic classification completion

Created: 2026-08-16 UTC

## Outcome

PROMPT C1 is complete.

- The six requested missing classification cells were each completed for seeds 42, 43, and 44: 18 new final runs.
- The six immutable DOFA / EuroSAT final runs were not retrained, modified, or used as resume targets.
- The final C1 gate reports **18 PROMOTABLE, 0 BLOCKED, 0 MISSING, and 0 AMBIGUOUS**.
- Combining these 18 runs with the six separately validated immutable DOFA / EuroSAT runs gives three valid deterministic seeds for all eight final classification cells.
- Test metrics were report-only. Validation NLL alone selected checkpoints and controlled early stopping.
- No ensemble, MC Dropout, Temperature Scaling, or other UQ inference was performed in C1.

The machine-readable source of truth is:

- `reports/c1_classification_run_audits/20260816T063820Z/c1_run_audit.json`
- `reports/c1_classification_run_audits/20260816T063820Z/c1_run_audit.csv`
- `reports/c1_classification_run_audits/20260816T063820Z/c1_run_audit.md`

The JSON audit SHA256 is `bd27381a6e77793a3b36b79f1e51ab2f6ea46eebcea0b63c1c437a060dda8817`.

## Three-seed deterministic test results

Values are mean ± sample standard deviation over seeds 42, 43, and 44. The exact per-seed metrics and run IDs are in the C1 audit CSV above.

| Dataset | Model | Adaptation | Accuracy | Macro F1 | NLL | Brier | ECE-15 |
|---|---|---|---:|---:|---:|---:|---:|
| EuroSAT | Panopticon | frozen | 0.983296 ± 0.000425 | 0.982642 ± 0.000510 | 0.050461 ± 0.003633 | 0.025635 ± 0.000931 | 0.006621 ± 0.002629 |
| EuroSAT | Panopticon | full_finetune | 0.962663 ± 0.007317 | 0.960747 ± 0.008207 | 0.112616 ± 0.037419 | 0.055755 ± 0.013622 | 0.009453 ± 0.005022 |
| TreeSatAI | DOFA | frozen | 0.256667 ± 0.001528 | 0.223165 ± 0.003694 | 0.255844 ± 0.000742 | 0.073981 ± 0.000202 | 0.010807 ± 0.000883 |
| TreeSatAI | DOFA | full_finetune | 0.266833 ± 0.002754 | 0.243823 ± 0.012392 | 0.246859 ± 0.003015 | 0.071690 ± 0.000975 | 0.012555 ± 0.003901 |
| TreeSatAI | Panopticon | frozen | 0.237833 ± 0.002887 | 0.210847 ± 0.005483 | 0.256998 ± 0.002091 | 0.075375 ± 0.000618 | 0.009496 ± 0.000896 |
| TreeSatAI | Panopticon | full_finetune | 0.251500 ± 0.031277 | 0.219752 ± 0.024360 | 0.250458 ± 0.004507 | 0.073186 ± 0.001611 | 0.007630 ± 0.003557 |

TreeSatAI is a 15-label multilabel task. Its reported accuracy is strict sample-level exact match at threshold 0.5, not labelwise accuracy. Macro F1 is the unweighted mean of positive-class F1 over labels. NLL and Brier are averaged over sample-label binary decisions. ECE-15 flattens binary decisions and uses confidence `max(p, 1-p)`. These values must not be interpreted using EuroSAT's multiclass metric conventions.

## Artifact completeness

Every new run has:

- `best.pt` and `last.pt` with verified loadability and metadata;
- a schema-v2 deterministic test export at `predictions/test/deterministic/predictions.parquet`;
- a representation sidecar at `predictions/test/deterministic/embeddings.npz`;
- sample IDs, labels, logits, probabilities, predictions, correctness, confidence, predictive entropy, and backbone representations;
- exact official test sample order and labels;
- finite logits, probabilities, metrics, and representations;
- a prediction manifest bound to the byte hash of `best.pt`;
- recorded and verified dataset, split/manifest, pretrained-weight, training-protocol, environment, and code-snapshot provenance.

The six EuroSAT exports each contain 2,714 samples, 10 output logits/probabilities, and a `2714 × 768` representation array. The twelve TreeSatAI exports each contain 2,000 samples, 15 output logits/probabilities, and a `2000 × 768` representation array.

All 18 live checkpoint hashes, Parquets, and representation sidecars were rechecked after the final audit. No checked artifact is newer than the audit.

## Learning-curve audit

“Best/stop” lists the NLL-selected epoch and the last completed epoch for seeds 42/43/44.

| Dataset | Model | Adaptation | Best epochs | Stop epochs | Finding |
|---|---|---|---|---|---|
| EuroSAT | Panopticon | frozen | 10 / 6 / 9 | 20 / 16 / 19 | Stable across seeds; mild post-best deterioration only. |
| EuroSAT | Panopticon | full_finetune | 1 / 1 / 2 | 16 / 16 / 17 | Early optimum followed by optimization instability and catastrophic degradation under the frozen LR schedule. |
| TreeSatAI | DOFA | frozen | 27 / 14 / 13 | 37 / 24 / 23 | Stable plateau; shallow post-best deterioration. |
| TreeSatAI | DOFA | full_finetune | 5 / 4 / 4 | 20 / 19 / 19 | Severe post-best NLL/calibration overfitting. |
| TreeSatAI | Panopticon | frozen | 9 / 12 / 15 | 19 / 22 / 25 | Stable plateau; shallow post-best deterioration. |
| TreeSatAI | Panopticon | full_finetune | 3 / 2 / 1 | 18 / 17 / 16 | Severe post-best NLL/calibration overfitting and material seed variability. |

Every run stopped exactly according to the frozen rule: ten non-improving epochs for frozen adaptation and fifteen for full fine-tuning. There is no early-stopping off-by-one error. All required epoch/batch values and gradient norms are finite. Frozen backbone gradients are zero; all intended full-finetune backbone and head tensors received finite, nonzero gradients in the parameter-level audit.

All nine full-finetune checkpoints were selected by epoch 5; eight were selected before the five-epoch warmup completed. This makes the full-finetune results strongly schedule-dominated and must be disclosed in thesis interpretation. It does not make the recorded, validation-selected checkpoints invalid.

For TreeSatAI full fine-tuning, final validation NLL is 2.33–2.94 times the selected minimum while train exact-match reaches 0.890–0.997. For EuroSAT full fine-tuning, training loss and accuracy themselves often degrade after the early optimum; this is better described as optimization instability/catastrophic forgetting than ordinary overfitting. EuroSAT full runs also contain isolated but finite batch global-gradient spikes with clipping disabled. No NaN, Inf, missing-gradient, zero-gradient, OOM, or checkpoint corruption occurred.

## Resume provenance

EuroSAT / Panopticon / frozen seed 43 was interrupted after epoch 5 when training was paused for a network change. It resumed in the same run from the last complete epoch boundary. The original epoch-5 checkpoint was archived, original provenance was preserved byte-for-byte, a separate resume code snapshot/event was appended, and the train-loader generator was deterministically reconstructed through five completed sampler epochs. The audit verified continuous epochs and global steps, the sampler-state SHA, unchanged epoch-1 gradient audit, and the complete resume chain. It is a valid independent final run and was not retrained from scratch.

## Immutable DOFA-EuroSAT verification

`reports/dofa_eurosat_final_manifest.json` remains mode `0444` with SHA256 `bdea21e9901219b6fa6d587ef41917ad781ee9eaf076f95a0ad2c09178c77cd1`.

- All six checkpoint hashes, six prediction hashes, and six config hashes still match.
- All six run trees contain no write-bit entries.
- The recursive metadata fingerprint was unchanged before and after validation.
- The safe module-form validation command returned `validated: true` without training or inference.

## Reporting caveats and inconsistencies

These issues do not invalidate the 18 checkpoints, but they must be handled correctly in thesis reporting:

1. The current dashboard labels TreeSatAI loss as cross-entropy/CE even though the multilabel implementation uses binary cross-entropy with logits.
2. TreeSatAI dashboard accuracy is strict exact match; labelwise accuracy and macro F1 exist in `training_history.json` but are omitted from the dashboard/CSV fields.
3. TreeSatAI predictive entropy is summed over 15 labels while confidence is averaged. Plotting both on one axis, or comparing that entropy directly with EuroSAT categorical entropy, is not scientifically meaningful.
4. Audit fields named `best_validation_accuracy` and `best_validation_macro_f1` mean the metric value at the NLL-selected epoch, not the maximum of that metric's curve.
5. The stored overfitting heuristic is not a reliable scientific classifier. It flags 15/18 runs, calls unstable EuroSAT full-finetune trajectories “overfitting,” and misses mild deterioration in all three EuroSAT-frozen curves. The raw histories and NLL-selected checkpoint rule are authoritative.
6. Frozen histories use `NaN` for the non-applicable backbone learning rate. This is an N/A sentinel, not numerical divergence, but strict JSON parsers may reject it.
7. The immutable historical DOFA-EuroSAT runs and the new Panopticon-EuroSAT runs retain their already frozen normalization provenance difference. The historical DOFA runs use the repository Sentinel-2 fallback constants, while Panopticon uses final-train-only RGB DN statistics. This is a cross-model comparison caveat; historical runs were deliberately not altered or retrained.
8. Direct execution of `scripts/build_dofa_eurosat_ensembles.py` does not resolve project imports; the supported module form `python -m scripts.build_dofa_eurosat_ensembles` works.

## Verification commands and status

- Final artifact gate: `python -m scripts.audit_c1_classification_runs --seeds 42 43 44 --fail-on-nonpromotable`
  - Result: 18 promotable, 0 blocked, 0 missing, 0 ambiguous.
- Immutable-run validation: `python -m scripts.build_dofa_eurosat_ensembles --project-root /workspace --validate-only`
  - Result: validated true; no immutable metadata changes.
- Regression suite: `python -m unittest discover -s tests -v`
  - Result: 58 tests passed.
- GPU process check after completion: no active NVIDIA compute process.

