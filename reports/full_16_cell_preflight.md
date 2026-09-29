# Full 16-cell deterministic preflight

**Outcome:** 16/16 cells are **READY**; 0 are **BLOCKED**.

The final evidence run used CUDA with PyTorch 2.5.1+cu124 and strict deterministic algorithms (`warn_only: false`). Each cell consumed one training batch and one validation batch, performed exactly one optimizer update, and then released the model. No epoch loop and no final training were performed. The machine-readable source of truth is [`preflight_results.json`](../results/preflight/full_16_cell_20260812T192427Z/preflight_results.json).

## Acceptance table

`PASS` means the check completed with real downloaded benchmark data, the real DOFA or Panopticon pretrained backbone, and no mock model. The validation numbers below only prove that metric code executed on a tiny, essentially untrained validation batch; they are not thesis performance results.

| Task | Dataset | Model | Adaptation | Dataloader | Metadata | Forward | Loss | Gradients | Validation metric | Checkpoint writing | Prediction export | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Classification | EuroSAT | DOFA | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (accuracy 0.0000) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Classification | EuroSAT | DOFA | full_finetune | PASS | PASS | PASS | PASS | PASS (169 backbone tensors) | PASS (accuracy 0.0000) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Classification | EuroSAT | Panopticon | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (accuracy 0.0000) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Classification | EuroSAT | Panopticon | full_finetune | PASS | PASS | PASS | PASS | PASS (179 backbone tensors) | PASS (accuracy 0.0000) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Classification | TreeSatAI | DOFA | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (macro-F1 0.0000) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Classification | TreeSatAI | DOFA | full_finetune | PASS | PASS | PASS | PASS | PASS (169 backbone tensors) | PASS (macro-F1 0.0667) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Classification | TreeSatAI | Panopticon | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (macro-F1 0.0000) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Classification | TreeSatAI | Panopticon | full_finetune | PASS | PASS | PASS | PASS | PASS (179 backbone tensors) | PASS (macro-F1 0.0444) | PASS (strict reload) | PASS (2 samples) | **READY** |
| Segmentation | CloudSEN12 | DOFA | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (mIoU 0.0981) | PASS (strict reload) | PASS (1 sample) | **READY** |
| Segmentation | CloudSEN12 | DOFA | full_finetune | PASS | PASS | PASS | PASS | PASS (169 backbone tensors) | PASS (mIoU 0.0915) | PASS (strict reload) | PASS (1 sample) | **READY** |
| Segmentation | CloudSEN12 | Panopticon | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (mIoU 0.0570) | PASS (strict reload) | PASS (1 sample) | **READY** |
| Segmentation | CloudSEN12 | Panopticon | full_finetune | PASS | PASS | PASS | PASS | PASS (179 backbone tensors) | PASS (mIoU 0.0527) | PASS (strict reload) | PASS (1 sample) | **READY** |
| Segmentation | SpaceNet7 | DOFA | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (mIoU 0.2519) | PASS (strict reload) | PASS (1 sample) | **READY** |
| Segmentation | SpaceNet7 | DOFA | full_finetune | PASS | PASS | PASS | PASS | PASS (169 backbone tensors) | PASS (mIoU 0.2527) | PASS (strict reload) | PASS (1 sample) | **READY** |
| Segmentation | SpaceNet7 | Panopticon | frozen | PASS | PASS | PASS | PASS | PASS (backbone frozen) | PASS (mIoU 0.1797) | PASS (strict reload) | PASS (1 sample) | **READY** |
| Segmentation | SpaceNet7 | Panopticon | full_finetune | PASS | PASS | PASS | PASS | PASS (179 backbone tensors) | PASS (mIoU 0.1920) | PASS (strict reload) | PASS (1 sample) | **READY** |

## What was verified

- **Dataloader:** official manifests were opened and yielded real samples. Observed split sizes were EuroSAT 18,866/2,707/2,713/2,714 (train/validation/calibration/test), TreeSatAI 4,000/1,000/2,000, CloudSEN12 4,000/535/975, and SpaceNet7 3,500/652/1,152.
- **Metadata:** sample IDs, tensor shapes, labels/masks, band order, and canonical nanometre wavelengths were checked. TreeSatAI batches followed the shared timestamp-wise encoder contract and included a temporal mask.
- **Forward and loss:** all logits and losses were finite. Shapes were `[2,10]` for EuroSAT, `[2,15]` for TreeSatAI, `[1,4,224,224]` for CloudSEN12, and `[1,2,224,224]` for SpaceNet7.
- **Gradients:** every frozen cell had zero trainable backbone tensors and no backbone gradients. Every full-finetune cell had present, finite, nonzero gradients in all intended trainable backbone and head tensors.
- **Validation metrics:** classification metrics and segmentation mIoU, per-class IoU, pixel accuracy, NLL, Brier, and ECE-15 executed. SpaceNet7 foreground, classwise, and boundary calibration also executed.
- **Checkpoint writing:** every cell wrote a temporary checkpoint, computed its SHA256, loaded it back with strict state-dict validation, and then removed it. No preflight checkpoint was retained or promoted to a thesis result.
- **Prediction export:** classification exports contain the required per-sample fields plus backbone representations in Parquet/NPZ form. Segmentation exports contain masks, logits, probabilities, predictions, correctness, confidence, entropy, valid masks, and sample IDs in compressed NPZ form. Every export passed its schema and numerical round-trip validator.

## Determinism issue resolved during preflight

The first segmentation attempt exposed that PyTorch's CUDA spatial NLL kernel is nondeterministic under strict mode. The common segmentation loss now flattens spatial pixels before cross-entropy. This is mathematically the same mean per-pixel objective, including ignore-index handling, but selects PyTorch's deterministic two-dimensional implementation. A regression test confirms equality with the spatial formulation. The final 16-cell evidence run completed with strict deterministic algorithms and no warn-only exceptions.

## Artifacts and limitations

- Final evidence root: [`full_16_cell_20260812T192427Z`](../results/preflight/full_16_cell_20260812T192427Z/)
- Per-cell `summary.json` files record all checks, gradient audits, temporary-checkpoint hashes, timing, and peak GPU memory.
- Per-cell prediction exports are retained only as preflight evidence.
- No reported metric is suitable for a thesis results table because each model received only one optimizer update and was evaluated on one tiny validation batch.
- No final training was launched.
