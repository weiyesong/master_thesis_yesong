# MC Dropout Protocol

Created: 2026-08-25T22:48:06.191548+00:00

Status: frozen prospectively from `reports/pre_uq_protocol_freeze.md`.

## Method scope

The thesis method is **downstream-head/decoder MC Dropout**. DOFA and Panopticon backbone architectures, native dropout, DropPath, and stochastic depth are not altered or enabled. Classification uses one `Dropout(p=0.10)` immediately before the final linear classifier (after EuroSAT's fixed BatchNorm). Segmentation uses one `Dropout2d(p=0.10)` on the common decoder's final feature map immediately before its 1×1 classifier. Placement is identical across backbone families within each task.

The deterministic comparator retains its original zero-dropout head/decoder. The MC model is newly trained with the one designated dropout layer; a deterministic checkpoint is never converted by inference-only dropout insertion.

## Training and checkpoint selection

- Classification: minimum validation NLL; earliest strict tie.
- Segmentation: maximum validation mIoU; earliest strict tie.
- Validation used for checkpoint selection is deterministic with dropout disabled.
- Test metrics, calibration, and uncertainty never participate in selection.

## Stochastic inference

- Freeze model weights; call full-model evaluation mode, then activate exactly the one designated downstream dropout module.
- BatchNorm remains in evaluation mode and its running buffers must be unchanged.
- Use `torch.no_grad()` and identical preprocessed input tensors for all passes.
- Run exactly `T=30` passes for final experiments. Arithmetic-mean probabilities—not logits or decisions—are predictive probabilities.
- Preserve predictive entropy, expected entropy, their difference as MI-style disagreement, and per-class/per-pixel probability variance.

## Pass-count validation rule

Pilots evaluate nested prefixes T=10/20/30/50 on validation only. T=30 is accepted relative to T=50 only when probability MAE ≤0.005, maximum absolute probability difference ≤0.05, each requested metric changes by ≤0.01, and each mean uncertainty summary changes by ≤0.01. Failure blocks the full launch; thresholds are not adjusted after results.

## Output schema

Classification saves `[N,T,C]` raw logits/probabilities plus sample ID, label, probability-mean prediction, predictive entropy, expected entropy, MI-style disagreement, probability variance `[N,C]`, and backbone representation.

Segmentation saves full-test probability-mean maps, prediction/confidence maps, predictive entropy, expected entropy, disagreement, and variance. Full `[N_subset,T,C,H,W]` stochastic probabilities are retained only for the predeclared research subset when full storage is excessive. The exact final CloudSEN12 and SpaceNet7 test IDs were fixed by label/prediction-independent SHA256 ranking before pilot stochastic inference in `reports/mc_dropout_segmentation_research_subset_ids.json`.

## Replication

The frozen core is 24 new runs: seed 42 for all 16 cells, plus seeds 43/44 only for EuroSAT/DOFA/frozen, TreeSatAI/Panopticon/full-finetune, CloudSEN12/Panopticon/frozen, and SpaceNet7/DOFA/full-finetune. MC passes are repeated posterior-style draws, not independent training replicates.

## Frozen values

| Quantity | Value |
|---|---|
| Dropout probability | 0.10 for both tasks |
| Classification placement | after optional fixed BatchNorm, before final Linear |
| Segmentation placement | final decoder feature map, before 1×1 classifier |
| Backbone stochasticity | disabled during MC inference |
| Final stochastic passes | 30 |
| Calibration bins | 15 |
| Aggregation | arithmetic mean of probabilities |

Pilot inference code snapshot: `sha256:a9222f99461259165c6a145b161c32cf91cfcf25a4ae4c886ec01491ea6b7d04` (103 files).
