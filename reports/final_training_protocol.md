# Final training protocol

**Protocol ID:** `B5-2026-08-15-v1`  
**Status:** FROZEN FOR FUTURE FINAL RUNS  
**Evidence base:** successful 16-cell real-model preflight in [`full_16_cell_preflight.md`](full_16_cell_preflight.md) and the immutable DOFA–EuroSAT registry in [`dofa_eurosat_final_manifest.json`](dofa_eurosat_final_manifest.json).

This document is the authoritative training protocol for every new deterministic thesis run. A setting may not be changed silently after inspecting validation or test results. Any necessary change requires a dated protocol revision that states the reason and applies to every affected model comparison. No final training was launched while creating this document.

## 1. Scope and immutable exception

The final matrix has 16 dataset × model × adaptation cells and three independent seeds per cell (`42`, `43`, `44`). The following six runs already fill the two DOFA–EuroSAT cells and are grandfathered final artifacts:

- DOFA / EuroSAT / frozen / seeds 42, 43, 44
- DOFA / EuroSAT / full fine-tuning / seeds 42, 43, 44

Their read-only run directories, checkpoints, predictions, resolved configurations, hashes, and metadata must not be edited, resumed, replaced, or retrained. Their exact allowlist is the final manifest above. `best.pt`, selected by validation NLL, is authoritative; `last.pt` is provenance only.

The remaining 14 cells require 42 new final runs. All three seeds are reported; a seed must never be selected or discarded because of its test performance. The same three checkpoints become the deterministic members of the corresponding Deep Ensemble.

## 2. Frozen optimization matrix for new runs

All batch sizes below are physical per-step batch sizes on one GPU, not nominal effective sizes. Gradient accumulation, mixed precision, automatic LR scaling, and model-specific automatic batch fallback are disabled.

| Dataset/task | Model(s) | Adaptation | Optimizer and weight decay | Backbone LR | Head/decoder LR | LR policy | Batch | Epoch cap | Early stopping | Best checkpoint |
|---|---|---|---|---:|---:|---|---:|---:|---|---|
| EuroSAT classification | Panopticon | frozen | AdamW, WD `0.01` | not trainable | `1e-3` | constant | 64 | 50 | patience 10 | minimum validation NLL |
| EuroSAT classification | Panopticon | full fine-tuning | AdamW, WD `0.0` | `4e-4` | `4e-3` | 5-epoch linear warm-up, then constant | 64 | 100 | patience 15 | minimum validation NLL |
| TreeSatAI multilabel classification | DOFA and Panopticon | frozen | AdamW, WD `0.01` | not trainable | `1e-3` | constant | 64 | 50 | patience 10 | minimum validation NLL |
| TreeSatAI multilabel classification | DOFA and Panopticon | full fine-tuning | AdamW, WD `0.01` | `1e-4` | `1e-3` | 5-epoch linear warm-up, then constant | 64 | 100 | patience 15 | minimum validation NLL |
| CloudSEN12 segmentation | DOFA and Panopticon | frozen | AdamW, WD `0.01` | not trainable | `1e-3` | constant | 8 | 50 | patience 10 | maximum validation mIoU |
| CloudSEN12 segmentation | DOFA and Panopticon | full fine-tuning | AdamW, WD `0.01` | `1e-4` | `1e-3` | constant | 8 | 50 | patience 15 | maximum validation mIoU |
| SpaceNet7 segmentation | DOFA and Panopticon | frozen | AdamW, WD `0.01` | not trainable | `1e-3` | constant | 8 | 50 | patience 10 | maximum validation mIoU |
| SpaceNet7 segmentation | DOFA and Panopticon | full fine-tuning | AdamW, WD `0.01` | `1e-4` | `1e-3` | constant | 8 | 50 | patience 15 | maximum validation mIoU |

### Optimizer definition

- PyTorch `AdamW` is used with beta values `(0.9, 0.999)`, epsilon `1e-8`, `amsgrad=false`, and the table's weight decay.
- The configured weight decay applies uniformly to every trainable tensor; there is no bias or normalization-parameter exemption.
- Frozen runs instantiate only a head/decoder optimizer group. The backbone remains non-trainable and in the already validated frozen-mode policy.
- Full fine-tuning uses separate backbone and head/decoder groups with the table's learning rates. DOFA's fixed sinusoidal positional embedding and Panopticon's inactive SAR-only channel embeddings remain the already validated structural exceptions, not adaptation freezes.
- No gradient clipping and no layer-wise LR decay are used. Training is FP32 under strict deterministic mode.

### Learning-rate schedule

- “Constant” means the base group LR from epoch 1 through stopping; there is no cosine, step, or plateau decay.
- Five-epoch linear warm-up uses epoch factors `0.2`, `0.4`, `0.6`, `0.8`, and `1.0`, followed by the constant base LR.
- EuroSAT Panopticon inherits the exact optimization schedule of the immutable DOFA comparator. It is not independently tuned by default.

## 3. Batch-size policy

- Classification uses batch 64 for both models and both adaptation regimes. TreeSatAI's official artifact currently has one available timestamp, processed through the common timestamp-wise encoder and mean-representation path.
- Segmentation uses batch 8 for both models and both adaptation regimes.
- A lower batch for one model is not allowed merely because it is more convenient. B4 validated the complete computation paths at batch 2 for classification and batch 1 for segmentation; it did **not** validate these final batch capacities.
- Before the first final run, the exact final batch must pass a memory-only forward/backward checkpoint smoke test for both models. An out-of-memory result is a protocol blocker, not permission for a silent per-model change. Any revised shared batch size must be documented in a new protocol version before any affected final seed is trained.

## 4. Epoch limit, early stopping, and checkpoint selection

- Validation is run once after each complete epoch.
- Early stopping monitors the same quantity used to write `best.pt`, with `min_delta=0.0`. Only a strict improvement resets patience; an exact tie retains the earlier checkpoint.
- Patience counts consecutive completed validation epochs without improvement, including warm-up epochs.
- Classification checkpoints are selected by minimum full-validation NLL. For TreeSatAI this is multilabel binary cross-entropy with logits averaged over sample-label entries. Accuracy and Macro-F1 are reported but cannot select a checkpoint.
- Segmentation checkpoints are selected by maximum full-validation mIoU with the dataset's fixed ignore handling. Pixel accuracy, NLL, Brier, ECE-15, and SpaceNet7 foreground/boundary calibration are reported but cannot select a checkpoint.
- The epoch cap is a hard maximum, not a target that overrides early stopping.
- `last.pt` may support interruption recovery but can never replace validation-selected `best.pt` in final evaluation. Resume must restore optimizer, epoch, best metric, best epoch, and the accumulated early-stopping counter exactly.

Validation-NLL selection for classification preserves the historical DOFA–EuroSAT rule and aligns checkpoint selection with the calibration focus. Segmentation retains mIoU selection so calibration metrics are outcomes rather than optimization targets.

## 5. Augmentation and deterministic preprocessing

No stochastic data augmentation is used for any dataset or split. This means no random crop, flip, rotation, color/spectral jitter, mixup, CutMix, or test-time augmentation. Validation, calibration, and test preprocessing is deterministic.

- EuroSAT RGB images use the full 64×64 sample resized bilinearly to 224×224.
- TreeSatAI applies identical deterministic preprocessing to every timestamp, encodes timestamps with the same backbone, and means valid timestamp representations before the common classification head.
- CloudSEN12 and SpaceNet7 images are resized bilinearly from 512×512 to 224×224. Masks use nearest-neighbor resizing only. SpaceNet7 remains a static per-sample task.
- The classification head remains conceptually identical across backends within a dataset: non-affine BatchNorm-plus-linear for EuroSAT and a zero-dropout linear head for TreeSatAI. Both segmentation backends use the same common UNet-style decoder `[256,128,64,32]` with no added dropout.

The absence of augmentation is deliberate: it matches the immutable EuroSAT training protocol and the computation paths validated by B4, and avoids introducing an unvalidated model-dependent stochastic transformation.

## 6. Frozen normalization

Normalization is channelwise z-score normalization, `(x - mean) / std`, using fixed constants established before final training. Constants cannot be recomputed per seed. Within every new paired DOFA/Panopticon comparison, both backends receive the same normalized input tensor.

| Dataset | Channels | Fixed normalization source |
|---|---|---|
| EuroSAT, new Panopticon runs | B04, B03, B02 | Final-train-only raw-DN statistics: mean `[936.085209866211, 1031.3388562784282, 1111.4795678522003]`, std `[589.387623763833, 388.3087839027959, 327.14241012761033]` |
| TreeSatAI | B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09 | Pinned GEO-Bench-2 training statistics: mean `[245.31068420410156, 387.63568115234375, 248.4667205810547, 2825.93603515625, 625.9300537109375, 2118.83740234375, 2709.37890625, 2982.208740234375, 1316.7186279296875, 594.203369140625, 265.8070068359375, 2962.182373046875]`; std `[117.73491668701172, 130.0995635986328, 129.66375732421875, 756.8175659179688, 191.35238647460938, 517.2822265625, 691.1488037109375, 754.9419555664062, 411.339111328125, 234.48863220214844, 125.9928207397461, 674.169189453125]` |
| CloudSEN12 | B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12 | Pinned GEO-Bench-2 training statistics: mean `[2030.244384765625, 2074.817138671875, 2209.807373046875, 2247.927490234375, 2589.593505859375, 3103.521240234375, 3277.909423828125, 3331.6318359375, 3377.544677734375, 4038.193115234375, 2448.748046875, 1907.728515625]`; std `[2723.43603515625, 2691.302734375, 2539.91357421875, 2538.520751953125, 2504.328369140625, 2241.74462890625, 2145.667724609375, 2176.997802734375, 2066.763671875, 3083.179931640625, 1595.065185546875, 1474.11767578125]` |
| SpaceNet7 | red, green, blue | Pinned GEO-Bench-2 training statistics: mean `[116.94474029541016, 103.55889129638672, 76.77427673339844]`, std `[61.655845642089844, 49.64897537231445, 45.88066864013672]` |

The six immutable DOFA–EuroSAT runs instead used historical raw-DN RGB constants, mean `[1136.89, 1120.77, 1184.39]` and std `[965.23, 712.12, 650.20]`. Their derivation provenance is not preserved. Those runs must not be altered, and this unavoidable normalization difference from new Panopticon–EuroSAT runs must be disclosed in every cross-model interpretation; it must not be hidden by retraining DOFA.

Normalization is dataset preprocessing. Spectral metadata remains canonical in nm, converted to µm only at the DOFA adapter and retained in nm at the Panopticon adapter.

## 7. Validation-only learning-rate contingency

Default action is **no LR tuning**. A lower validation score than the other model is not a reason to tune. The contingency below may be opened only for objective optimizer instability—non-finite loss/gradients, repeat divergence, or failure to produce a valid validation checkpoint—documented before any test access.

EuroSAT is excluded from model-specific tuning because its DOFA comparator is already immutable. Panopticon uses the table's inherited settings. If those settings are objectively unstable, the EuroSAT comparison is paused and requires a protocol amendment; the DOFA artifacts remain untouched.

For TreeSatAI, CloudSEN12, or SpaceNet7, an affected dataset × adaptation pair receives exactly the same tuning budget for DOFA and Panopticon:

1. Three pilot attempts per model, all with seed 42 and the frozen batch/preprocessing protocol.
2. LR multipliers `{1/3, 1, 3}` applied to every active LR group while preserving the backbone-to-head/decoder ratio.
3. Exactly 10 epochs per attempt, with no early termination and no replacement attempt if a candidate fails.
4. No test dataloader, test metric, test prediction, or test export is permitted. Pilot outputs are labelled `PILOT_ONLY` and cannot become ensemble members or final checkpoints.
5. Select each model's multiplier by its best validation NLL (classification) or best validation mIoU (segmentation) within the 10 epochs. An exact tie prefers multiplier `1`, then `1/3`, then `3`.
6. Record the complete candidate table, including failed candidates. Then train fresh final runs for seeds 42, 43, and 44 with the selected setting.

No additional candidates, seeds, epochs, or post-test adjustments are allowed. Opening this contingency for one model automatically spends the same three-attempt budget on the other model in that dataset × adaptation pair.

## 8. Split isolation and test embargo

- Training updates use only the official training split.
- LR contingency, early stopping, and checkpoint selection use only the official validation split.
- EuroSAT calibration samples are not used for optimization, early stopping, or checkpoint selection. Any later Temperature Scaling uses calibration data under its separate protocol.
- Test labels, predictions, and metrics remain inaccessible until preprocessing, LR, epoch policy, and checkpoint-selection rules are locked and the final training run has ended.
- After loading `best.pt`, deterministic test evaluation is performed once per final seed for artifact generation. Test outcomes cannot trigger retraining, a different seed, a different checkpoint, or a protocol amendment.

## 9. Reproducibility and required artifacts

- Final seeds are exactly `42`, `43`, and `44`; each begins from the same official pretrained-weight artifact and a freshly initialized downstream head/decoder.
- Strict deterministic execution remains enabled with `warn_only=false`.
- Each run records the resolved configuration, dataset/manifest hashes, pretrained-weight identity, code snapshot/hash, environment, model and optimizer-group audits, epoch history, best/last checkpoint metadata, and final prediction export.
- Model summaries are mean and standard deviation across all three seeds. Ensemble aggregation uses all three corresponding final members; no member is chosen by test performance.

## 10. Pre-launch gates

The protocol is frozen, but the current final-run configuration/code is not yet fully compliant. These are engineering gates before any final run, not invitations to change the scientific settings:

1. Reconcile `configs/treesatai_classification.yaml`, whose current epoch-1/batch-2/seed-42 values are smoke-test settings, with this protocol.
2. Implement and smoke-test segmentation early stopping with the exact patience/counter semantics above. The current segmentation loop writes the best checkpoint but ignores `early_stopping`.
3. Add a true validation-only execution mode before any LR contingency. The current segmentation runner evaluates test after training and therefore cannot be used for tuning.
4. Persist and restore the classification early-stopping counter during resume, or forbid resume. Resetting patience after interruption would change the frozen stopping rule.
5. Exercise the final batch capacities as a memory-only smoke check; do not launch an epoch or alter batch size automatically.
6. Wire the already validated segmentation prediction exporter into the final segmentation run path and re-smoke checkpoint-to-export execution.
7. Verify every future resolved config and code snapshot against this document before launch. Historical DOFA–EuroSAT metadata remains untouched.

Passing these gates authorizes later final training; creating this protocol does not. No final run, LR pilot, test evaluation, calibration fitting, or new inference was performed for B5.
