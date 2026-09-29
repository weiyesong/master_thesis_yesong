# Pre-UQ Protocol Freeze — Final Version

**Freeze date:** 2026-08-25  
**Status:** FINAL for the thesis experiments covered here  
**Scope:** future MC Dropout work and final reporting; no deterministic artifact is changed

## Executive decision

The completed deterministic benchmark remains valid. The checkpoint-sensitivity audit found substantial counterfactual changes in some runs, but it did **not** identify a severe scientific defect in the historical validation-only policies. The observed pattern is scientifically interpretable: classification NLL selection favors probabilistic fit, whereas segmentation mIoU selection protects task performance and, on SpaceNet7, avoids background-dominated early checkpoints with almost no useful building segmentation.

The final policy is therefore:

| Experiment family | Frozen checkpoint criterion |
|---|---|
| Classification deterministic | Minimum validation NLL |
| Classification MC Dropout | Minimum validation NLL |
| Segmentation deterministic | Maximum validation mIoU |
| Segmentation MC Dropout | Maximum validation mIoU |

Ties use the earliest epoch. Checkpoint selection is validation-only. Test labels, test metrics, MC uncertainty results, and downstream calibration results must never be used to select or revisit a checkpoint.

**No completed deterministic experiment requires retraining, repeat inference, or checkpoint replacement.** The checkpoint audit is not a blocker to within-task UQ work. It is a blocker to strong causal interpretation of raw classification-versus-segmentation calibration differences, so those claims are prohibited below.

No MC Dropout training is authorized by the creation of this report itself, and none was started during this freeze.

## Evidence reviewed and precedence

This decision was made after reviewing the available final and historical evidence, including:

- `reports/checkpoint_selection_sensitivity.md`;
- the final C1 deterministic completion report and all C1 promotion-audit reports;
- the C2 execution/completion record and all C2 promotion-audit reports, including the final 24/24 promotion audit;
- `reports/c3_temperature_scaling_and_ensembles.md`, `reports/c3_treesatai_temperature_scaling.md`, and the C3 classification and segmentation result tables;
- the frozen dataset/training protocols for EuroSAT, TreeSatAI, CloudSEN12, and SpaceNet7; and
- `reports/code_audit.md` and the current MC collection/model code.

Where an earlier report conflicts with this document on a reporting choice, this dated freeze takes precedence. Earlier artifacts remain immutable provenance. In particular, the existing TreeSatAI validation-fitted Temperature Scaling artifacts are retained as historical/diagnostic artifacts but are excluded from final thesis results under the rule below.

The early repository-wide code audit predates completion of much of the deterministic matrix. Its old completion counts are superseded by the final C1/C2 audits, while its methodological warning that standard MC Dropout requires training-time non-zero dropout remains applicable.

## 1. Checkpoint-selection policy

### 1.1 Interpretation of the sensitivity audit

The audit reconstructed validation-level alternative selection for all 48 promoted deterministic runs without retraining or inference. It found:

- a changed epoch in 43/48 runs;
- 26/48 MATERIAL or SEVERE ratings under its descriptive per-run rubric;
- 20/24 changed classification epochs, with the six TreeSatAI full-finetune task-metric alternatives rated SEVERE; and
- 23/24 changed segmentation epochs, with systematic SpaceNet7 sensitivity and all pixel-NLL minima occurring at epochs 1–3.

A per-run **SEVERE sensitivity rating is not, by itself, a severe scientific defect**. It says that a counterfactual policy would select a meaningfully different operating point. Here the direction of the changes supports the task-specific policy rather than invalidating it:

- TreeSatAI full-finetune Macro-F1 maxima occur later but have very large NLL and ECE increases. Minimum NLL is consistent with the predeclared classification objective and avoids the documented post-best probabilistic instability.
- SpaceNet7 minimum pixel-NLL checkpoints gain background-dominated aggregate likelihood/accuracy while sharply reducing building IoU. Selecting them would make the nominally uniform probabilistic rule scientifically misleading for the foreground task.
- Every retained selector is validation-only, and the final C1 and C2 promotion audits found no artifact defect that invalidates a promoted checkpoint.

Accordingly, there is no checkpoint-selection blocker for within-task deterministic-versus-MC comparisons. The material confound applies to absolute cross-task calibration interpretation and is handled by the claim restrictions in Section 6.

### 1.2 Locked use of checkpoints

- Historical deterministic `best.pt` artifacts remain authoritative and must not be replaced or relabeled.
- Future MC Dropout models are new dropout-enabled training runs. They start from the same official pretrained foundation weights and a fresh downstream head/decoder under the corresponding deterministic cell protocol; an existing dropout-zero deterministic checkpoint must not be converted into an MC model by changing dropout only at inference.
- Classification MC Dropout validation uses minimum NLL; segmentation MC Dropout validation uses maximum mIoU. Dropout is disabled during validation used for checkpoint selection, making the selector deterministic and directly aligned with its deterministic comparator.
- Within a task, deterministic and MC Dropout comparisons therefore share the same split, primary preprocessing/training protocol, and selection principle. The intended dropout mechanism is the experimental difference.
- Historical counterfactual epochs that were not retained must not be recreated. The sensitivity values remain an audit result, not a request for retroactive checkpoints.

## 2. Temperature Scaling policy

### 2.1 TreeSatAI

**Final decision: TreeSatAI Temperature Scaling is `N/A` and is not reported as a thesis result.** It is not a failed experiment.

Reason: the completed TreeSatAI benchmark has no independent calibration split. Its official validation split was already used for checkpoint selection. Reusing that same split to fit a temperature would conflate selection and calibration, while the test labels are embargoed from model fitting and protocol decisions.

Therefore:

- do not fit or refit a final TreeSatAI temperature;
- do not use TreeSatAI test labels for temperature fitting or model selection;
- do not retrain TreeSatAI merely to manufacture a new calibration split;
- retain existing validation-fitted temperature artifacts unchanged for provenance/diagnostic purposes only;
- exclude those historical values from final comparative tables and replace the method entry with `N/A — no independent calibration split under the completed benchmark protocol`; and
- do not count the `N/A` entry as a failure, missing run, or zero-valued result.

### 2.2 EuroSAT

EuroSAT Temperature Scaling remains valid because its fixed protocol includes a dedicated calibration split that is disjoint from training, validation/checkpoint selection, and test. Existing EuroSAT temperatures/results may be reported under the established C3 protocol. They are not affected by the TreeSatAI decision.

## 3. SpaceNet7 calibration reporting

Every deterministic, MC Dropout, and applicable ensemble SpaceNet7 result must report at least the following, using the same valid-pixel/ignore mask and the fixed calibration-bin convention:

| Required quantity | Frozen interpretation |
|---|---|
| Overall pixel ECE | Top-label ECE across all valid pixels |
| Foreground/building ECE | Binary calibration of building probability against the building indicator |
| Classwise calibration | One-vs-rest calibration for both `background` and `building`; do not report only their average |
| Foreground NLL | Binary building-vs-not-building NLL over valid pixels |
| Foreground Brier | Binary squared error of building probability over valid pixels |
| Boundary ECE | Top-label ECE on valid boundary pixels; boundary foreground ECE should also be retained/reported when available |

Boundary handling is immutable:

- A per-image boundary ECE with zero valid boundary pixels is `NaN`/undefined.
- It must never be replaced with zero, because zero would falsely encode perfect calibration.
- Per-image summaries exclude undefined values and report the number of images contributing a valid boundary value.
- Dataset-level boundary ECE is computed by pooling calibration-bin counts/confidence/outcomes from **valid boundary pixels only**. It is not an unweighted mean of per-image ECE values, and zero-boundary images contribute no pixels.
- Ignore-index pixels never enter the boundary or non-boundary calibration denominators.

The consistently low SpaceNet7 building IoU across seeds is frozen as a genuine model-performance result. The final C2 audit promoted 24/24 segmentation runs and identified no concrete mask, class-mapping, metric, or export implementation error that explains the low foreground score. Low IoU alone is not a reason to retrain, change the selector, alter class balance, or suppress the result. This decision may be revisited only if a later read-only audit demonstrates a specific implementation defect; it must not be revisited merely because the number is unfavorable.

## 4. CloudSEN12 numerical reproducibility warning

The CloudSEN12 / DOFA / full-finetune / seed-44 audit difference is recorded as a **numerical reproducibility warning only**:

- float32 CPU/GPU evaluation differs by one ULP at one class-boundary pixel;
- the single affected pixel can change the argmax column of the confusion matrix;
- exported logits, probabilities, and predictions are internally consistent; and
- no broader target-row, sample-order, checkpoint, or artifact-integrity discrepancy was found.

The run remains promoted. No inference, export regeneration, checkpoint change, training, or retraining is required. Any reproducibility statement should disclose the one-pixel boundary case and avoid promising bitwise CPU/GPU argmax identity for float32 ties.

## 5. MC Dropout method and compute freeze

### 5.1 Method contract

The following prospective choices are fixed before any MC Dropout training so that the method cannot be tuned after seeing UQ or test results:

- **Training-time dropout is mandatory.** Inference-only alteration of a deterministic dropout-zero checkpoint is prohibited.
- **Dropout rate:** `p = 0.10` for every cell.
- **Classification placement:** one standard dropout layer immediately before the final linear classifier (after the fixed BatchNorm in the EuroSAT head, and before the linear layer in the TreeSatAI head).
- **Segmentation placement:** one channelwise spatial dropout (`Dropout2d`) on the common decoder's final feature map, immediately before the 1×1 classifier.
- **Backbone stochasticity:** do not enable or alter foundation-backbone dropout, DropPath, or stochastic depth. This keeps the intervention identical across DOFA and Panopticon and makes the method specifically downstream-head/decoder MC Dropout.
- **Checkpoint validation:** ordinary evaluation mode with dropout disabled; apply the task-specific criterion in Section 1.
- **MC inference:** exactly 30 stochastic forward passes per sample. Keep the full model in evaluation mode except for the single designated dropout module, so BatchNorm and other normalization/statistics remain fixed.
- **Aggregation:** arithmetic mean of per-pass probabilities is the predictive probability. Do not average class decisions or logits as the primary MC prediction.
- **Uncertainty decomposition:** retain raw per-pass probabilities and compute predictive entropy, expected entropy, mutual information, and probability variance at the natural output unit (per label for TreeSatAI and per pixel for segmentation) before any documented summary reduction. MC passes are repeated draws, not independent training seeds.
- **No adaptive tuning:** dropout rate, placement, pass count, seed set, replication subset, and checkpoint may not be changed in response to validation calibration, test performance, or attractive uncertainty maps.

This contract intentionally uses only the shared downstream module. It avoids a model-family confound from different native backbone dropout/DropPath implementations and must be described in the thesis as **head/decoder MC Dropout**, not as full-backbone Bayesian inference.

The existing generic `samples: 10` entry and commented historical examples are not authoritative. This freeze supersedes them for the final MC experiments. Implementing this prospective path may add new MC-specific configuration/code, but it must not modify existing model artifacts or reinterpret deterministic checkpoints.

### 5.2 Minimum scientifically defensible replication

The core design is **24 new MC Dropout training runs**, not 48:

1. Run one predeclared training seed, **seed 42**, for every model × dataset × adaptation cell: 16 runs.
2. Pair each result with the completed deterministic run from seed 42 in the identical cell. The primary per-cell contrast is therefore seed-matched.
3. Add seeds 43 and 44 in the following four predeclared robustness cells: 8 additional runs.

| Dataset | Model | Adaptation | MC training seeds | Reason for inclusion |
|---|---|---|---|---|
| EuroSAT | DOFA | frozen | 42, 43, 44 | Stable multiclass classification reference |
| TreeSatAI | Panopticon | full fine-tune | 42, 43, 44 | Multilabel classification and a sensitive full-finetune regime |
| CloudSEN12 | Panopticon | frozen | 42, 43, 44 | Four-class segmentation with frozen transfer |
| SpaceNet7 | DOFA | full fine-tune | 42, 43, 44 | Imbalanced foreground segmentation and a sensitive full-finetune regime |

This subset covers every dataset, both tasks, both foundation models, both adaptation regimes, multiclass and multilabel classification, multiclass segmentation, and imbalanced binary foreground segmentation. It also balances DOFA/Panopticon and frozen/full representation across the four cells.

Reporting rules for this design:

- For all 16 cells, report the seed-42 deterministic-versus-MC paired difference.
- For the four robustness cells, additionally report the three seed-level paired differences and their mean and standard deviation. Do not treat 30 MC passes as `n = 30` independent replicates.
- Use the three deterministic seeds and the already completed Deep Ensemble as complementary evidence about training-seed and ensemble uncertainty, but do not present the three-member Deep Ensemble versus one-seed MC contrast as a controlled causal comparison.
- General claims about seed robustness must be limited to the four replicated cells. The other 12 MC cells are single-training-seed estimates.
- The subset cannot be changed after inspecting MC results. An unfavorable or favorable seed-42 result is not a reason to add seeds selectively.

This is the minimum scientifically defensible scheme for the core thesis because it preserves full factorial cell coverage, pairs the main comparison, and directly tests training-seed robustness on a predeclared balanced subset while using half the training count of three seeds in every cell. If the thesis regulations or supervisor explicitly require seed-level inferential claims for every MC cell, then all 16 cells must use seeds 42/43/44 (48 runs); that is a different prospective requirement, not the default policy and not a reason to retrain deterministic models.

## 6. UQ comparison and claim policy

### 6.1 Primary quantitative evidence: within classification

Allowed:

- deterministic versus MC Dropout comparisons within the same classification dataset/model/adaptation, paired by seed where available;
- DOFA versus Panopticon or frozen versus full-finetune comparisons within the same classification dataset, with the relevant seed/replication limitation stated;
- comparisons of NLL, Brier, ECE, predictive performance, and uncertainty summaries when their definitions are held fixed within the classification task; and
- EuroSAT Temperature Scaling and classification Deep Ensemble comparisons under their valid method-specific protocols.

TreeSatAI Temperature Scaling must appear as `N/A`, so it cannot enter a quantitative method ranking.

### 6.2 Primary quantitative evidence: within segmentation

Allowed:

- deterministic versus MC Dropout comparisons within the same segmentation dataset/model/adaptation, paired by seed where available;
- DOFA versus Panopticon or frozen versus full-finetune comparisons within the same segmentation dataset;
- within-dataset comparisons of mIoU, per-class IoU, pixel accuracy, NLL, Brier, ECE, and the frozen SpaceNet7 foreground/boundary metrics; and
- segmentation Deep Ensemble comparisons under the existing probability-mean aggregation protocol.

### 6.3 Cross-task comparisons

Cross-task classification-versus-segmentation discussion is restricted to **qualitative, explicitly caveated patterns**. It may describe, for example, that calibration challenges appear differently for multilabel images, dense multiclass pixels, and imbalanced building boundaries. It may not turn raw metric differences into an architecture- or task-caused effect.

Every cross-task calibration discussion must state that absolute comparability is confounded by:

1. different primary task metrics and label semantics;
2. different checkpoint selectors (classification minimum NLL; segmentation maximum mIoU);
3. different output structures and effective observation units (image/category, multilabel outputs, and spatially correlated pixels); and
4. task-specific calibration summaries, including foreground and boundary conditioning for SpaceNet7.

Prohibited claims include:

- “classification is better calibrated than segmentation” based only on lower raw ECE/NLL/Brier;
- “segmentation is intrinsically more uncertain” based only on predictive entropy or variance at a different output unit;
- attributing a classification-versus-segmentation gap causally to DOFA, Panopticon, adaptation, or UQ method without a design that removes the selector/output-structure confounds;
- treating pixel count as independent sample replication; and
- pooling classification and segmentation metrics into one inferential test, average rank, or universal method leaderboard.

Permitted wording is descriptive: “Under their task-specific protocols, the observed metric was X for classification and Y for segmentation; the absolute difference is not causally interpretable because checkpoint selection, metric definitions, and output structures differ.”

## 7. Immutable-artifact and retraining decision

### Completed deterministic experiments

**Decision: none requires retraining.** No final C1 or C2 run is invalidated by the checkpoint sensitivity, TreeSatAI calibration-split limitation, SpaceNet7 foreground result, or the CloudSEN12 one-pixel numerical warning.

Specifically:

- do not recreate alternative checkpoint epochs;
- do not rerun deterministic inference for this freeze;
- do not retrain TreeSatAI to create a calibration split;
- do not retrain SpaceNet7 because building IoU is low;
- do not rerun CloudSEN12 for the one-ULP boundary case; and
- do not modify, delete, overwrite, or relabel existing checkpoints, predictions, histories, C3 outputs, or manifests.

### Future MC Dropout experiments

Training a new dropout-enabled MC model is **new UQ training**, not repair/retraining of the completed deterministic model. It must use the locked method, selection, seed, and reporting rules above. This document does not start that work.

## Final freeze checklist

- Classification checkpoint criterion: **minimum validation NLL** for deterministic and MC Dropout.
- Segmentation checkpoint criterion: **maximum validation mIoU** for deterministic and MC Dropout.
- TreeSatAI Temperature Scaling: **N/A; not failed; no independent calibration split**.
- EuroSAT Temperature Scaling: **retained as valid under the dedicated calibration split**.
- SpaceNet7: **overall, foreground, classwise, foreground NLL/Brier, and valid-boundary-only calibration reporting required; zero-boundary per-image ECE stays undefined**.
- CloudSEN12 one-pixel discrepancy: **numerical warning only**.
- MC Dropout replication: **seed 42 in all 16 cells plus seeds 43/44 in four fixed robustness cells (24 training runs total)**.
- Cross-task claims: **qualitative and caveated only; within-task comparisons are the primary quantitative evidence**.
- Completed deterministic retraining: **none required**.
- MC Dropout launch during this freeze: **not started**.

## Artifact-integrity statement

No existing experiment artifact or model was modified. No training, retraining, model inference, temperature fitting, checkpoint reselection, or test-label access was performed. This report is the only artifact created by the protocol-freeze task.
