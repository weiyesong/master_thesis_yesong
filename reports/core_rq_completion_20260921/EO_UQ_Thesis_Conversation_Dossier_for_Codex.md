# EO Foundation Model Uncertainty / Calibration Thesis
## Complete conversation-derived research dossier for Codex

**Purpose.** This document reconstructs, as comprehensively as possible, the full line of thinking developed across the user's conversations with ChatGPT about uncertainty quantification (UQ), calibration, Earth Observation (EO) foundation models, the user's TUM master's thesis, the experiment pipeline, theoretical extensions, implementation decisions, audit findings, and thesis-writing strategy. It is not a polished thesis chapter. It is a **research memory, decision log, and writing specification** intended to be given to Codex or another coding/writing agent so that it can write the thesis without losing the reasoning that led to the final design.

**Important hierarchy of authority for Codex.** When historical ideas conflict, use this order:

1. The supervisor's original thesis requirements and the final three research questions.
2. The final audited experimental protocol and actually completed experiments/artifacts.
3. Explicitly accepted later decisions, especially the final model/dataset/method matrix.
4. Mechanism analyses and theoretical extensions that were proposed as optional additions.
5. Earlier exploratory ideas that were later narrowed, replaced, or deferred.

Do **not** silently merge all historical ideas into the Methods section. Several directions were intentionally explored and then demoted to Discussion/Future Work. This document marks those distinctions.

---

# 1. Current thesis state and the core research problem

The thesis evolved from a broad topic — **uncertainty quantification of deep learning in Earth Observation** — into a much more specific empirical study of **calibration and uncertainty in downstream models built on EO foundation models**.

The final main models are **DOFA** and **Panopticon**. The final main datasets are:

- **Classification:** EuroSAT and TreeSatAI.
- **Segmentation:** CloudSEN12 and SpaceNet7.

The two main downstream adaptation regimes are:

- **Frozen backbone / trained downstream head** (linear probing for classification; trainable adapter/decoder for segmentation while the FM backbone remains frozen).
- **Full fine-tuning** of the foundation model plus downstream head/decoder.

The final main prediction/calibration/UQ conditions are:

- **Deterministic baseline**.
- **Temperature Scaling (TS)** — treated primarily as a post-hoc calibration method, not as a method that creates new epistemic evidence.
- **MC Dropout (MCD)**.
- **Deep Ensemble (DE)**.

The final main evaluation metrics are:

- **Classification:** Accuracy, NLL, multiclass Brier score, ECE with 15 bins, reliability diagrams, plus derived confidence/calibration diagnostics where available.
- **Segmentation:** mIoU, per-class IoU, pixel accuracy, NLL, Brier, ECE-15; for SpaceNet7, additional foreground/building calibration and boundary-oriented calibration diagnostics are important because background dominance can make global pixel metrics misleading.

The experiment/audit workflow has essentially been completed through the A–C stages, and the user has repeatedly stated that the central experiments are complete. The current goal is to use existing validated results, reports, logits/probabilities, calibration outputs, ensemble/MCD outputs, and audit artifacts to write the thesis and answer the original three questions, rather than to keep expanding the experimental matrix.

A formal caveat remains around the historical **Pre-UQ Protocol Freeze**: on 25 August 2026 it was explicitly reported as not yet executed, while later the user reported the broader A–C program as complete. Codex should therefore inspect the actual repository/artifacts before claiming that a named “Pre-UQ Protocol Freeze” document was generated. The important scientific policies that were to be frozen are nevertheless clear and are documented below.

---

# 2. The supervisor's original thesis requirement

The original supervisor framing is the anchor for the entire thesis.

The research question was not simply “can we output uncertainty?” It was closer to:

> EO foundation models are increasingly fine-tuned for downstream tasks because they offer high performance at relatively low adaptation cost. After downstream adaptation, are their predictive probabilities actually reliable and well calibrated?

The intended pipeline was:

1. Use EO datasets and pretrained models available through or compatible with **TorchGeo**.
2. Compare multiple pretrained EO foundation models.
3. Cover both **classification and segmentation** tasks.
4. Train/adapt models on the training split.
5. Evaluate both task performance and **calibration** on validation/evaluation data.
6. Compare different adaptation regimes / fine-tuning strategies.
7. In a second stage, introduce UQ/calibration methods through **lightning-uq-box** or equivalent implementations.
8. Determine whether these methods improve calibration and whether they reduce accuracy or segmentation performance.

This created the thesis's final causal chain:

**foundation model → downstream adaptation → predictions → raw calibration → UQ/calibration intervention → change in calibration and task performance**.

This chain should remain the organizing logic of the thesis.

---

# 3. Final three research questions

Across the later conversations, the user's own formulation stabilized around three required questions. Wording varied slightly, but the meaning was consistent.

## RQ1 — Calibration after downstream adaptation

**After downstream fine-tuning/adaptation, are the predictive probabilities of different EO foundation models well calibrated?**

This is fundamentally a descriptive baseline question. It must be answered using deterministic downstream models before arguing that any UQ method helps.

The evidence should compare DOFA and Panopticon across classification and segmentation, datasets, and adaptation regimes, while keeping task-specific protocols explicit.

## RQ2 — Effect of adaptation / fine-tuning strategy

**Does the way the foundation model is adapted — especially frozen backbone versus full fine-tuning — systematically affect calibration?**

The key comparison is not just which regime gets higher Accuracy or mIoU. The central object is the joint change in:

- task performance,
- NLL,
- Brier,
- ECE,
- reliability shape,
- and, where useful, confidence on errors / foreground or class-specific calibration.

A recurring conceptual point was that **better accuracy does not imply better calibration**. Full fine-tuning can improve classification/segmentation performance while making the output distribution sharper or more overconfident.

## RQ3 — Effect of UQ/calibration interventions

**Do Temperature Scaling, MC Dropout, and Deep Ensembles improve calibration, and what is the cost in accuracy/mIoU and computation?**

This question is answered even if a method fails. “No improvement” is still an empirical answer.

The required comparison is method versus its matched deterministic baseline, not just absolute metrics across unrelated models.

For TS, an important implementation invariant is that standard scalar temperature scaling should preserve the argmax class and therefore should not materially change classification accuracy. A reported accuracy change after pure scalar TS is a reason to investigate the evaluation pipeline.

---

# 4. Chronological evolution of the thesis thinking

This section is deliberately chronological. It records how the user's research thinking evolved so that Codex understands why some ideas are central and others are extensions.

## 4.1 2025: broad UQ framing

The earliest thesis discussions treated the topic broadly as **uncertainty quantification of deep learning in Earth Observation**. The main conceptual split was the standard one:

- **Aleatoric uncertainty:** variability/noise or ambiguity associated with data and observation conditions.
- **Epistemic uncertainty:** uncertainty related to insufficient model knowledge, limited data support, model parameters, or distribution shift.

Methods discussed at this stage included:

- Bayesian Neural Networks (BNNs),
- MC Dropout,
- Deep Ensembles,
- Evidential Deep Learning (EDL),
- post-hoc calibration such as Temperature Scaling,
- uncertainty maps for spatial tasks,
- calibration metrics including ECE, NLL, and Brier score.

EO-specific examples of aleatoric-looking effects included sensor noise, clouds, smoke, shadows, mixed pixels, label noise, and temporal variability. Epistemic-looking effects included underrepresented regions, unseen sensors, geographic/seasonal shift, and limited training support.

At this stage, the framework was conventional and method-oriented: compare methods and uncertainty types.

## 4.2 September 2025 to January 2026: shift toward calibration of EO foundation models

The thesis then became more concrete. The supervisor's plan focused on whether **fine-tuned EO foundation models are calibrated**, using TorchGeo and lightning-uq-box, comparing two or three pretrained models across classification and segmentation.

DOFA and Panopticon emerged as the main candidates. EuroSAT and RESISC45 were early classification candidates; segmentation was to be added when the pipeline was ready.

Temperature Scaling, MC Dropout, Deep Ensembles, and at times Laplace approximation were discussed as representative calibration/UQ approaches. The key evaluation axes became:

- predictive performance,
- calibration,
- computational cost.

The thesis title “Uncertainty Quantification for Earth Observation Foundation Models” was discussed and retained as a good high-level title even though the empirical center of gravity became calibration.

## 4.3 May–June 2026: construction of the first reproducible experimental pipeline

The practical pipeline was first made concrete with **EuroSAT RGB** because it was the fastest controlled setting for debugging the architecture and calibration metrics.

An early formal DOFA classification configuration was:

- EuroSAT RGB using B04/B03/B02,
- image size around 224×224,
- pretrained DOFA backbone,
- frozen backbone,
- 10-class linear classification head,
- cross-entropy loss,
- AdamW,
- head learning rate around 1e-3,
- weight decay 1e-4,
- batch size 32,
- 20 epochs,
- seed 42 in the early configuration,
- validation every epoch,
- best checkpoint selected by validation NLL for classification,
- metrics: Accuracy, NLL, ECE, Brier, reliability diagrams.

A ResNet18 baseline was used or proposed as a smoke-test/control baseline. The intent was not to make standard CNN benchmarking the thesis contribution, but to ensure the calibration pipeline behaved sensibly before using EO FMs.

The proposed sequence at this time was roughly:

1. EuroSAT RGB smoke test.
2. Calibration metrics and artifact export.
3. Multispectral extension.
4. DOFA and Panopticon comparison.
5. Temperature Scaling.
6. MC Dropout / Deep Ensembles.
7. Segmentation.

Limited-label regimes (100%, 20%, 10%, 5%) and distribution-shift experiments were discussed as useful scientific extensions, but these did not remain part of the mandatory final core matrix.

## 4.4 June 2026: user rejects “metrics-only” UQ and asks for causal explanation

A major intellectual shift occurred when the user explicitly said that simply reporting Accuracy, ECE, NLL, and Brier was insufficient. The user wanted to know **why uncertainty and calibration succeed or fail**.

The desired analysis expanded to:

- uncertainty versus error,
- confidence on wrong predictions,
- risk–coverage/selective prediction,
- uncertainty quantiles,
- per-class calibration,
- class imbalance,
- label ambiguity,
- OOD/domain shift,
- corruption sensitivity,
- logit magnitude and softmax geometry,
- differences between frozen, partial, and full fine-tuning.

The assistant recommended diagnostics such as:

- binning samples by uncertainty and plotting error rate,
- AURC / risk-coverage curves,
- error-detection AUROC/AUPRC,
- inspecting confidently wrong cases,
- per-class ECE/confidence/confusion,
- corruption tests (noise, blur, brightness/contrast, cloud masks, band dropout, resolution degradation),
- domain-shift tests,
- comparing linear probe / partial FT / full FT,
- checking whether calibration changes track representation changes rather than accuracy alone.

This broader causal/mechanistic mindset remained important even after the final core experiments were narrowed.

## 4.5 June–July 2026: critique of fixed aleatoric/epistemic decomposition

The user then made a deeper conceptual objection: **aleatoric and epistemic uncertainty should not be treated as fixed intrinsic properties of a sample or dataset independent of the model**.

The user's central intuition was that the observed uncertainty depends on the combination of:

- input x,
- training data,
- model architecture and representation,
- adaptation procedure,
- task definition,
- labels,
- observation modality / information content.

A useful conditional notation proposed in the discussions was conceptually like:

**U = U(Y | X, D_train, M, T)**

or more generally **U = U(M, D, x, task, estimator)**.

The practical implication was: the thesis should avoid language suggesting that a measured entropy term is “the true aleatoric uncertainty” or “the true epistemic uncertainty.” Instead, these should be treated as operational quantities or diagnostic proxies under a particular model and predictive distribution.

## 4.6 June 30–July 2026: Wimmer23a and the TU/AU/EU debate

A detailed theoretical branch examined the common information-theoretic decomposition:

- Total uncertainty: predictive entropy.
- Aleatoric term: expected conditional entropy.
- Epistemic term: mutual information / disagreement.

For a distribution Q over categorical probability vectors θ, the standard equations discussed were:

- TU = H(E_Q[θ])
- AU = E_Q[H(θ)]
- EU = H(E_Q[θ]) − E_Q[H(θ)] = I(Y; Θ)

The conversations emphasized an important distinction:

**The identity TU = AU + EU can be mathematically exact under the chosen entropy/scoring-rule construction, but that does not prove that the two terms correspond to stable, independent, uniquely identifiable physical sources of uncertainty.**

Several objections were developed:

1. **Same predictive mean, different underlying Q.** Two predictive distributions can have the same mean probability vector and therefore the same predictive entropy while representing radically different uncertainty structures.
2. **Conditional entropy depends on epistemic state.** If the model family or posterior approximation changes, the measured “AU” can change even when the observed data x is identical.
3. **Mutual information is disagreement, not automatically ignorance.** It can be high because predictors disagree, but “disagreement” is not identical to a universal epistemic quantity.
4. **The semantics are model- and scoring-rule-dependent.** With proper scoring rules other than log loss, the decomposition changes.

A canonical binary counterexample discussed was:

- Q = δ_(0.5, 0.5): every predictor says 50/50. This looks like pure ambiguity under the predictive model.
- Q = 0.5 δ_(1,0) + 0.5 δ_(0,1): predictors are individually certain but mutually opposed. This looks like pure disagreement.

Both can have the same mean prediction (0.5, 0.5), yet their expected entropy and mutual information differ drastically.

Another illustrative example compared a uniform distribution over Bernoulli probabilities Q = U[0,1] with a two-point extreme mixture. Predictive entropy can be maximal in both, but expected entropy and MI differ.

The strongest conclusion was not “TU = AU + EU is false.” The conclusion was:

- the additive identity can be valid by definition;
- **source identifiability**, **source separability**, and **intervention stability** are separate questions;
- calling the two mathematical terms “aleatoric” and “epistemic” can overstate their semantic purity.

## 4.7 July 2026: AU–EU coupling as a possible original research direction

The user considered making **AU/EU coupling** a deeper research contribution rather than merely applying existing UQ methods.

The assistant proposed distinguishing:

- mathematical additivity of a chosen decomposition,
- structural/source separability,
- estimator cross-contamination.

A possible interaction test was formulated using a two-factor interaction contrast:

C = T(a1,e1) − T(a1,e0) − T(a0,e1) + T(a0,e0)

and local interaction via a mixed partial derivative:

κ = ∂²T / (∂a ∂e)

with possible global interaction indices.

A toy multiplicative uncertainty surface was discussed to show that an observed total uncertainty can contain interaction between controllable “ambiguity” and “knowledge” factors even when a chosen entropy decomposition remains algebraically additive.

The thesis direction proposed at this point was theory-first:

1. synthetic categorical example with known uncertainty structure,
2. controllable EO mixed-pixel benchmark,
3. frozen foundation representations,
4. model-specific mechanism analysis.

However, this branch was later deliberately demoted from the mandatory thesis core because the supervisor's original empirical requirements had to be completed first.

## 4.8 August 2026: re-freezing the thesis around the supervisor's core

By early August, the user accepted that the thesis needed a clean, defensible core before any ambitious AU/EU theory.

The thesis scope was explicitly refocused to:

- DOFA + Panopticon,
- classification + segmentation,
- frozen versus full fine-tuning,
- deterministic calibration,
- Temperature Scaling,
- MC Dropout,
- Deep Ensembles,
- Accuracy/mIoU plus ECE/NLL/Brier/reliability,
- careful audit and reproducibility.

OOD experiments, AU/EU coupling, representation-level uncertainty, and information-theoretic theory were retained as **extensions / Discussion / Future Work**, not required to answer the supervisor's core questions.

## 4.9 August 2026: model and dataset finalization

### Why DOFA and Panopticon

Both are sensor-flexible EO foundation models, which makes them a controlled but scientifically meaningful pair.

**DOFA** uses wavelength-conditioned dynamic embedding / a dynamic weight-generating mechanism so that spectral metadata influences how input channels are embedded.

**Panopticon** uses channel/spectral-aware mechanisms and cross-channel/cross-view processing designed to support variable EO inputs.

The comparison is therefore interesting because both attempt sensor flexibility but use different representation mechanisms.

The key caution was repeated many times:

> Higher accuracy of one foundation model must not be interpreted as evidence of better calibration.

### Dataset evolution

The classification datasets changed several times because of contamination, benchmark overlap, sample size, and calibration-split concerns.

Early candidates included:

- EuroSAT,
- RESISC45,
- So2Sat,
- GEO-Bench-2 subsets,
- Cameroon-like alternatives.

So2Sat was at one point planned as the full original LCZ42 dataset because the GEO-Bench subset was considered too small for stable ECE/NLL/Brier estimates. A detailed S2-only protocol was designed.

Later, the classification design shifted to **TreeSatAI**, in part because it reduced known overlap with the original DOFA/Panopticon evaluation ecosystem and was viewed as a cleaner controlled choice. The final classification pair became **EuroSAT + TreeSatAI**.

The final segmentation datasets became **CloudSEN12 + SpaceNet7**.

The contamination concern was not only literal sample leakage. It also included using a benchmark or dataset heavily entangled with the same research ecosystem or prior evaluation of the chosen foundation model, which could weaken claims about general downstream reliability.

## 4.10 August 2026: execution and audit-first workflow

A key methodological decision was to stop writing ad hoc training scripts and instead use a staged, auditable workflow.

The coding-agent prompts were organized around dependency boundaries rather than many tiny tasks. The recurring required sections were:

- objective,
- scientific context,
- required changes,
- prohibited shortcuts / things not to modify,
- acceptance tests,
- completion report.

Critical audit targets included:

- dataset split integrity,
- no calibration leakage,
- wavelength/channel semantics,
- wavelength units (DOFA vs Panopticon),
- correct frozen/full gradient behavior,
- optimizer parameter groups,
- real model loading,
- dropout stochasticity,
- consistent prediction export,
- ignore-label handling for segmentation,
- reproducible checkpoint selection,
- preservation of legacy artifacts.

The implementation philosophy was: **do not launch the full matrix until the real backbones, channels, gradients, and metrics pass smoke tests.**

## 4.11 Late August 2026: experiment completion and known anomalies

The audit eventually reported **62 tests passed**.

Known issues and their interpretations included:

### CloudSEN12 DOFA-full seed 44

One boundary pixel out of roughly 48.9 million pixels had a GPU/CPU argmax discrepancy of one float32 ULP. Exported logits, probabilities, and predictions were otherwise internally consistent.

Interpretation: numerical edge case, not evidence of an invalid run. No retraining required.

### SpaceNet7 low building IoU

Building/foreground IoU remained low across seeds.

Interpretation: likely genuine foreground under-recovery rather than a broken evaluation. This means global pixel calibration can be dominated by background and must be supplemented by foreground/building metrics.

### Boundary ECE undefined on some images

Some images have no valid boundary pixels, so per-image boundary ECE can be mathematically undefined.

Interpretation: keep it undefined/NaN at the per-image level rather than fabricating zero; aggregate boundary metrics remain meaningful when computed over valid boundary pixels.

### Abandoned zero-epoch directories

Two seed-42 initialization directories with zero training epochs were preserved but explicitly excluded from valid experiment aggregation.

### EuroSAT wavelength metadata

An older DOFA EuroSAT artifact lacked explicit wavelength fields. The pipeline later audited/derived the intended RGB wavelengths around 665/560/490 nm for B04/B03/B02.

### Panopticon validation

The real Panopticon backbone was checked for:

- weight loading,
- wavelength units,
- input resolution/dynamic range,
- feature dimensions,
- frozen/full gradient behavior.

These checks passed and were part of the reason the final results can be treated as model-valid rather than merely interface-valid.

## 4.12 September 2026: mechanism explanation returns as an optional enhancement

Once the core empirical program was stable, the user again asked how to explain **why** the models display the observed uncertainty behavior.

Proposed model-internal analyses included:

- representation similarity / CKA before and after fine-tuning,
- representation drift,
- logit norm and margin geometry,
- softmax scale analysis,
- gradient sensitivity / gradient visualization,
- controlled ablations on which blocks are trainable,
- curvature / Laplace / Hessian-style diagnostics,
- Jacobian covariance propagation JΣJᵀ,
- analysis of DOFA wavelength-conditioned embedding behavior,
- analysis of Panopticon channel-invariance/cross-channel behavior,
- dedicated foreground/background/boundary analysis on SpaceNet7.

A proposed intervention ladder was:

1. frozen backbone,
2. only input-adaptation layer trainable,
3. Transformer body trainable with input mechanism constrained,
4. full fine-tuning.

The intent was to locate where calibration changes originate.

This is valuable but **optional**. It should not block the thesis if the three core RQs are already answered convincingly.

---

# 5. The user's core conceptual positions

These are not incidental remarks. They are recurring intellectual positions that should shape the Discussion.

## 5.1 Calibration is not accuracy

A model can become more accurate while becoming less calibrated.

Example mechanism: training/fine-tuning may increase the logit norm or margin for most samples. Correct samples become more confident, but a small number of remaining errors can become extremely confident. Accuracy improves; NLL may worsen; ECE may also worsen.

Therefore every comparison should keep **performance** and **probability quality** separate.

## 5.2 Softmax confidence is scale-dependent

Multiplying logits by a positive scalar changes the concentration of the softmax distribution without changing the class ordering.

Thus:

- the prediction can become lower entropy / more confident,
- no new sample evidence has been added,
- argmax remains unchanged.

This was central to the user's intuition that confidence should not be confused with evidence.

## 5.3 Temperature Scaling is calibration, not new evidence

For logits z and scalar T > 0:

p_T(y|x) = softmax(z/T)

- T > 1 softens predictions.
- T < 1 sharpens predictions.
- For positive scalar T, class ordering is preserved.

TS learns how logit scale should map to empirical correctness on held-out labeled data. It does **not** create new epistemic information about the sample.

For this thesis, TS is best described as a **post-hoc calibration intervention / calibration baseline**, not as a full model-based UQ method.

## 5.4 Predictive uncertainty is conditional on the model–data–task system

The user repeatedly rejected the idea that a sample has one immutable aleatoric uncertainty regardless of model.

Measured uncertainty can change when:

- representation quality changes,
- fine-tuning changes the feature space,
- input wavelengths/modalities change,
- labels/task definition change,
- training support changes,
- posterior/ensemble approximation changes.

Thus the thesis should use cautious phrases such as:

- “predictive uncertainty,”
- “entropy-based uncertainty,”
- “model disagreement,”
- “expected predictive entropy,”
- “MI-style epistemic proxy,”

rather than claiming access to metaphysically “true” AU/EU.

## 5.5 Good UQ should be behaviorally meaningful

The user's desired standard was not “the model emits a high uncertainty number.” Good uncertainty should correlate with conditions where decisions are less reliable.

Desirable behavior includes:

- higher uncertainty on errors than correct predictions,
- higher uncertainty under distribution shift or corrupted inputs,
- uncertainty increasing with severity when evidence degrades,
- lower uncertainty where the model has reliable support,
- useful risk–coverage behavior,
- avoiding confidently wrong predictions.

These were proposed as deeper diagnostics, even if the final mandatory thesis matrix emphasizes calibration metrics.

---

# 6. Calibration metrics: meaning, use, and limitations

## 6.1 Expected Calibration Error (ECE)

The thesis uses **ECE-15**, generally with 15 equal-width confidence bins.

For bin B_m:

ECE = Σ_m (|B_m|/n) · |acc(B_m) − conf(B_m)|

Interpretation: ECE estimates the mismatch between average confidence and empirical accuracy across confidence bins.

Limitations repeatedly discussed:

- binning makes it sensitive to bin count and sample size,
- top-label ECE ignores the full probability vector,
- low global ECE can hide class-specific or foreground failures,
- segmentation ECE can be dominated by background pixels,
- ECE is not a complete measure of uncertainty quality.

Therefore ECE should never be reported alone.

## 6.2 Negative Log-Likelihood (NLL)

For the true class y:

NLL = −log p(y|x)

NLL strongly penalizes confidently wrong predictions.

It is a proper scoring rule and useful for calibration-sensitive model selection, but it is not a pure “calibration-only” metric because it also reflects predictive discrimination.

The thesis uses validation NLL as the classification checkpoint criterion.

## 6.3 Brier score

For multiclass one-hot target y and probability vector p:

Brier = Σ_c (p_c − y_c)^2

It evaluates the full probability vector and complements NLL and ECE. It is less singularly dominated by very confident errors than log loss.

## 6.4 Reliability diagrams

Reliability diagrams compare empirical accuracy versus mean confidence by bin. They provide visual information that a scalar ECE can hide, such as systematic overconfidence or underconfidence at particular confidence ranges.

They should be included for representative experiments and not used merely as decorative plots.

## 6.5 Mean confidence and signed calibration gap

These were discussed as useful descriptive supplements, especially when interpreting whether fine-tuning simply sharpens outputs.

A model whose accuracy and ECE change should also be examined for the direction of confidence shift.

---

# 7. Standard predictive uncertainty quantities and the caveats around them

For stochastic predictions p_t(y|x) from MC Dropout samples, ensemble members, or Bayesian weight samples:

Mean predictive probability:

p̄ = (1/T) Σ_t p_t

## 7.1 Predictive entropy

H(p̄) = −Σ_c p̄_c log p̄_c

Often called “total predictive uncertainty.” It reflects the uncertainty of the averaged prediction.

## 7.2 Expected entropy

E_t[H(p_t)]

Often interpreted as an aleatoric-like component because it measures uncertainty within individual predictive members.

## 7.3 Mutual information / disagreement term

MI = H(p̄) − E_t[H(p_t)]

Often interpreted as an epistemic-like component because it increases when individual predictive members disagree.

## 7.4 Thesis wording requirement

Do not present these as guaranteed true AU/EU. The safer interpretation is:

- predictive entropy = aggregate predictive uncertainty,
- expected entropy = average within-member uncertainty,
- MI = between-member predictive disagreement / epistemic-style proxy.

Raw stochastic predictions should be preserved so that these quantities can be recomputed rather than hard-coded into one interpretation.

---

# 8. UQ and calibration methods discussed across the project

## 8.1 Deterministic softmax baseline — final core

A standard trained downstream model produces one set of logits and one softmax distribution.

This baseline is scientifically essential because all later interventions need a matched reference.

The first question is not “which UQ method is best?” but “how calibrated is the downstream EO FM before any explicit intervention?”

## 8.2 Temperature Scaling — final core

Role: **post-hoc calibration**.

Procedure:

1. Freeze the trained model.
2. Obtain logits on an independent calibration/validation split.
3. Fit one positive scalar T by minimizing NLL.
4. Apply softmax(z/T) to evaluation logits.
5. Recompute ECE, NLL, Brier, reliability.

Key invariants:

- standard scalar TS should preserve argmax predictions,
- it usually does not change accuracy/mIoU label predictions in classification-like settings,
- it does not add sample evidence,
- it can fail when miscalibration is class-dependent or sample-dependent.

A major practical issue in the final project is that **TreeSatAI did not provide a clean independent calibration split under the final artifact protocol**. Therefore TS was deliberately not generated in at least one audited TreeSatAI configuration rather than fitting temperature on a split already used for checkpoint selection and creating leakage.

## 8.3 MC Dropout — final core

Role: stochastic approximate model-based UQ through repeated dropout masks.

At inference, dropout remains active and the same input is passed through the model T times. The predictive mean and disagreement can then be computed.

Important cautions:

- it is an approximation, not exact Bayesian posterior sampling,
- results depend on dropout placement and dropout probability,
- repeated forward passes increase inference cost,
- using a pretrained FM without meaningful dropout paths can make the method ineffective unless dropout is introduced in the downstream head/decoder.

A later recommended protocol was:

- prioritize dropout in the downstream head/decoder while preserving the pretrained backbone,
- pilot T ∈ {10,20,30,50},
- freeze a practical value around T≈30 if convergence/stability is adequate,
- keep raw [N,T,C] or pixel-equivalent stochastic probabilities where feasible.

Earlier discussions often used T=20 as an illustrative default. Codex should inspect the final repository to report the actually frozen value.

## 8.4 Deep Ensembles — final core

Role: estimate predictive mean plus between-model disagreement using independently trained models or valid independent seeds.

Advantages:

- strong empirical robustness,
- often good calibration and OOD behavior,
- conceptually simple.

Costs:

- approximately M× training and storage,
- diversity depends on initialization/training differences,
- an ensemble does not automatically disentangle aleatoric uncertainty unless member outputs explicitly model observation noise.

A 3-model ensemble was repeatedly suggested as a practical compromise. Existing DOFA–EuroSAT frozen/full checkpoints from multiple valid seeds were considered reusable as ensemble members rather than retraining identical experiments solely for the ensemble.

## 8.5 Bayesian Neural Networks — background / not final core

BNNs place a posterior distribution over network weights and approximate the posterior predictive distribution by integrating over weights.

Key limitations discussed:

- exact inference is intractable for large networks,
- mean-field or other approximate posteriors can be biased and unstable,
- a single sampled network can perform badly,
- meaningful prediction requires posterior predictive averaging,
- foundation-model scale makes BNNs expensive.

BNNs helped establish the conceptual foundation of uncertainty decomposition but were not retained as a main method in the final thesis matrix.

## 8.6 Evidential Deep Learning — background / deferred

EDL predicts distributional evidence, often Dirichlet parameters for classification, in one forward pass.

Potential advantages:

- single-pass uncertainty,
- explicit distribution over class probabilities.

Concerns discussed:

- training instability,
- sensitivity to evidence regularization,
- uncertainty semantics can be hard to interpret,
- fewer mature EO comparisons,
- reproducibility risk.

It was not retained in the final core.

## 8.7 Laplace approximation — background / deferred

Laplace was discussed as a compromise approximation around a trained solution, often using local curvature to model parameter uncertainty.

It remained useful conceptually for later mechanism analysis but was not part of the final central four-condition matrix.

## 8.8 Conformal prediction / RAPS — exploratory / deferred

Conformal methods were discussed because they provide coverage-oriented uncertainty sets rather than confidence calibration in the same sense as ECE.

They are scientifically valuable but answer a somewhat different question and were therefore not necessary for the supervisor's core RQs.

## 8.9 Test-Time Augmentation and other methods — exploratory

TTA, DropConnect, Gaussian-process style methods, SWAG-like approaches, isotonic/Platt/vector/matrix scaling, and related methods appeared in reading/research discussions. They should remain in Related Work or Future Work unless actual artifacts exist.

---

# 9. Foundation models

## 9.1 DOFA

DOFA was chosen as a central model because it is designed for EO inputs with variable spectral configurations.

The important representation idea discussed was **wavelength-conditioned dynamic embedding**. Instead of treating every sensor as a fixed RGB-like channel stack, wavelength metadata helps generate or modulate input embedding behavior.

Implications for the thesis:

- explicit wavelengths are scientifically meaningful, not cosmetic metadata,
- channel order and wavelength units must be audited,
- same wavelength configuration should produce consistent inference behavior,
- changing wavelengths/channels can change the generated input embedding behavior,
- fine-tuning may change calibration through both downstream representation adaptation and logit geometry.

The historical EuroSAT RGB central wavelengths were audited as approximately:

- B04 / red: 665 nm,
- B03 / green: 560 nm,
- B02 / blue: 490 nm.

When using DOFA, the codebase at one stage expected wavelength units different from Panopticon; unit conversion was an explicit audit target.

## 9.2 Panopticon

Panopticon was selected as the second main FM because it is also sensor/channel flexible but uses a different architectural strategy.

The discussions emphasized spectral/channel-aware modeling, channel interaction, cross-channel/cross-view processing, and training strategies intended to make the representation robust across sensor configurations.

Why this is a useful controlled comparison:

- both models target flexible EO inputs,
- both are Transformer-family EO FMs,
- they differ in how channel/spectral information enters the representation,
- calibration differences therefore cannot be reduced simply to “one is a generic CNN and one is an EO FM.”

The real model integration was audited for weights, wavelength units, input shape/dynamic range, feature dimensions, and trainability behavior.

## 9.3 Possible third model

TerraMind and SMARTIES were discussed as possible third foundation models, partly for architectural diversity and networking value.

However, the thesis core does not require a third FM if DOFA and Panopticon already provide a complete, controlled comparison. Codex should not invent results for a third FM unless such artifacts actually exist.

---

# 10. Dataset evolution and final dataset choices

## 10.1 EuroSAT — final classification dataset

Role:

- early smoke-test dataset,
- retained as a final classification benchmark,
- supports controlled comparison and straightforward calibration metrics.

Historical implementation first used RGB B04/B03/B02, then broader multispectral considerations were discussed.

Important audit issue: historical DOFA artifacts did not always store explicit wavelength metadata, so the final reporting should describe the intended wavelength mapping and the audit that reconciled it.

## 10.2 TreeSatAI — final classification dataset

TreeSatAI replaced earlier alternatives because it offered a cleaner benchmark choice with lower known entanglement with the original DOFA/Panopticon evaluation ecosystem and a more controlled test of transfer reliability.

Important limitation:

- the final artifact protocol did **not** provide a clearly independent calibration split after accounting for checkpoint selection.

Therefore TS on TreeSatAI must be reported carefully. Do not imply that every dataset has a leakage-free TS result if it does not.

## 10.3 CloudSEN12 — final segmentation dataset

Used to test dense prediction calibration and uncertainty.

The audited DOFA-full seed 44 run contained one GPU/CPU float32 ULP argmax edge case at a boundary pixel. This is a numerical reproducibility note, not a scientific failure.

## 10.4 SpaceNet7 — final segmentation dataset

Used for building/foreground segmentation.

Key observed behavior:

- building IoU remained low,
- foreground under-recovery is genuine across seeds,
- background dominates the pixel population,
- therefore global accuracy/ECE can appear acceptable even when foreground performance/calibration is poor.

This motivated additional metrics:

- foreground/building IoU,
- foreground/building ECE,
- classwise calibration,
- foreground NLL/Brier,
- boundary calibration.

Per-image boundary metrics can be undefined on images with no boundary pixels. Aggregate evaluation should explicitly handle valid-pixel masks.

## 10.5 RESISC45 — earlier candidate, not final main dataset

RESISC45 was part of the early classification plan. It was later displaced as the thesis became more concerned with contamination/benchmark overlap and sensor-aware EO FM transfer.

Do not write it as a final dataset unless there are final artifacts explicitly retained for an auxiliary experiment.

## 10.6 So2Sat / LCZ42 — serious intermediate candidate, later replaced

So2Sat was studied in substantial implementation detail. The original full LCZ42 dataset was preferred over the much smaller GEO-Bench subset because calibration metrics need adequate sample sizes.

A detailed protocol was designed around Sentinel-2 channels, official splits, lazy HDF5 access, stable sample IDs, and avoiding accidental double reflectance scaling.

However, this was later replaced by TreeSatAI in the final design. It should therefore appear only in the research-development history or possibly Related Work/ablation history, not as a final core result.

## 10.7 GEO-Bench subsets

GEO-Bench was useful for standardized EO task definitions and segmentation splits, but the discussions repeatedly warned that benchmark subsampling can be problematic for calibration estimates and that benchmark provenance may overlap with FM evaluation histories.

---

# 11. Adaptation regimes

## 11.1 Frozen backbone / linear probe

For classification:

- pretrained FM backbone frozen,
- downstream classification head trained,
- this isolates the quality of the pretrained representation and minimizes representation drift.

For segmentation:

- FM backbone frozen,
- model-specific feature adapter and a shared lightweight decoder remain trainable.

Scientific interpretation:

- calibration reflects how a fixed pretrained representation supports the downstream classifier/decoder,
- changes are concentrated in the task head rather than the full FM.

## 11.2 Full fine-tuning

All or essentially all FM parameters plus the downstream head/decoder are trainable.

This is a model-side intervention that can change:

- representation geometry,
- logit margins,
- confidence sharpness,
- class separation,
- overfitting behavior,
- calibration.

The thesis should not assume that full FT is “better” just because it improves task performance.

## 11.3 Partial fine-tuning — explored but not final core

Partial fine-tuning was repeatedly proposed as a scientifically interesting middle condition.

Possible versions included:

- train only input adaptation,
- train last blocks,
- train Transformer body but freeze sensor-specific input mechanism.

It was ultimately not needed for the minimum supervisor-aligned factorial design, which stabilized around frozen versus full FT.

Partial FT is best used as an optional mechanism ablation if already implemented.

---

# 12. Downstream heads and segmentation decoder

## 12.1 Classification head

The cleanest frozen-representation probe is a linear classification head.

An MLP head was discussed at points, but the scientific reason for preferring a simple head is to avoid giving one model extra nonlinear capacity that could confound attribution of calibration differences to the foundation representation.

## 12.2 Segmentation decoder

A purely linear segmentation probe was considered too weak for realistic dense prediction.

The recommended compromise was a **shared lightweight UNet-style decoder** with only model-specific adapters needed to convert each FM's dense features into the common decoder interface.

Why this matters:

- a common decoder reduces architecture confounding,
- enough spatial/nonlinear capacity remains to make segmentation feasible,
- the FM comparison is still interpretable.

Caveat: because the decoder itself is nonlinear, segmentation results cannot be attributed solely to the backbone. The thesis should state that the controlled object is the **FM + standardized downstream decoder system**.

---

# 13. Checkpoint-selection policy

This became a significant methodological issue.

## Classification

Best checkpoint selected by **minimum validation NLL**.

Rationale: classification calibration/probability quality is central, and NLL is a proper scoring rule.

## Segmentation

Best checkpoint selected by **maximum validation mIoU**.

Rationale: segmentation training must avoid selecting a model that is well calibrated only because it predicts background or underfits the foreground task.

## Cross-task caveat

Because classification and segmentation use different checkpoint criteria, a cross-task statement such as “segmentation is more/less calibrated than classification” can be confounded by selection policy.

The recommended response was not automatic retraining. Instead:

- run a checkpoint-selection sensitivity audit if needed,
- state the task-specific criteria explicitly,
- restrict cross-task claims to robust qualitative patterns rather than pretending the selection policies are identical.

---

# 14. Training configuration details that appeared in the conversations

The exact final configurations must be read from repository configs for manuscript-level reporting. However, recurring concrete settings included:

### Early DOFA classification baseline

- seed: 42,
- epochs: 20,
- batch size: 32,
- optimizer: AdamW,
- head LR: 1e-3,
- weight decay: 1e-4,
- loss: cross entropy,
- ECE bins: 15,
- checkpoint by validation NLL.

### Fine-tuning optimizer grouping

A later coding-audit recommendation was to use distinct learning-rate groups, for example:

- backbone around 1e-5,
- head/decoder around 1e-3.

This was a protocol recommendation and may not describe every final run. Codex must verify the actual YAML/config before writing the numerical optimizer values in Methods.

### Seeds

Multiple seeds were used; seed 42 and seed 44 explicitly appear in the audit history. The final exact seed set should be taken from the experiment registry, not inferred from conversation summaries.

---

# 15. Artifact preservation requirements

A recurring recommendation was to preserve sample-level outputs so the thesis can be re-analyzed without retraining.

For deterministic runs, preserve where possible:

- sample ID,
- ground-truth label/mask,
- predicted label/mask,
- logits,
- probabilities,
- confidence,
- metadata needed to trace dataset split.

For MC Dropout:

- raw stochastic probabilities/logits across T passes,
- not only the averaged probability.

For Deep Ensemble:

- raw member predictions,
- not only the ensemble average.

For future AU/EU work, preferred array concepts were:

- classification MCD: [N, T, C],
- classification ensemble: [N, M, C],
- segmentation equivalents with pixel dimensions or chunked storage.

This enables recomputation of:

- predictive entropy,
- expected entropy,
- mutual information,
- variance/disagreement,
- error-detection scores,
- classwise calibration,
- risk-coverage.

---

# 16. Calibration leakage and split discipline

The thesis discussions were strict about calibration leakage.

Key rules:

1. The test set must never be used to fit temperature.
2. A split used to select the best training checkpoint is not automatically a clean calibration split for TS.
3. If a dataset lacks an independent calibration split, do not silently reuse the validation set and then claim leakage-free evaluation.
4. If necessary, omit the TS result for that dataset or document the limitation explicitly.
5. Ensemble member selection and MC dropout evaluation must use the same frozen dataset protocol as the deterministic baseline.

The TreeSatAI case is the main concrete example where this mattered.

---

# 17. Segmentation-specific calibration issues

Dense prediction introduces problems that classification metrics can hide.

## 17.1 Pixel dependence

Pixels are not independent samples, so standard confidence-bin statistics can look artificially precise if interpreted as i.i.d. observations.

The thesis can still report pixel ECE, but should not overstate nominal sample size.

## 17.2 Class imbalance

In SpaceNet7, background pixels dominate. A model can achieve high pixel accuracy and reasonable global ECE while failing to recover buildings.

Therefore report:

- global metrics,
- foreground/building metrics,
- classwise metrics,
- boundary metrics.

## 17.3 Ignore pixels

All probability and task metrics must apply the same ignore mask. Inconsistent ignore handling would invalidate cross-metric comparisons.

## 17.4 Boundary calibration

Boundaries are often the most ambiguous and operationally difficult pixels. Boundary calibration is therefore scientifically useful.

But if an image contains zero valid boundary pixels, per-image boundary ECE is undefined. Preserve NaN/undefined semantics rather than substituting zero.

---

# 18. Distribution shift and corruption experiments — proposed extension

A substantial set of conversations explored whether calibrated uncertainty should respond to degraded evidence.

Proposed corruptions included:

- Gaussian noise,
- blur,
- brightness/contrast shift,
- cloud masking,
- band dropout,
- resolution degradation.

Possible protocol:

- train on clean data,
- calibrate on clean calibration data,
- test on clean and corrupted/shifted data,
- keep TS fixed when moving to shift conditions,
- measure accuracy, ECE, NLL, Brier, mean confidence, entropy/disagreement,
- examine monotonicity versus corruption severity.

These experiments would be useful for demonstrating whether UQ is behaviorally meaningful, but they were not part of the final mandatory core after scope was refrozen.

---

# 19. Label-scarcity experiments — proposed extension

Because foundation models are often justified by label efficiency, the user considered experiments at label fractions such as:

- 100%,
- 20%,
- 10%,
- 5%.

The scientific question would be whether the FM preserves calibration as supervision decreases.

This direction is consistent with the thesis theme but was not needed to answer the final three RQs and should not be added to Results unless actual experiments exist.

---

# 20. Risk–coverage and error-detection analysis — recommended deeper diagnostics

Even though the final core metrics emphasize ECE/NLL/Brier, several conversations argued that uncertainty should also be evaluated by decision utility.

## 20.1 Risk–coverage

Sort samples by an uncertainty score. Retain only the most certain fraction and compute risk/error as coverage decreases.

A useful uncertainty measure should allow the model to reduce error by abstaining on uncertain cases.

AURC can summarize this behavior.

## 20.2 Error-detection AUROC/AUPRC

Treat “prediction is wrong” as a binary target and use uncertainty as the score.

This measures whether uncertainty discriminates errors from correct predictions.

## 20.3 Confidently wrong samples

The user repeatedly wanted qualitative inspection of cases where:

- confidence is high,
- prediction is wrong,
- entropy is low,
- or UQ methods disagree with deterministic confidence.

This can reveal systematic semantic confusions or representation failures that scalar calibration metrics cannot explain.

These analyses are highly suitable for Discussion/diagnostic subsections if sample-level outputs already exist.

---

# 21. Model-internal explanation of uncertainty

This was the main proposed enhancement after the core experiments.

## 21.1 Representation drift and CKA

Compare frozen versus fine-tuned representations.

Questions:

- How much does the representation change?
- Are calibration shifts associated with stronger representation drift?
- Do the two FMs preserve class geometry differently?

CKA or related representation similarity measures were proposed.

## 21.2 Logit geometry

Analyze:

- logit norms,
- top-1/top-2 margins,
- classwise margin distributions,
- relation between margin, correctness, and calibration,
- how fine-tuning changes logit scale.

This directly connects to the user's concern that confidence can change without new evidence.

## 21.3 Gradient-based diagnostics

Gradient visualization was discussed as one way to see which inputs/features drive predictions and whether uncertain samples show unstable or diffuse sensitivity.

Potential objects:

- input gradients,
- feature gradients,
- gradient norms,
- sensitivity around boundaries,
- layer-wise gradient changes under frozen versus full FT.

The thesis should avoid claiming that gradient magnitude is “uncertainty” by itself. It is a mechanism diagnostic.

## 21.4 Curvature / Laplace / Hessian analysis

A local approximation around trained parameters can estimate sensitivity to parameter perturbation.

The conceptual expression JΣJᵀ was discussed as a way to propagate parameter covariance through the predictive function.

This could connect representation geometry to epistemic-style uncertainty, but it is an advanced optional section.

## 21.5 Architecture-specific mechanism hypotheses

### DOFA

Wavelength-conditioned dynamic embeddings may affect robustness/calibration by changing how spectral evidence is mapped into the common latent space.

Possible analysis:

- perturb wavelength metadata,
- compare representation/logit sensitivity,
- inspect whether fine-tuning makes the model over-specialize to a fixed channel configuration.

### Panopticon

Channel-aware/cross-channel invariance mechanisms may produce different confidence behavior under sensor/channel changes.

Possible analysis:

- channel ablations,
- representation consistency,
- confidence under missing/degraded channels.

Again, these were proposed hypotheses, not established results.

---

# 22. AU/EU theory: what can safely be claimed

This is the most important theoretical caution for Codex.

## 22.1 Safe mathematical statement

Under the standard entropy decomposition for a predictive distribution over class probabilities:

TU = H(E_Q[P])

AU = E_Q[H(P)]

EU = H(E_Q[P]) − E_Q[H(P])

Therefore:

TU = AU + EU

by construction.

## 22.2 What this identity does not prove

It does not prove that:

- AU is a pure property of the data independent of the model,
- EU is pure ignorance,
- the two causal sources are independent,
- the decomposition is unique across scoring rules,
- a measured AU value is intervention-invariant.

## 22.3 Intervention-stability test

A key proposed argument was:

If the same input and label data produce different measured “AU” after changing from frozen to full fine-tuning, then the measured term is not purely a data-intrinsic quantity. It is at least partly a function of the predictive model/estimator.

This is a strong Discussion point if supported by actual stochastic outputs.

## 22.4 Model × data interaction

A more general hypothesis was:

U = Model + Data + Model×Data

in an ANOVA-like conceptual sense.

Evidence for model×data interaction could include:

- rank reversals across datasets,
- uncertainty changes that depend on both architecture and dataset,
- interaction terms improving error prediction.

## 22.5 “Uncertainty as residue of compression” idea

One conceptual metaphor developed in July was:

- representation/model = compression scheme,
- predictive probability = compression claim,
- NLL = coding cost / surprise of the observed label,
- entropy = perceived ambiguity under the model,
- calibration = agreement between claimed certainty and empirical outcomes,
- overconfident high-NLL predictions = “compression hallucinations.”

This is intellectually interesting but should be used cautiously, likely in Discussion rather than as the formal theoretical foundation unless the thesis explicitly develops it.

---

# 23. What the final thesis should say about Temperature Scaling

This topic was revisited repeatedly and is easy to miswrite.

Correct framing:

- TS rescales logits using one learned scalar temperature.
- It is fitted on held-out labeled calibration data by minimizing NLL.
- Positive scalar scaling preserves class ordering.
- Therefore standard TS should preserve classification argmax predictions.
- TS can substantially change softmax entropy and confidence without adding new evidence.
- It is useful because raw neural-network logit scale is often not aligned with empirical correctness.
- It is limited when miscalibration is class-dependent, sample-dependent, or shifts across domains.

Incorrect framing to avoid:

- “TS quantifies epistemic uncertainty.”
- “TS adds information about the sample.”
- “TS should improve accuracy.”
- “Lower entropy after scaling means more evidence.”

---

# 24. What the final thesis should say about MC Dropout and Deep Ensembles

## MC Dropout

Use repeated stochastic forward passes. Report predictive mean and, where relevant, disagreement/entropy quantities.

Do not claim that MC Dropout samples are exact posterior samples.

## Deep Ensemble

Use independently trained valid members. Average member probabilities, not raw logits unless a specific justified aggregation is used.

Disagreement can be interpreted as epistemic-style variability, but the thesis should avoid claiming a perfect source decomposition.

For both methods, calibration is an empirical property to measure. A UQ method can produce stochastic predictions and still be miscalibrated.

---

# 25. Audit findings and how they should appear in the thesis

Most audit details belong in Methods / Reproducibility / Limitations, not the main Results narrative.

## Include explicitly

- real DOFA and Panopticon backbones were validated,
- channel/wavelength semantics were audited,
- frozen/full gradient behavior was checked,
- prediction artifacts were preserved,
- invalid zero-epoch directories were excluded,
- ignore masks and segmentation metrics were checked,
- 62 tests passed in the audit suite.

## Include only if relevant

- the single 1-ULP boundary pixel issue can be mentioned in a reproducibility appendix or audit note rather than in the main scientific Results.

## Important limitation

- TreeSatAI's lack of an independent calibration split affects TS claims.

## Important scientific observation

- low SpaceNet7 building IoU is not an implementation bug; it changes how calibration must be interpreted.

---

# 26. How to answer RQ1 from the existing results

RQ1 asks whether downstream-adapted EO FMs are calibrated.

Recommended evidence structure:

For every valid deterministic cell:

- model,
- dataset,
- task,
- adaptation regime,
- seed,
- Accuracy or mIoU,
- NLL,
- Brier,
- ECE-15,
- mean confidence if available,
- reliability diagram reference.

Then aggregate by model/dataset/adaptation with mean ± std where multiple seeds exist.

The narrative should distinguish:

- well-calibrated versus systematically over/underconfident regimes,
- classification versus segmentation,
- global versus foreground/classwise segmentation behavior,
- whether calibration rankings agree with accuracy rankings.

Do not create a binary “calibrated/not calibrated” claim based on one arbitrary ECE threshold unless such a threshold is justified. Prefer comparative language and actual metric magnitudes.

---

# 27. How to answer RQ2 from the existing results

RQ2 should be analyzed as **paired changes from frozen to full fine-tuning**.

For each model–dataset pair, compute:

ΔAccuracy or ΔmIoU

ΔNLL

ΔBrier

ΔECE

and optionally Δmean confidence.

Key interpretation patterns:

### Pattern A: performance ↑, calibration improves

Fine-tuning improves both representation/task fit and probability quality.

### Pattern B: performance ↑, calibration worsens

This is scientifically important. It supports the thesis claim that adaptation can sharpen or distort confidence even while improving task accuracy.

### Pattern C: performance unchanged, calibration changes

This suggests probability geometry changes independently of decision boundaries.

### Pattern D: performance and calibration both worsen

Fine-tuning may be overfitting, unstable, or poorly matched to the dataset.

Use paired comparisons rather than comparing unrelated absolute values.

---

# 28. How to answer RQ3 from the existing results

For each deterministic baseline, compare matched TS, MCD, and DE results where valid.

Calculate:

ΔECE_method = ECE_method − ECE_det

ΔNLL_method = NLL_method − NLL_det

ΔBrier_method = Brier_method − Brier_det

ΔPerf_method = Accuracy/mIoU_method − Accuracy/mIoU_det

For TS, ΔAccuracy should be approximately zero under standard classification evaluation.

For MCD/DE, performance can shift because averaged predictive probabilities can change argmax decisions.

The narrative should report:

- which methods improve calibration consistently,
- whether improvement depends on model/dataset/task,
- whether lower ECE agrees with NLL/Brier,
- any performance cost,
- computational cost / extra inference passes / ensemble members.

Do not force a universal “best method” if the evidence is heterogeneous.

---

# 29. Statistical analysis recommended after A–C completion

Once the main runs are complete, the priority is analysis, not more training.

Recommended additions:

## 29.1 Evidence matrix

Create one master table with one row per valid run or aggregated condition and columns for:

- model,
- dataset,
- task,
- adaptation,
- method,
- seed/member,
- checkpoint criterion,
- performance metrics,
- calibration metrics,
- artifact paths,
- validity/audit status.

## 29.2 Mean ± standard deviation

For multi-seed deterministic runs and applicable UQ results.

## 29.3 Paired deltas

Especially for RQ2 and RQ3.

## 29.4 Bootstrap confidence intervals

Paired bootstrap or sample-level resampling was recommended where sample-level predictions are available.

For segmentation, bootstrap at image/tile level rather than naïvely treating every pixel as independent.

## 29.5 Compute cost

Report at least coarse training/inference overhead:

- deterministic = 1× inference,
- TS = negligible post-hoc/inference overhead,
- MCD = T stochastic passes,
- DE = M models / M passes and M× storage/training.

## 29.6 SpaceNet7 focused analysis

Foreground/building and boundary results should be shown separately because global calibration can be misleading.

---

# 30. Figures that best support the thesis

Recommended high-value figures:

1. **Thesis pipeline schematic:** EO FM → adaptation → deterministic prediction → calibration evaluation → TS/MCD/DE → recalibration/UQ evaluation.
2. **Reliability diagrams** for representative DOFA/Panopticon frozen/full cells.
3. **Performance vs ECE scatter** with points labeled by model/adaptation.
4. **Frozen→full paired arrows** showing how performance and calibration move together or in opposite directions.
5. **Method delta plot** for TS/MCD/DE relative to deterministic baseline.
6. **SpaceNet7 global vs foreground calibration** comparison.
7. Optional: risk–coverage curve if uncertainty discrimination outputs exist.
8. Optional: representation/logit mechanism figure if internal analysis is added.

Avoid producing dozens of nearly identical reliability plots in the main body. Put exhaustive per-run plots in an appendix.

---

# 31. Results-writing logic

The Results chapter should follow the research questions, not the chronological execution order.

## Results section A — RQ1

Raw deterministic calibration of each EO FM after downstream adaptation.

## Results section B — RQ2

Effect of frozen versus full fine-tuning.

## Results section C — RQ3

Effect of TS, MCD, and DE relative to deterministic baselines.

## Results section D — cross-task / failure-case analysis

- classification vs segmentation caveats,
- SpaceNet7 foreground/boundary behavior,
- reliability diagrams,
- statistically robust recurring patterns.

Mechanism analysis should come after the basic empirical findings, not before them.

---

# 32. Discussion-writing logic

The Discussion should explain observations using hypotheses that are consistent with the experiments, while clearly separating evidence from speculation.

Potential themes:

## 32.1 Foundation-model representation quality does not guarantee calibrated probabilities

Pretraining optimizes representation transfer, not necessarily probability calibration after a new downstream head or full FT.

## 32.2 Fine-tuning changes both features and confidence geometry

Full FT can alter logit scale, margin distributions, and representation geometry. This can improve accuracy while worsening calibration.

## 32.3 Post-hoc calibration and model-based UQ solve different problems

TS corrects probability scale using labeled calibration data.

MCD/DE add stochastic/model-disagreement information but are still not automatically calibrated.

## 32.4 Dense EO tasks expose class-imbalance and spatial-structure limitations of standard calibration metrics

SpaceNet7 is the clearest case.

## 32.5 “Aleatoric vs epistemic” should be interpreted operationally

Stochastic entropy decomposition can be reported, but the thesis should not claim that the terms are unique model-independent physical quantities.

---

# 33. Known pitfalls Codex must not introduce

1. Do not equate **uncertainty** with **miscalibration**.
2. Do not equate **calibration** with **accuracy**.
3. Do not describe TS as adding epistemic evidence.
4. Do not claim TS should improve accuracy.
5. Do not call predictive entropy “true total uncertainty” without qualifying the modeling context.
6. Do not call expected entropy “ground-truth aleatoric uncertainty.”
7. Do not call MI “pure epistemic uncertainty.”
8. Do not compare ECE values without considering sample size/binning/class imbalance.
9. Do not fit temperature on the test set.
10. Do not silently reuse a validation set for both checkpoint selection and calibration without reporting it.
11. Do not interpret SpaceNet7 global ECE alone.
12. Do not replace undefined boundary metrics with zero.
13. Do not include abandoned zero-epoch runs.
14. Do not treat the CloudSEN12 1-ULP issue as a training failure.
15. Do not report historical candidate datasets as if they were final.
16. Do not invent a third FM result.
17. Do not claim that all proposed mechanism analyses were executed unless repository artifacts prove it.
18. Do not claim a universal ranking of UQ methods if results vary across tasks.
19. Do not use only ECE to claim improved reliability; cross-check NLL/Brier and reliability shape.
20. Do not attribute all segmentation behavior directly to the backbone because the decoder is part of the downstream system.

---

# 34. Historical ideas that were discussed but should normally be excluded from the final core Methods

Unless actual final artifacts exist, keep these in Related Work, Discussion, or Future Work:

- Bayesian Neural Networks as a main experimental condition,
- Evidential Deep Learning,
- Laplace as a main condition,
- conformal prediction/RAPS,
- Platt/vector/matrix scaling,
- TTA,
- extensive corruption/OOD matrix,
- label-scarcity sweep,
- partial fine-tuning sweep,
- a third EO FM,
- AU/EU coupling experiments,
- CKA/gradient/Hessian/JΣJᵀ analyses.

These ideas are valuable, but adding them to Methods without corresponding artifacts would make the thesis internally inconsistent.

---

# 35. What additional experiments are actually necessary now?

The repeated recommendation after A–C completion was: **probably no new core training**.

Additional work is justified only if one of the following is true:

1. A required RQ cell is missing.
2. An experiment is invalid because of leakage, corrupted artifacts, or inconsistent protocol.
3. The raw outputs needed to verify a central claim do not exist.
4. A result rests on one anomalous run with no reproducibility support.
5. A mechanism claim is important enough to the thesis that an explicit ablation is required.

Otherwise, prioritize:

- evidence consolidation,
- statistical analysis,
- figure generation,
- RQ-aligned writing,
- limitations.

The deeper AU/EU theory and model-internal mechanism work is best treated as an optional contribution after the core thesis is already defensible.

---

# 36. Suggested thesis chapter structure

## 1. Introduction

- EO foundation models and downstream adaptation.
- Why probability reliability matters in remote sensing.
- Gap: high downstream accuracy does not establish calibrated probabilities.
- Three RQs.
- Contributions.

## 2. Background and Related Work

- EO foundation models.
- Calibration and proper scoring rules.
- UQ taxonomy.
- Temperature Scaling.
- MC Dropout.
- Deep Ensembles.
- Calibration/UQ in remote sensing.
- Short caution on AU/EU semantics.

## 3. Methodology

- models: DOFA, Panopticon,
- datasets: EuroSAT, TreeSatAI, CloudSEN12, SpaceNet7,
- adaptation: frozen vs full,
- downstream heads/decoder,
- deterministic baseline,
- TS/MCD/DE,
- metrics,
- checkpoint selection,
- split/calibration policy,
- statistical aggregation,
- reproducibility/audit.

## 4. Results

- RQ1 baseline calibration,
- RQ2 adaptation effect,
- RQ3 UQ/calibration interventions,
- segmentation foreground/boundary analysis,
- representative reliability diagrams.

## 5. Discussion

- accuracy–calibration decoupling,
- adaptation mechanisms,
- differences between TS and stochastic UQ,
- EO-specific segmentation challenges,
- AU/EU interpretation limits,
- threats to validity.

## 6. Conclusion

- direct answers to the three RQs,
- practical implications,
- concise future work.

Appendix:

- exhaustive tables,
- all reliability plots,
- audit details,
- configs,
- optional mechanism analyses.

---

# 37. Codex-ready writing instructions

Codex should treat this dossier as the conceptual and historical specification. It should then inspect the actual repository, results CSV/JSON files, reports, YAML configs, checkpoints, prediction exports, and audit logs before inserting numerical claims.

When writing:

1. First reconstruct one **master experiment registry** from actual artifacts.
2. Mark every cell valid/invalid and identify deterministic/TS/MCD/DE relations.
3. Verify final dataset/model/adaptation/method names from config files.
4. Verify exact seeds, learning rates, epochs, dropout configuration, ensemble size, and MCD sample count from artifacts — do not infer them from historical discussions if repository values differ.
5. Build the Results chapter around RQ1–RQ3.
6. Use the historical theory sections only to interpret results, not to overwrite the actual experiment protocol.
7. Preserve the distinction between **post-hoc calibration** and **model-based stochastic UQ**.
8. Use cautious wording for AU/EU.
9. Explicitly mention dataset-specific limitations such as TreeSatAI calibration-split constraints and SpaceNet7 foreground imbalance.
10. Treat the 1-ULP discrepancy as a minor numerical audit observation, not as a failed experiment.

---

# 38. Suggested master evidence table schema for Codex

Recommended columns:

- experiment_id
- task
- dataset
- model
- adaptation
- method
- seed
- checkpoint_path
- checkpoint_criterion
- calibration_split
- valid_run
- exclusion_reason
- accuracy
- f1_if_available
- miou
- per_class_iou
- pixel_accuracy
- nll
- brier
- ece15
- mean_confidence
- signed_calibration_gap
- foreground_iou
- foreground_ece
- foreground_nll
- foreground_brier
- boundary_ece
- temperature
- mcd_T
- ensemble_M
- train_time
- inference_time
- logits_path
- probs_path
- stochastic_probs_path
- audit_status
- notes

From this table, generate paired delta tables for RQ2/RQ3 automatically.

---

# 39. Suggested direct answer templates for the three RQs

These are structural templates, not conclusions; Codex must fill them from actual results.

## RQ1 answer template

“Across [datasets/tasks], deterministic downstream models built on DOFA and Panopticon showed [pattern] calibration as measured by ECE-15, NLL, Brier, and reliability diagrams. Calibration quality did/did not track predictive performance consistently. The strongest deviations occurred in [conditions], indicating that high downstream accuracy alone is insufficient to establish reliable probabilities.”

## RQ2 answer template

“Changing from a frozen backbone to full fine-tuning produced [consistent/heterogeneous] changes in calibration. In [conditions], fine-tuning improved task performance while [improving/worsening] ECE/NLL/Brier, demonstrating that adaptation affects probability geometry independently of decision accuracy. The effect was model- and dataset-dependent rather than universally monotonic.”

## RQ3 answer template

“Temperature Scaling, MC Dropout, and Deep Ensembles produced different calibration–performance–cost trade-offs. TS primarily corrected probability scale at negligible inference cost and preserved class ordering. MCD and DE introduced stochastic predictive variation and, in [conditions], improved [metrics] at the cost of repeated inference / multiple models. No method should be described as universally superior unless supported across all tasks.”

---

# 40. Threats to validity already identified in the conversations

## Internal validity

- calibration split reuse/leakage,
- checkpoint-selection differences,
- inconsistent wavelength metadata,
- dropout placement,
- ensemble independence,
- invalid/abandoned run directories.

## Construct validity

- ECE bin sensitivity,
- global pixel metrics hiding foreground errors,
- entropy/MI not being uniquely identifiable uncertainty sources,
- calibration not equivalent to uncertainty usefulness.

## External validity

- only two main EO FMs,
- selected datasets/tasks,
- possible benchmark/pretraining ecosystem overlap,
- EO sensor/domain specificity.

## Statistical validity

- limited seeds,
- correlated pixels in segmentation,
- undefined per-image boundary metrics,
- calibration estimates depending on sample size.

---

# 41. High-value optional analyses that do not require retraining

If raw artifacts exist, Codex can often add scientific depth without new training:

- confidence histograms for correct vs incorrect predictions,
- uncertainty/error AUROC,
- risk–coverage/AURC,
- classwise ECE,
- calibration by confidence decile,
- paired bootstrap CIs,
- per-image segmentation calibration,
- foreground/background split,
- boundary vs interior split,
- logit-norm distribution,
- margin distribution,
- correlations between ECE/NLL/Brier and performance across cells,
- predictive entropy / expected entropy / MI from existing MCD/ensemble arrays.

These are preferable to starting a new large training campaign.

---

# 42. Summary of “user ideas” versus “assistant recommendations”

## User-originated recurring ideas

- The thesis should not stop at a few calibration metrics; it should explain why uncertainty behaves as observed.
- AU/EU are not fixed, universally separable properties of data and model.
- The same sample's uncertainty can change with model, representation, task, and training distribution.
- Scaling logits can reduce entropy without adding evidence.
- Good UQ should respond to error, ambiguity, and distribution shift.
- The thesis should directly analyze EO FM design if possible, not only treat models as black boxes.
- After the experimental matrix is complete, the priority should be to answer the original three questions rather than keep adding methods.

## Assistant-originated recurring recommendations

- Separate calibration from accuracy and from UQ.
- Use ECE jointly with NLL, Brier, and reliability diagrams.
- Use deterministic baselines before UQ methods.
- Treat TS as calibration.
- Use MCD and Deep Ensembles as representative stochastic UQ methods.
- Preserve raw stochastic predictions.
- Compare frozen vs full fine-tuning through paired deltas.
- Add risk–coverage/error-detection analyses where possible.
- Audit wavelength units, splits, gradients, ignore masks, and checkpoint criteria.
- Use a shared lightweight segmentation decoder.
- Stop core training once the RQ evidence matrix is complete.
- Move AU/EU coupling and model-internal mechanism work to optional extensions unless needed to explain a central result.

---

# 43. Final decision log

## Final / authoritative for core thesis

- Thesis topic: uncertainty/calibration of EO foundation models after downstream adaptation.
- Main FMs: DOFA, Panopticon.
- Main classification datasets: EuroSAT, TreeSatAI.
- Main segmentation datasets: CloudSEN12, SpaceNet7.
- Adaptation: frozen vs full fine-tuning.
- Methods: deterministic, TS, MCD, Deep Ensemble.
- Classification checkpoint: validation NLL.
- Segmentation checkpoint: validation mIoU.
- Classification metrics: Accuracy, ECE-15, NLL, Brier, reliability.
- Segmentation metrics: mIoU, per-class IoU, pixel accuracy, ECE-15, NLL, Brier.
- SpaceNet7: add foreground/building and boundary calibration diagnostics.
- Preserve sample-level and stochastic prediction artifacts.
- Answer RQ1–RQ3 before any extension.

## Important final limitations / caveats

- TreeSatAI independent calibration split issue.
- Cross-task checkpoint criteria differ.
- SpaceNet7 low building IoU is genuine and global pixel metrics can mislead.
- Per-image boundary ECE may be undefined.
- AU/EU labels are operational, not absolute.

## Strong optional additions

- risk–coverage / error detection,
- logit/margin analysis,
- representation drift / CKA,
- model-internal ablations,
- predictive entropy/expected entropy/MI from stored stochastic outputs.

## Deferred / historical

- RESISC45 and So2Sat as main final datasets,
- BNN/EDL/Laplace/conformal as main final method conditions,
- full corruption/OOD sweep,
- label-scarcity sweep,
- third FM,
- theory-first AU/EU coupling thesis.

---

# 44. Minimal glossary

**Calibration:** Agreement between predicted confidence/probabilities and empirical outcome frequencies.

**Confidence:** Usually the maximum predicted class probability for top-label classification; not identical to evidence.

**Predictive uncertainty:** Uncertainty represented by the predictive distribution of a specific model/system.

**ECE:** Binned discrepancy between empirical accuracy and mean confidence.

**NLL:** Proper scoring rule that strongly penalizes confident errors.

**Brier score:** Squared probability error over the full class-probability vector.

**Reliability diagram:** Plot of empirical accuracy versus confidence across bins.

**Temperature Scaling:** One-parameter post-hoc logit rescaling fitted on labeled calibration data.

**MC Dropout:** Repeated stochastic inference using dropout masks.

**Deep Ensemble:** Aggregation of independently trained model predictions.

**Predictive entropy:** Entropy of the averaged predictive distribution.

**Expected entropy:** Average entropy of stochastic member predictions.

**Mutual information / disagreement:** Difference between predictive entropy and expected entropy.

**Frozen backbone:** Foundation-model parameters fixed; downstream task components trained.

**Full fine-tuning:** Foundation model and downstream task components trained jointly.

**mIoU:** Mean intersection-over-union for segmentation.

**Foreground calibration:** Calibration computed on the positive/foreground class rather than all pixels.

**Boundary calibration:** Calibration restricted to segmentation boundary pixels.

---

# 45. Final instruction to Codex

Before drafting prose, first inspect the repository and map actual artifacts to this dossier. Resolve any remaining discrepancy in exact seeds, hyperparameters, MCD sample count, ensemble size, TS availability, and the formal Pre-UQ freeze status from the files themselves.

Then write the thesis so that the central story is simple and defensible:

1. EO foundation models can produce strong downstream predictions, but probability reliability is not guaranteed.
2. Measure the raw calibration of DOFA and Panopticon on classification and segmentation.
3. Determine how frozen versus full fine-tuning changes both task performance and calibration.
4. Determine whether TS, MC Dropout, and Deep Ensembles improve calibration and at what performance/computational cost.
5. Interpret the results using model–data interaction, logit geometry, and cautious uncertainty semantics.
6. Treat AU/EU decomposition, distribution shift, and model-internal mechanism analysis as deeper extensions unless supported by completed artifacts.

The manuscript should answer the three research questions explicitly and separately in the Results and again in the Conclusion.

---

# 46. Detailed chronological conversation ledger

This ledger is intentionally redundant with the thematic sections above. Its purpose is to preserve the actual evolution of the discussion, including ideas that were later revised. Dates are approximate conversation dates from the recovered chat history.

## 23 January 2025 — earliest UQ/EO framing recovered

The user's research context already involved uncertainty quantification for deep learning in EO, including MC Dropout and Bayesian neural networks and EO tasks such as classification, segmentation, and anomaly detection.

The assistant's framing emphasized confidence/reliability, interpretability, robustness, and uncertainty sources in environmental monitoring and land-use applications.

At this point the discussion was broad and did not yet focus on EO foundation models or downstream calibration.

## 8 July 2025 — broad thesis framework

The thesis was described as uncertainty quantification in machine learning for EO/remote sensing.

The assistant proposed a generic framework covering:

- Bayesian learning,
- ensemble learning,
- epistemic versus aleatoric uncertainty,
- MC Dropout,
- Deep Ensembles,
- uncertainty maps,
- Accuracy, Brier, ECE,
- data quality and label noise,
- interpretability,
- task context,
- OOD/domain-shift degradation,
- computation/inference trade-offs.

This was still method-taxonomy driven rather than calibration-first.

## 24 July 2025 — UQ as a research hotspot

In a PhD-oriented context, EO UQ was framed as an active research direction involving Bayesian methods, ensembles, post-hoc calibration, and conformal prediction.

Applications discussed included crop classification, atmospheric retrieval, and hazard mapping.

## 28 July 2025 — first extended method limitations discussion

The user requested a complete thesis-style first draft and later asked that method limitations be written naturally rather than as isolated bullet points.

The assistant discussed:

### Bayesian Neural Networks

- posterior distributions over weights,
- theoretically principled epistemic UQ,
- intractable exact inference,
- reliance on approximations such as variational inference,
- posterior approximation bias,
- scaling problems for modern deep networks.

### MC Dropout

- repeated dropout masks at inference,
- approximate Bayesian interpretation,
- repeated forward-pass cost,
- dependence on dropout rate and placement,
- not a strict posterior sampler.

### Deep Ensembles

- independently initialized/trained models,
- robust empirical performance,
- substantial compute/storage cost,
- ensemble diversity is not guaranteed,
- ordinary classification ensembles do not automatically model aleatoric noise.

### Evidential Deep Learning

- direct prediction of distributional evidence,
- potential single-pass UQ,
- training instability,
- sensitivity to regularization,
- difficult semantics,
- relatively limited EO benchmarking/reproducibility.

EO uncertainty sources were also separated into sensor/environment/label ambiguity versus model/data-support limitations. This conventional separation was later challenged by the user.

## 15–16 September 2025 — supervisor-calibration plan becomes central

The supervisor's intended project was reconstructed as:

- use TorchGeo and lightning-uq-box,
- compare 2–3 pretrained EO models,
- begin with classification because segmentation resources were initially less settled,
- use EuroSAT/RESISC45-like datasets,
- evaluate both predictive performance and calibration,
- then add UQ methods and measure calibration gains and accuracy effects.

On 15 September, ECE was explained with its binning formula and a worked numerical example. NLL and Temperature Scaling were also discussed in detail.

Important conclusions established here:

- NLL can worsen even while accuracy improves because confident errors become more heavily penalized;
- scalar TS is fitted by validation NLL;
- scalar TS preserves argmax and therefore should preserve classification accuracy;
- the test set must not be used to fit T.

## 15 October 2025 — research-learning roadmap

The user asked for a pipeline to become research-qualified in UQ.

The assistant proposed a learning path spanning:

1. EO/DL/statistical foundations,
2. aleatoric and epistemic uncertainty,
3. BNNs,
4. MC Dropout,
5. Deep Ensembles,
6. Evidential Deep Learning,
7. calibration and Temperature Scaling,
8. ECE/NLL/Brier/reliability diagrams,
9. interval coverage/sharpness,
10. uncertainty maps,
11. EO datasets/tasks,
12. gaps in spatial calibration, sensor transfer, physics integration, multimodal uncertainty propagation.

This established the broader intellectual background later used in Related Work.

## 20 October 2025 — predictive uncertainty and method comparison

The assistant again framed predictive uncertainty conventionally as epistemic + aleatoric and reviewed BNN, MC Dropout, Deep Ensembles, and EDL.

Representative literature was discussed, including Gal & Ghahramani for MC Dropout, Lakshminarayanan et al. for Deep Ensembles, and Guo et al. for Temperature Scaling.

TS was explicitly categorized as calibration/post-hoc adjustment rather than a UQ method in the same sense as stochastic model uncertainty.

## 29 December 2025 — milestone-style thesis execution

An early milestone architecture was proposed:

- Milestone 0: precise research claim,
- Milestone 1: reusable reproducible pipeline,
- Milestone 2: baseline calibration without explicit UQ,
- later UQ experiments.

The assistant stressed that a good thesis requires reproducible controlled comparisons and systematic calibration analysis, not one impressive metric.

## 15–16 January 2026 — EO FM calibration thesis formalization

The user described the thesis as evaluating whether fine-tuned EO foundation-model predictions are reliable/calibrated, comparing pretrained ViT-style FMs and CNN baselines across classification/segmentation with TorchGeo and Lightning-UQ-Box.

The assistant proposed:

- Temperature Scaling,
- MC Dropout,
- Deep Ensembles,
- Laplace approximation,
- ECE and reliability diagrams,
- accuracy/calibration/compute trade-offs.

The title **“Uncertainty Quantification for Earth Observation Foundation Models”** was considered suitable.

## 20 May 2026 — Monte Carlo predictive averaging

The discussion clarified that Bayesian predictive inference averages predictions over a distribution of model parameters, approximated with Monte Carlo samples.

Key point: T=1 is just one model sample; meaningful uncertainty usually requires repeated predictive samples.

MC Dropout was described as generating stochastic model realizations by varying dropout masks.

Recommended metrics included Accuracy, NLL, Brier, ECE, reliability, and confidence histograms.

## 22–23 May 2026 — limited-label foundation-model motivation

The user emphasized that EO FMs are valuable under limited labels and that classification should be treated as a controlled calibration/UQ benchmark rather than the ultimate EO problem.

The assistant suggested comparing label fractions and studying calibration under reduced supervision. This later remained an extension rather than a core experiment.

## 31 May 2026 — first concrete DOFA/EuroSAT experiment prompt

The pipeline was operationalized.

Sequence proposed:

- EuroSAT RGB first,
- then multispectral,
- DOFA and Panopticon,
- calibration metrics,
- Temperature Scaling,
- MC Dropout / ensembles,
- segmentation.

Early formal config:

- EuroSAT RGB B04/B03/B02,
- pretrained DOFA,
- frozen backbone,
- 10-class linear head,
- AdamW,
- LR 1e-3,
- weight decay 1e-4,
- batch 32,
- 20 epochs,
- CE loss,
- validation Accuracy/NLL/ECE/Brier,
- checkpoint by validation NLL.

A ResNet18 baseline and shared split/metrics were to be preserved.

Also on this date, NLL was explicitly explained as a metric that is sensitive to confident errors but is not a pure calibration statistic.

## 1 June 2026 — calibration methods and conceptual deepening

The user asked for detailed Platt scaling and Temperature Scaling.

The assistant explained:

- binary Platt scaling: sigmoid(az+b),
- multiclass vector/matrix scaling,
- TS: softmax(z/T),
- T>1 softens overconfidence,
- global T has limited flexibility,
- TS should be fitted on held-out data by minimizing NLL.

Later that day the user asked a more important question: the thesis should not merely compute ECE/NLL/Brier; it should explain the cause of uncertainty and miscalibration.

The assistant proposed:

- uncertainty quantiles versus error,
- risk–coverage/AURC,
- confidently wrong case review,
- per-class calibration,
- corruption/OOD testing,
- class imbalance and label ambiguity,
- linear/partial/full FT comparison.

The user then challenged the usual aleatoric/epistemic distinction, arguing that uncertainty changes with model–dataset combination, representation, labels, task, and modality.

The assistant agreed with a conditional framing and recommended avoiding claims that the observed uncertainty can always be uniquely assigned to “data uncertainty” versus “model uncertainty.”

## 5 June 2026 — BNN and Deep Ensemble uncertainty formulas

For BNN-style stochastic predictions, the assistant explained:

- predictive entropy as aggregate predictive uncertainty,
- expected entropy as an aleatoric-like term,
- MI as a disagreement/epistemic-like term,
- variation ratio and probability variance as alternative disagreement measures.

For regression, the law-of-total-variance-style decomposition was discussed:

Var(y|x,D) = E_q[σ_w²] + Var_q[μ_w].

A critical BNN implementation point was raised: averaging over posterior predictive probabilities is essential; one sampled network is not representative.

Deep Ensembles were explained similarly:

- average probabilities,
- entropy of the average,
- member variance/disagreement,
- MI-style decomposition,
- compute cost roughly proportional to ensemble size.

A small three-member ensemble was proposed as a practical EO compromise.

## 6 June 2026 — broader UQ related-work map

The assistant discussed OOD uncertainty, calibration, quantile regression, MC-DropConnect, Gaussian processes, conformal prediction, Platt/isotonic scaling, and related alternatives.

The recommended empirical line remained:

**deterministic baseline → calibration evaluation → TS/MCD/DE → Accuracy/ECE/NLL/Brier/reliability**.

## 8 June 2026 — thesis becomes representation/adaptation-centric

The user explicitly rejected a shallow “run a few methods” thesis and wanted transferable understanding of model principles.

The assistant proposed a conceptual learning/research ladder:

1. statistical learning and generalization,
2. representation learning,
3. foundation-model pretraining paradigms,
4. logits/softmax and calibration,
5. UQ,
6. EO spatiotemporal/multimodal structure,
7. OOD/distribution shift,
8. cross-domain reliability.

The experimental axis proposed:

- ResNet baseline,
- DOFA frozen,
- partial FT,
- full FT,
- TS,
- MCD,
- DE,
- label scarcity,
- domain shift.

The user supplied an early configuration with seed 42, 20 epochs, batch 32, AdamW, LR 0.001, wd 0.0001, CE, ECE-15, UQ=none.

The assistant emphasized that an EO FM should not merely become confident; it should maintain uncertainty under ambiguous or shifted evidence.

## 9 June 2026 — DOFA mechanism clarification

DOFA was discussed as a wavelength-conditioned dynamic patch-embedding generator with a shared Transformer backbone.

Important conceptual detail: wavelength configuration affects generated embedding weights. This made wavelength metadata part of the scientific protocol, not merely engineering metadata.

## 11 June 2026 — DOFA prior evaluation and contamination awareness

The assistant summarized DOFA/DOFA+ pretraining and downstream benchmark history, including GEO-Bench/PANGEA-type tasks.

The user's thesis gap was sharpened:

> strong transfer accuracy does not prove reliable calibrated probabilities.

This also contributed to later concern that using datasets already embedded in a model's evaluation ecosystem weakens independent reliability claims.

## 17 June 2026 — field positioning

UQ was positioned at the intersection of:

- probabilistic ML,
- Bayesian deep learning,
- calibration,
- trustworthy/reliable AI,
- OOD/distribution shift.

The distinction between calibration and UQ was repeated.

## 18–19 June 2026 — distribution-shift extension

The assistant proposed clean training/validation and corrupted evaluation using noise, blur, brightness changes and multiple severities.

Metrics:

- accuracy,
- ECE,
- NLL,
- mean confidence,
- uncertainty separation,
- risk–coverage,
- error-detection performance.

The scientific target was whether uncertainty increases as evidence quality degrades and whether calibration fails under shift.

## 20 June 2026 — feasibility refocus

The user reiterated the supervisor requirements and favored a feasible thesis over turning the project into an unbounded general theory of foundation-model uncertainty.

The assistant recommended retaining sensor/spatial/temporal shift as a possible extension but prioritizing the supervisor's model × adaptation × calibration/UQ matrix.

## 30 June 2026 — Wimmer23a critique enters

The standard entropy decomposition was analyzed:

TU = H(Y), AU = H(Y|Θ), EU = I(Y;Θ).

The assistant stressed:

- the information identity is valid,
- the semantic interpretation is the disputed part,
- MI is disagreement and is not automatically synonymous with ignorance,
- the expected entropy term can depend on the learner's epistemic state.

Examples compared a continuous distribution over Bernoulli probabilities with an extreme two-point mixture to show how the same predictive uncertainty can mask very different member structures.

A model + data + model×data interaction analysis was suggested.

## 6 July 2026 — deeper Wimmer interpretation

The same Q-over-probability-vectors view was elaborated:

- TU depends only on the mean predictive distribution,
- AU/EU depend on the shape of Q,
- same mean can correspond to ambiguity or conflict.

The distinction between an information-theoretic identity and a causal uncertainty-source claim was reinforced.

## 9–14 July 2026 — proper scoring rules and AU/EU coupling

The decomposition was generalized beyond log loss.

For a proper scoring rule/loss l, the discussion introduced an abstract form like:

- TU_l = E_Q[L_l(θ̄, θ)],
- AU_l = E_Q[L_l(θ, θ)],
- EU_l = E_Q[D_l(θ̄, θ)] = TU_l − AU_l.

For log loss, EU becomes KL/MI-like disagreement.

For Brier-like scoring, the decomposition has a Gini/squared-distance geometry.

For zero-one-style quantities, disagreement can miss probability-shape differences if members retain the same top class.

This led to a major conclusion:

**TU=AU+EU is exact relative to a chosen decomposition/scoring rule, not a universal physical law that uniquely identifies uncertainty sources.**

The user wanted to make AU–EU coupling a genuine research contribution. The assistant proposed:

- interaction contrast C,
- mixed derivative κ,
- global interaction measures,
- leakage/contamination matrices,
- synthetic ground-truth uncertainty,
- EO mixed-pixel experiments,
- then FM-specific analysis.

By 13–14 July, the conceptual claim was refined to:

> formal additive decomposition may hold while source identifiability, intervention stability, and model–data separability fail.

## 20 July 2026 — broader research positioning

For PhD positioning, the thesis was described as studying whether uncertainty measures remain stable/identifiable across architecture, sensor modality, adaptation, and data distribution.

The assistant cautioned not to overstate experiments that had not yet been completed.

## 3 August 2026 — preserving future AU/EU research while finishing the empirical thesis

The assistant proposed RQs around adaptation/calibration/UQ and recommended preserving raw logits/probabilities/stochastic predictions for later theory.

A key wording recommendation was repeated:

- predictive entropy / expected entropy / MI are mathematically standard;
- their interpretation as true AU/EU remains unresolved;
- save raw outputs so alternative analyses remain possible.

## 8 August 2026 — supervisor scope reasserted

The user restated the original supervisor assignment and asked whether full fine-tuning was even necessary for UQ.

The assistant answered:

- UQ can be performed with a frozen FM + trained task head;
- however fine-tuning should remain a research axis because the thesis asks about downstream-adapted/fine-tuned EO FMs;
- compare frozen/linear-probe and full FT at minimum;
- partial FT is optional.

The user accepted Panopticon as the controlled second model.

The assistant articulated the main thesis question in a compact form:

> Given comparable downstream tasks and evaluation conditions, do different EO FM representation mechanisms lead to systematically different calibration behavior, and to what extent can post-hoc calibration or UQ methods mitigate these differences?

The thesis core was then explicitly separated from extensions:

- core = DOFA/Panopticon + classification/segmentation + adaptation + calibration/UQ;
- extensions = OOD, AU/EU coupling, representation uncertainty, information theory.

## 9 August 2026 — coding-agent architecture and scientific invariants

The assistant recommended about four major Codex prompts rather than many fragmented prompts.

Critical invariants:

- wavelength semantics,
- frozen gradients,
- optimizer groups,
- dropout stochasticity,
- split integrity,
- calibration leakage,
- prediction export,
- acceptance tests before large training.

A provisional model × adaptation × method factorial design was formalized.

The user also explored So2Sat versus benchmark subsets; the assistant warned that very small validation/test subsets can make ECE/NLL/Brier noisy.

A full original LCZ42 import protocol was designed before the dataset decision later changed.

## 10 August 2026 — TreeSatAI and segmentation protocol

The classification design shifted away from RESISC45/So2Sat to TreeSatAI.

The assistant recommended TreeSatAI because of standardized protocol, lower known evaluation overlap, and useful spectral/temporal properties.

For segmentation, the assistant proposed:

- DOFA/Panopticon × CloudSEN12/SpaceNet7,
- shared lightweight UNet-style decoder,
- only model-specific dense feature adapters,
- frozen = backbone fixed, adapter+decoder trainable,
- full = all trainable,
- mIoU/per-class IoU/pixel accuracy/NLL/Brier/ECE-15,
- SpaceNet7 foreground/classwise/boundary calibration.

A linear segmentation head was retained only as a possible diagnostic because it was considered too weak as the main dense-prediction head.

## 23 August 2026 — extended final protocol suggestions

The assistant recommended a richer analysis layer:

- mean confidence and signed calibration gap,
- clean-to-shift testing,
- multiple seeds,
- risk–coverage/AURC,
- entropy/MI,
- sample-level artifact storage,
- SpaceNet7 foreground/boundary diagnostics,
- optional CKA/representation drift.

Some of these were enhancements beyond the mandatory matrix.

## 25 August 2026 — C-stage audit, Pre-UQ freeze, MCD protocol

The user's audit reported:

- deterministic and TS/ensemble artifacts largely valid,
- CloudSEN12 one-pixel ULP issue,
- SpaceNet7 low building IoU genuine,
- boundary-free images causing undefined per-image boundary ECE,
- TreeSatAI no independent calibration split,
- abandoned zero-epoch seed-42 directories excluded.

At this point **Pre-UQ Protocol Freeze had not yet been executed**.

Recommended order:

1. checkpoint-selection sensitivity audit,
2. Pre-UQ Protocol Freeze,
3. MCD pilot/protocol,
4. full/targeted MCD execution,
5. result synthesis.

MCD recommendation:

- downstream head/decoder dropout,
- backbone preserved,
- T in {10,20,30,50} pilot,
- freeze about T=30 if stable,
- avoid unnecessarily doing 3 seeds for every MCD cell if compute can be focused on a predefined robustness subset.

## 26 August 2026 — A–C considered complete; stop expanding

The user stated that all A–C execution was complete and asked how to use reports and results to write the thesis.

The assistant's main recommendation was:

**stop expanding the main experiment matrix.**

Instead:

- create a thesis master evidence matrix,
- build paired delta tables,
- compute mean±std and CIs,
- run foreground/boundary analysis,
- summarize compute cost,
- write Methods → Results → Discussion around RQ1–RQ3.

New training should occur only if a core evidence cell is missing or invalid.

## 5 September 2026 — direct model-mechanism analysis

The user asked how to explain uncertainty sources theoretically by studying the actual model rather than only experiment outputs. The user was willing to retrain and now had 4×3090 GPUs.

The assistant proposed:

- representation information loss,
- DOFA wavelength-conditioned embeddings,
- Panopticon cross-view/channel mechanisms,
- CKA/representation probing,
- logit geometry and softmax scaling,
- frozen vs full FT,
- shared-backbone UQ blind spots,
- Laplace/Hessian curvature,
- JΣJᵀ propagation.

A four-condition intervention ladder was proposed:

- frozen,
- only input-adaptation trainable,
- only Transformer trainable,
- full FT.

Three seeds per model/condition were suggested for a dedicated mechanism study, parallelized on 4×3090.

This was an optional deeper study, not required to complete the supervisor's main thesis.

## 18 September 2026 — multi-agent audit criteria

The user wanted Codex and Claude Code to audit whether the thesis actually answered the three required questions.

The assistant emphasized:

- distinguish “question answered” from “method successful”;
- a null result still answers RQ3;
- standard scalar TS should not change classification argmax;
- clearly distinguish training seeds, MCD sample count, and ensemble member count;
- audit omissions/inconsistent comparisons rather than simply looking for more experiments.

## 22 September 2026 — current-state summary

The final system was summarized as:

- DOFA/Panopticon,
- EuroSAT/TreeSatAI,
- CloudSEN12/SpaceNet7,
- frozen/full,
- deterministic/TS/MCD/DE,
- classification checkpoint by validation NLL,
- segmentation checkpoint by mIoU,
- 62 audit tests passed,
- known numerical and segmentation caveats documented.

Mechanism work remained an enhancement after the core RQs.

## 25 September 2026 — thesis-writing transition

The user again stated that all A–C experiments were complete and asked how to use existing data/reports for the initial thesis draft.

The assistant reiterated:

- evidence tables must map every experiment directly to RQ1/RQ2/RQ3,
- statistical analysis should be done before adding new experiments,
- new experiments are only justified if existing evidence cannot answer a question or lacks minimum robustness/ablation support.

This is the immediate state from which Codex should proceed.

---

# 47. Proper-scoring-rule view of uncertainty decomposition

This theoretical branch was important enough to preserve separately because it prevents the thesis from overstating entropy-based AU/EU.

Let Q represent a distribution over predictive probability vectors θ, and let θ̄ = E_Q[θ]. A family of uncertainty decompositions can be constructed from a proper scoring rule or associated entropy/divergence.

Conceptually:

- total uncertainty is the expected loss associated with using the mean predictive distribution,
- aleatoric-like uncertainty is the expected intrinsic entropy/loss of each member distribution,
- epistemic-like uncertainty is the divergence between member distributions and the mean prediction.

For log loss:

- total uncertainty reduces to Shannon entropy of θ̄,
- expected member entropy is the conditional-entropy term,
- the difference is an average KL divergence / mutual-information quantity.

For Brier-type scores:

- the geometry becomes squared Euclidean distance / Gini-type uncertainty,
- the decomposition remains valid but the numerical AU/EU values change.

For zero-one-style disagreement measures:

- two members with the same top class can have radically different probability shapes yet register no top-class disagreement.

Therefore, even before asking whether AU/EU are “real” causal sources, the measured decomposition depends on what scoring geometry the researcher chooses.

This supports cautious thesis language:

> The study uses entropy/disagreement quantities as operational uncertainty diagnostics associated with the predictive distribution, rather than assuming a unique source decomposition independent of the scoring rule and model.

---

# 48. Full inventory of ideas proposed during the project, classified by status

This section exists so Codex does not accidentally omit an idea when scanning the history.

## A. Core and completed/expected to be completed

- deterministic EO FM downstream calibration,
- DOFA,
- Panopticon,
- EuroSAT,
- TreeSatAI,
- CloudSEN12,
- SpaceNet7,
- frozen adaptation,
- full fine-tuning,
- Temperature Scaling,
- MC Dropout,
- Deep Ensembles,
- Accuracy,
- mIoU,
- per-class IoU,
- pixel accuracy,
- NLL,
- Brier,
- ECE-15,
- reliability diagrams,
- foreground/building calibration,
- boundary calibration,
- task-specific checkpointing,
- wavelength/channel audit,
- prediction/logit/probability export,
- reproducibility tests,
- invalid-run exclusion.

## B. Strong analysis-only additions if artifacts permit

- paired frozen→full delta tables,
- paired deterministic→method delta tables,
- mean±std,
- bootstrap CIs,
- confidence histograms,
- signed calibration gap,
- classwise ECE,
- error-detection AUROC/AUPRC,
- risk–coverage/AURC,
- predictive entropy,
- expected entropy,
- mutual information,
- member variance/disagreement,
- foreground/background stratification,
- boundary/interior stratification,
- logit norms,
- logit margins.

## C. Mechanism additions requiring limited extra analysis or retraining

- CKA / representation similarity,
- representation drift,
- input-adapter-only fine-tuning,
- Transformer-only fine-tuning,
- gradient visualization,
- gradient norm/sensitivity analysis,
- Hessian/Laplace curvature,
- JΣJᵀ predictive variance propagation,
- wavelength perturbation tests,
- channel ablations,
- feature geometry analysis.

## D. Distribution/robustness extensions

- Gaussian noise,
- blur,
- brightness/contrast changes,
- cloud masking,
- band dropout,
- resolution degradation,
- geographic/sensor/seasonal shift,
- clean calibration then shifted testing,
- uncertainty-severity monotonicity.

## E. Data-regime extensions

- 100/20/10/5% label fractions,
- controlled mixed pixels,
- synthetic known-AU benchmark,
- repeated-data versus optimization variance.

## F. Methods considered but not retained in final core

- BNN,
- Evidential Deep Learning,
- Laplace as a main experimental condition,
- conformal prediction,
- RAPS,
- Platt scaling,
- vector scaling,
- matrix scaling,
- isotonic regression,
- TTA,
- DropConnect,
- GP/DGP approaches,
- SWAG-like methods.

## G. Candidate models/datasets considered but not final core

- ResNet18 / CNN baseline as primary comparison rather than smoke/control,
- generic ViT baseline,
- third EO FM such as TerraMind or SMARTIES,
- RESISC45,
- So2Sat/LCZ42,
- GEO-Bench classification subsets,
- Cameroon alternative.

## H. Theoretical research directions

- AU/EU identifiability,
- intervention stability,
- model×data interaction,
- source separability,
- interaction contrast C,
- mixed derivative κ,
- global interaction indices,
- estimator leakage matrix,
- proper-scoring-rule dependence,
- uncertainty as residual compressibility,
- “compression hallucination” interpretation of overconfident high-NLL errors.

None of these theoretical directions should be silently promoted to a completed thesis contribution unless supported by an explicit derivation and experiment in the final repository/manuscript.
