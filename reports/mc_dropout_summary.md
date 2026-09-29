# Final MC Dropout Results

Created: 2026-08-26T17:14:28.738319+00:00

All 24 models are newly trained downstream-head/decoder MC-Dropout models. No deterministic checkpoint was modified or retrained. The primary analysis is the predeclared seed-42 paired comparison in all 16 cells. Each predictive distribution is the arithmetic mean of 30 probability passes; logits are never averaged.

## Primary paired matrix (seed 42)

| Task | Dataset | Model | Adaptation | Accuracy or mIoU MC / deterministic | Macro-F1 or pixel accuracy MC / deterministic | NLL MC / deterministic | Brier MC / deterministic | ECE-15 MC / deterministic | Predictive entropy | Expected predictive entropy | MI-style disagreement |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| classification | eurosat | dofa | frozen | 0.985262 / 0.983051 | 0.984358 / 0.982032 | 0.049743 / 0.058109 | 0.024561 / 0.026906 | 0.008762 / 0.004231 | 0.058280 | 0.050708 | 0.007572 |
| classification | eurosat | dofa | full_finetune | 0.958732 / 0.970892 | 0.956901 / 0.970073 | 0.125732 / 0.083691 | 0.063580 / 0.043979 | 0.010823 / 0.007804 | 0.130661 | 0.129792 | 0.000869 |
| classification | eurosat | panopticon | frozen | 0.983051 / 0.983788 | 0.982313 / 0.983230 | 0.048680 / 0.051499 | 0.024773 / 0.025839 | 0.004673 / 0.007696 | 0.040987 | 0.037133 | 0.003854 |
| classification | eurosat | panopticon | full_finetune | 0.957996 / 0.967944 | 0.955553 / 0.967447 | 0.122141 / 0.098654 | 0.064204 / 0.048209 | 0.006936 / 0.005855 | 0.117148 | 0.116263 | 0.000885 |
| classification | treesatai | dofa | frozen | 0.255000 / 0.255000 | 0.216242 / 0.219526 | 0.256165 / 0.256531 | 0.074336 / 0.074177 | 0.008883 / 0.009970 | 3.708322 | 3.638894 | 0.069429 |
| classification | treesatai | dofa | full_finetune | 0.282500 / 0.265000 | 0.239423 / 0.248274 | 0.247913 / 0.249935 | 0.071544 / 0.072594 | 0.015532 / 0.014093 | 3.933397 | 3.916784 | 0.016613 |
| classification | treesatai | panopticon | frozen | 0.238000 / 0.239500 | 0.204819 / 0.206796 | 0.258400 / 0.258193 | 0.075799 / 0.075721 | 0.010090 / 0.010479 | 3.943363 | 3.896736 | 0.046627 |
| classification | treesatai | panopticon | full_finetune | 0.290000 / 0.283500 | 0.250963 / 0.244701 | 0.249833 / 0.249930 | 0.072897 / 0.072844 | 0.011062 / 0.011267 | 3.553768 | 3.528246 | 0.025523 |
| segmentation | cloudsen12 | dofa | frozen | 0.600460 / 0.600316 | 0.830804 / 0.832242 | 0.464807 / 0.462548 | 0.242407 / 0.240743 | 0.033069 / 0.019295 | 0.362334 | 0.353328 | 0.009004 |
| segmentation | cloudsen12 | dofa | full_finetune | 0.658796 / 0.656149 | 0.864361 / 0.863083 | 0.456192 / 0.495201 | 0.206561 / 0.212455 | 0.055433 / 0.068018 | 0.205589 | 0.200766 | 0.004823 |
| segmentation | cloudsen12 | panopticon | frozen | 0.643686 / 0.654635 | 0.854819 / 0.859185 | 0.396616 / 0.437820 | 0.207317 / 0.210702 | 0.017883 / 0.049961 | 0.346061 | 0.336721 | 0.009340 |
| segmentation | cloudsen12 | panopticon | full_finetune | 0.649443 / 0.670490 | 0.860373 / 0.870085 | 0.396055 / 0.395478 | 0.203098 / 0.192828 | 0.035384 / 0.042889 | 0.279718 | 0.272622 | 0.007096 |
| segmentation | spacenet7 | dofa | frozen | 0.485379 / 0.487917 | 0.927078 / 0.926494 | 0.277597 / 0.282278 | 0.127304 / 0.127705 | 0.036594 / 0.035681 | 0.114715 | 0.114332 | 0.000383 |
| segmentation | spacenet7 | dofa | full_finetune | 0.489874 / 0.497224 | 0.930730 / 0.923429 | 0.219157 / 0.560002 | 0.116258 / 0.136851 | 0.021510 / 0.056913 | 0.147067 | 0.146476 | 0.000591 |
| segmentation | spacenet7 | panopticon | frozen | 0.500655 / 0.500110 | 0.921697 / 0.918576 | 0.317323 / 0.354484 | 0.132092 / 0.136818 | 0.038801 / 0.041095 | 0.103745 | 0.103408 | 0.000337 |
| segmentation | spacenet7 | panopticon | full_finetune | 0.501278 / 0.509020 | 0.923054 / 0.913300 | 0.414734 / 0.480196 | 0.131250 / 0.144632 | 0.044222 / 0.052910 | 0.077765 | 0.077297 | 0.000468 |

## Primary segmentation class and SpaceNet7 calibration detail

| Dataset | Model | Adaptation | MC per-class IoU | Foreground ECE-15 | Foreground NLL | Foreground Brier | Background classwise ECE-15 | Building classwise ECE-15 | Boundary ECE-15 | Boundary foreground ECE-15 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| cloudsen12 | dofa | frozen | clear=0.815022; cloud shadow=0.485580; thick cloud=0.744020; thin cloud=0.357219 | — | — | — | — | — | — | — |
| cloudsen12 | dofa | full_finetune | clear=0.853208; cloud shadow=0.557121; thick cloud=0.789917; thin cloud=0.434940 | — | — | — | — | — | — | — |
| cloudsen12 | panopticon | frozen | clear=0.842358; cloud shadow=0.518959; thick cloud=0.781562; thin cloud=0.431864 | — | — | — | — | — | — | — |
| cloudsen12 | panopticon | full_finetune | clear=0.848996; cloud shadow=0.528762; thick cloud=0.783501; thin cloud=0.436512 | — | — | — | — | — | — | — |
| spacenet7 | dofa | frozen | background=0.926833; building=0.043926 | 0.041210 | 0.277597 | 0.063652 | 0.041210 | 0.041210 | 0.135423 | 0.142759 |
| spacenet7 | dofa | full_finetune | background=0.930480; building=0.049268 | 0.023024 | 0.219157 | 0.058129 | 0.023024 | 0.023024 | 0.089222 | 0.091644 |
| spacenet7 | panopticon | frozen | background=0.921159; building=0.080151 | 0.050167 | 0.317317 | 0.066046 | 0.050167 | 0.050167 | 0.134699 | 0.155018 |
| spacenet7 | panopticon | full_finetune | background=0.922535; building=0.080021 | 0.054116 | 0.414733 | 0.065625 | 0.054116 | 0.054116 | 0.151363 | 0.170727 |

## Protocol and archive validation

- Frozen dropout probability: `0.1`.
- Frozen stochastic pass count: `30`.
- Classification uses one designated downstream-head `Dropout`; segmentation uses one designated final-decoder-feature `Dropout2d`. Foundation-model native dropout, DropPath, and stochastic depth remain inactive at inference.
- BatchNorm stayed in evaluation mode, inference used no gradients, and checkpoint hashes were verified before and after inference.
- Classification archives contain the complete `[N,T,C]` logits and probabilities plus mean probabilities, labels/predictions, confidence, backbone representations, predictive entropy, expected predictive entropy, MI-style disagreement, and per-class probability variance.
- TreeSatAI retains unreduced per-label Bernoulli entropy/disagreement arrays. Its scalar per-sample and reported entropy/disagreement summaries sum across labels, matching the existing multilabel prediction-export convention.
- Segmentation archives contain complete-test mean-probability, prediction, confidence, predictive-entropy, expected-predictive-entropy, disagreement, and variance maps. Raw `[32,T,C,H,W]` probabilities are retained for each dataset's prospectively fixed research subset.
- SpaceNet7 metrics include overall and building/foreground calibration, one-vs-rest classwise calibration, foreground NLL/Brier, and valid-boundary-only calibration. Images with no valid boundary keep undefined per-image boundary metrics.

## Interpretation limits

`Expected predictive entropy` is the entropy averaged across stochastic predictions. `MI-style disagreement` is reported as stochastic disagreement/an epistemic proxy; neither is claimed to identify true aleatoric or true epistemic uncertainty. The 30 passes are not independent training replicates. Seed-robust claims are limited to the four prospectively replicated cells in the separate robustness report.

EuroSAT retains the frozen backend-specific preprocessing history: DOFA MC runs mirror the immutable DOFA comparator's historical RGB normalization, while Panopticon MC runs mirror the newer final-train-statistics protocol. Cross-model EuroSAT interpretation must disclose this confound. The C4 DOFA pilot used the newer statistics only for technical dropout validation and is not a final result.

Machine-readable results: `/workspace/reports/mc_dropout_results.csv`.
