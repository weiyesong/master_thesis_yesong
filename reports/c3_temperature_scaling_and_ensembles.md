# C3 — Temperature Scaling and Deep Ensembles

Created: 2026-08-24T16:57:37.657095+00:00

No model training was performed. EuroSAT Temperature Scaling uses its dedicated 2,713-sample calibration split; TreeSatAI uses its official validation split because the benchmark has no independent calibration split. Both are evaluated on untouched official test logits. Deep Ensembles use seeds 42/43/44 and arithmetic means of probabilities, never logits.

## Temperature Scaling — EuroSAT

| Model | Adaptation | Seed | T | Accuracy before/after | NLL before/after | Brier before/after | ECE-15 before/after |
|---|---|---:|---:|---:|---:|---:|---:|
| dofa | frozen | 42 | 0.982541 | 0.983051 / 0.983051 | 0.058109 / 0.058096 | 0.026906 / 0.026865 | 0.004231 / 0.003991 |
| dofa | frozen | 43 | 1.014027 | 0.984156 / 0.984156 | 0.050331 / 0.050340 | 0.024243 / 0.024272 | 0.007125 / 0.007518 |
| dofa | frozen | 44 | 0.955440 | 0.983051 / 0.983051 | 0.054730 / 0.054766 | 0.026848 / 0.026787 | 0.003668 / 0.003440 |
| dofa | full_finetune | 42 | 1.085933 | 0.970892 / 0.970892 | 0.083691 / 0.084893 | 0.043979 / 0.044263 | 0.007804 / 0.007663 |
| dofa | full_finetune | 43 | 1.177231 | 0.960575 / 0.960575 | 0.121394 / 0.120048 | 0.059371 / 0.059326 | 0.008804 / 0.006851 |
| dofa | full_finetune | 44 | 1.409426 | 0.963154 / 0.963154 | 0.115748 / 0.107377 | 0.056458 / 0.054499 | 0.016744 / 0.007498 |
| panopticon | frozen | 42 | 1.124308 | 0.983788 / 0.983788 | 0.051499 / 0.050107 | 0.025839 / 0.025500 | 0.007696 / 0.006102 |
| panopticon | frozen | 43 | 1.010639 | 0.983051 / 0.983051 | 0.046422 / 0.046382 | 0.024620 / 0.024597 | 0.003625 / 0.003809 |
| panopticon | frozen | 44 | 1.111715 | 0.983051 / 0.983051 | 0.053462 / 0.052007 | 0.026448 / 0.026216 | 0.008543 / 0.007056 |
| panopticon | full_finetune | 42 | 1.238321 | 0.967944 / 0.967944 | 0.098654 / 0.100563 | 0.048209 / 0.048618 | 0.005855 / 0.009531 |
| panopticon | full_finetune | 43 | 1.124329 | 0.965733 / 0.965733 | 0.084185 / 0.085478 | 0.047577 / 0.047421 | 0.007314 / 0.007554 |
| panopticon | full_finetune | 44 | 1.224584 | 0.954311 / 0.954311 | 0.155008 / 0.148824 | 0.071480 / 0.070815 | 0.015190 / 0.007396 |

## Temperature Scaling — TreeSatAI

One positive scalar temperature was fitted per checkpoint on official validation logits only. Validation was also used for checkpoint selection; this declared dual use is required because the official benchmark has no separate calibration split. Test logits were untouched during fitting and no test result informed temperature acceptance or selection.

| Model | Adaptation | Seed | T | Accuracy before/after | Macro-F1 before/after | NLL before/after | Brier before/after | ECE-15 before/after | Argmax unchanged |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| dofa | frozen | 42 | 1.085235 | 0.255000 / 0.255000 | 0.219526 / 0.219526 | 0.256531 / 0.256121 | 0.074177 / 0.074355 | 0.009970 / 0.011744 | true |
| dofa | frozen | 43 | 1.027817 | 0.258000 / 0.258000 | 0.226912 / 0.226912 | 0.255057 / 0.255108 | 0.073773 / 0.073864 | 0.010719 / 0.011696 | true |
| dofa | frozen | 44 | 1.015037 | 0.257000 / 0.257000 | 0.223055 / 0.223055 | 0.255944 / 0.255984 | 0.073994 / 0.074045 | 0.011730 / 0.012304 | true |
| dofa | full_finetune | 42 | 1.027804 | 0.265000 / 0.265000 | 0.248274 / 0.248274 | 0.249935 / 0.250613 | 0.072594 / 0.072761 | 0.014093 / 0.017499 | true |
| dofa | full_finetune | 43 | 1.119862 | 0.265500 / 0.265500 | 0.229821 / 0.229821 | 0.243910 / 0.244470 | 0.070658 / 0.070935 | 0.008119 / 0.013037 | true |
| dofa | full_finetune | 44 | 0.992058 | 0.270000 / 0.270000 | 0.253375 / 0.253375 | 0.246731 / 0.246567 | 0.071820 / 0.071779 | 0.015452 / 0.014460 | true |
| panopticon | frozen | 42 | 1.056240 | 0.239500 / 0.239500 | 0.206796 / 0.206796 | 0.258193 / 0.258659 | 0.075721 / 0.075909 | 0.010479 / 0.012975 | true |
| panopticon | frozen | 43 | 1.067136 | 0.239500 / 0.239500 | 0.217086 / 0.217086 | 0.254583 / 0.254919 | 0.074662 / 0.074837 | 0.008727 / 0.011121 | true |
| panopticon | frozen | 44 | 1.079910 | 0.234500 / 0.234500 | 0.208658 / 0.208658 | 0.258218 / 0.258241 | 0.075744 / 0.075916 | 0.009281 / 0.011152 | true |
| panopticon | full_finetune | 42 | 1.148707 | 0.283500 / 0.283500 | 0.244701 / 0.244701 | 0.249930 / 0.250548 | 0.072844 / 0.073222 | 0.011267 / 0.015708 | true |
| panopticon | full_finetune | 43 | 1.069310 | 0.250000 / 0.250000 | 0.218528 / 0.218528 | 0.246238 / 0.246611 | 0.071773 / 0.071852 | 0.004158 / 0.008068 | true |
| panopticon | full_finetune | 44 | 1.031815 | 0.221000 / 0.221000 | 0.196027 / 0.196027 | 0.255205 / 0.255699 | 0.074941 / 0.075052 | 0.007464 / 0.011037 | true |
## Classification Deep Ensembles

| Dataset | Model | Adaptation | Accuracy | Macro-F1 | NLL | Brier | ECE-15 |
|---|---|---|---:|---:|---:|---:|---:|
| eurosat | dofa | frozen | 0.983788 | 0.982741 | 0.051996 | 0.024748 | 0.005501 |
| eurosat | dofa | full_finetune | 0.982682 | 0.982283 | 0.063951 | 0.030502 | 0.027278 |
| eurosat | panopticon | frozen | 0.982682 | 0.981954 | 0.047755 | 0.024903 | 0.006773 |
| eurosat | panopticon | full_finetune | 0.976419 | 0.975289 | 0.072574 | 0.036389 | 0.021987 |
| treesatai | dofa | frozen | 0.260500 | 0.224077 | 0.253751 | 0.073480 | 0.010428 |
| treesatai | dofa | full_finetune | 0.274000 | 0.237644 | 0.237553 | 0.068733 | 0.017046 |
| treesatai | panopticon | frozen | 0.239000 | 0.207461 | 0.254693 | 0.074749 | 0.008805 |
| treesatai | panopticon | full_finetune | 0.265500 | 0.224298 | 0.241657 | 0.070432 | 0.009754 |

TreeSatAI metrics follow its multilabel protocol: accuracy is strict exact match, Macro-F1 is label-macro positive-class F1, NLL/Brier average sample-label decisions, and ECE-15 flattens binary decisions. They are not directly comparable to EuroSAT multiclass metric definitions.

## Segmentation Deep Ensembles

| Dataset | Model | Adaptation | mIoU | Pixel accuracy | NLL | Brier | ECE-15 |
|---|---|---|---:|---:|---:|---:|---:|
| cloudsen12 | dofa | frozen | 0.620746 | 0.843823 | 0.424634 | 0.223812 | 0.016911 |
| cloudsen12 | dofa | full_finetune | 0.671393 | 0.870869 | 0.398249 | 0.191970 | 0.040211 |
| cloudsen12 | panopticon | frozen | 0.667274 | 0.867130 | 0.365808 | 0.192717 | 0.024999 |
| cloudsen12 | panopticon | full_finetune | 0.690278 | 0.879842 | 0.335453 | 0.174708 | 0.020214 |
| spacenet7 | dofa | frozen | 0.481794 | 0.929942 | 0.241043 | 0.121549 | 0.030411 |
| spacenet7 | dofa | full_finetune | 0.486771 | 0.930647 | 0.321206 | 0.121998 | 0.039981 |
| spacenet7 | panopticon | frozen | 0.491314 | 0.929230 | 0.248144 | 0.120909 | 0.030904 |
| spacenet7 | panopticon | full_finetune | 0.495910 | 0.929689 | 0.260364 | 0.121392 | 0.028380 |

## Protocol differences

- EuroSAT fits Temperature Scaling on its independent dedicated calibration split.
- TreeSatAI has no independent official calibration split, so its temperature is fitted on official validation logits. That split was also used for checkpoint selection; this dual use is documented rather than concealed.
- Every official test set remained untouched during fitting, and no temperature was selected or rejected using test performance.

## Artifact rules

- TreeSatAI Temperature Scaling artifacts have their own extension code snapshot: `sha256:aae9032c5cb2313f956aa04f1aa9a2e5f9a24102e689a8c4783fa2b662b81a3c` (99 files).

- DOFA–EuroSAT ensembles are the previously frozen direct-Parquet artifacts and were revalidated against the immutable member Parquets without inference.
- New classification ensemble bundles contain aligned member logits/probabilities and ensemble probabilities.
- Segmentation manifests SHA256-pin the existing complete member prediction bundles; these 1–1.7 GB files are retained in place rather than duplicated. Ensemble probability/confidence/entropy maps are saved separately.
- C3 analysis code snapshot: `sha256:4b7c40588caef7fc94dd7b3c7dea0bf83800e38b906abfbdf32c0640bc4ae4ba` (99 files).
