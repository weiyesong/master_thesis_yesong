# MC Dropout Robustness Subset

The four cells below were fixed before any final MC-Dropout result was inspected. Each contains independently trained MC models at seeds 42/43/44 and is paired seed-by-seed with the corresponding deterministic model. Thirty stochastic passes within one checkpoint are not treated as independent replicates.

## eurosat / dofa / frozen

| Seed | MC accuracy | Deterministic accuracy | Paired delta | MC macro_f1 | Paired macro_f1 delta | MC NLL | Paired NLL delta | MC Brier | Paired Brier delta | MC ECE-15 | Paired ECE delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.985262 | 0.983051 | 0.002211 | 0.984358 | 0.002326 | 0.049743 | -0.008366 | 0.024561 | -0.002344 | 0.008762 | 0.004531 |
| 43 | 0.983419 | 0.984156 | -0.000737 | 0.982413 | -0.000720 | 0.057210 | 0.006879 | 0.028099 | 0.003856 | 0.011169 | 0.004044 |
| 44 | 0.981945 | 0.983051 | -0.001105 | 0.980816 | -0.001043 | 0.056449 | 0.001719 | 0.026876 | 0.000028 | 0.004424 | 0.000755 |

Paired-difference mean ± sample SD:

- `accuracy`: 0.000123 ± 0.001818
- `macro_f1`: 0.000188 ± 0.001859
- `nll`: 0.000077 ± 0.007754
- `brier`: 0.000513 ± 0.003129
- `ece_15`: 0.003110 ± 0.002054

## treesatai / panopticon / full_finetune

| Seed | MC accuracy | Deterministic accuracy | Paired delta | MC macro_f1 | Paired macro_f1 delta | MC NLL | Paired NLL delta | MC Brier | Paired Brier delta | MC ECE-15 | Paired ECE delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.290000 | 0.283500 | 0.006500 | 0.250963 | 0.006262 | 0.249833 | -0.000097 | 0.072897 | 0.000053 | 0.011062 | -0.000205 |
| 43 | 0.251000 | 0.250000 | 0.001000 | 0.217284 | -0.001244 | 0.245164 | -0.001073 | 0.071465 | -0.000308 | 0.005186 | 0.001028 |
| 44 | 0.263000 | 0.221000 | 0.042000 | 0.210075 | 0.014048 | 0.247515 | -0.007690 | 0.072539 | -0.002401 | 0.006610 | -0.000854 |

Paired-difference mean ± sample SD:

- `accuracy`: 0.016500 ± 0.022254
- `macro_f1`: 0.006355 ± 0.007646
- `nll`: -0.002954 ± 0.004131
- `brier`: -0.000886 ± 0.001325
- `ece_15`: -0.000010 ± 0.000956

## cloudsen12 / panopticon / frozen

| Seed | MC miou | Deterministic miou | Paired delta | MC pixel_accuracy | Paired pixel_accuracy delta | MC NLL | Paired NLL delta | MC Brier | Paired Brier delta | MC ECE-15 | Paired ECE delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.643686 | 0.654635 | -0.010949 | 0.854819 | -0.004366 | 0.396616 | -0.041205 | 0.207317 | -0.003385 | 0.017883 | -0.032078 |
| 43 | 0.649931 | 0.653777 | -0.003845 | 0.861069 | -0.000021 | 0.399377 | -0.006947 | 0.202912 | -0.001337 | 0.037028 | -0.001416 |
| 44 | 0.653344 | 0.654758 | -0.001414 | 0.861053 | 0.000446 | 0.396751 | -0.000243 | 0.202515 | -0.000903 | 0.036994 | 0.000540 |

Paired-difference mean ± sample SD:

- `miou`: -0.005403 ± 0.004954
- `pixel_accuracy`: -0.001314 ± 0.002653
- `nll`: -0.016132 ± 0.021971
- `brier`: -0.001875 ± 0.001326
- `ece_15`: -0.010984 ± 0.018293

## spacenet7 / dofa / full_finetune

| Seed | MC miou | Deterministic miou | Paired delta | MC pixel_accuracy | Paired pixel_accuracy delta | MC NLL | Paired NLL delta | MC Brier | Paired Brier delta | MC ECE-15 | Paired ECE delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.489874 | 0.497224 | -0.007351 | 0.930730 | 0.007301 | 0.219157 | -0.340845 | 0.116258 | -0.020593 | 0.021510 | -0.035404 |
| 43 | 0.486278 | 0.496559 | -0.010280 | 0.929595 | 0.004241 | 0.279288 | -0.133505 | 0.122067 | -0.007524 | 0.037163 | -0.007835 |
| 44 | 0.493969 | 0.499638 | -0.005669 | 0.927484 | 0.006776 | 0.358214 | -0.079221 | 0.127082 | -0.008704 | 0.044401 | -0.002631 |

Paired-difference mean ± sample SD:

- `miou`: -0.007767 ± 0.002334
- `pixel_accuracy`: 0.006106 ± 0.001636
- `nll`: -0.184523 ± 0.138072
- `brier`: -0.012274 ± 0.007229
- `ece_15`: -0.015290 ± 0.017612

## Scope

These three-seed summaries support robustness statements only for these four cells. They do not turn stochastic passes into sample size and do not justify seed-general claims for the other 12 cells.
