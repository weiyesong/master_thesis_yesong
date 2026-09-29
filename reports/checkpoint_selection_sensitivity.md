# Checkpoint Selection Sensitivity Audit

Date: 2026-08-25

## Executive decision

The difference between the current classification and segmentation checkpoint-selection policies is a **material cross-task confound** if calibration or uncertainty results are compared directly across tasks without conditioning on the selection rule. It is not merely a naming difference:

- the counterfactual selection epoch changes in **43/48** final deterministic runs;
- **26/48** runs are MATERIAL or SEVERE under the thresholds below;
- classification contributes 8/24 MATERIAL or SEVERE cases, concentrated in TreeSatAI, especially full fine-tuning;
- segmentation contributes 18/24 MATERIAL or SEVERE cases, with the strongest and most systematic effect on SpaceNet7.

The recommended policy for the remainder of the thesis is **OPTION C: keep the current task-specific policies** (classification: minimum validation NLL; segmentation: maximum validation mIoU). This recommendation is conditional on explicitly reporting checkpoint policy as part of each task protocol and not interpreting raw cross-task calibration differences as pure model effects. It preserves the completed, validation-only protocol and avoids replacing task-competent SpaceNet7 checkpoints with background-dominated, pixel-NLL minima.

**No existing deterministic experiment actually needs retraining under the recommended policy.** No artifact is corrupt or invalidated by this audit, and no training or inference was run. If the thesis were instead migrated retroactively to Option A, 20/24 classification alternative model states would have to be recreated by training because their selected epochs were not retained. Option B would analogously require recreation of 23/24 segmentation alternative model states. Those would be policy migrations, not repairs required by a defect in the existing experiments.

## Scope and method

The audit covers the 48 promoted final deterministic runs:

- 24 classification runs: EuroSAT and TreeSatAI, DOFA and Panopticon, frozen and full fine-tuning, seeds 42/43/44;
- 24 segmentation runs: CloudSEN12 and SpaceNet7 with the same model, adaptation, and seed matrix.

Dry runs, preflights, auxiliary/legacy non-matrix baselines, failed runs, and the two documented zero-epoch segmentation initialization attempts were excluded. The promoted-run boundaries agree with the existing C1 and C2 promotion audits.

Selection used only the saved per-epoch validation histories. Test results were not used to select an epoch or compute any sensitivity result. Exact ties use the earliest epoch, matching the training code's strict-improvement behavior. For classification, the task-performance counterfactual is validation Accuracy for multiclass EuroSAT and validation Macro-F1 for multilabel TreeSatAI; TreeSatAI's stored `accuracy` is strict sample-level exact match and is therefore retained as a reported secondary metric. For segmentation, the two selectors are validation mIoU and pixelwise validation NLL.

No inference was necessary. The histories already contain all metrics needed for the comparisons that can be made. In particular, running inference on a retained checkpoint cannot recover the model state of an unretained epoch.

## 1. Inventory of checkpoint-selection evidence

### Per-epoch evidence

Each statement in the table applies to every promoted run in that dataset row, not just one representative run.

| Dataset | Final runs | Completed epochs | Checkpoint at every epoch | Retained model states | Val NLL | Val Accuracy | Val Macro-F1 | Val mIoU / class IoU | Val pixel accuracy | Val Brier | Val ECE | Training history |
|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| EuroSAT | 12 | 16–25 | No | `best.pt`, `last.pt` | Yes | Yes | **No** | N/A | N/A | Yes | Yes | Yes |
| TreeSatAI | 12 | 16–37 | No | `best.pt`, `last.pt` | Yes | Yes (exact match) | Yes | N/A | N/A | Yes | Yes | Yes |
| CloudSEN12 | 12 | 26–50 | No | `best.pt`, `last.pt` | Yes | N/A | N/A | Yes / Yes | Yes | Yes | Yes (`ece_15`) | Yes |
| SpaceNet7 | 12 | 31–50 | No | `best.pt`, `last.pt` | Yes | N/A | N/A | Yes / Yes | Yes | Yes | Yes (`ece_15`) | Yes |

All 48 histories are contiguous from epoch 1 through the recorded final epoch. Classification histories store validation Accuracy, NLL, Brier, and ECE for every epoch; TreeSatAI additionally stores labelwise accuracy and Macro-F1. EuroSAT does not store per-epoch Macro-F1, so numeric EuroSAT Macro-F1 sensitivity cannot be recovered from the existing histories. Segmentation histories store mIoU, per-class IoU, pixel accuracy, NLL, Brier, ECE-15, valid/ignored pixel counts, and the confusion matrix at every epoch. SpaceNet7 additionally stores foreground, classwise, and boundary calibration metrics.

The ordinary checkpoint policy stores only the current selector's `best.pt` and the terminal/early-stopping `last.pt`. Three resume-provenance snapshots also exist: EuroSAT/Panopticon/frozen/seed43 at epoch 5, CloudSEN12/DOFA/frozen/seed42 at epoch 1, and SpaceNet7/DOFA/frozen/seed42 at epoch 2. None is the counterfactual epoch identified below. A `last.pt` history contains prior scalar metrics, but its model state is only the final epoch's state; it is not a collection of prior epoch models.

### What can be reconstructed without retraining

Two meanings of “reconstruct” must be separated:

1. **Validation selection sensitivity:** fully reconstructable for all 48 runs. The selected epoch and all requested stored metrics at that epoch can be read directly from `training_history.json`.
2. **Runnable alternative checkpoint:** recoverable only when the counterfactual epoch happens to equal an existing retained checkpoint epoch. This occurs in 4/24 classification runs and 1/24 segmentation runs, all because the two selectors choose the same epoch. None of the 43 changed selections has a retained model state.

Consequently, this audit can determine validation-level dependence without retraining, but it cannot generate counterfactual test predictions or use most counterfactual epochs as MC Dropout starting points. That limitation is caused by checkpoint retention, not missing validation evidence.

## 2. Classification sensitivity

### Representative coverage first: seed 42

`A` is minimum validation NLL. `B` is maximum validation Accuracy for EuroSAT and maximum validation Macro-F1 for TreeSatAI. Values are `A -> B`. EuroSAT Macro-F1 is `NR` because it was not recorded per epoch.

| Run | Epoch A→B | Accuracy | Macro-F1 | NLL | Brier | ECE | Rating |
|---|---:|---:|---:|---:|---:|---:|---|
| EuroSAT / DOFA / frozen / s42 | 7→14 | 0.9771→0.9778 | NR | 0.0687→0.0704 | 0.0352→0.0344 | 0.0047→0.0049 | NEGLIGIBLE |
| EuroSAT / DOFA / full / s42 | 1→2 | 0.9586→0.9631 | NR | 0.1122→0.1425 | 0.0605→0.0609 | 0.0046→0.0191 | SMALL |
| EuroSAT / Panopticon / frozen / s42 | 10→14 | 0.9804→0.9808 | NR | 0.0613→0.0643 | 0.0295→0.0306 | 0.0073→0.0074 | NEGLIGIBLE |
| EuroSAT / Panopticon / full / s42 | 1→10 | 0.9527→0.9549 | NR | 0.1267→0.1485 | 0.0717→0.0695 | 0.0205→0.0124 | SMALL |
| TreeSatAI / DOFA / frozen / s42 | 27→35 | 0.2170→0.2020 | 0.2008→0.2154 | 0.2675→0.2704 | 0.0797→0.0810 | 0.0099→0.0115 | SMALL |
| TreeSatAI / DOFA / full / s42 | 5→18 | 0.2000→0.2430 | 0.2670→0.3081 | 0.2693→0.6776 | 0.0805→0.0961 | 0.0061→0.0879 | SEVERE |
| TreeSatAI / Panopticon / frozen / s42 | 9→15 | 0.2130→0.2090 | 0.1695→0.2107 | 0.2699→0.2731 | 0.0809→0.0817 | 0.0083→0.0122 | MATERIAL |
| TreeSatAI / Panopticon / full / s42 | 3→12 | 0.2300→0.2100 | 0.2241→0.3114 | 0.2695→0.5395 | 0.0796→0.1012 | 0.0179→0.0879 | SEVERE |

This representative slice already shows the interaction: EuroSAT is insensitive or small, TreeSatAI frozen is small-to-material, and TreeSatAI full fine-tuning is severe.

### All 24 classification runs

The `|Δ|` vector is primary task metric / NLL / Brier / ECE. `B state` describes whether the counterfactual model weights exist. A same-epoch result has `best.pt`; `not retained` means the validation values exist but the epoch's model state does not.

| Run | eNLL→eTask | Accuracy A→B | Macro-F1 A→B | NLL A→B | Brier A→B | ECE A→B | Abs Δ primary/NLL/Brier/ECE | Rating | B state |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| EuroSAT/DOFA/frozen/s42 | 7→14 | 0.9771→0.9778 | NR | 0.0687→0.0704 | 0.0352→0.0344 | 0.0047→0.0049 | 0.0007/0.0017/0.0008/0.0002 | NEGLIGIBLE | not retained |
| EuroSAT/DOFA/frozen/s43 | 11→6 | 0.9767→0.9775 | NR | 0.0682→0.0692 | 0.0344→0.0345 | 0.0062→0.0054 | 0.0007/0.0010/0.0001/0.0007 | NEGLIGIBLE | not retained |
| EuroSAT/DOFA/frozen/s44 | 9→3 | 0.9771→0.9771 | NR | 0.0706→0.0759 | 0.0349→0.0367 | 0.0067→0.0095 | 0.0000/0.0052/0.0018/0.0028 | NEGLIGIBLE | not retained |
| EuroSAT/DOFA/full/s42 | 1→2 | 0.9586→0.9631 | NR | 0.1122→0.1425 | 0.0605→0.0609 | 0.0046→0.0191 | 0.0044/0.0303/0.0004/0.0145 | SMALL | not retained |
| EuroSAT/DOFA/full/s43 | 7→1 | 0.9682→0.9723 | NR | 0.1026→0.1201 | 0.0500→0.0477 | 0.0059→0.0139 | 0.0041/0.0175/0.0022/0.0080 | SMALL | not retained |
| EuroSAT/DOFA/full/s44 | 10→10 | 0.9590→0.9590 | NR (=same model) | 0.1338→0.1338 | 0.0610→0.0610 | 0.0135→0.0135 | 0.0000/0.0000/0.0000/0.0000 | NEGLIGIBLE | best.pt |
| EuroSAT/Panopticon/frozen/s42 | 10→14 | 0.9804→0.9808 | NR | 0.0613→0.0643 | 0.0295→0.0306 | 0.0073→0.0074 | 0.0004/0.0030/0.0011/0.0001 | NEGLIGIBLE | not retained |
| EuroSAT/Panopticon/frozen/s43 | 6→4 | 0.9786→0.9815 | NR | 0.0615→0.0629 | 0.0304→0.0297 | 0.0077→0.0044 | 0.0030/0.0014/0.0007/0.0033 | NEGLIGIBLE | not retained |
| EuroSAT/Panopticon/frozen/s44 | 9→9 | 0.9819→0.9819 | NR (=same model) | 0.0636→0.0636 | 0.0306→0.0306 | 0.0094→0.0094 | 0.0000/0.0000/0.0000/0.0000 | NEGLIGIBLE | best.pt |
| EuroSAT/Panopticon/full/s42 | 1→10 | 0.9527→0.9549 | NR | 0.1267→0.1485 | 0.0717→0.0695 | 0.0205→0.0124 | 0.0022/0.0218/0.0022/0.0082 | SMALL | not retained |
| EuroSAT/Panopticon/full/s43 | 1→1 | 0.9712→0.9712 | NR (=same model) | 0.1045→0.1045 | 0.0480→0.0480 | 0.0094→0.0094 | 0.0000/0.0000/0.0000/0.0000 | NEGLIGIBLE | best.pt |
| EuroSAT/Panopticon/full/s44 | 2→2 | 0.9542→0.9542 | NR (=same model) | 0.1764→0.1764 | 0.0727→0.0727 | 0.0094→0.0094 | 0.0000/0.0000/0.0000/0.0000 | NEGLIGIBLE | best.pt |
| TreeSatAI/DOFA/frozen/s42 | 27→35 | 0.2170→0.2020 | 0.2008→0.2154 | 0.2675→0.2704 | 0.0797→0.0810 | 0.0099→0.0115 | 0.0146/0.0029/0.0013/0.0015 | SMALL | not retained |
| TreeSatAI/DOFA/frozen/s43 | 14→21 | 0.2130→0.2150 | 0.1984→0.2141 | 0.2674→0.2721 | 0.0803→0.0817 | 0.0042→0.0067 | 0.0157/0.0047/0.0014/0.0025 | SMALL | not retained |
| TreeSatAI/DOFA/frozen/s44 | 13→22 | 0.2160→0.2020 | 0.1989→0.2115 | 0.2689→0.2694 | 0.0806→0.0810 | 0.0033→0.0075 | 0.0127/0.0005/0.0005/0.0042 | SMALL | not retained |
| TreeSatAI/DOFA/full/s42 | 5→18 | 0.2000→0.2430 | 0.2670→0.3081 | 0.2693→0.6776 | 0.0805→0.0961 | 0.0061→0.0879 | 0.0411/0.4083/0.0156/0.0819 | SEVERE | not retained |
| TreeSatAI/DOFA/full/s43 | 4→14 | 0.2220→0.1920 | 0.1888→0.2946 | 0.2610→0.6579 | 0.0776→0.1056 | 0.0167→0.0937 | 0.1059/0.3969/0.0279/0.0770 | SEVERE | not retained |
| TreeSatAI/DOFA/full/s44 | 4→9 | 0.2140→0.2040 | 0.2257→0.3174 | 0.2640→0.4060 | 0.0797→0.0979 | 0.0094→0.0699 | 0.0917/0.1420/0.0182/0.0605 | SEVERE | not retained |
| TreeSatAI/Panopticon/frozen/s42 | 9→15 | 0.2130→0.2090 | 0.1695→0.2107 | 0.2699→0.2731 | 0.0809→0.0817 | 0.0083→0.0122 | 0.0412/0.0032/0.0008/0.0039 | MATERIAL | not retained |
| TreeSatAI/Panopticon/frozen/s43 | 12→17 | 0.2280→0.2190 | 0.1934→0.2079 | 0.2665→0.2741 | 0.0799→0.0820 | 0.0093→0.0120 | 0.0145/0.0077/0.0022/0.0027 | SMALL | not retained |
| TreeSatAI/Panopticon/frozen/s44 | 15→22 | 0.2260→0.2230 | 0.1984→0.2227 | 0.2684→0.2711 | 0.0802→0.0812 | 0.0125→0.0138 | 0.0243/0.0027/0.0009/0.0013 | MATERIAL | not retained |
| TreeSatAI/Panopticon/full/s42 | 3→12 | 0.2300→0.2100 | 0.2241→0.3114 | 0.2695→0.5395 | 0.0796→0.1012 | 0.0179→0.0879 | 0.0873/0.2700/0.0216/0.0700 | SEVERE | not retained |
| TreeSatAI/Panopticon/full/s43 | 2→16 | 0.2040→0.2270 | 0.1800→0.3093 | 0.2658→0.6325 | 0.0799→0.1033 | 0.0115→0.0935 | 0.1293/0.3667/0.0235/0.0820 | SEVERE | not retained |
| TreeSatAI/Panopticon/full/s44 | 1→14 | 0.2000→0.2150 | 0.1688→0.2944 | 0.2703→0.6348 | 0.0815→0.1000 | 0.0068→0.0902 | 0.1256/0.3646/0.0185/0.0833 | SEVERE | not retained |

Classification summary:

- selected epoch changes in 20/24 runs;
- ratings are 9 NEGLIGIBLE, 7 SMALL, 2 MATERIAL, and 6 SEVERE;
- EuroSAT's mean absolute change is 0.00129 Accuracy, 0.00683 NLL, 0.00078 Brier, and 0.00314 ECE;
- TreeSatAI's mean absolute change is 0.05864 Macro-F1, 0.16417 NLL, 0.01102 Brier, and 0.03922 ECE;
- all six TreeSatAI full-finetune runs are SEVERE. Their Macro-F1 maxima occur later and improve Macro-F1, but do so with very large NLL and ECE increases. This is consistent with the already documented post-best instability, not a test-set effect.

As a robustness check on the `Accuracy or Macro-F1` wording, maximum exact-match Accuracy rather than Macro-F1 changes the TreeSatAI epoch in 9/12 runs and all 6 full-finetune runs. It does not remove the full-finetune sensitivity; Macro-F1 is used in the main counterfactual because exact-match Accuracy is an unusually discontinuous primary criterion for a 15-label multilabel task.

## 3. Segmentation sensitivity

### Representative coverage first: seed 42

`A` is maximum validation mIoU and `B` is minimum pixelwise validation NLL. Values are `A -> B`.

| Run | Epoch A→B | mIoU | Per-class IoU | Pixel accuracy | NLL | Brier | ECE | Rating |
|---|---:|---:|---|---:|---:|---:|---:|---|
| CloudSEN12 / DOFA / frozen / s42 | 16→12 | 0.6123→0.5960 | clear .8097→.8137; thick .7330→.7444; thin .4256→.3840; shadow .4808→.4417 | 0.8292→0.8329 | 0.4749→0.4565 | 0.2483→0.2407 | 0.0203→0.0214 | SMALL |
| CloudSEN12 / DOFA / full / s42 | 50→13 | 0.6590→0.6281 | clear .8491→.8314; thick .7823→.7637; thin .4720→.4690; shadow .5324→.4482 | 0.8605→0.8487 | 0.5237→0.4142 | 0.2201→0.2183 | 0.0725→0.0219 | SEVERE |
| CloudSEN12 / Panopticon / frozen / s42 | 29→11 | 0.6715→0.6581 | clear .8488→.8497; thick .7845→.7816; thin .5114→.4864; shadow .5412→.5147 | 0.8624→0.8613 | 0.4369→0.3863 | 0.2079→0.2018 | 0.0489→0.0281 | MATERIAL |
| CloudSEN12 / Panopticon / full / s42 | 44→28 | 0.6805→0.6677 | clear .8594→.8583; thick .7897→.7846; thin .5089→.4911; shadow .5639→.5367 | 0.8687→0.8661 | 0.3970→0.3761 | 0.1973→0.1954 | 0.0466→0.0300 | SMALL |
| SpaceNet7 / DOFA / frozen / s42 | 23→1 | 0.4923→0.4713 | background .9279→.9312; building .0566→.0114 | 0.9283→0.9312 | 0.2588→0.1931 | 0.1210→0.1108 | 0.0336→0.0147 | MATERIAL |
| SpaceNet7 / DOFA / full / s42 | 23→1 | 0.5036→0.4668 | background .9276→.9311; building .0796→.0025 | 0.9280→0.9311 | 0.4954→0.2100 | 0.1268→0.1133 | 0.0517→0.0137 | SEVERE |
| SpaceNet7 / Panopticon / frozen / s42 | 43→1 | 0.5155→0.4719 | background .9228→.9313; building .1081→.0125 | 0.9235→0.9313 | 0.3292→0.1908 | 0.1251→0.1089 | 0.0354→0.0083 | MATERIAL |
| SpaceNet7 / Panopticon / full / s42 | 40→1 | 0.5242→0.4856 | background .9178→.9313; building .1306→.0399 | 0.9188→0.9315 | 0.3959→0.1940 | 0.1312→0.1105 | 0.0454→0.0136 | SEVERE |

The representative SpaceNet7 rows show why pixelwise NLL is not a neutral replacement for mIoU: the NLL selector favors epochs 1–3 with slightly higher overall pixel accuracy but very low building IoU. The dominant background class improves aggregate NLL and accuracy while the foreground task degrades.

### All 24 segmentation runs

The `|Δ|` vector is mIoU / NLL / Brier / ECE. Per-class names are abbreviated only to keep the table readable.

| Run | emIoU→eNLL | mIoU A→B | Per-class IoU A→B | Pixel accuracy A→B | NLL A→B | Brier A→B | ECE A→B | Abs Δ mIoU/NLL/Brier/ECE | Rating | B state |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|
| CloudSEN12/DOFA/frozen/s42 | 16→12 | 0.6123→0.5960 | clear .8097→.8137; thick .7330→.7444; thin .4256→.3840; shadow .4808→.4417 | 0.8292→0.8329 | 0.4749→0.4565 | 0.2483→0.2407 | 0.0203→0.0214 | 0.0163/0.0184/0.0075/0.0011 | SMALL | not retained |
| CloudSEN12/DOFA/frozen/s43 | 24→9 | 0.6202→0.6049 | clear .8193→.8070; thick .7458→.7315; thin .4399→.4319; shadow .4759→.4490 | 0.8364→0.8289 | 0.4759→0.4559 | 0.2407→0.2418 | 0.0486→0.0137 | 0.0154/0.0199/0.0011/0.0349 | MATERIAL | not retained |
| CloudSEN12/DOFA/frozen/s44 | 24→18 | 0.6153→0.6082 | clear .8223→.8202; thick .7372→.7389; thin .4279→.4007; shadow .4739→.4731 | 0.8338→0.8350 | 0.4839→0.4631 | 0.2459→0.2409 | 0.0503→0.0343 | 0.0071/0.0208/0.0050/0.0159 | SMALL | not retained |
| CloudSEN12/DOFA/full/s42 | 50→13 | 0.6590→0.6281 | clear .8491→.8314; thick .7823→.7637; thin .4720→.4690; shadow .5324→.4482 | 0.8605→0.8487 | 0.5237→0.4142 | 0.2201→0.2183 | 0.0725→0.0219 | 0.0309/0.1095/0.0018/0.0507 | SEVERE | not retained |
| CloudSEN12/DOFA/full/s43 | 49→22 | 0.6678→0.6516 | clear .8509→.8433; thick .7903→.7787; thin .4798→.4907; shadow .5500→.4939 | 0.8637→0.8575 | 0.4958→0.4130 | 0.2132→0.2106 | 0.0674→0.0392 | 0.0161/0.0829/0.0026/0.0283 | MATERIAL | not retained |
| CloudSEN12/DOFA/full/s44 | 42→24 | 0.6600→0.6446 | clear .8477→.8450; thick .7815→.7715; thin .4837→.4590; shadow .5271→.5030 | 0.8596→0.8531 | 0.4576→0.4109 | 0.2139→0.2137 | 0.0571→0.0354 | 0.0154/0.0467/0.0002/0.0217 | MATERIAL | not retained |
| CloudSEN12/Panopticon/frozen/s42 | 29→11 | 0.6715→0.6581 | clear .8488→.8497; thick .7845→.7816; thin .5114→.4864; shadow .5412→.5147 | 0.8624→0.8613 | 0.4369→0.3863 | 0.2079→0.2018 | 0.0489→0.0281 | 0.0134/0.0507/0.0061/0.0209 | MATERIAL | not retained |
| CloudSEN12/Panopticon/frozen/s43 | 21→8 | 0.6702→0.6604 | clear .8475→.8388; thick .7871→.7797; thin .5112→.5080; shadow .5351→.5153 | 0.8630→0.8553 | 0.4038→0.3879 | 0.2046→0.2080 | 0.0394→0.0138 | 0.0098/0.0159/0.0034/0.0255 | MATERIAL | not retained |
| CloudSEN12/Panopticon/frozen/s44 | 16→16 | 0.6704→0.6704 | clear .8469→.8469; thick .7825→.7825; thin .5183→.5183; shadow .5340→.5340 | 0.8617→0.8617 | 0.3899→0.3899 | 0.2032→0.2032 | 0.0352→0.0352 | 0.0000/0.0000/0.0000/0.0000 | NEGLIGIBLE | best.pt |
| CloudSEN12/Panopticon/full/s42 | 44→28 | 0.6805→0.6677 | clear .8594→.8583; thick .7897→.7846; thin .5089→.4911; shadow .5639→.5367 | 0.8687→0.8661 | 0.3970→0.3761 | 0.1973→0.1954 | 0.0466→0.0300 | 0.0128/0.0209/0.0019/0.0166 | SMALL | not retained |
| CloudSEN12/Panopticon/full/s43 | 39→38 | 0.6840→0.6835 | clear .8632→.8641; thick .7962→.7950; thin .5139→.5431; shadow .5628→.5318 | 0.8710→0.8723 | 0.3823→0.3646 | 0.1923→0.1881 | 0.0405→0.0346 | 0.0005/0.0176/0.0042/0.0059 | SMALL | not retained |
| CloudSEN12/Panopticon/full/s44 | 44→23 | 0.6819→0.6731 | clear .8629→.8565; thick .7890→.7879; thin .5162→.5095; shadow .5597→.5386 | 0.8696→0.8665 | 0.3951→0.3630 | 0.1941→0.1937 | 0.0451→0.0259 | 0.0088/0.0321/0.0004/0.0192 | SMALL | not retained |
| SpaceNet7/DOFA/frozen/s42 | 23→1 | 0.4923→0.4713 | background .9279→.9312; building .0566→.0114 | 0.9283→0.9312 | 0.2588→0.1931 | 0.1210→0.1108 | 0.0336→0.0147 | 0.0210/0.0656/0.0102/0.0189 | MATERIAL | not retained |
| SpaceNet7/DOFA/frozen/s43 | 31→3 | 0.4978→0.4682 | background .9271→.9312; building .0685→.0053 | 0.9275→0.9312 | 0.2840→0.1956 | 0.1217→0.1119 | 0.0361→0.0227 | 0.0296/0.0884/0.0098/0.0134 | MATERIAL | not retained |
| SpaceNet7/DOFA/frozen/s44 | 44→3 | 0.4975→0.4774 | background .9246→.9312; building .0705→.0237 | 0.9250→0.9313 | 0.3047→0.1921 | 0.1255→0.1105 | 0.0378→0.0184 | 0.0201/0.1126/0.0150/0.0194 | MATERIAL | not retained |
| SpaceNet7/DOFA/full/s42 | 23→1 | 0.5036→0.4668 | background .9276→.9311; building .0796→.0025 | 0.9280→.9311 | 0.4954→0.2100 | 0.1268→0.1133 | 0.0517→0.0137 | 0.0367/0.2854/0.0134/0.0380 | SEVERE | not retained |
| SpaceNet7/DOFA/full/s43 | 16→2 | 0.5085→0.4709 | background .9275→.9315; building .0896→.0104 | 0.9280→.9315 | 0.3715→0.1997 | 0.1203→0.1097 | 0.0394→0.0213 | 0.0376/0.1718/0.0106/0.0180 | MATERIAL | not retained |
| SpaceNet7/DOFA/full/s44 | 24→1 | 0.5048→0.4701 | background .9256→.9311; building .0840→.0091 | 0.9261→.9312 | 0.3806→0.1967 | 0.1240→0.1103 | 0.0419→0.0147 | 0.0347/0.1839/0.0136/0.0272 | MATERIAL | not retained |
| SpaceNet7/Panopticon/frozen/s42 | 43→1 | 0.5155→0.4719 | background .9228→.9313; building .1081→.0125 | 0.9235→0.9313 | 0.3292→0.1908 | 0.1251→0.1089 | 0.0354→0.0083 | 0.0436/0.1384/0.0162/0.0271 | MATERIAL | not retained |
| SpaceNet7/Panopticon/frozen/s43 | 50→3 | 0.5209→0.4906 | background .9207→.9317; building .1211→.0494 | 0.9215→0.9320 | 0.3270→0.1917 | 0.1259→0.1089 | 0.0376→0.0147 | 0.0303/0.1354/0.0169/0.0229 | MATERIAL | not retained |
| SpaceNet7/Panopticon/frozen/s44 | 49→1 | 0.5153→0.4785 | background .9214→.9314; building .1093→.0256 | 0.9222→0.9316 | 0.3286→0.1921 | 0.1266→0.1083 | 0.0374→0.0062 | 0.0368/0.1365/0.0183/0.0313 | MATERIAL | not retained |
| SpaceNet7/Panopticon/full/s42 | 40→1 | 0.5242→0.4856 | background .9178→.9313; building .1306→.0399 | 0.9188→0.9315 | 0.3959→0.1940 | 0.1312→0.1105 | 0.0454→0.0136 | 0.0386/0.2019/0.0207/0.0318 | SEVERE | not retained |
| SpaceNet7/Panopticon/full/s43 | 23→2 | 0.5179→0.4681 | background .9244→.9313; building .1114→.0048 | 0.9251→.9313 | 0.4335→0.1996 | 0.1255→0.1101 | 0.0456→0.0079 | 0.0499/0.2339/0.0154/0.0377 | SEVERE | not retained |
| SpaceNet7/Panopticon/full/s44 | 42→1 | 0.5161→0.4802 | background .9185→.9317; building .1136→.0288 | 0.9194→0.9318 | 0.2929→0.1922 | 0.1322→0.1087 | 0.0256→0.0086 | 0.0358/0.1007/0.0235/0.0169 | MATERIAL | not retained |

Segmentation summary:

- selected epoch changes in 23/24 runs;
- ratings are 1 NEGLIGIBLE, 5 SMALL, 14 MATERIAL, and 4 SEVERE;
- CloudSEN12's mean absolute change is 0.01221 mIoU, 0.03628 NLL, 0.00285 Brier, and 0.02005 ECE;
- SpaceNet7's mean absolute change is 0.03456 mIoU, 0.15455 NLL, 0.01531 Brier, and 0.02524 ECE;
- minimum-NLL selection decreases mIoU in all 23 changed runs. It decreases Brier in 21/23 and ECE in 22/23, demonstrating the expected calibration/performance trade-off;
- all 12 SpaceNet7 minimum-NLL epochs occur at epochs 1–3. Building IoU falls sharply even while pixel accuracy often rises, which identifies class imbalance as the mechanism rather than random checkpoint noise.

## 4. Sensitivity thresholds and aggregate assessment

The rating is the worst band reached by any of four absolute changes. Task metrics, Brier, and ECE are proportions; for example, 0.005 is 0.5 percentage points. NLL is not bounded, so its cutoffs are pragmatic within-run effect-size cutoffs rather than claims that categorical, multilabel, and pixelwise NLL have identical scales.

| Rating | Required absolute changes |
|---|---|
| NEGLIGIBLE | all of: primary task metric < 0.005; NLL < 0.01; Brier < 0.005; ECE < 0.005 |
| SMALL | not negligible, and all of: primary < 0.02; NLL < 0.05; Brier < 0.02; ECE < 0.02 |
| MATERIAL | not small, and all of: primary < 0.05; NLL < 0.20; Brier < 0.05; ECE < 0.05 |
| SEVERE | any of: primary ≥ 0.05; NLL ≥ 0.20; Brier ≥ 0.05; ECE ≥ 0.05 |

The primary metric is Accuracy for EuroSAT, Macro-F1 for TreeSatAI, and mIoU for segmentation. This is a descriptive sensitivity rubric, not a significance test. With only three seeds per cell and shared fixed validation splits, the audit should not be read as estimating a population-level causal effect.

| Scope | Runs | Epoch changes | NEGLIGIBLE | SMALL | MATERIAL | SEVERE | Mean absolute primary Δ | Mean absolute NLL Δ | Mean absolute Brier Δ | Mean absolute ECE Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Classification | 24 | 20 | 9 | 7 | 2 | 6 | 0.02997 | 0.08550 | 0.00590 | 0.02118 |
| Segmentation | 24 | 23 | 1 | 5 | 14 | 4 | 0.02338 | 0.09541 | 0.00908 | 0.02264 |
| Total | 48 | 43 | 10 | 12 | 16 | 10 | — | — | — | — |

The aggregate means should not be used to compare raw performance across tasks because the primary metrics and NLL definitions differ. They are shown only to summarize within-run selector dependence. The rating counts, dataset-specific patterns, and per-run tables are the authoritative audit result.

## 5. Policy evaluation and recommendation

| Consideration | Option A: task-performance selection | Option B: probabilistic selection | Option C: current task-specific policies |
|---|---|---|---|
| Rule | EuroSAT Accuracy; TreeSatAI Macro-F1; segmentation mIoU | classification NLL; segmentation pixelwise NLL | classification NLL; segmentation mIoU |
| Scientific fairness | Aligns every task to its downstream performance objective, but the objectives remain structurally different | Uses a common proper-scoring-rule concept, but pixel averaging makes it unfair to rare segmentation classes | Task-aware and protocol-consistent, but the different selectors are a cross-task calibration confound |
| Calibration-selection bias | Lowest direct NLL-selection bias; calibration remains an outcome | Highest: NLL is both selector and a reported calibration/probabilistic outcome | Present for classification, absent for segmentation; must be disclosed |
| Compatibility | Would change 20/24 classification epochs and cannot recover those model states | Would change 23/24 segmentation epochs and cannot recover those model states | Fully compatible with all completed deterministic artifacts |
| Retraining cost if applied retroactively | 20 classification runs need recreated alternative states under the defined Accuracy/Macro-F1 rule | 23 segmentation runs need recreated alternative states | None |
| Cross-task interpretability | Better task-performance symmetry, but not metric identity | Superficially strongest common selector; empirically produces background-dominated SpaceNet7 choices | Weaker direct calibration comparability, but preserves usable task models and permits explicit stratification by task |
| Empirical warning from this audit | TreeSatAI full-finetune task maxima have severely worse NLL/ECE | SpaceNet7 NLL minima have near-zero building IoU | Sensitivity itself must accompany all cross-task claims |

### Recommendation: Option C

Use the already selected deterministic `best.pt` for each run as the fixed starting point for all remaining MC Dropout experiments. Apply MC Dropout to the same task-specific checkpoint that defines the deterministic baseline; do not use MC results or test results to revisit checkpoint selection.

Option C is recommended because:

1. the thesis protocols were frozen and completed under these validation-only rules;
2. changing policy now would require substantial retraining solely because intermediate model states were not retained;
3. Option B's nominal cross-task symmetry is scientifically misleading for SpaceNet7: pixelwise NLL is dominated by background and selects models with much poorer building IoU;
4. Option A is a reasonable design for a future prospective study with all epoch checkpoints retained, but in these completed TreeSatAI full-finetune trajectories it selects substantially less probabilistically sound epochs;
5. retaining the current policy keeps calibration-selection bias asymmetric but observable. The correct remedy in this thesis is disclosure and sensitivity reporting, not silently changing only the experiments whose alternative checkpoints happen to survive.

For thesis interpretation, report deterministic and MC Dropout results within each task first. Any classification-versus-segmentation calibration comparison must state that classification checkpoints were NLL-selected while segmentation checkpoints were mIoU-selected, and should cite the dataset-level sensitivity above. Do not present the selector difference as a model-architecture effect.

## Retraining statement

**No promoted deterministic experiment needs retraining.** The current checkpoints remain valid under the predeclared validation-only protocols, the sensitivity can be quantified from complete saved histories, and the recommended remainder policy uses the existing authoritative checkpoints. Retraining would become necessary only if a new, retroactive common selection policy were mandated and counterfactual checkpoint weights or test/MC outputs were required. This audit does not authorize or initiate that change.

## Artifact-integrity statement

No existing experiment artifact was modified. No model was trained, resumed, or evaluated. This report is the only new file created by the audit.
