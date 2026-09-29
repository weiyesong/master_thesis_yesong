# C3 — TreeSatAI validation-fitted Temperature Scaling

Created: 2026-08-24T21:30:50.312553+00:00

TreeSatAI uses the downloaded official GEO-Bench-2 train/validation/test artifact. It has no independent calibration split. Under the final thesis protocol, one positive scalar temperature is therefore fitted to each checkpoint's official validation logits. This official validation split was also used for checkpoint selection; that dual use is an explicit protocol difference from EuroSAT, which uses a dedicated calibration split.

The official test split remained untouched during fitting. No temperature was selected, rejected, or adjusted from test performance, and no neural-network weight was changed. All reported test results apply the validation-fitted temperature prospectively to the existing saved test logits.

TreeSatAI is multilabel: Accuracy is strict exact match, Macro-F1 is label-macro positive-class F1, NLL and Brier average all sample-label decisions, and ECE-15 uses flattened binary-decision confidence. A positive scalar leaves both each 0.5-threshold prediction vector and the per-sample argmax label unchanged; both invariants were verified for every checkpoint.

## Results

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

## Saved artifacts

Each checkpoint directory contains raw and calibrated test probabilities in Parquet and NPZ formats, a paired raw/calibrated reliability diagram, and a metrics/provenance manifest. The validation-logit exports and their checkpoint hashes are recorded in those manifests.

TreeSatAI Temperature Scaling code snapshot: `sha256:aae9032c5cb2313f956aa04f1aa9a2e5f9a24102e689a8c4783fa2b662b81a3c` (99 files).

The optional EuroSAT validation-fit sensitivity analysis was not run; it is not needed to resolve the required TreeSatAI result and no protocol was chosen using test performance.
