# C1 deterministic classification run audit

Created: 2026-08-15T23:12:01.777369+00:00

Promotion is based only on the frozen protocol, validation behavior, provenance, checkpoint integrity, and prediction-artifact integrity. Test metric values are reported for completeness and are explicitly excluded from every promotion decision.

Summary: **6 promotable**, **0 blocked**, **0 missing**, **0 ambiguous**.

| Dataset | Model | Adaptation | Seed | Run ID | Status | Best val NLL | Test accuracy (excluded) | Errors |
|---|---|---|---:|---|---|---:|---:|---:|
| eurosat | panopticon | frozen | 42 | `20260815T195254997519Z_eurosat_panopticon_frozen_bnlinear_seed42_74d738b8` | **PROMOTABLE** | 0.06125512 | 0.98378777 | 0 |
| eurosat | panopticon | full_finetune | 42 | `20260815T210816909227Z_eurosat_panopticon_full_finetune_seed42_d280d48a` | **PROMOTABLE** | 0.12665270 | 0.96794399 | 0 |
| treesatai | dofa | frozen | 42 | `20260815T202926010453Z_treesatai_dofa_frozen_seed42_3dae0e5b` | **PROMOTABLE** | 0.26749730 | 0.25500000 | 0 |
| treesatai | dofa | full_finetune | 42 | `20260815T222120553265Z_treesatai_dofa_full_finetune_seed42_9845090f` | **PROMOTABLE** | 0.26934287 | 0.26500000 | 0 |
| treesatai | panopticon | frozen | 42 | `20260815T204959852998Z_treesatai_panopticon_frozen_seed42_5c8cc8e4` | **PROMOTABLE** | 0.26986948 | 0.23950000 | 0 |
| treesatai | panopticon | full_finetune | 42 | `20260815T223733845323Z_treesatai_panopticon_full_finetune_seed42_432b3047` | **PROMOTABLE** | 0.26948667 | 0.28350000 | 0 |
## Decision rule

A run is promotable only when every structural check passes. No minimum accuracy, Macro-F1, NLL, Brier, or ECE threshold is implemented for test data, and test values cannot select runs, seeds, checkpoints, or learning rates.
