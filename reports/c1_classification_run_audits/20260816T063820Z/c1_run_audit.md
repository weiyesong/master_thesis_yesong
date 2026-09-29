# C1 deterministic classification run audit

Created: 2026-08-16T06:38:20.708519+00:00

Promotion is based only on the frozen protocol, validation behavior, provenance, checkpoint integrity, and prediction-artifact integrity. Test metric values are reported for completeness and are explicitly excluded from every promotion decision.

Summary: **18 promotable**, **0 blocked**, **0 missing**, **0 ambiguous**.

| Dataset | Model | Adaptation | Seed | Run ID | Status | Best val NLL | Test accuracy (excluded) | Errors |
|---|---|---|---:|---|---|---:|---:|---:|
| eurosat | panopticon | frozen | 42 | `20260815T195254997519Z_eurosat_panopticon_frozen_bnlinear_seed42_74d738b8` | **PROMOTABLE** | 0.06125512 | 0.98378777 | 0 |
| eurosat | panopticon | full_finetune | 42 | `20260815T210816909227Z_eurosat_panopticon_full_finetune_seed42_d280d48a` | **PROMOTABLE** | 0.12665270 | 0.96794399 | 0 |
| treesatai | dofa | frozen | 42 | `20260815T202926010453Z_treesatai_dofa_frozen_seed42_3dae0e5b` | **PROMOTABLE** | 0.26749730 | 0.25500000 | 0 |
| treesatai | dofa | full_finetune | 42 | `20260815T222120553265Z_treesatai_dofa_full_finetune_seed42_9845090f` | **PROMOTABLE** | 0.26934287 | 0.26500000 | 0 |
| treesatai | panopticon | frozen | 42 | `20260815T204959852998Z_treesatai_panopticon_frozen_seed42_5c8cc8e4` | **PROMOTABLE** | 0.26986948 | 0.23950000 | 0 |
| treesatai | panopticon | full_finetune | 42 | `20260815T223733845323Z_treesatai_panopticon_full_finetune_seed42_432b3047` | **PROMOTABLE** | 0.26948667 | 0.28350000 | 0 |
| eurosat | panopticon | frozen | 43 | `20260815T231227548291Z_eurosat_panopticon_frozen_bnlinear_seed43_743e193d` | **PROMOTABLE** | 0.06154155 | 0.98305085 | 0 |
| eurosat | panopticon | full_finetune | 43 | `20260816T023205454878Z_eurosat_panopticon_full_finetune_seed43_e304fee6` | **PROMOTABLE** | 0.10445461 | 0.96573324 | 0 |
| treesatai | dofa | frozen | 43 | `20260816T012149418043Z_treesatai_dofa_frozen_seed43_d484c9b3` | **PROMOTABLE** | 0.26743665 | 0.25800000 | 0 |
| treesatai | dofa | full_finetune | 43 | `20260816T050303117850Z_treesatai_dofa_full_finetune_seed43_e40f79f2` | **PROMOTABLE** | 0.26098755 | 0.26550000 | 0 |
| treesatai | panopticon | frozen | 43 | `20260816T014811037405Z_treesatai_panopticon_frozen_seed43_baa4ef7a` | **PROMOTABLE** | 0.26648262 | 0.23950000 | 0 |
| treesatai | panopticon | full_finetune | 43 | `20260816T053348883249Z_treesatai_panopticon_full_finetune_seed43_b7ae4087` | **PROMOTABLE** | 0.26581973 | 0.25000000 | 0 |
| eurosat | panopticon | frozen | 44 | `20260816T004654290690Z_eurosat_panopticon_frozen_bnlinear_seed44_4f21cdf8` | **PROMOTABLE** | 0.06362555 | 0.98305085 | 0 |
| eurosat | panopticon | full_finetune | 44 | `20260816T034514128137Z_eurosat_panopticon_full_finetune_seed44_0883c676` | **PROMOTABLE** | 0.17644331 | 0.95431098 | 0 |
| treesatai | dofa | frozen | 44 | `20260816T013516128703Z_treesatai_dofa_frozen_seed44_f16a2a61` | **PROMOTABLE** | 0.26890326 | 0.25700000 | 0 |
| treesatai | dofa | full_finetune | 44 | `20260816T051814359617Z_treesatai_dofa_full_finetune_seed44_6a6f8fd9` | **PROMOTABLE** | 0.26400521 | 0.27000000 | 0 |
| treesatai | panopticon | frozen | 44 | `20260816T020829175076Z_treesatai_panopticon_frozen_seed44_341d0eca` | **PROMOTABLE** | 0.26839864 | 0.23450000 | 0 |
| treesatai | panopticon | full_finetune | 44 | `20260816T060624868595Z_treesatai_panopticon_full_finetune_seed44_d6d97c6a` | **PROMOTABLE** | 0.27025956 | 0.22100000 | 0 |
## Decision rule

A run is promotable only when every structural check passes. No minimum accuracy, Macro-F1, NLL, Brier, or ECE threshold is implemented for test data, and test values cannot select runs, seeds, checkpoints, or learning rates.
