# C1 deterministic classification run audit

Created: 2026-08-16T01:35:35.741417+00:00

Promotion is based only on the frozen protocol, validation behavior, provenance, checkpoint integrity, and prediction-artifact integrity. Test metric values are reported for completeness and are explicitly excluded from every promotion decision.

Summary: **1 promotable**, **0 blocked**, **5 missing**, **0 ambiguous**.

| Dataset | Model | Adaptation | Seed | Run ID | Status | Best val NLL | Test accuracy (excluded) | Errors |
|---|---|---|---:|---|---|---:|---:|---:|
| eurosat | panopticon | frozen | 43 | `—` | **MISSING** | — | — | 1 |
| eurosat | panopticon | full_finetune | 43 | `—` | **MISSING** | — | — | 1 |
| treesatai | dofa | frozen | 43 | `20260816T012149418043Z_treesatai_dofa_frozen_seed43_d484c9b3` | **PROMOTABLE** | 0.26743665 | 0.25800000 | 0 |
| treesatai | dofa | full_finetune | 43 | `—` | **MISSING** | — | — | 1 |
| treesatai | panopticon | frozen | 43 | `—` | **MISSING** | — | — | 1 |
| treesatai | panopticon | full_finetune | 43 | `—` | **MISSING** | — | — | 1 |

## Non-promotable targets

### eurosat|panopticon|frozen|seed43

- No completed non-dry-run candidate was discovered or supplied.

### eurosat|panopticon|full_finetune|seed43

- No completed non-dry-run candidate was discovered or supplied.

### treesatai|dofa|full_finetune|seed43

- No completed non-dry-run candidate was discovered or supplied.

### treesatai|panopticon|frozen|seed43

- No completed non-dry-run candidate was discovered or supplied.

### treesatai|panopticon|full_finetune|seed43

- No completed non-dry-run candidate was discovered or supplied.

## Decision rule

A run is promotable only when every structural check passes. No minimum accuracy, Macro-F1, NLL, Brier, or ECE threshold is implemented for test data, and test values cannot select runs, seeds, checkpoints, or learning rates.
