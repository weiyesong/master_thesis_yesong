# Deterministic classification baseline gap analysis

Scope: EuroSAT RGB classification, deterministic inference only, target seeds 42/43/44. A target is `DONE` only when the registry contains a matching canonical run with status `VALIDATED`. It is `NEEDS_RERUN` when an attempt exists but no matching validated result survives (for example only `FAILED`, `EXCLUDED`, incomplete, or incompatible-protocol artifacts). It is `MISSING` when no matching run attempt exists.

| Model | Adaptation | Seed | Target status | Matching run | Reason |
|---|---|---:|---|---|---|
| DOFA ViT-Base | frozen | 42 | DONE | `20260807T103233022656Z_eurosat_dofa_frozen_bnlinear_seed42_338e2700` | Canonical run is `VALIDATED`; complete best checkpoint, metrics, and deterministic test predictions exist. |
| DOFA ViT-Base | frozen | 43 | DONE | `20260807T104927933179Z_eurosat_dofa_frozen_bnlinear_seed43_3567033c` | Canonical run is `VALIDATED`; complete best checkpoint, metrics, and deterministic test predictions exist. |
| DOFA ViT-Base | frozen | 44 | DONE | `20260807T110954653187Z_eurosat_dofa_frozen_bnlinear_seed44_36c48b49` | Canonical run is `VALIDATED`; complete best checkpoint, metrics, and deterministic test predictions exist. |
| DOFA ViT-Base | full fine-tuning | 42 | DONE | `20260808T211626550846Z_eurosat_dofa_full_finetune_seed42_466f9f00` | Canonical run is `VALIDATED`; instability after the best epoch is documented and does not invalidate the validation-selected checkpoint. |
| DOFA ViT-Base | full fine-tuning | 43 | DONE | `20260808T215707981044Z_eurosat_dofa_full_finetune_seed43_53c8b069` | Canonical run is `VALIDATED`; instability after the best epoch is documented and does not invalidate the validation-selected checkpoint. |
| DOFA ViT-Base | full fine-tuning | 44 | DONE | `20260808T225252226548Z_eurosat_dofa_full_finetune_seed44_a9291cb9` | Canonical run is `VALIDATED`; instability after the best epoch is documented and does not invalidate the validation-selected checkpoint. |
| Panopticon | frozen | 42 | MISSING | — | No Panopticon implementation/config/run artifact was found. |
| Panopticon | frozen | 43 | MISSING | — | No Panopticon implementation/config/run artifact was found. |
| Panopticon | frozen | 44 | MISSING | — | No Panopticon implementation/config/run artifact was found. |
| Panopticon | full fine-tuning | 42 | MISSING | — | No Panopticon implementation/config/run artifact was found. |
| Panopticon | full fine-tuning | 43 | MISSING | — | No Panopticon implementation/config/run artifact was found. |
| Panopticon | full fine-tuning | 44 | MISSING | — | No Panopticon implementation/config/run artifact was found. |

## Summary

| Status | Count |
|---|---:|
| DONE | 6 |
| NEEDS_RERUN | 0 |
| MISSING | 6 |

The deterministic DOFA classification baseline matrix is complete for both adaptation regimes and all three seeds. The entire Panopticon half of the matrix is missing. The legacy DOFA run, all dry runs, the failed preflight, and the auxiliary ResNet experiment remain in the registry for provenance but do not fill target cells.

No training was run while creating this analysis.
