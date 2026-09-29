# Uncertainty Quantification for Earth Observation Foundation Models

**English** | [Deutsch](readme.de.md)

This repository contains the code and results documentation for a master's thesis. It studies probability calibration and uncertainty quantification (UQ) of Earth observation (EO) foundation models (**DOFA** and **Panopticon**) after downstream adaptation.

**Status (updated 2026-09-29): all planned experiments are complete, and the thesis is in the writing phase.** The following are all finished:

- 16 dataset × model × adaptation cells (72 training runs)
- Temperature Scaling, MC Dropout and Deep Ensemble evaluation
- the audit and the follow-up analyses (A/E)

No further training or model inference is needed.

## Research questions

- **RQ1**: How well calibrated are the predicted probabilities of different EO foundation models after downstream adaptation?
- **RQ2**: Does the fine-tuning strategy (frozen vs. full fine-tuning) affect calibration and task performance, and if so, how?
- **RQ3**: Do Temperature Scaling, MC Dropout and Deep Ensembles improve calibration, and do they cost task performance?

The audit standard is in [EO_FM_UQ_Core_RQ_Audit_Standard.md](EO_FM_UQ_Core_RQ_Audit_Standard.md).

## Where to start reading

Most of the linked reports are written in Chinese.

| Document | Content |
|---|---|
| [Final results package](reports/core_rq_completion_20260921/published/final_thesis_results.md) | All final tables, reliability diagrams, performance–calibration plots and segmentation uncertainty maps |
| [RQ1–RQ3 results section](reports/core_rq_completion_20260921/RESULTS_SECTION.md) | Answer to each RQ, with counterexamples and the limits of each conclusion |
| [Audit completion report](reports/core_rq_completion_20260921/COMPLETION_REPORT.md) | 29 checks: 27 PASS and 2 UNKNOWN (both about missing historical provenance) |
| [A/E follow-up report](reports/thesis_followup_execution_20260926/FOLLOWUP_COMPLETION_REPORT.md) | Error detection and selective prediction, confidence scale, UQ proxies, analytic reference experiment |
| [Draft of the thesis results chapter](reports/thesis_followup_execution_20260926/THESIS_RESULTS_DRAFT.md) | Paragraphs ready for the results chapter |
| [Thesis readiness and outline](reports/thesis_readiness_20260926/THESIS_READINESS_AND_OUTLINE.md) | Chapter structure and completeness assessment |
| [Training protocol](reports/final_training_protocol.md) / [Dataset protocols](reports/final_dataset_protocols.md) / [Pre-UQ protocol freeze](reports/pre_uq_protocol_freeze.md) | The frozen experimental settings |
| [Evidence matrix](reports/thesis_evidence_matrix.md) | The evidence file behind each claim |

## Experimental matrix

| Dimension | Setting |
|---|---|
| Foundation models | DOFA ViT-Base, Panopticon ViT-B/14 |
| Datasets | EuroSAT (10-class classification), TreeSatAI (15-label multi-label classification), CloudSEN12 (4-class cloud segmentation), SpaceNet7 (building segmentation) |
| Adaptation | Frozen backbone (only the head/decoder is trained), full fine-tuning |
| Seeds | 42, 43, 44 (all reported, none selected or dropped) |
| UQ methods | Deterministic; Temperature Scaling (EuroSAT only, fitted on a dedicated calibration split); MC Dropout (head only, p=0.1, T=30); Deep Ensemble (probability mean of the 3 seed members) |
| Checkpoint selection | Classification: minimum validation NLL. Segmentation: maximum validation mIoU |
| Metrics | Classification: accuracy, Macro-F1, NLL, Brier, ECE-15. Segmentation: mIoU, per-class IoU, pixel accuracy, NLL, Brier, ECE-15; SpaceNet7 also reports building and boundary ECE |

There are 64 method cells. 52 have results. The other 12 are Temperature Scaling cells for TreeSatAI and segmentation, which the protocol marks as N/A. In addition, there are 24 same-checkpoint comparisons of MC Dropout against dropout-off inference.

Data splits:

- EuroSAT uses a project-built 70/10/10/10 split with spatial grouping ([splits/eurosat_70_10_10_10_spatial20m/](splits/eurosat_70_10_10_10_spatial20m/)).
- TreeSatAI, CloudSEN12 and SpaceNet7 use the official GEO-Bench-2 splits.

## Main results (summary)

Full numbers are in the [final results package](reports/core_rq_completion_20260921/published/final_thesis_results.md). The table shows deterministic EuroSAT results as mean ± SD over 3 seeds:

| Model | Adaptation | Accuracy | NLL ↓ | ECE-15 ↓ |
|---|---|---:|---:|---:|
| DOFA | frozen | 0.9834 ± 0.0006 | 0.0544 ± 0.0039 | 0.0050 ± 0.0019 |
| DOFA | full | 0.9649 ± 0.0054 | 0.1069 ± 0.0203 | 0.0111 ± 0.0049 |
| Panopticon | frozen | 0.9833 ± 0.0004 | 0.0505 ± 0.0036 | 0.0066 ± 0.0026 |
| Panopticon | full | 0.9627 ± 0.0073 | 0.1126 ± 0.0374 | 0.0095 ± 0.0050 |

Key findings (each holds only within the configurations studied):

- **RQ1**: Which model is better calibrated depends on the task, the adaptation, the metric and the binning. No model is best across all conditions. A low overall ECE does not mean that every class is well calibrated.
- **RQ2**: The effect of switching from frozen to full fine-tuning depends on the condition, and the inconsistent direction is itself a finding.
  - On EuroSAT, full fine-tuning is worse.
  - On CloudSEN12 and TreeSatAI, full fine-tuning is better.
  - The two recipes use different learning rates and schedules, so the differences cannot be attributed to freezing alone.
- **RQ3**:
  - Temperature Scaling does not change the predicted class and lowers ECE and NLL in some cells.
  - Deep Ensembles raise mIoU and lower NLL and Brier on segmentation, but give a worse ECE for EuroSAT full fine-tuning.
  - Compared with dropout-off inference, MC Dropout improves ECE, NLL and Brier on segmentation. On classification the effect is inconsistent.
- **Follow-up analyses A/E**:
  - The maximum softmax probability (MSP) works for error detection and selective prediction (EuroSAT error-detection AUROC 0.92–0.98).
  - On TreeSatAI, mutual information (MI) ranks errors worse than MSP in all 10 MC/DE cases.
  - Better calibration does not guarantee better error ranking.
  - The analytic reference experiment (E) shows that posterior expected entropy cannot be read directly as the conditional entropy of the data-generating process.
- **Known limits**:
  - The statistical origin of the historical DOFA–EuroSAT normalization constants cannot be traced; these are the 2 UNKNOWN checks.
  - Segmentation has only one Deep Ensemble per configuration.
  - MC Dropout is applied only in the head.

## Code layout

```text
configs/                  # Final YAMLs for the 16 cells (*_final.yaml, eurosat_*) and c5_mc_dropout/
scripts/
├── run_experiments.py              # Classification training/evaluation entry point
├── segmentation_pipeline.py        # Segmentation training/evaluation
├── geobench_datasets.py            # TreeSatAI/CloudSEN12/SpaceNet7 loaders
├── create_eurosat_splits.py        # Creates and validates the fixed EuroSAT split
├── experiment_manager.py           # Config, run_id, seeding, environment metadata
├── prediction_export.py            # Per-sample prediction export, loading, metric recomputation
├── c3_calibration_ensembles.py     # Temperature Scaling and Deep Ensembles
├── c4_mc_dropout_pilots.py / c5_*  # MC Dropout preparation and inference
├── complete_segmentation_dropout_off.py
├── c6_build_final_results.py       # Builds the final tables and figures
└── build_*.py / audit_*.py         # Effect tables, evidence matrix, audits
models/calibration.py     # NLL, Brier, ECE, reliability diagrams
reports/                  # Protocols, audits, results, follow-up analyses (see table above)
tests/                    # Unit tests
DOFA/                     # Upstream DOFA code (weights not tracked)
```

## Usage

```bash
# Classification (example: EuroSAT Panopticon frozen, runs seeds 42/43/44 in turn)
python scripts/run_experiments.py --config configs/eurosat_panopticon_frozen_baseline.yaml
# Add --dry-run to check the pipeline only (1 epoch, a few batches, written to dry_runs/)

# Validate the EuroSAT split (does not overwrite it)
python scripts/create_eurosat_splits.py --data-root data \
  --output-dir splits/eurosat_70_10_10_10_spatial20m --validate-only

# Tests
python -m unittest discover -s tests -v
```

To re-run the A/E follow-up analyses, see [reports/thesis_followup_execution_20260926/README.md](reports/thesis_followup_execution_20260926/README.md). They need only a CPU.

## Not tracked in Git

Datasets, checkpoints, raw predictions and training outputs total several hundred GB, so they are kept only locally. This covers `data/`, `datasets/`, `research_data/`, `results/`, `checkpoints/`, `RS3DBench/` and all `*.pt`/`*.pth`/`*.npy` files. The reports record the SHA256 hash of each of these artifacts.

## Other

The following belong to historical or alternative pipelines and are not part of the thesis protocol:

- `RS3DBench/` (depth estimation)
- `eo_uq_experiments/`
- `results/first_stage_rgb/`
- early configs such as `config.yaml` and `eurosat_dofa_rgb.yaml`

Do not pool their results with the final results.
