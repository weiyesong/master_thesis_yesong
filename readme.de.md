# Unsicherheitsquantifizierung für Foundation Models der Erdbeobachtung

[English](readme.md) | **Deutsch**

Dieses Repository enthält den Code und die Ergebnisdokumentation einer Masterarbeit. Untersucht werden die Wahrscheinlichkeitskalibrierung und die Unsicherheitsquantifizierung (UQ) von Foundation Models der Erdbeobachtung (EO) (**DOFA** und **Panopticon**) nach der Anpassung an Downstream-Aufgaben.

**Stand (aktualisiert am 29.09.2026): Alle geplanten Experimente sind abgeschlossen, die Arbeit befindet sich in der Schreibphase.** Abgeschlossen sind:

- 16 Zellen aus Datensatz × Modell × Anpassung (72 Trainingsläufe)
- die Auswertung von Temperature Scaling, MC Dropout und Deep Ensembles
- das Audit und die Zusatzanalysen (A/E)

Weiteres Training oder weitere Modellinferenz ist nicht nötig.

## Forschungsfragen

- **RQ1**: Wie gut sind die vorhergesagten Wahrscheinlichkeiten verschiedener EO-Foundation-Models nach der Downstream-Anpassung kalibriert?
- **RQ2**: Beeinflusst die Fine-Tuning-Strategie (eingefroren vs. vollständiges Fine-Tuning) Kalibrierung und Aufgabenleistung, und wenn ja, wie?
- **RQ3**: Verbessern Temperature Scaling, MC Dropout und Deep Ensembles die Kalibrierung, und geht das auf Kosten der Aufgabenleistung?

Der Audit-Standard steht in [EO_FM_UQ_Core_RQ_Audit_Standard.md](EO_FM_UQ_Core_RQ_Audit_Standard.md).

## Einstieg in die Dokumentation

Die meisten verlinkten Berichte sind auf Chinesisch verfasst.

| Dokument | Inhalt |
|---|---|
| [Finales Ergebnispaket](reports/core_rq_completion_20260921/published/final_thesis_results.md) | Alle finalen Tabellen, Reliability-Diagramme, Leistungs-Kalibrierungs-Grafiken und Unsicherheitskarten der Segmentierung |
| [Ergebnisteil zu RQ1–RQ3](reports/core_rq_completion_20260921/RESULTS_SECTION.md) | Antwort auf jede Forschungsfrage, mit Gegenbeispielen und den Grenzen jeder Schlussfolgerung |
| [Audit-Abschlussbericht](reports/core_rq_completion_20260921/COMPLETION_REPORT.md) | 29 Prüfungen: 27 PASS und 2 UNKNOWN (beide wegen fehlender historischer Herkunftsnachweise) |
| [Bericht zu den Zusatzanalysen A/E](reports/thesis_followup_execution_20260926/FOLLOWUP_COMPLETION_REPORT.md) | Fehlererkennung und selektive Vorhersage, Konfidenzskala, UQ-Proxys, analytisches Referenzexperiment |
| [Entwurf des Ergebniskapitels](reports/thesis_followup_execution_20260926/THESIS_RESULTS_DRAFT.md) | Absätze, die direkt ins Ergebniskapitel übernommen werden können |
| [Stand und Gliederung der Arbeit](reports/thesis_readiness_20260926/THESIS_READINESS_AND_OUTLINE.md) | Kapitelstruktur und Bewertung der Vollständigkeit |
| [Trainingsprotokoll](reports/final_training_protocol.md) / [Datensatzprotokolle](reports/final_dataset_protocols.md) / [Protokollfixierung vor UQ](reports/pre_uq_protocol_freeze.md) | Die festgeschriebenen Versuchseinstellungen |
| [Evidenzmatrix](reports/thesis_evidence_matrix.md) | Die Belegdatei zu jeder Aussage |

## Versuchsmatrix

| Dimension | Einstellung |
|---|---|
| Foundation Models | DOFA ViT-Base, Panopticon ViT-B/14 |
| Datensätze | EuroSAT (Klassifikation mit 10 Klassen), TreeSatAI (Multi-Label-Klassifikation mit 15 Labels), CloudSEN12 (Wolkensegmentierung mit 4 Klassen), SpaceNet7 (Gebäudesegmentierung) |
| Anpassung | Eingefrorenes Backbone (nur Head/Decoder wird trainiert), vollständiges Fine-Tuning |
| Seeds | 42, 43, 44 (alle berichtet, keiner ausgewählt oder verworfen) |
| UQ-Methoden | Deterministisch; Temperature Scaling (nur EuroSAT, auf einem eigenen Kalibrierungs-Split angepasst); MC Dropout (nur im Head, p=0.1, T=30); Deep Ensemble (Mittelwert der Wahrscheinlichkeiten der 3 Seed-Modelle) |
| Checkpoint-Auswahl | Klassifikation: minimale Validierungs-NLL. Segmentierung: maximale Validierungs-mIoU |
| Metriken | Klassifikation: Accuracy, Macro-F1, NLL, Brier, ECE-15. Segmentierung: mIoU, IoU pro Klasse, Pixel-Accuracy, NLL, Brier, ECE-15; bei SpaceNet7 zusätzlich Gebäude- und Rand-ECE |

Es gibt 64 Methodenzellen. 52 davon haben Ergebnisse. Die übrigen 12 sind Temperature-Scaling-Zellen für TreeSatAI und die Segmentierung, die das Protokoll als N/A führt. Zusätzlich gibt es 24 Vergleiche von MC Dropout mit Inferenz ohne Dropout, jeweils mit demselben Checkpoint.

Datensplits:

- EuroSAT verwendet einen projekteigenen 70/10/10/10-Split mit räumlicher Gruppierung ([splits/eurosat_70_10_10_10_spatial20m/](splits/eurosat_70_10_10_10_spatial20m/)).
- TreeSatAI, CloudSEN12 und SpaceNet7 verwenden die offiziellen GEO-Bench-2-Splits.

## Wichtigste Ergebnisse (Zusammenfassung)

Alle Werte stehen im [finalen Ergebnispaket](reports/core_rq_completion_20260921/published/final_thesis_results.md). Die Tabelle zeigt die deterministischen EuroSAT-Ergebnisse als Mittelwert ± Standardabweichung über 3 Seeds:

| Modell | Anpassung | Accuracy | NLL ↓ | ECE-15 ↓ |
|---|---|---:|---:|---:|
| DOFA | eingefroren | 0.9834 ± 0.0006 | 0.0544 ± 0.0039 | 0.0050 ± 0.0019 |
| DOFA | vollständig | 0.9649 ± 0.0054 | 0.1069 ± 0.0203 | 0.0111 ± 0.0049 |
| Panopticon | eingefroren | 0.9833 ± 0.0004 | 0.0505 ± 0.0036 | 0.0066 ± 0.0026 |
| Panopticon | vollständig | 0.9627 ± 0.0073 | 0.1126 ± 0.0374 | 0.0095 ± 0.0050 |

Zentrale Befunde (jeweils nur innerhalb der untersuchten Konfigurationen gültig):

- **RQ1**: Welches Modell besser kalibriert ist, hängt von Aufgabe, Anpassung, Metrik und Binning ab. Kein Modell ist unter allen Bedingungen am besten. Ein niedriger Gesamt-ECE bedeutet nicht, dass jede Klasse gut kalibriert ist.
- **RQ2**: Der Effekt des Wechsels von eingefroren zu vollständigem Fine-Tuning hängt von der Bedingung ab; die uneinheitliche Richtung ist selbst ein Befund.
  - Auf EuroSAT schneidet vollständiges Fine-Tuning schlechter ab.
  - Auf CloudSEN12 und TreeSatAI schneidet es besser ab.
  - Die beiden Rezepte verwenden unterschiedliche Lernraten und Trainingspläne, daher lassen sich die Unterschiede nicht allein auf das Einfrieren zurückführen.
- **RQ3**:
  - Temperature Scaling ändert die vorhergesagte Klasse nicht und senkt in einigen Zellen ECE und NLL.
  - Deep Ensembles erhöhen bei der Segmentierung die mIoU und senken NLL und Brier, verschlechtern aber den ECE bei EuroSAT mit vollständigem Fine-Tuning.
  - Gegenüber Inferenz ohne Dropout verbessert MC Dropout bei der Segmentierung ECE, NLL und Brier. Bei der Klassifikation ist der Effekt uneinheitlich.
- **Zusatzanalysen A/E**:
  - Die maximale Softmax-Wahrscheinlichkeit (MSP) eignet sich zur Fehlererkennung und selektiven Vorhersage (AUROC der Fehlererkennung auf EuroSAT: 0.92–0.98).
  - Auf TreeSatAI ordnet die Mutual Information (MI) Fehler in allen 10 MC/DE-Fällen schlechter als MSP.
  - Bessere Kalibrierung garantiert keine bessere Fehlerrangfolge.
  - Das analytische Referenzexperiment (E) zeigt, dass die erwartete Entropie unter der Posterior-Verteilung nicht direkt als bedingte Entropie des datenerzeugenden Prozesses gelesen werden kann.
- **Bekannte Grenzen**:
  - Die statistische Herkunft der historischen DOFA–EuroSAT-Normalisierungskonstanten lässt sich nicht nachvollziehen; das sind die 2 UNKNOWN-Prüfungen.
  - Für die Segmentierung gibt es nur ein Deep Ensemble pro Konfiguration.
  - MC Dropout wird nur im Head angewendet.

## Codestruktur

```text
configs/                  # Finale YAMLs der 16 Zellen (*_final.yaml, eurosat_*) und c5_mc_dropout/
scripts/
├── run_experiments.py              # Einstiegspunkt für Training/Auswertung der Klassifikation
├── segmentation_pipeline.py        # Training/Auswertung der Segmentierung
├── geobench_datasets.py            # Datenlader für TreeSatAI/CloudSEN12/SpaceNet7
├── create_eurosat_splits.py        # Erzeugt und prüft den festen EuroSAT-Split
├── experiment_manager.py           # Konfiguration, run_id, Seeds, Umgebungsmetadaten
├── prediction_export.py            # Export, Laden und Metrik-Neuberechnung der Vorhersagen pro Sample
├── c3_calibration_ensembles.py     # Temperature Scaling und Deep Ensembles
├── c4_mc_dropout_pilots.py / c5_*  # Vorbereitung und Inferenz für MC Dropout
├── complete_segmentation_dropout_off.py
├── c6_build_final_results.py       # Erzeugt die finalen Tabellen und Abbildungen
└── build_*.py / audit_*.py         # Effekttabellen, Evidenzmatrix, Audits
models/calibration.py     # NLL, Brier, ECE, Reliability-Diagramme
reports/                  # Protokolle, Audits, Ergebnisse, Zusatzanalysen (siehe Tabelle oben)
tests/                    # Unit-Tests
DOFA/                     # Upstream-Code von DOFA (Gewichte nicht versioniert)
```

## Verwendung

```bash
# Klassifikation (Beispiel: EuroSAT Panopticon eingefroren, führt die Seeds 42/43/44 nacheinander aus)
python scripts/run_experiments.py --config configs/eurosat_panopticon_frozen_baseline.yaml
# Mit --dry-run wird nur die Pipeline geprüft (1 Epoche, wenige Batches, Ausgabe in dry_runs/)

# EuroSAT-Split prüfen (wird nicht überschrieben)
python scripts/create_eurosat_splits.py --data-root data \
  --output-dir splits/eurosat_70_10_10_10_spatial20m --validate-only

# Tests
python -m unittest discover -s tests -v
```

Wie die Zusatzanalysen A/E erneut ausgeführt werden, steht in [reports/thesis_followup_execution_20260926/README.md](reports/thesis_followup_execution_20260926/README.md). Dafür genügt eine CPU.

## Nicht in Git enthalten

Datensätze, Checkpoints, Rohvorhersagen und Trainingsausgaben umfassen zusammen mehrere hundert GB und liegen daher nur lokal vor. Das betrifft `data/`, `datasets/`, `research_data/`, `results/`, `checkpoints/`, `RS3DBench/` sowie alle Dateien vom Typ `*.pt`/`*.pth`/`*.npy`. Die Berichte halten für jedes dieser Artefakte den SHA256-Hash fest.

## Sonstiges

Folgende Teile gehören zu historischen oder alternativen Pipelines und sind nicht Teil des Protokolls der Arbeit:

- `RS3DBench/` (Tiefenschätzung)
- `eo_uq_experiments/`
- `results/first_stage_rgb/`
- frühe Konfigurationen wie `config.yaml` und `eurosat_dofa_rgb.yaml`

Ihre Ergebnisse dürfen nicht mit den finalen Ergebnissen zusammengefasst werden.
