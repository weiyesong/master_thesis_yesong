# EO UQ Experiments

EuroSAT RGB classification experiments for comparing backbone accuracy and
calibration under the same deterministic train/validation split.

## Experiments

Run the ResNet18 RGB smoke test:

```bash
python main.py --config configs/eurosat_resnet18_rgb.yaml
```

Run the first-stage DOFA RGB linear-probe experiment:

```bash
python main.py --config configs/eurosat_dofa_rgb.yaml
```

The DOFA RGB experiment uses EuroSAT RGB imagery in Sentinel-2 visible-band
order. The input channels are passed to DOFA with the matching wavelengths:

- B04 = red = 0.665 micrometers
- B03 = green = 0.560 micrometers
- B02 = blue = 0.490 micrometers

The DOFA backbone is frozen by default and only the 10-class linear head is
trained with cross-entropy and AdamW. Validation accuracy, NLL, ECE, and Brier
score are evaluated after every epoch. The best checkpoint is selected by
validation NLL, and the best run summary is appended to `outputs/results.csv`.

## DOFA Setup

The default config expects:

- DOFA source at `../DOFA`
- pretrained weights at `../DOFA/checkpoints/DOFA_ViT_base_e100.pth`

If those paths differ, update `model.dofa_root` and `model.checkpoint_path` in
`configs/eurosat_dofa_rgb.yaml`. The paths in that config are resolved relative
to the config file. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```
