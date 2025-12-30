# DOFA + TorchGeo + Lightning Training Pipeline

Complete training pipeline for EuroSAT classification using DOFA foundation model.

## ✅ What's Included

- **[main.py](main.py)** - Complete training script with all integrations
- **[experiments/dofa_classifier.py](experiments/dofa_classifier.py)** - DOFA model wrapper
- **[checkpoints/DOFA_ViT_base_e100.pth](checkpoints/DOFA_ViT_base_e100.pth)** - Pretrained weights (427 MB)
- **Data**: Auto-downloads EuroSAT dataset (2.07 GB)

## 🔧 Key Features

### Data Processing
✅ **TorchGeo EuroSATDataModule** integration  
✅ **Automatic shape fixing**: `[B,1,C,H,W]` → `[B,C,H,W]`  
✅ **Resize to 224×224** (DOFA requirement)  
✅ **Sentinel-2 normalization** (band-specific mean/std)  
✅ **Dict format handling**: `{'image': x, 'label': y}`

### Model
✅ **DOFA ViT-Base** pretrained on multi-modal EO data  
✅ **Sentinel-2 wavelength embedding** (13 bands)  
✅ **111M parameters** (pretrained backbone + classification head)  
✅ **Freeze/unfreeze backbone** option

### Training
✅ **Lightning-UQ-Box** integration (optional, falls back to standard Lightning)  
✅ **Cross-entropy loss** + **Accuracy metrics**  
✅ **ModelCheckpoint** (saves top-3 + best)  
✅ **Early stopping** (patience=10)  
✅ **TensorBoard logging**  
✅ **Mixed precision** support

## 🚀 Quick Start

### 1. Basic Training (5 epochs)
```bash
python main.py \
    --data_root ./data \
    --batch_size 32 \
    --max_epochs 5 \
    --experiment_name "dofa_full_finetune"
```

### 2. Linear Probing (freeze backbone)
```bash
python main.py \
    --data_root ./data \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --max_epochs 20 \
    --freeze_backbone \
    --experiment_name "dofa_linear_probe"
```

### 3. Full Fine-tuning (recommended)
```bash
python main.py \
    --data_root ./data \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --max_epochs 50 \
    --patience 10 \
    --experiment_name "dofa_full_50ep"
```

### 4. With GPU
```bash
python main.py \
    --data_root ./data \
    --batch_size 64 \
    --gpus 1 \
    --precision 16-mixed \
    --max_epochs 50
```

## 📊 Expected Results

### Shape Verification
```
Image Batch Shape: [32, 13, 224, 224] ✓
  - Batch: 32
  - Channels: 13 (Sentinel-2)
  - Size: 224×224

Label Batch Shape: [32] ✓

DOFA forward pass: [32, 10] ✓
```

### Training Output
```
[1/4] Setting up data module...
✓ Data module configured
  - Root: ./data
  - Batch size: 32
  - Transforms: Resize(224x224) + Normalize(Sentinel-2)
  - Format: Dict {'image': [B,C,H,W], 'label': [B]}

[2/4] Creating DOFA classifier...
✓ DOFA Classifier created
  - Model size: base
  - Total parameters: 111,319,818
  - Trainable parameters: 111,168,522
  - Wavelengths: 13 Sentinel-2 bands

[3/4] Wrapping with Lightning module...
✓ Using Lightning-UQ-Box wrapper (CCE Loss + Uncertainty)

[4/4] Setting up trainer...
✓ Trainer configured
  - Max epochs: 5
  - Accelerator: GPU
  - Devices: 1
```

## 📁 Project Structure

```
workspace/
├── main.py                          # ← Main training script
├── experiments/
│   ├── dofa_classifier.py          # DOFA model wrapper
│   └── __init__.py
├── checkpoints/
│   └── DOFA_ViT_base_e100.pth     # Pretrained weights (427 MB)
├── data/                           # Auto-downloaded EuroSAT
│   └── eurosat/
├── results/                        # Training outputs
│   ├── checkpoints/
│   │   ├── dofa-epoch=XX-val_acc=X.XXXX.ckpt
│   │   └── last.ckpt
│   └── dofa_eurosat/
│       └── {experiment_name}/
│           └── events.out.tfevents.*
└── test_main.sh                    # Quick test script
```

## 🔍 Command Line Arguments

### Data Parameters
```bash
--data_root ./data              # EuroSAT dataset root
--num_classes 10                # Number of classes
```

### Model Parameters
```bash
--model_size base               # 'small', 'base', or 'large'
--pretrained_path ./checkpoints/DOFA_ViT_base_e100.pth
--freeze_backbone              # Freeze DOFA backbone (linear probing)
```

### Training Parameters
```bash
--batch_size 32                # Batch size
--learning_rate 1e-4           # Learning rate
--weight_decay 1e-4            # Weight decay
--max_epochs 5                 # Max epochs
--patience 10                  # Early stopping patience
```

### System Parameters
```bash
--gpus 1                       # Number of GPUs
--num_workers 4                # Data loading workers
--precision 32                 # '32', '16-mixed', 'bf16-mixed'
--seed 42                      # Random seed
```

### Output Parameters
```bash
--output_dir ./results         # Output directory
--experiment_name dofa_base    # Experiment name
```

## 📈 Monitor Training

```bash
tensorboard --logdir ./results
```

Open http://localhost:6006 in your browser.

## 🐛 Troubleshooting

### Issue: Shape mismatch
**Symptom**: `RuntimeError: shape mismatch`  
**Solution**: The script automatically handles this via `EuroSATDataModuleWrapper`

### Issue: Out of memory
**Symptom**: `CUDA out of memory`  
**Solution**: Reduce batch size:
```bash
python main.py --batch_size 16
```

### Issue: Slow data loading
**Symptom**: Training slow  
**Solution**: Increase workers:
```bash
python main.py --num_workers 8
```

### Issue: Values not normalized
**Symptom**: Training unstable  
**Solution**: Already handled with Sentinel-2 statistics in the script

## 🧪 Testing

Quick test run (1 epoch):
```bash
./test_main.sh
```

Or manually:
```bash
python main.py \
    --batch_size 4 \
    --max_epochs 1 \
    --num_workers 0 \
    --experiment_name "test"
```

## 📝 Key Implementation Details

### 1. Sentinel-2 Wavelengths
```python
SENTINEL2_WAVELENGTHS = [
    0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783,
    0.842, 0.865, 0.945, 1.375, 1.610, 2.190
]
```

### 2. Shape Fixing
```python
# TorchGeo returns [B, 1, C, H, W]
if images.ndim == 5:
    images = images.squeeze(1)  # → [B, C, H, W]
```

### 3. Normalization
```python
# Band-specific Sentinel-2 statistics
K.Normalize(mean=s2_mean, std=s2_std)
```

### 4. Model Instantiation
```python
dofa_model = DOFAClassifier(
    num_classes=10,
    model_size='base',
    weights_path='./checkpoints/DOFA_ViT_base_e100.pth',
    wavelengths=SENTINEL2_WAVELENGTHS,
    freeze_backbone=False,
)
```

### 5. Lightning Integration
```python
# Option 1: UQ-Box (with uncertainty)
model = DeterministicClassification(
    model=dofa_model,
    num_classes=10,
    learning_rate=1e-4,
)

# Option 2: Standard Lightning (fallback)
model = DOFALightningModule(
    model=dofa_model,
    num_classes=10,
    learning_rate=1e-4,
)
```

## 🎯 Expected Performance

Based on DOFA paper and EuroSAT benchmarks:

| Model | Accuracy | Training Time (5 epochs) |
|-------|----------|--------------------------|
| DOFA-Base (frozen) | ~92-94% | ~10 min (GPU) |
| DOFA-Base (full) | ~95-97% | ~15 min (GPU) |

## 📚 References

- **DOFA**: [Neural Plasticity-Inspired Foundation Model](https://arxiv.org/abs/2403.15356)
- **EuroSAT**: [Land Use and Land Cover Classification](https://github.com/phelber/EuroSAT)
- **TorchGeo**: [Microsoft TorchGeo](https://github.com/microsoft/torchgeo)
- **Lightning-UQ-Box**: [Uncertainty Quantification](https://github.com/lightning-uq-box/lightning-uq-box)

---

**Ready to train!** 🚀

```bash
python main.py
```
