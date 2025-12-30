"""
Main training script for EuroSAT classification using:
- DOFA foundation model (pretrained)
- TorchGeo EuroSAT dataset
- Lightning-UQ-Box for uncertainty quantification

This script handles all the necessary data preprocessing, format conversion,
and integration between the three frameworks.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import argparse

# TorchGeo imports
from torchgeo.datamodules import EuroSATDataModule
from torchgeo.transforms import AugmentationSequential
import kornia.augmentation as K

# Lightning-UQ-Box imports
try:
    from lightning_uq_box.uq_methods import DeterministicClassification
    UQBOX_AVAILABLE = True
    print("✓ Lightning-UQ-Box available")
except ImportError:
    UQBOX_AVAILABLE = False
    print("⚠️  Lightning-UQ-Box not available, using standard PyTorch Lightning")

# Custom DOFA model
from experiments.dofa_classifier import DOFAClassifier


# ============================================================================
# Sentinel-2 Wavelengths Configuration
# ============================================================================
SENTINEL2_WAVELENGTHS = [
    0.443,  # Band 1  - Coastal aerosol
    0.490,  # Band 2  - Blue
    0.560,  # Band 3  - Green
    0.665,  # Band 4  - Red
    0.705,  # Band 5  - Red Edge 1
    0.740,  # Band 6  - Red Edge 2
    0.783,  # Band 7  - Red Edge 3
    0.842,  # Band 8  - NIR
    0.865,  # Band 8a - Narrow NIR
    0.945,  # Band 9  - Water vapor
    1.375,  # Band 10 - SWIR - Cirrus
    1.610,  # Band 11 - SWIR 1
    2.190,  # Band 12 - SWIR 2
]


# ============================================================================
# DataModule Wrapper with Format Adaptation
# ============================================================================
class EuroSATDataModuleWrapper(pl.LightningDataModule):
    """
    Wrapper around TorchGeo's EuroSATDataModule that:
    1. Applies proper transforms (resize to 224x224)
    2. Fixes shape issues (squeeze temporal dimension)
    3. Normalizes with Sentinel-2 statistics
    4. Converts dict format to tuple for compatibility
    """
    
    def __init__(
        self,
        root: str = './data',
        batch_size: int = 32,
        num_workers: int = 4,
        download: bool = True,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        
        # Sentinel-2 normalization statistics
        self.s2_mean = torch.tensor([
            1370.19, 1184.39, 1120.77, 1136.89, 1263.73,
            1645.40, 1846.87, 1762.59, 1972.62, 2197.07,
            2383.72, 2093.36, 1517.16
        ])
        self.s2_std = torch.tensor([
            633.15, 650.20, 712.12, 965.23, 948.99,
            1108.06, 1258.36, 1233.18, 1364.38, 1497.52,
            1602.03, 1505.44, 1084.63
        ])
        
    def prepare_data(self):
        """Download data if needed."""
        if self.download:
            base_dm = EuroSATDataModule(
                root=self.root,
                batch_size=self.batch_size,
                num_workers=0,
                download=True,
            )
            base_dm.prepare_data()
    
    def setup(self, stage=None):
        """Setup datasets with proper transforms."""
        
        # Define transforms with resize and normalization
        train_transform = AugmentationSequential(
            K.Resize(size=(224, 224)),  # Resize to 224x224 for DOFA
            K.Normalize(mean=self.s2_mean, std=self.s2_std),
            data_keys=["image"],
        )
        
        val_transform = AugmentationSequential(
            K.Resize(size=(224, 224)),
            K.Normalize(mean=self.s2_mean, std=self.s2_std),
            data_keys=["image"],
        )
        
        # Create base datamodule
        self.base_dm = EuroSATDataModule(
            root=self.root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            download=False,
        )
        
        # Setup base datamodule
        self.base_dm.setup(stage)
        
        # Apply transforms
        if stage in (None, 'fit'):
            self.base_dm.train_dataset.transforms = train_transform
            self.base_dm.val_dataset.transforms = val_transform
        
        if stage in (None, 'test'):
            self.base_dm.test_dataset.transforms = val_transform
    
    def _fix_batch(self, batch):
        """
        Fix batch format:
        1. Squeeze temporal dimension: [B, 1, C, H, W] -> [B, C, H, W]
        2. Keep dict format (Lightning-UQ-Box can handle it)
        """
        images = batch['image']
        labels = batch['label']
        
        # Squeeze temporal dimension if present
        if images.ndim == 5:  # [B, T, C, H, W]
            images = images.squeeze(1)  # -> [B, C, H, W]
        
        return {'image': images, 'label': labels}
    
    def train_dataloader(self):
        """Return train dataloader with batch fixing."""
        return FixedDataLoader(self.base_dm.train_dataloader(), self._fix_batch)
    
    def val_dataloader(self):
        """Return validation dataloader with batch fixing."""
        return FixedDataLoader(self.base_dm.val_dataloader(), self._fix_batch)
    
    def test_dataloader(self):
        """Return test dataloader with batch fixing."""
        return FixedDataLoader(self.base_dm.test_dataloader(), self._fix_batch)


class FixedDataLoader:
    """Wrapper to apply batch fixing on-the-fly."""
    
    def __init__(self, base_loader, fix_fn):
        self.base_loader = base_loader
        self.fix_fn = fix_fn
    
    def __iter__(self):
        for batch in self.base_loader:
            yield self.fix_fn(batch)
    
    def __len__(self):
        return len(self.base_loader)


# ============================================================================
# Lightning Module Wrapper
# ============================================================================
class DOFALightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for DOFA classifier.
    Handles dict input format from TorchGeo.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        from torchmetrics import Accuracy
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        
        self.log('test/loss', loss, on_epoch=True, sync_dist=True)
        self.log('test/acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


# ============================================================================
# Main Training Function
# ============================================================================
def main(args):
    """Main training function."""
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    print("=" * 70)
    print("DOFA + TorchGeo + Lightning Training Pipeline")
    print("=" * 70)
    
    # ========================================================================
    # 1. Data Loading with proper transforms and format conversion
    # ========================================================================
    print("\n[1/4] Setting up data module...")
    
    datamodule = EuroSATDataModuleWrapper(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,
    )
    
    print(f"✓ Data module configured:")
    print(f"  - Root: {args.data_root}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Transforms: Resize(224x224) + Normalize(Sentinel-2)")
    print(f"  - Format: Dict {{'image': [B,C,H,W], 'label': [B]}}")
    
    # ========================================================================
    # 2. Model Setup: DOFA Classifier
    # ========================================================================
    print("\n[2/4] Creating DOFA classifier...")
    
    # Instantiate DOFA model with Sentinel-2 wavelengths
    dofa_model = DOFAClassifier(
        num_classes=args.num_classes,
        embed_dim=768,  # Base model
        model_size=args.model_size,
        weights_path=args.pretrained_path,
        wavelengths=SENTINEL2_WAVELENGTHS,
        freeze_backbone=args.freeze_backbone,
    )
    
    total_params = sum(p.numel() for p in dofa_model.parameters())
    trainable_params = sum(p.numel() for p in dofa_model.parameters() if p.requires_grad)
    
    print(f"✓ DOFA Classifier created:")
    print(f"  - Model size: {args.model_size}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Wavelengths: {len(SENTINEL2_WAVELENGTHS)} Sentinel-2 bands")
    print(f"  - Freeze backbone: {args.freeze_backbone}")
    
    # ========================================================================
    # 3. Wrap with Lightning Module (UQ-Box or standard)
    # ========================================================================
    print("\n[3/4] Wrapping with Lightning module...")
    
    if UQBOX_AVAILABLE:
        # Try to use UQ-Box wrapper
        try:
            model = DeterministicClassification(
                model=dofa_model,
                num_classes=args.num_classes,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            print("✓ Using Lightning-UQ-Box wrapper (CCE Loss + Uncertainty)")
        except Exception as e:
            print(f"⚠️  UQ-Box wrapper failed ({e}), using standard Lightning")
            model = DOFALightningModule(
                model=dofa_model,
                num_classes=args.num_classes,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
            )
    else:
        model = DOFALightningModule(
            model=dofa_model,
            num_classes=args.num_classes,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        print("✓ Using standard PyTorch Lightning wrapper")
    
    # ========================================================================
    # 4. Training Setup: Trainer + Callbacks
    # ========================================================================
    print("\n[4/4] Setting up trainer...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='dofa-{epoch:02d}-{val/acc:.4f}',
            monitor='val/acc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/acc',
            patience=args.patience,
            mode='max',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='dofa_eurosat',
        version=args.experiment_name,
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if torch.cuda.is_available() else 1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
    )
    
    print(f"✓ Trainer configured:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Accelerator: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Devices: {args.gpus if torch.cuda.is_available() else 1}")
    print(f"  - Precision: {args.precision}")
    print(f"  - Output: {output_dir}")
    
    # ========================================================================
    # Training
    # ========================================================================
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
    trainer.fit(model, datamodule=datamodule)
    
    # ========================================================================
    # Testing
    # ========================================================================
    print("\n" + "=" * 70)
    print("Running Final Test...")
    print("=" * 70 + "\n")
    
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best')
    
    # ========================================================================
    # Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for metric, value in test_results[0].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nBest model: {callbacks[0].best_model_path}")
    print(f"Logs: {output_dir / 'dofa_eurosat' / args.experiment_name}")
    print(f"\nView with: tensorboard --logdir {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DOFA on EuroSAT with TorchGeo and Lightning'
    )
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for EuroSAT dataset')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes (EuroSAT has 10)')
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='DOFA model size')
    parser.add_argument('--pretrained_path', type=str,
                        default='./checkpoints/DOFA_ViT_base_e100.pth',
                        help='Path to pretrained DOFA weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze DOFA backbone (linear probing)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # System parameters
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--precision', type=str, default='32',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='dofa_base',
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Run training
    main(args)
