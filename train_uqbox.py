"""
Complete training pipeline for EuroSAT classification using:
- DOFA foundation model (pretrained on multi-modal EO data)
- TorchGeo EuroSAT datamodule
- Lightning-UQ-Box for uncertainty quantification

This script demonstrates how to integrate all three components for
robust satellite image classification with uncertainty estimation.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path
import argparse

# TorchGeo imports
from torchgeo.datamodules import EuroSATDataModule

# Lightning-UQ-Box imports
try:
    from lightning_uq_box.uq_methods import DeterministicClassification
    UQBOX_AVAILABLE = True
except ImportError:
    print("Warning: Lightning-UQ-Box not available. Using standard Lightning module.")
    UQBOX_AVAILABLE = False

# Custom DOFA model
from experiments.dofa_classifier import create_dofa_classifier


def create_dofa_uqbox_model(
    model_size: str = 'base',
    num_classes: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    pretrained_path: str = None,
    freeze_backbone: bool = False,
):
    """
    Create DOFA model wrapped with Lightning-UQ-Box for uncertainty quantification.
    
    Args:
        model_size: Size of DOFA model ('small', 'base', 'large')
        num_classes: Number of output classes
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        pretrained_path: Path to pretrained DOFA weights
        freeze_backbone: Whether to freeze the DOFA backbone
    
    Returns:
        Lightning module with UQ capabilities
    """
    # Create DOFA backbone
    dofa_model = create_dofa_classifier(
        model_size=model_size,
        num_classes=num_classes,
        pretrained=True,
        weights_path=pretrained_path,
        freeze_backbone=freeze_backbone,
    )
    
    if UQBOX_AVAILABLE:
        # Wrap with UQ-Box for uncertainty quantification
        # DeterministicClassification provides:
        # - Cross-entropy loss
        # - Accuracy metrics
        # - Expected Calibration Error (ECE)
        # - Proper logging and checkpointing
        model = DeterministicClassification(
            model=dofa_model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        print("✓ Using Lightning-UQ-Box wrapper (with uncertainty metrics)")
    else:
        # Fallback to standard Lightning module
        model = StandardDOFAModule(
            model=dofa_model,
            num_classes=num_classes,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        print("✓ Using standard PyTorch Lightning module")
    
    return model


class StandardDOFAModule(pl.LightningModule):
    """
    Standard PyTorch Lightning module (fallback if UQ-Box not available).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Use torchmetrics for metrics
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
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, labels)
        
        self.log('test/loss', loss, on_epoch=True)
        self.log('test/acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
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


def main(args):
    """Main training function."""
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    print("=" * 70)
    print("DOFA + TorchGeo + Lightning-UQ-Box Training Pipeline")
    print("=" * 70)
    
    # ========================================================================
    # 1. Data Loading: TorchGeo EuroSAT DataModule
    # ========================================================================
    print("\n[1/4] Setting up data module...")
    
    datamodule = EuroSATDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=True,  # Automatically download if missing
    )
    
    # Setup the datamodule to download/prepare data
    datamodule.setup()
    
    print(f"✓ EuroSAT DataModule configured:")
    print(f"  - Root: {args.data_root}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Num workers: {args.num_workers}")
    print(f"  - Classes: 10 (AnnualCrop, Forest, HerbaceousVegetation, etc.)")
    
    # ========================================================================
    # 2. Model: DOFA Classifier with UQ-Box Integration
    # ========================================================================
    print("\n[2/4] Creating DOFA model with UQ-Box wrapper...")
    
    model = create_dofa_uqbox_model(
        model_size=args.model_size,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created:")
    print(f"  - Backbone: DOFA ViT-{args.model_size}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Freeze backbone: {args.freeze_backbone}")
    
    # ========================================================================
    # 3. Training Setup: Callbacks and Logger
    # ========================================================================
    print("\n[3/4] Setting up training configuration...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Callbacks
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='dofa-{epoch:02d}-{val/acc:.4f}',
            monitor='val/acc',
            mode='max',
            save_top_k=3,
            save_last=True,
        ),
        # Early stopping
        EarlyStopping(
            monitor='val/acc',
            patience=args.patience,
            mode='max',
            verbose=True,
        ),
        # Learning rate monitor
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
        enable_progress_bar=True,
    )
    
    print(f"✓ Trainer configured:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Accelerator: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Precision: {args.precision}")
    print(f"  - Output: {output_dir}")
    
    # ========================================================================
    # 4. Training and Testing
    # ========================================================================
    print("\n[4/4] Starting training...")
    print("=" * 70)
    
    # Fit the model
    trainer.fit(model, datamodule=datamodule)
    
    print("\n" + "=" * 70)
    print("Training completed! Running final test...")
    print("=" * 70)
    
    # Test the model
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path='best')
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for metric, value in test_results[0].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nBest model saved at: {callbacks[0].best_model_path}")
    print(f"TensorBoard logs: {output_dir / 'dofa_eurosat' / args.experiment_name}")
    print(f"\nView logs with: tensorboard --logdir {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train DOFA on EuroSAT with TorchGeo and Lightning-UQ-Box'
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
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--experiment_name', type=str, default='dofa_uqbox',
                        help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Run training
    main(args)
