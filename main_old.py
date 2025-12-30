"""
Main training script for EuroSAT classification using DOFA foundation model.

This script integrates:
- DOFA pretrained backbone
- EuroSAT dataset (13-band Sentinel-2 imagery) via TorchGeo
- PyTorch Lightning training framework
- Lightning-UQ-Box for uncertainty quantification
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from pathlib import Path

# TorchGeo imports
from torchgeo.datamodules import EuroSATDataModule

# Lightning-UQ-Box imports
from lightning_uq_box.uq_methods import DeterministicClassification

# Import custom modules
from experiments.dofa_classifier import create_dofa_classifier


class DOFALightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for DOFA-based EuroSAT classification.
    """
    
    def __init__(
        self,
        model_size: str = 'base',
        num_classes: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = False,
        pretrained_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create DOFA classifier
        self.model = create_dofa_classifier(
            model_size=model_size,
            num_classes=num_classes,
            pretrained=True,
            weights_path=pretrained_path,
            freeze_backbone=freeze_backbone,
        )
        
        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = pl.metrics.Accuracy(task='multiclass', num_classes=num_classes)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, labels)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        logits = self(images)
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, labels)
        
        # Log metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        # Separate learning rates for backbone and classifier head
        if self.hparams.freeze_backbone:
            # Only optimize classifier head
            params = self.model.classifier.parameters()
        else:
            # Different learning rates for backbone and head
            params = [
                {'params': self.model.backbone.parameters(), 'lr': self.hparams.learning_rate * 0.1},
                {'params': self.model.classifier.parameters(), 'lr': self.hparams.learning_rate},
            ]
        
        optimizer = torch.optim.AdamW(
            params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Cosine annealing scheduler
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
    
    # Create data module
    datamodule = EuroSATDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    
    # Create model
    model = DOFALightningModule(
        model_size=args.model_size,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_backbone=args.freeze_backbone,
        pretrained_path=args.pretrained_path,
    )
    
    # Callbacks
    callbacks = [
        # Save best model based on validation accuracy
        ModelCheckpoint(
            dirpath=args.output_dir / 'checkpoints',
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
        save_dir=args.output_dir,
        name='dofa_eurosat',
        version=args.experiment_name,
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=args.grad_clip,
        deterministic=True,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training: {args.experiment_name}")
    print(f"{'='*60}")
    print(f"Model: DOFA-{args.model_size}")
    print(f"Dataset: EuroSAT ({args.data_root})")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    trainer.fit(model, datamodule=datamodule)
    
    # Test on best model
    print(f"\n{'='*60}")
    print("Testing best model...")
    print(f"{'='*60}\n")
    
    trainer.test(model, datamodule=datamodule, ckpt_path='best')
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best model saved at: {callbacks[0].best_model_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DOFA on EuroSAT dataset')
    
    # Data parameters
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory for EuroSAT dataset')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes (EuroSAT has 10)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    
    # Model parameters
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['small', 'base', 'large'],
                        help='DOFA model size')
    parser.add_argument('--pretrained_path', type=str, 
                        default='./checkpoints/DOFA_ViT_base_e100.pth',
                        help='Path to pretrained DOFA weights')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze DOFA backbone (only train classifier head)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping value')
    
    # System parameters
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for logs and checkpoints')
    parser.add_argument('--experiment_name', type=str, default='dofa_base_eurosat',
                        help='Experiment name for logging')
    
    args = parser.parse_args()
    
    # Convert output_dir to Path
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    main(args)
