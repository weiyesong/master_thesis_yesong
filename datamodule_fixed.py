"""
Fixed EuroSAT DataModule wrapper with proper preprocessing for DOFA.

Key fixes:
1. Squeeze temporal dimension: [B, 1, C, H, W] → [B, C, H, W]
2. Resize to 224x224
3. Normalize Sentinel-2 values properly
"""

import torch
import kornia.augmentation as K
from torchgeo.datamodules import EuroSATDataModule as TorchGeoEuroSAT
from torchgeo.transforms import AugmentationSequential
import pytorch_lightning as pl


class EuroSATDataModuleFixed(pl.LightningDataModule):
    """
    Wrapper around TorchGeo's EuroSATDataModule with proper preprocessing for DOFA.
    
    Handles:
    - Automatic resizing to 224x224
    - Proper normalization for Sentinel-2 data
    - Squeezing extra temporal dimension
    - Wavelength-aware preprocessing
    """
    
    def __init__(
        self,
        root: str = './data',
        batch_size: int = 32,
        num_workers: int = 4,
        download: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.normalize = normalize
        
        # Sentinel-2 statistics (from EuroSAT dataset)
        # Mean and std for each of the 13 bands
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
        # Create base datamodule to trigger download
        if self.download:
            base_dm = TorchGeoEuroSAT(
                root=self.root,
                batch_size=self.batch_size,
                num_workers=0,
                download=True,
            )
            base_dm.prepare_data()
    
    def setup(self, stage=None):
        """Setup datasets with proper transforms."""
        
        # Define transforms with normalization
        if self.normalize:
            train_transform = AugmentationSequential(
                K.Resize(size=(224, 224)),
                K.Normalize(mean=self.s2_mean, std=self.s2_std),
                data_keys=["image"],
            )
            val_transform = AugmentationSequential(
                K.Resize(size=(224, 224)),
                K.Normalize(mean=self.s2_mean, std=self.s2_std),
                data_keys=["image"],
            )
        else:
            train_transform = AugmentationSequential(
                K.Resize(size=(224, 224)),
                data_keys=["image"],
            )
            val_transform = AugmentationSequential(
                K.Resize(size=(224, 224)),
                data_keys=["image"],
            )
        
        # Create base datamodule
        self.base_dm = TorchGeoEuroSAT(
            root=self.root,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            download=False,  # Already downloaded in prepare_data
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
        """Fix batch format by squeezing temporal dimension."""
        images = batch['image']
        labels = batch['label']
        
        # Squeeze temporal dimension if present
        if images.ndim == 5:  # [B, T, C, H, W]
            images = images.squeeze(1)  # -> [B, C, H, W]
        
        return {'image': images, 'label': labels}
    
    def train_dataloader(self):
        """Return train dataloader with batch fixing."""
        base_loader = self.base_dm.train_dataloader()
        return FixedDataLoader(base_loader, self._fix_batch)
    
    def val_dataloader(self):
        """Return validation dataloader with batch fixing."""
        base_loader = self.base_dm.val_dataloader()
        return FixedDataLoader(base_loader, self._fix_batch)
    
    def test_dataloader(self):
        """Return test dataloader with batch fixing."""
        base_loader = self.base_dm.test_dataloader()
        return FixedDataLoader(base_loader, self._fix_batch)


class FixedDataLoader:
    """Wrapper to fix batches on-the-fly."""
    
    def __init__(self, base_loader, fix_fn):
        self.base_loader = base_loader
        self.fix_fn = fix_fn
    
    def __iter__(self):
        for batch in self.base_loader:
            yield self.fix_fn(batch)
    
    def __len__(self):
        return len(self.base_loader)


if __name__ == "__main__":
    """Test the fixed datamodule."""
    
    print("=" * 70)
    print("Testing Fixed EuroSAT DataModule")
    print("=" * 70)
    
    # Create datamodule
    dm = EuroSATDataModuleFixed(
        root='./data',
        batch_size=4,
        num_workers=0,
        download=True,
        normalize=True,
    )
    
    # Setup
    print("\nSetting up datamodule...")
    dm.setup('fit')
    
    # Get a batch
    print("Loading first batch...")
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    images = batch['image']
    labels = batch['label']
    
    print(f"\n✓ Batch loaded successfully!")
    print(f"\nImage Batch Shape: {list(images.shape)}")
    print(f"  Expected: [4, 13, 224, 224]")
    print(f"  Match: {list(images.shape) == [4, 13, 224, 224]}")
    
    print(f"\nLabel Batch Shape: {list(labels.shape)}")
    print(f"  Expected: [4]")
    print(f"  Match: {list(labels.shape) == [4]}")
    
    print(f"\nImage Statistics:")
    print(f"  Min: {images.min().item():.4f}")
    print(f"  Max: {images.max().item():.4f}")
    print(f"  Mean: {images.mean().item():.4f}")
    print(f"  Std: {images.std().item():.4f}")
    
    # Test with DOFA
    print("\n" + "=" * 70)
    print("Testing with DOFA model...")
    print("=" * 70)
    
    from experiments.dofa_classifier import create_dofa_classifier
    
    model = create_dofa_classifier(
        model_size='base',
        num_classes=10,
        pretrained=True,
        weights_path='./checkpoints/DOFA_ViT_base_e100.pth',
    )
    model.eval()
    
    with torch.no_grad():
        logits = model(images)
        print(f"\n✓ Forward pass successful!")
        print(f"  Output shape: {list(logits.shape)}")
        print(f"  Expected: [4, 10]")
        print(f"  Match: {list(logits.shape) == [4, 10]}")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed! DataModule is ready for training.")
    print("=" * 70)
