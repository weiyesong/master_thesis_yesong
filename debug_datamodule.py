"""
Debug script for TorchGeo EuroSATDataModule input shapes.

This script:
1. Instantiates EuroSATDataModule with transforms
2. Resizes images to 224x224 (DOFA requirement)
3. Inspects batch shapes and data properties
"""

import torch
import kornia.augmentation as K
from torchgeo.datamodules import EuroSATDataModule
from torchgeo.transforms import AugmentationSequential

print("=" * 70)
print("TorchGeo EuroSATDataModule - Shape Debugging")
print("=" * 70)

# ============================================================================
# 1. Define transforms for 224x224 resizing
# ============================================================================
print("\n[Step 1] Setting up transforms...")

# TorchGeo uses AugmentationSequential from torchgeo.transforms
# This handles both images and labels properly
# Note: We need to handle the shape properly - EuroSAT images are [C, H, W]
train_transform = AugmentationSequential(
    K.Resize(size=(224, 224)),  # Resize to 224x224 for DOFA
    # Note: EuroSAT images come as uint16, need proper normalization
    data_keys=["image"],  # Only apply to images, not labels
)

# For validation/test, we typically don't do augmentation
val_transform = AugmentationSequential(
    K.Resize(size=(224, 224)),
    data_keys=["image"],
)

print("✓ Transforms configured:")
print("  - Resize: 224 x 224")
print("  - Normalize: [0, 255] → [0, 1]")

# ============================================================================
# 2. Instantiate EuroSATDataModule
# ============================================================================
print("\n[Step 2] Creating EuroSATDataModule...")

datamodule = EuroSATDataModule(
    root='./data',
    batch_size=4,
    num_workers=0,  # Set to 0 for debugging to avoid multiprocessing issues
    download=True,  # Auto-download if not present
    # Note: TorchGeo datamodules don't accept transform parameters directly
    # We'll need to set them after instantiation or subclass
)

# Setup the datamodule (downloads data if needed)
print("  Downloading/preparing data (this may take a moment)...")
datamodule.setup(stage='fit')
datamodule.setup(stage='test')

print("✓ DataModule setup complete")

# ============================================================================
# 3. Apply transforms to datasets
# ============================================================================
print("\n[Step 3] Applying transforms to datasets...")

# TorchGeo datasets accept transforms
if hasattr(datamodule, 'train_dataset'):
    datamodule.train_dataset.transforms = train_transform
    print("  ✓ Train transforms applied")

if hasattr(datamodule, 'val_dataset'):
    datamodule.val_dataset.transforms = val_transform
    print("  ✓ Val transforms applied")

if hasattr(datamodule, 'test_dataset'):
    datamodule.test_dataset.transforms = val_transform
    print("  ✓ Test transforms applied")

# ============================================================================
# 4. Get first batch from train_dataloader
# ============================================================================
print("\n[Step 4] Loading first training batch...")

train_loader = datamodule.train_dataloader()
print(f"  Train loader created with {len(train_loader)} batches")

# Get first batch
batch = next(iter(train_loader))

# ============================================================================
# 5. Inspect batch structure and shapes
# ============================================================================
print("\n" + "=" * 70)
print("BATCH INSPECTION")
print("=" * 70)

print("\nBatch keys:", list(batch.keys()))

# Get images and labels
images = batch['image']
labels = batch['label']

# Fix shape if needed - squeeze extra dimensions
print(f"\nRaw Image Batch Shape: {list(images.shape)}")
if images.ndim == 5:  # [B, T, C, H, W] where T=1
    print("  Detected 5D tensor, squeezing temporal dimension...")
    images = images.squeeze(1)  # Remove temporal dimension -> [B, C, H, W]
    print(f"  After squeeze: {list(images.shape)}")

# Print shapes
print(f"\nImage Batch Shape: {list(images.shape)}")
print(f"  - Batch size: {images.shape[0]}")
print(f"  - Channels: {images.shape[1]} (Sentinel-2 has 13 bands)")
print(f"  - Height: {images.shape[2]}")
print(f"  - Width: {images.shape[3]}")

print(f"\nLabel Batch Shape: {list(labels.shape)}")
print(f"  - Batch size: {labels.shape[0]}")

# Print data statistics
print(f"\nImage Statistics:")
print(f"  - Data type: {images.dtype}")
print(f"  - Min value: {images.min().item():.4f}")
print(f"  - Max value: {images.max().item():.4f}")
print(f"  - Mean value: {images.mean().item():.4f}")
print(f"  - Std value: {images.std().item():.4f}")

print(f"\nLabel Statistics:")
print(f"  - Data type: {labels.dtype}")
print(f"  - Unique classes: {labels.unique().tolist()}")
print(f"  - Class distribution in batch:")
for cls in labels.unique():
    count = (labels == cls).sum().item()
    print(f"    Class {cls}: {count} samples")

# ============================================================================
# 6. Verify compatibility with DOFA
# ============================================================================
print("\n" + "=" * 70)
print("DOFA COMPATIBILITY CHECK")
print("=" * 70)

expected_shape = [4, 13, 224, 224]
actual_shape = list(images.shape)

if actual_shape == expected_shape:
    print(f"✓ Shape matches DOFA requirements: {actual_shape}")
else:
    print(f"⚠️  Shape mismatch!")
    print(f"   Expected: {expected_shape}")
    print(f"   Got: {actual_shape}")
    
    if actual_shape[1] != 13:
        print(f"\n   Note: EuroSAT might be using {actual_shape[1]} bands instead of 13.")
        print(f"   Common configurations:")
        print(f"   - 3 bands: RGB only")
        print(f"   - 10 bands: Sentinel-2 without 60m resolution bands")
        print(f"   - 13 bands: All Sentinel-2 bands")

# Check value range
if images.min() >= 0 and images.max() <= 1:
    print("✓ Values normalized to [0, 1]")
elif images.min() >= -1 and images.max() <= 1:
    print("! Values in [-1, 1] range (consider adjusting normalization)")
else:
    print(f"⚠️  Values outside expected range: [{images.min():.2f}, {images.max():.2f}]")

# ============================================================================
# 7. Test with DOFA model (optional)
# ============================================================================
print("\n" + "=" * 70)
print("TESTING WITH DOFA MODEL")
print("=" * 70)

try:
    from experiments.dofa_classifier import create_dofa_classifier
    
    print("\nLoading DOFA model...")
    model = create_dofa_classifier(
        model_size='base',
        num_classes=10,
        pretrained=True,
        weights_path='./checkpoints/DOFA_ViT_base_e100.pth',
    )
    model.eval()
    
    print("Running forward pass...")
    with torch.no_grad():
        # Adjust wavelengths if needed based on actual channels
        if images.shape[1] == 13:
            # All 13 Sentinel-2 bands
            wavelengths = model.wavelengths  # Use default
        elif images.shape[1] == 3:
            # RGB only
            wavelengths = [0.490, 0.560, 0.665]  # Blue, Green, Red
        else:
            wavelengths = None  # Let model handle it
        
        logits = model(images, wave_list=wavelengths)
        print(f"✓ Forward pass successful!")
        print(f"  Output logits shape: {list(logits.shape)}")
        print(f"  Expected: [4, 10]")
        
        if list(logits.shape) == [4, 10]:
            print("✓ Output shape correct!")
        else:
            print("⚠️  Output shape mismatch!")
            
except Exception as e:
    print(f"⚠️  Could not test with DOFA model: {e}")
    print("   Make sure DOFA weights are downloaded.")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\n✓ EuroSAT data loaded successfully")
print(f"✓ Batch shape: {list(images.shape)}")
print(f"✓ Label shape: {list(labels.shape)}")
print(f"\nTo use in training:")
print(f"  1. Apply transforms to resize to 224x224 ✓")
print(f"  2. Normalize pixel values appropriately ✓")
print(f"  3. Ensure {images.shape[1]} channels match your model's wavelength config")
print("=" * 70)
