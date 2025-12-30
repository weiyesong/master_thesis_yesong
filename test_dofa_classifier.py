"""
Test script for DOFAClassifier with EuroSAT/Sentinel-2 configuration.

This script demonstrates:
1. Defining Sentinel-2 wavelengths
2. Loading pretrained DOFA weights
3. Instantiating the classifier
4. Running a sanity check with dummy data
"""

import torch
from experiments.dofa_classifier import DOFAClassifier, create_dofa_classifier

# ============================================================================
# 1. Define Sentinel-2 wavelengths (in micrometers)
# ============================================================================
# EuroSAT uses all 13 Sentinel-2 bands
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

print("=" * 70)
print("DOFA Classifier for EuroSAT - Sanity Check")
print("=" * 70)
print(f"\nSentinel-2 Bands: {len(SENTINEL2_WAVELENGTHS)} channels")
print(f"Wavelength range: {min(SENTINEL2_WAVELENGTHS)} - {max(SENTINEL2_WAVELENGTHS)} μm")

# ============================================================================
# 2. Define path to pretrained weights
# ============================================================================
WEIGHTS_PATH = "checkpoints/DOFA_ViT_base_e100.pth"
print(f"\nPretrained weights path: {WEIGHTS_PATH}")

# ============================================================================
# 3. Instantiate DOFAClassifier
# ============================================================================
print("\n" + "-" * 70)
print("Instantiating DOFA Classifier...")
print("-" * 70)

# Method 1: Using the factory function (recommended)
model = create_dofa_classifier(
    model_size='base',
    num_classes=10,  # EuroSAT has 10 land use classes
    pretrained=True,
    weights_path=WEIGHTS_PATH,
    freeze_backbone=False,  # Set to True for linear probing
)

# Method 2: Direct instantiation (alternative)
# model = DOFAClassifier(
#     num_classes=10,
#     embed_dim=768,  # Base model
#     model_size='base',
#     weights_path=WEIGHTS_PATH,
#     wavelengths=SENTINEL2_WAVELENGTHS,
#     freeze_backbone=False,
# )

print(f"\nModel created successfully!")
print(f"- Model size: base (ViT-Base)")
print(f"- Embedding dimension: {model.embed_dim}")
print(f"- Number of classes: {model.num_classes}")
print(f"- Wavelengths configured: {len(model.wavelengths)} bands")
print(f"- Backbone frozen: {model.freeze_backbone}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 4. Sanity Check: Forward pass with dummy data
# ============================================================================
print("\n" + "-" * 70)
print("Running Sanity Check...")
print("-" * 70)

# Create dummy EuroSAT batch
# Shape: (batch_size, channels, height, width)
batch_size = 2
channels = 13  # All Sentinel-2 bands
height = 224
width = 224

dummy_batch = torch.randn(batch_size, channels, height, width)
print(f"\nInput shape: {dummy_batch.shape}")
print(f"  - Batch size: {batch_size}")
print(f"  - Channels: {channels} (Sentinel-2 bands)")
print(f"  - Image size: {height}×{width}")

# Set model to evaluation mode
model.eval()

# Forward pass (no gradient computation needed for testing)
with torch.no_grad():
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    dummy_batch = dummy_batch.to(device)
    
    print(f"\nDevice: {device}")
    
    # Test feature extraction
    print("\nTesting feature extraction...")
    features = model.forward_features(dummy_batch)
    print(f"Feature shape: {features.shape}")
    print(f"  - Expected: [{batch_size}, {model.embed_dim}]")
    assert features.shape == (batch_size, model.embed_dim), "Feature shape mismatch!"
    print("✓ Feature extraction passed!")
    
    # Test full forward pass (classification)
    print("\nTesting classification...")
    logits = model(dummy_batch)
    print(f"Output shape: {logits.shape}")
    print(f"  - Expected: [{batch_size}, {model.num_classes}]")
    assert logits.shape == (batch_size, model.num_classes), "Output shape mismatch!"
    print("✓ Classification passed!")
    
    # Get predictions
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(probs, dim=1)
    
    print("\nPrediction details:")
    for i in range(batch_size):
        pred_class = predictions[i].item()
        confidence = probs[i, pred_class].item()
        print(f"  Sample {i+1}: Class {pred_class} (confidence: {confidence:.4f})")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✓ All sanity checks passed!")
print("=" * 70)
print("\nThe DOFA classifier is ready for EuroSAT training!")
print("\nExpected workflow:")
print("  1. Load EuroSAT dataset with 13-band Sentinel-2 images")
print("  2. Images will be automatically preprocessed to [B, 13, 224, 224]")
print("  3. Model will output logits of shape [B, 10]")
print("  4. Apply softmax for probabilities or CrossEntropyLoss for training")
print("\nNext steps:")
print("  - Run: python main.py --data_root ./data --batch_size 32")
print("=" * 70)
