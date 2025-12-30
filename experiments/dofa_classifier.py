"""
DOFA Classifier Wrapper for EuroSAT Classification

This module provides a PyTorch wrapper for the DOFA foundation model
to be used with the EuroSAT dataset for land use classification.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add DOFA to path
DOFA_PATH = Path(__file__).parent.parent / "DOFA"
if str(DOFA_PATH) not in sys.path:
    sys.path.insert(0, str(DOFA_PATH))

from dofa_v1 import vit_base_patch16, vit_large_patch16, vit_small_patch16


class DOFAClassifier(nn.Module):
    """
    DOFA-based classifier for EuroSAT land use classification.
    
    This wrapper integrates the DOFA foundation model with a custom
    classification head for the 10-class EuroSAT dataset.
    
    Args:
        num_classes (int): Number of output classes. Default: 10 (EuroSAT)
        embed_dim (int): Embedding dimension of the backbone. Default: 768 (base model)
        model_size (str): Size of DOFA model ('small', 'base', 'large'). Default: 'base'
        weights_path (str, optional): Path to pretrained DOFA weights (.pth file)
        img_size (int): Input image size. Default: 224
        patch_size (int): Patch size for ViT. Default: 16
        drop_rate (float): Dropout rate for classification head. Default: 0.1
        wavelengths (list, optional): List of wavelengths (in micrometers) for input channels.
                                     If None, uses standard Sentinel-2 13-band wavelengths.
        freeze_backbone (bool): Whether to freeze the DOFA backbone. Default: False
    
    Sentinel-2 Bands (EuroSAT uses all 13 bands):
        Band 1  - Coastal aerosol    - 0.443 μm
        Band 2  - Blue                - 0.490 μm  
        Band 3  - Green               - 0.560 μm
        Band 4  - Red                 - 0.665 μm
        Band 5  - Red Edge 1          - 0.705 μm
        Band 6  - Red Edge 2          - 0.740 μm
        Band 7  - Red Edge 3          - 0.783 μm
        Band 8  - NIR                 - 0.842 μm
        Band 8a - Narrow NIR          - 0.865 μm
        Band 9  - Water vapor         - 0.945 μm
        Band 10 - SWIR - Cirrus       - 1.375 μm
        Band 11 - SWIR 1              - 1.610 μm
        Band 12 - SWIR 2              - 2.190 μm
    """
    
    # Standard Sentinel-2 wavelengths for all 13 bands (in micrometers)
    SENTINEL2_WAVELENGTHS = [
        0.443,  # Band 1 - Coastal aerosol
        0.490,  # Band 2 - Blue
        0.560,  # Band 3 - Green
        0.665,  # Band 4 - Red
        0.705,  # Band 5 - Red Edge 1
        0.740,  # Band 6 - Red Edge 2
        0.783,  # Band 7 - Red Edge 3
        0.842,  # Band 8 - NIR
        0.865,  # Band 8a - Narrow NIR
        0.945,  # Band 9 - Water vapor
        1.375,  # Band 10 - SWIR - Cirrus
        1.610,  # Band 11 - SWIR 1
        2.190,  # Band 12 - SWIR 2
    ]
    
    def __init__(
        self,
        num_classes: int = 10,
        embed_dim: int = 768,
        model_size: str = 'base',
        weights_path: str = None,
        img_size: int = 224,
        patch_size: int = 16,
        drop_rate: float = 0.1,
        wavelengths: list = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.model_size = model_size
        self.freeze_backbone = freeze_backbone
        
        # Set wavelengths (use Sentinel-2 standard if not provided)
        self.wavelengths = wavelengths if wavelengths is not None else self.SENTINEL2_WAVELENGTHS
        
        # Initialize DOFA backbone based on model size
        model_factory = {
            'small': vit_small_patch16,
            'base': vit_base_patch16,
            'large': vit_large_patch16,
        }
        
        if model_size not in model_factory:
            raise ValueError(f"model_size must be one of {list(model_factory.keys())}, got {model_size}")
        
        # Create backbone without classification head (num_classes=0)
        self.backbone = model_factory[model_size](
            img_size=img_size,
            num_classes=0,  # No head in backbone
            drop_rate=0.0,  # No dropout in backbone
        )
        
        # Update embed_dim based on actual model
        if model_size == 'small':
            self.embed_dim = 384
        elif model_size == 'base':
            self.embed_dim = 768
        elif model_size == 'large':
            self.embed_dim = 1024
        
        # Load pretrained weights if provided
        if weights_path is not None:
            self.load_pretrained_weights(weights_path)
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize the classification head with proper initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def load_pretrained_weights(self, weights_path: str, strict: bool = False):
        """
        Load pretrained DOFA weights.
        
        Args:
            weights_path (str): Path to the .pth checkpoint file
            strict (bool): Whether to strictly enforce weight matching. Default: False
                          (allows loading weights with different head dimensions)
        """
        print(f"Loading pretrained weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out classification head weights (we're using custom head)
        backbone_state_dict = {}
        for k, v in state_dict.items():
            # Skip head weights
            if k.startswith('head.') or k.startswith('fc_norm.'):
                continue
            backbone_state_dict[k] = v
        
        # Load weights into backbone
        msg = self.backbone.load_state_dict(backbone_state_dict, strict=strict)
        print(f"Loaded weights with message: {msg}")
    
    def forward_features(self, x: torch.Tensor, wave_list: list = None) -> torch.Tensor:
        """
        Extract features from the DOFA backbone.
        
        Args:
            x (torch.Tensor): Input image batch [B, C, H, W]
            wave_list (list, optional): List of wavelengths for each channel.
                                       If None, uses self.wavelengths.
        
        Returns:
            torch.Tensor: Feature embeddings [B, embed_dim]
        """
        if wave_list is None:
            wave_list = self.wavelengths
        
        # Validate wavelength list matches input channels
        if x.shape[1] != len(wave_list):
            raise ValueError(
                f"Number of channels ({x.shape[1]}) doesn't match "
                f"number of wavelengths ({len(wave_list)})"
            )
        
        # Extract features using DOFA backbone
        features = self.backbone.forward_features(x, wave_list)
        return features
    
    def forward(self, x: torch.Tensor, wave_list: list = None) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            x (torch.Tensor): Input image batch [B, C, H, W]
                             Expected shape: [batch_size, 13, 224, 224] for EuroSAT
            wave_list (list, optional): List of wavelengths for each channel (in micrometers).
                                       If None, uses standard Sentinel-2 wavelengths.
        
        Returns:
            torch.Tensor: Raw logits [B, num_classes]
        """
        # Extract features
        features = self.forward_features(x, wave_list)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_feature_extractor(self):
        """
        Returns a feature extractor version of the model (without classification head).
        Useful for transfer learning or feature extraction tasks.
        
        Returns:
            nn.Module: Feature extractor that outputs embeddings
        """
        class FeatureExtractor(nn.Module):
            def __init__(self, dofa_classifier):
                super().__init__()
                self.backbone = dofa_classifier.backbone
                self.wavelengths = dofa_classifier.wavelengths
            
            def forward(self, x, wave_list=None):
                if wave_list is None:
                    wave_list = self.wavelengths
                return self.backbone.forward_features(x, wave_list)
        
        return FeatureExtractor(self)


def create_dofa_classifier(
    model_size: str = 'base',
    num_classes: int = 10,
    pretrained: bool = True,
    weights_path: str = None,
    freeze_backbone: bool = False,
    **kwargs
) -> DOFAClassifier:
    """
    Factory function to create a DOFA classifier.
    
    Args:
        model_size (str): Model size ('small', 'base', 'large')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load pretrained weights
        weights_path (str, optional): Custom path to weights. If None and pretrained=True,
                                     will use default DOFA weights
        freeze_backbone (bool): Whether to freeze the backbone
        **kwargs: Additional arguments passed to DOFAClassifier
    
    Returns:
        DOFAClassifier: Initialized model
    
    Example:
        >>> # Create model with pretrained weights
        >>> model = create_dofa_classifier(
        ...     model_size='base',
        ...     num_classes=10,
        ...     pretrained=True,
        ...     weights_path='/path/to/DOFA_ViT_base_e100.pth'
        ... )
        >>> 
        >>> # Inference
        >>> x = torch.randn(4, 13, 224, 224)  # Batch of EuroSAT images
        >>> logits = model(x)
        >>> print(logits.shape)  # [4, 10]
    """
    # Determine default weights path if pretrained
    if pretrained and weights_path is None:
        # Check for default weights in checkpoints directory
        default_path = Path(__file__).parent.parent / "DOFA" / "checkpoints" / f"DOFA_ViT_{model_size}_e100.pth"
        if default_path.exists():
            weights_path = str(default_path)
        else:
            print(f"Warning: No pretrained weights found at {default_path}")
            print("Set weights_path explicitly or download from Hugging Face:")
            print("https://huggingface.co/earthflow/DOFA")
    
    model = DOFAClassifier(
        num_classes=num_classes,
        model_size=model_size,
        weights_path=weights_path if pretrained else None,
        freeze_backbone=freeze_backbone,
        **kwargs
    )
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating DOFA Classifier for EuroSAT...")
    
    # Create model (without pretrained weights for testing)
    model = create_dofa_classifier(
        model_size='base',
        num_classes=10,
        pretrained=False,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    x = torch.randn(2, 13, 224, 224)  # Batch of 2 EuroSAT images (13 bands)
    
    with torch.no_grad():
        # Test feature extraction
        features = model.forward_features(x)
        print(f"Features shape: {features.shape}")  # [2, 768]
        
        # Test classification
        logits = model(x)
        print(f"Logits shape: {logits.shape}")  # [2, 10]
    
    print("\nModel structure:")
    print(f"- Backbone: DOFA ViT-Base (embed_dim={model.embed_dim})")
    print(f"- Classifier: Dropout + Linear({model.embed_dim} -> {model.num_classes})")
    print(f"- Wavelengths: {len(model.wavelengths)} bands (Sentinel-2)")
