from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(config: dict) -> nn.Module:
    """Build a classification model from config.

    Extension point: add DOFA, Panopticon, or other EO backbones by branching on
    `config["model"]["name"]` and returning the same kind of logits-producing module.
    """
    model_name = config["model"]["name"]
    num_classes = int(config["model"]["num_classes"])

    if model_name == "resnet18":
        return _build_resnet18(config, num_classes=num_classes)
    if model_name == "dofa":
        return _build_dofa(config, num_classes=num_classes)

    raise ValueError(f"Unsupported model name: {model_name}. Expected one of: resnet18, dofa")


def _build_resnet18(config: dict, num_classes: int) -> nn.Module:
    pretrained = bool(config["model"].get("pretrained", True))
    freeze_backbone = bool(config["model"].get("freeze_backbone", True))

    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


class DOFARGBClassifier(nn.Module):
    """Frozen DOFA RGB backbone with a trainable EuroSAT classification head."""

    # EuroSAT RGB images are rendered from Sentinel-2 bands in RGB channel order:
    # B04 = red   = 0.665 micrometers
    # B03 = green = 0.560 micrometers
    # B02 = blue  = 0.490 micrometers
    SENTINEL2_RGB_WAVELENGTHS = [0.665, 0.560, 0.490]

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int,
        freeze_backbone: bool = True,
        wavelengths: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.wavelengths = wavelengths or self.SENTINEL2_RGB_WAVELENGTHS
        self.freeze_backbone = freeze_backbone
        self.head = nn.Linear(embed_dim, num_classes)

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"DOFA expects input shape [batch, channels, height, width], got {images.shape}")
        if images.shape[1] != len(self.wavelengths):
            raise ValueError(
                "DOFA RGB configuration expects 3 input channels mapped to Sentinel-2 "
                f"B04/B03/B02 wavelengths {self.wavelengths}, got {images.shape[1]} channels."
            )

        features = self.backbone.forward_features(images, wave_list=self.wavelengths)
        return self.head(features)


def _build_dofa(config: dict, num_classes: int) -> nn.Module:
    model_config = config["model"]
    wavelengths_nm = [float(value) for value in config["data"].get("wavelengths_nm", [])]
    channels = config["data"].get("channels")
    if not wavelengths_nm or any(not 300.0 <= value <= 2500.0 for value in wavelengths_nm):
        raise ValueError("DOFA requires canonical optical data.wavelengths_nm in the range [300, 2500]")
    if channels is not None and len(channels) != len(wavelengths_nm):
        raise ValueError("data.channels and data.wavelengths_nm must have equal length")
    actual_wavelengths = [value / 1000.0 for value in wavelengths_nm]
    if [float(value) for value in model_config.get("actual_wavelengths", [])] != actual_wavelengths:
        raise ValueError("model.actual_wavelengths must equal data.wavelengths_nm converted from nm to micrometers")
    if model_config.get("actual_wavelength_units") != "micrometers":
        raise ValueError("DOFA model.actual_wavelength_units must be micrometers")
    model_size = str(model_config.get("model_size", "base")).lower()
    freeze_backbone = bool(model_config.get("freeze_backbone", True))
    pretrained = bool(model_config.get("pretrained", True))
    image_size = int(config["data"].get("image_size", 224))
    global_pool = bool(model_config.get("global_pool", False))

    dofa_root = _resolve_config_path(config, model_config.get("dofa_root", "../DOFA"))
    if not dofa_root.exists():
        raise RuntimeError(
            "DOFA source directory was not found. Set model.dofa_root in the config to the "
            "local DOFA checkout, or clone/download DOFA before running this experiment. "
            f"Configured path: {dofa_root}"
        )

    if str(dofa_root) not in sys.path:
        sys.path.insert(0, str(dofa_root))

    try:
        from dofa_v1 import vit_base_patch16, vit_large_patch16, vit_small_patch16
    except ImportError as exc:
        missing_name = exc.name or "required dependency"
        raise RuntimeError(
            "Could not import DOFA. Make sure model.dofa_root points to the DOFA source "
            "directory and install DOFA dependencies, especially timm. Example:\n"
            "  pip install timm\n"
            "  pip install -r ../DOFA/requirements.txt\n"
            f"Original missing module: {missing_name}"
        ) from exc

    factories = {
        "small": (vit_small_patch16, 384),
        "base": (vit_base_patch16, 768),
        "large": (vit_large_patch16, 1024),
    }
    if model_size not in factories:
        raise ValueError(f"Unsupported DOFA model_size: {model_size}. Expected one of: small, base, large")

    factory, embed_dim = factories[model_size]
    backbone = factory(img_size=image_size, num_classes=0, drop_rate=0.0, global_pool=global_pool)

    if pretrained:
        checkpoint_path = _resolve_config_path(
            config,
            model_config.get("checkpoint_path", dofa_root / "checkpoints" / "DOFA_ViT_base_e100.pth"),
        )
        if not checkpoint_path.is_file():
            raise RuntimeError(
                "Pretrained DOFA checkpoint was not found. Download DOFA_ViT_base_e100.pth "
                "from the DOFA/Hugging Face release or run DOFA/checkpoints/download_weights.py, "
                "then set model.checkpoint_path in the config.\n"
                f"Configured path: {checkpoint_path}"
            )
        _load_dofa_checkpoint(backbone, checkpoint_path)

    return DOFARGBClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        wavelengths=actual_wavelengths,
    )


def _resolve_config_path(config: dict, path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    config_dir = Path(config.get("_config_dir", "."))
    return (config_dir / path).resolve()


def _load_dofa_checkpoint(backbone: nn.Module, checkpoint_path: Path) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    load_message = backbone.load_state_dict(state_dict, strict=False)
    print(f"Loaded DOFA checkpoint from {checkpoint_path}")
    if load_message.missing_keys:
        print(f"DOFA missing keys ignored for linear probing: {load_message.missing_keys}")
    if load_message.unexpected_keys:
        print(f"DOFA unexpected keys ignored for linear probing: {load_message.unexpected_keys}")


def trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
