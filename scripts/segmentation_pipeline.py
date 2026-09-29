from __future__ import annotations

"""Common semantic-segmentation pipeline for DOFA and Panopticon.

The only model-specific components are the dense-token adapters. Both adapters
emit a [B, D, Hpatch, Wpatch] feature map consumed by the same decoder class.
"""

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from scripts.experiment_manager import seed_dataloader_worker
from scripts.geobench_datasets import GeoBenchSegmentationDataset
from scripts.mc_dropout import designate_mc_dropout, designated_mc_dropout_modules


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tokens_to_feature_map(tokens: torch.Tensor, grid_size: Optional[Sequence[int]] = None) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Dense adapter expected [B,N,D] patch tokens, got {tuple(tokens.shape)}")
    if grid_size is None:
        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise ValueError(f"Patch-token count {tokens.shape[1]} is not a square grid")
        height = width = side
    else:
        height, width = (int(value) for value in grid_size)
        if height * width != tokens.shape[1]:
            raise ValueError(
                f"Configured patch grid {(height, width)} does not match {tokens.shape[1]} patch tokens"
            )
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width).contiguous()


class DOFADenseFeatureAdapter(nn.Module):
    """Expose DOFA's final normalized patch tokens without global pooling."""

    def __init__(self, backbone: nn.Module, wavelengths_um: Sequence[float]) -> None:
        super().__init__()
        self.backbone = backbone
        self.wavelengths_um = tuple(float(value) for value in wavelengths_um)
        if not self.wavelengths_um or any(not 0.3 <= value <= 2.5 for value in self.wavelengths_um):
            raise ValueError("DOFA dense adapter requires optical wavelengths in micrometers")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != len(self.wavelengths_um):
            raise ValueError(
                f"DOFA dense input must be [B,{len(self.wavelengths_um)},H,W], got {tuple(image.shape)}"
            )
        waves = image.new_tensor(self.wavelengths_um)
        patches, _ = self.backbone.patch_embed(image, waves)
        if patches.ndim != 3:
            raise ValueError(f"DOFA patch_embed returned unexpected shape {tuple(patches.shape)}")
        position = self.backbone.pos_embed[:, 1 : 1 + patches.shape[1], :]
        if position.shape[1] != patches.shape[1]:
            raise ValueError("DOFA positional embedding does not match the configured input patch grid")
        patches = patches + position
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        tokens = torch.cat((cls_token.expand(patches.shape[0], -1, -1), patches), dim=1)
        for block in self.backbone.blocks:
            tokens = block(tokens)
        normalization = getattr(self.backbone, "norm", None) or getattr(self.backbone, "fc_norm", None)
        if normalization is None:
            raise ValueError("DOFA backbone exposes neither norm nor fc_norm for dense tokens")
        patch_tokens = normalization(tokens)[:, 1:, :]
        grid_size = getattr(getattr(self.backbone, "patch_embed", None), "grid_size", None)
        return _tokens_to_feature_map(patch_tokens, grid_size=grid_size)


class PanopticonDenseFeatureAdapter(nn.Module):
    """Expose Panopticon's normalized optical patch tokens as a dense map."""

    def __init__(self, backbone: nn.Module, channel_ids_nm: Sequence[float]) -> None:
        super().__init__()
        self.backbone = backbone
        self.channel_ids_nm = tuple(float(value) for value in channel_ids_nm)
        if not self.channel_ids_nm or any(not 300.0 <= value <= 2500.0 for value in self.channel_ids_nm):
            raise ValueError("Panopticon dense adapter requires optical channel IDs in nanometers")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != len(self.channel_ids_nm):
            raise ValueError(
                f"Panopticon dense input must be [B,{len(self.channel_ids_nm)},H,W], got {tuple(image.shape)}"
            )
        channel_ids = image.new_tensor(self.channel_ids_nm).unsqueeze(0).expand(image.shape[0], -1).clone()
        model = getattr(self.backbone, "model", None)
        if model is None or not callable(getattr(model, "forward_features", None)):
            raise ValueError("Panopticon backbone does not expose model.forward_features")
        tokens = model.forward_features({"imgs": image, "chn_ids": channel_ids})
        if tokens.ndim != 3:
            raise ValueError(f"Panopticon forward_features returned unexpected shape {tuple(tokens.shape)}")
        prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
        patch_tokens = tokens[:, prefix_tokens:, :]
        grid_size = getattr(getattr(model, "patch_embed", None), "grid_size", None)
        return _tokens_to_feature_map(patch_tokens, grid_size=grid_size)


class UNetDecoderBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(32, output_channels)
        while output_channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.interpolate(features, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.block(features)


class CommonUNetDecoder(nn.Module):
    """Shared skip-free U-Net-style upsampling decoder for transformer features."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not decoder_channels or any(int(value) <= 0 for value in decoder_channels):
            raise ValueError("decoder_channels must contain positive channel counts")
        blocks = []
        current_channels = int(input_channels)
        for output_channels in decoder_channels:
            blocks.append(UNetDecoderBlock(current_channels, int(output_channels)))
            current_channels = int(output_channels)
        self.blocks = nn.Sequential(*blocks)
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("segmentation decoder dropout must satisfy 0 <= p < 1")
        self.mc_dropout: nn.Module = nn.Identity()
        if dropout > 0.0:
            self.mc_dropout = designate_mc_dropout(
                nn.Dropout2d(dropout), "common_decoder_final_feature_map_before_1x1_classifier"
            )
        self.classifier = nn.Conv2d(current_channels, int(num_classes), kernel_size=1)

    def forward(self, features: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
        logits = self.classifier(self.mc_dropout(self.blocks(features)))
        if tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=tuple(output_size), mode="bilinear", align_corners=False)
        return logits


class FoundationSegmentationModel(nn.Module):
    """Common segmentation wrapper exposing backbone/head for optimizer auditing."""

    def __init__(self, dense_adapter: nn.Module, decoder: CommonUNetDecoder, freeze_backbone: bool) -> None:
        super().__init__()
        self.backbone = dense_adapter
        self.head = decoder
        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "FoundationSegmentationModel":
        super().train(mode)
        if mode and self.freeze_backbone:
            self.backbone.eval()
        return self

    def extract_dense_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.extract_dense_features(image), output_size=image.shape[-2:])


def _resolve_weights_path(model_cfg: Dict[str, Any]) -> Optional[Path]:
    configured = model_cfg.get("weights_path")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path if path.is_file() else None
    size = str(model_cfg.get("size", "base"))
    name = "DOFA_ViT_base_e100.pth" if size == "base" else f"DOFA_ViT_{size}_e100.pth"
    candidate = PROJECT_ROOT / "DOFA" / "checkpoints" / name
    return candidate if candidate.is_file() else None


def build_segmentation_model(config: Dict[str, Any], model_cfg: Dict[str, Any]) -> FoundationSegmentationModel:
    data_cfg = config["data"]
    image_size = int(data_cfg.get("image_size", data_cfg.get("model_image_size", 224)))
    num_classes = int(data_cfg["num_classes"])
    wavelengths_nm = [float(value) for value in data_cfg["wavelengths_nm"]]
    model_name = str(model_cfg["name"]).lower()
    model_size = str(model_cfg.get("size", "base"))
    if model_size != "base":
        raise ValueError("The common segmentation protocol currently supports base-size backbones only")

    if model_name == "dofa":
        dofa_root = Path(model_cfg.get("dofa_root", PROJECT_ROOT / "DOFA"))
        if not dofa_root.is_absolute():
            dofa_root = PROJECT_ROOT / dofa_root
        if str(dofa_root) not in sys.path:
            sys.path.insert(0, str(dofa_root))
        from dofa_v1 import vit_base_patch16

        backbone = vit_base_patch16(
            img_size=image_size,
            num_classes=0,
            drop_rate=float(model_cfg.get("drop_rate", 0.0)),
            global_pool=False,
        )
        weights_path = _resolve_weights_path(model_cfg)
        if bool(model_cfg.get("pretrained", True)) and weights_path is None:
            raise FileNotFoundError("Pretrained DOFA base weights are required for the segmentation protocol")
        if weights_path is not None:
            try:
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                checkpoint = torch.load(weights_path, map_location="cpu")
            state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
            message = backbone.load_state_dict(state, strict=False)
            backbone._pretrained_load_audit = {
                "checkpoint": str(weights_path),
                "missing_keys": list(message.missing_keys),
                "unexpected_keys": list(message.unexpected_keys),
            }
        adapter: nn.Module = DOFADenseFeatureAdapter(backbone, [value / 1000.0 for value in wavelengths_nm])
        embedding_dim = 768
    elif model_name == "panopticon":
        from torchgeo.models import Panopticon_Weights, panopticon_vitb14

        weights = None
        if bool(model_cfg.get("pretrained", True)):
            weights = Panopticon_Weights[str(model_cfg.get("weights", "VIT_BASE14"))]
        backbone = panopticon_vitb14(weights=weights, img_size=image_size)
        if not bool(model_cfg.get("freeze_backbone", True)):
            named_parameters = dict(backbone.named_parameters())
            for name in model_cfg.get("expected_frozen_backbone_parameters", []):
                if name not in named_parameters:
                    raise ValueError(f"Unknown Panopticon expected frozen parameter: {name}")
                named_parameters[name].requires_grad = False
        backbone._pretrained_load_audit = {
            "weights": weights.name if weights is not None else None,
            "source": weights.url if weights is not None else None,
        }
        adapter = PanopticonDenseFeatureAdapter(backbone, wavelengths_nm)
        embedding_dim = int(backbone.model.num_features)
    else:
        raise ValueError("Segmentation model must be dofa or panopticon")

    decoder_cfg = model_cfg.get("decoder", {})
    decoder = CommonUNetDecoder(
        input_channels=embedding_dim,
        num_classes=num_classes,
        decoder_channels=tuple(decoder_cfg.get("channels", (256, 128, 64, 32))),
        dropout=float(decoder_cfg.get("dropout", 0.0)),
    )
    return FoundationSegmentationModel(adapter, decoder, freeze_backbone=bool(model_cfg.get("freeze_backbone", True)))


def audit_segmentation_model(model: FoundationSegmentationModel, model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    backbone_total = sum(parameter.numel() for parameter in model.backbone.parameters())
    backbone_trainable = sum(parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad)
    decoder_total = sum(parameter.numel() for parameter in model.head.parameters())
    decoder_trainable = sum(parameter.numel() for parameter in model.head.parameters() if parameter.requires_grad)
    frozen = str(model_cfg.get("adaptation_mode")) == "frozen"
    if frozen and backbone_trainable:
        raise RuntimeError("Frozen segmentation adaptation has trainable backbone parameters")
    if not frozen and backbone_trainable == 0:
        raise RuntimeError("Full-finetune segmentation adaptation has no trainable backbone parameters")
    structurally_frozen = {
        name: parameter.numel() for name, parameter in model.backbone.named_parameters() if not parameter.requires_grad
    }
    if not frozen:
        expected_frozen = {f"backbone.{name}" for name in model_cfg.get("expected_frozen_backbone_parameters", [])}
        if set(structurally_frozen) != expected_frozen:
            raise RuntimeError(
                "Full-finetune segmentation freeze invariant failed: "
                f"expected={sorted(expected_frozen)}, actual={sorted(structurally_frozen)}"
            )
    if decoder_total == 0 or decoder_trainable != decoder_total:
        raise RuntimeError("The common segmentation decoder must be fully trainable")
    model.train()
    if frozen and model.backbone.training:
        raise RuntimeError("Frozen segmentation backbone must remain in eval mode")
    configured_dropout = float(model_cfg.get("decoder", {}).get("dropout", 0.0))
    designated = designated_mc_dropout_modules(model)
    if configured_dropout == 0.0 and designated:
        raise RuntimeError("Deterministic segmentation decoder unexpectedly designates MC Dropout")
    if configured_dropout > 0.0 and (
        len(designated) != 1
        or not isinstance(designated[0][1], nn.Dropout2d)
        or not math.isclose(float(designated[0][1].p), configured_dropout, rel_tol=0.0, abs_tol=0.0)
    ):
        raise RuntimeError("MC segmentation decoder must expose exactly one configured Dropout2d")
    return {
        "adaptation_mode": model_cfg.get("adaptation_mode"),
        "backbone_total_parameters": backbone_total,
        "backbone_trainable_parameters": backbone_trainable,
        "backbone_requires_grad_all_false": backbone_trainable == 0,
        "backbone_structurally_frozen_parameters": structurally_frozen,
        "decoder_total_parameters": decoder_total,
        "decoder_trainable_parameters": decoder_trainable,
        "decoder_class": type(model.head).__name__,
        "mc_dropout": [
            {
                "path": name,
                "class": type(module).__name__,
                "p": float(module.p),
                "placement": str(getattr(module, "_mc_dropout_placement")),
            }
            for name, module in designated
        ],
        "dense_adapter_class": type(model.backbone).__name__,
        "pretrained_load": getattr(model.backbone.backbone, "_pretrained_load_audit", None),
    }


def make_segmentation_dataloaders(
    config: Dict[str, Any], generator: Optional[torch.Generator] = None
) -> Dict[str, DataLoader]:
    data_cfg = config["data"]
    name = str(data_cfg["name"]).lower()
    if name not in {"cloudsen12", "spacenet7"}:
        raise ValueError("Segmentation dataset must be cloudsen12 or spacenet7")
    root = Path(data_cfg.get("root", f"./datasets/{name}"))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    image_size = int(data_cfg.get("image_size", data_cfg.get("model_image_size", 224)))
    datasets = {
        split: GeoBenchSegmentationDataset(
            name=name,
            root=root,
            split=split,
            image_size=image_size,
            download=bool(data_cfg.get("download", False)) and split == "train",
        )
        for split in ("train", "val", "test")
    }
    identities = {split: set(dataset.sample_ids) for split, dataset in datasets.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = identities[left] & identities[right]
        if overlap:
            raise ValueError(f"{name} sample identity leakage between {left} and {right}: {len(overlap)}")
    loaders: Dict[str, DataLoader] = {}
    for offset, split in enumerate(("train", "val", "test")):
        split_generator = generator if split == "train" and generator is not None else torch.Generator().manual_seed(
            (int(config["seed"]) + offset) % (2**63)
        )
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=int(config["training"].get("batch_size", 8)),
            shuffle=split == "train",
            num_workers=int(data_cfg.get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_dataloader_worker,
            generator=split_generator,
        )
    return loaders


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int]) -> torch.Tensor:
    if logits.ndim != 4 or target.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError(f"Segmentation logits/target shapes disagree: {tuple(logits.shape)}, {tuple(target.shape)}")
    ignored = -100 if ignore_index is None else int(ignore_index)
    valid = target.ne(ignored)
    if not bool(valid.any()):
        raise ValueError("Segmentation batch contains no valid pixels")
    invalid = valid & ((target < 0) | (target >= logits.shape[1]))
    if bool(invalid.any()):
        raise ValueError(f"Segmentation target contains invalid class values: {torch.unique(target[invalid]).tolist()}")
    # CUDA's spatial NLL kernel is nondeterministic. Flattening the spatial
    # dimensions preserves the exact per-pixel objective while selecting the
    # deterministic two-dimensional cross-entropy implementation.
    flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    flat_target = target.reshape(-1).long()
    return F.cross_entropy(flat_logits, flat_target, ignore_index=ignored)


def _boundary_mask(target: torch.Tensor, valid: torch.Tensor, radius: int) -> torch.Tensor:
    boundary = torch.zeros_like(valid)
    horizontal = valid[:, :, 1:] & valid[:, :, :-1] & target[:, :, 1:].ne(target[:, :, :-1])
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    vertical = valid[:, 1:, :] & valid[:, :-1, :] & target[:, 1:, :].ne(target[:, :-1, :])
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    if radius > 0:
        kernel = 2 * int(radius) + 1
        boundary = F.max_pool2d(boundary[:, None].float(), kernel, stride=1, padding=radius)[:, 0].bool()
    return boundary & valid


class _CalibrationBins:
    def __init__(self, n_bins: int) -> None:
        self.n_bins = int(n_bins)
        self.count = torch.zeros(self.n_bins, dtype=torch.float64)
        self.confidence = torch.zeros(self.n_bins, dtype=torch.float64)
        self.outcome = torch.zeros(self.n_bins, dtype=torch.float64)

    def update(self, confidence: torch.Tensor, outcome: torch.Tensor) -> None:
        confidence = confidence.detach().double().cpu().reshape(-1)
        outcome = outcome.detach().double().cpu().reshape(-1)
        if confidence.numel() != outcome.numel():
            raise ValueError("Calibration confidence/outcome size mismatch")
        if not confidence.numel():
            return
        indices = (torch.ceil(confidence.clamp(0, 1) * self.n_bins).long() - 1).clamp(0, self.n_bins - 1)
        self.count += torch.bincount(indices, minlength=self.n_bins).double()
        self.confidence.scatter_add_(0, indices, confidence)
        self.outcome.scatter_add_(0, indices, outcome)

    def ece(self) -> float:
        total = float(self.count.sum().item())
        if total == 0:
            return math.nan
        nonempty = self.count > 0
        mean_confidence = self.confidence[nonempty] / self.count[nonempty]
        mean_outcome = self.outcome[nonempty] / self.count[nonempty]
        return float(((mean_confidence - mean_outcome).abs() * self.count[nonempty]).sum().item() / total)


class SegmentationMetricAccumulator:
    """Streaming pixel metrics with one consistent ignore mask."""

    def __init__(
        self,
        num_classes: int,
        class_names: Sequence[str],
        ignore_index: Optional[int],
        n_bins: int = 15,
        foreground_class_index: Optional[int] = None,
        boundary_radius: int = 1,
    ) -> None:
        if len(class_names) != num_classes:
            raise ValueError("class_names and num_classes disagree")
        self.num_classes = int(num_classes)
        self.class_names = tuple(class_names)
        self.ignore_index = ignore_index
        self.n_bins = int(n_bins)
        self.foreground_class_index = foreground_class_index
        self.boundary_radius = int(boundary_radius)
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.valid_pixels = 0
        self.ignored_pixels = 0
        self.nll_sum = 0.0
        self.brier_sum = 0.0
        self.top_label_bins = _CalibrationBins(n_bins)
        self.classwise_bins = [_CalibrationBins(n_bins) for _ in range(num_classes)]
        self.foreground_nll_sum = 0.0
        self.foreground_brier_sum = 0.0
        self.boundary_pixels = 0
        self.foreground_bins = _CalibrationBins(n_bins)
        self.boundary_top_label_bins = _CalibrationBins(n_bins)
        self.boundary_foreground_bins = _CalibrationBins(n_bins)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.ndim != 4 or target.shape != (logits.shape[0], *logits.shape[-2:]):
            raise ValueError("Segmentation metric logits/target shapes disagree")
        if logits.shape[1] != self.num_classes:
            raise ValueError("Segmentation metric class dimension disagrees with protocol")
        valid = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid &= target.ne(int(self.ignore_index))
        self.ignored_pixels += int((~valid).sum().item())
        if not bool(valid.any()):
            return
        invalid = valid & ((target < 0) | (target >= self.num_classes))
        if bool(invalid.any()):
            raise ValueError(f"Target contains invalid class values: {torch.unique(target[invalid]).tolist()}")
        probabilities = logits.softmax(dim=1)
        valid_target = target[valid].long()
        valid_probabilities = probabilities.permute(0, 2, 3, 1)[valid]
        predictions = valid_probabilities.argmax(dim=1)
        count = valid_target.numel()
        self.valid_pixels += int(count)
        encoded = valid_target * self.num_classes + predictions
        self.confusion += torch.bincount(
            encoded.cpu(), minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        selected = valid_probabilities.gather(1, valid_target[:, None]).squeeze(1).clamp_min(1.0e-12)
        self.nll_sum += float(-selected.log().sum().item())
        targets_one_hot = F.one_hot(valid_target, num_classes=self.num_classes).float()
        self.brier_sum += float(torch.square(valid_probabilities - targets_one_hot).sum(dim=1).sum().item())
        confidence, prediction = valid_probabilities.max(dim=1)
        self.top_label_bins.update(confidence, prediction.eq(valid_target))
        for class_index in range(self.num_classes):
            self.classwise_bins[class_index].update(
                valid_probabilities[:, class_index], valid_target.eq(class_index)
            )

        if self.foreground_class_index is None:
            return
        foreground = int(self.foreground_class_index)
        foreground_probability = valid_probabilities[:, foreground].clamp(1.0e-12, 1.0 - 1.0e-7)
        foreground_target = valid_target.eq(foreground).float()
        self.foreground_bins.update(foreground_probability, foreground_target)
        self.foreground_nll_sum += float(
            -(
                foreground_target * foreground_probability.log()
                + (1.0 - foreground_target) * (1.0 - foreground_probability).log()
            ).sum().item()
        )
        self.foreground_brier_sum += float(torch.square(foreground_probability - foreground_target).sum().item())

        boundary = _boundary_mask(target, valid, self.boundary_radius)
        if bool(boundary.any()):
            boundary_probabilities = probabilities.permute(0, 2, 3, 1)[boundary]
            boundary_target = target[boundary].long()
            boundary_confidence, boundary_prediction = boundary_probabilities.max(dim=1)
            self.boundary_pixels += int(boundary_target.numel())
            self.boundary_top_label_bins.update(boundary_confidence, boundary_prediction.eq(boundary_target))
            self.boundary_foreground_bins.update(
                boundary_probabilities[:, foreground], boundary_target.eq(foreground)
            )

    def compute(self) -> Dict[str, Any]:
        if self.valid_pixels == 0:
            raise ValueError("No valid pixels were accumulated")
        confusion = self.confusion.double()
        intersection = confusion.diag()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
        iou = torch.where(union > 0, intersection / union, torch.full_like(union, math.nan))
        metrics: Dict[str, Any] = {
            "miou": float(torch.nanmean(iou).item()),
            "per_class_iou": {
                name: (float(iou[index].item()) if not torch.isnan(iou[index]) else None)
                for index, name in enumerate(self.class_names)
            },
            "pixel_accuracy": float(intersection.sum().item() / self.valid_pixels),
            "nll": self.nll_sum / self.valid_pixels,
            "brier": self.brier_sum / self.valid_pixels,
            f"ece_{self.n_bins}": self.top_label_bins.ece(),
            "valid_pixels": self.valid_pixels,
            "ignored_pixels": self.ignored_pixels,
            "confusion_matrix": self.confusion.tolist(),
        }
        if self.foreground_class_index is not None:
            metrics.update(
                {
                    f"foreground_ece_{self.n_bins}": self.foreground_bins.ece(),
                    "classwise_calibration": {
                        name: {f"ece_{self.n_bins}": self.classwise_bins[index].ece()}
                        for index, name in enumerate(self.class_names)
                    },
                    "foreground_nll": self.foreground_nll_sum / self.valid_pixels,
                    "foreground_brier": self.foreground_brier_sum / self.valid_pixels,
                    "boundary_calibration": {
                        "radius_pixels": self.boundary_radius,
                        "pixel_count": self.boundary_pixels,
                        f"ece_{self.n_bins}": self.boundary_top_label_bins.ece(),
                        f"foreground_ece_{self.n_bins}": self.boundary_foreground_bins.ece(),
                    },
                }
            )
        return metrics


def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_names: Sequence[str],
    ignore_index: Optional[int] = None,
    n_bins: int = 15,
    foreground_class_index: Optional[int] = None,
    boundary_radius: int = 1,
) -> Dict[str, Any]:
    accumulator = SegmentationMetricAccumulator(
        num_classes=logits.shape[1],
        class_names=class_names,
        ignore_index=ignore_index,
        n_bins=n_bins,
        foreground_class_index=foreground_class_index,
        boundary_radius=boundary_radius,
    )
    accumulator.update(logits, target)
    return accumulator.compute()


def export_segmentation_predictions(
    output_dir: str | Path,
    *,
    sample_ids: Sequence[str],
    masks: torch.Tensor,
    logits: torch.Tensor,
    class_names: Sequence[str],
    ignore_index: Optional[int],
    model_name: str,
    dataset: str,
    adaptation_mode: str,
    split: str,
    checkpoint: str,
    representations: Optional[torch.Tensor] = None,
    per_image_results: Optional[Sequence[Dict[str, Any]]] = None,
    source: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Write a lossless, sample-aligned semantic-segmentation prediction bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = np.asarray([str(value) for value in sample_ids])
    target = masks.detach().long().cpu()
    raw_logits = logits.detach().float().cpu()
    if raw_logits.ndim != 4 or target.shape != (raw_logits.shape[0], *raw_logits.shape[-2:]):
        raise ValueError("Segmentation export logits/masks have inconsistent shapes")
    if len(ids) != raw_logits.shape[0] or len(set(ids.tolist())) != len(ids):
        raise ValueError("Segmentation export requires one unique sample_id per sample")
    if raw_logits.shape[1] != len(class_names) or not torch.isfinite(raw_logits).all():
        raise ValueError("Segmentation export has invalid class dimension or non-finite logits")
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid &= target.ne(int(ignore_index))
    probabilities = raw_logits.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correctness = prediction.eq(target) & valid
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    bundle_path = output_dir / "predictions.npz"
    np.savez_compressed(
        bundle_path,
        sample_id=ids,
        label=target.numpy(),
        logits=raw_logits.numpy(),
        probabilities=probabilities.numpy(),
        prediction=prediction.numpy(),
        correctness=correctness.numpy(),
        confidence=confidence.numpy(),
        predictive_entropy=entropy.numpy(),
        valid_mask=valid.numpy(),
    )
    artifacts: Dict[str, Path] = {"bundle": bundle_path}
    representation_shape = None
    if representations is not None:
        pooled = representations.detach().float().cpu()
        if pooled.ndim != 2 or pooled.shape[0] != len(ids):
            raise ValueError("Segmentation representations must have shape [N,D]")
        if not torch.isfinite(pooled).all():
            raise ValueError("Segmentation representations contain NaN or Inf")
        representation_path = output_dir / "representations.npz"
        np.savez_compressed(representation_path, sample_id=ids, representation=pooled.numpy())
        artifacts["representations"] = representation_path
        representation_shape = list(pooled.shape)

    if per_image_results is not None:
        rows = [dict(row) for row in per_image_results]
        if len(rows) != len(ids) or [str(row.get("sample_id")) for row in rows] != ids.tolist():
            raise ValueError("Per-image results must align exactly with exported sample IDs")
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        per_image_path = output_dir / "per_image_metrics.csv"
        with per_image_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        artifacts["per_image_metrics"] = per_image_path

    manifest_source = {
        "model_name": model_name,
        "dataset": dataset,
        "adaptation_mode": adaptation_mode,
        "split": split,
        "checkpoint": checkpoint,
    }
    manifest_source.update(source or {})
    manifest = {
        "schema_version": 2,
        "task": "semantic_segmentation",
        "sample_count": len(ids),
        "sample_ids_unique": len(set(ids.tolist())) == len(ids),
        "class_names": list(class_names),
        "ignore_index": ignore_index,
        "arrays": {
            "label": list(target.shape),
            "logits": list(raw_logits.shape),
            "probabilities": list(probabilities.shape),
            "prediction": list(prediction.shape),
            "correctness": list(correctness.shape),
            "confidence": list(confidence.shape),
            "predictive_entropy": list(entropy.shape),
            "valid_mask": list(valid.shape),
        },
        "representations": {
            "available": representations is not None,
            "shape": representation_shape,
            "aggregation": "global_mean_over_final_dense_backbone_feature_map",
        },
        "per_image_metrics": per_image_results is not None,
        "source": manifest_source,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    validation = validate_segmentation_prediction_export(output_dir)
    validation_path = output_dir / "validation_report.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["valid"]:
        raise RuntimeError(f"Segmentation prediction export validation failed: {validation['errors']}")
    artifacts.update({"manifest": manifest_path, "validation": validation_path})
    return artifacts


@torch.no_grad()
def collect_segmentation_predictions(
    model: FoundationSegmentationModel,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    class_names: Sequence[str],
    ignore_index: Optional[int],
    n_bins: int = 15,
    foreground_class_index: Optional[int] = None,
    boundary_radius: int = 1,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect one full deterministic split with pooled dense representations."""
    model.eval()
    sample_ids: List[str] = []
    masks: List[torch.Tensor] = []
    logits_parts: List[torch.Tensor] = []
    representations: List[torch.Tensor] = []
    per_image_results: List[Dict[str, Any]] = []
    aggregate = SegmentationMetricAccumulator(
        len(class_names), class_names, ignore_index, n_bins, foreground_class_index, boundary_radius
    )
    for batch_index, raw_batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        batch = move_segmentation_batch(raw_batch, device)
        dense_features = model.extract_dense_features(batch["image"])
        batch_logits = model.head(dense_features, output_size=batch["image"].shape[-2:])
        if not torch.isfinite(batch_logits).all() or not torch.isfinite(dense_features).all():
            raise FloatingPointError("Non-finite segmentation output or dense representation")
        aggregate.update(batch_logits, batch["mask"])
        batch_ids = [str(value) for value in raw_batch["sample_id"]]
        sample_ids.extend(batch_ids)
        masks.append(batch["mask"].detach().cpu())
        logits_parts.append(batch_logits.detach().float().cpu())
        representations.append(dense_features.mean(dim=(-2, -1)).detach().float().cpu())
        for index, sample_id in enumerate(batch_ids):
            item_metrics = segmentation_metrics(
                batch_logits[index : index + 1],
                batch["mask"][index : index + 1],
                class_names,
                ignore_index=ignore_index,
                n_bins=n_bins,
                foreground_class_index=foreground_class_index,
                boundary_radius=boundary_radius,
            )
            row: Dict[str, Any] = {
                "sample_id": sample_id,
                "miou": item_metrics["miou"],
                "pixel_accuracy": item_metrics["pixel_accuracy"],
                "nll": item_metrics["nll"],
                "brier": item_metrics["brier"],
                f"ece_{n_bins}": item_metrics[f"ece_{n_bins}"],
                "valid_pixels": item_metrics["valid_pixels"],
                "ignored_pixels": item_metrics["ignored_pixels"],
            }
            for class_name, value in item_metrics["per_class_iou"].items():
                row[f"iou_{class_name}"] = value
            if foreground_class_index is not None:
                row.update(
                    {
                        f"foreground_ece_{n_bins}": item_metrics[f"foreground_ece_{n_bins}"],
                        "foreground_nll": item_metrics["foreground_nll"],
                        "foreground_brier": item_metrics["foreground_brier"],
                        f"boundary_ece_{n_bins}": item_metrics["boundary_calibration"][f"ece_{n_bins}"],
                        f"boundary_foreground_ece_{n_bins}": item_metrics["boundary_calibration"][
                            f"foreground_ece_{n_bins}"
                        ],
                    }
                )
            per_image_results.append(row)
    if not sample_ids:
        raise ValueError("Segmentation prediction loader produced no batches")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Segmentation prediction collection found duplicate sample IDs")
    return {
        "sample_ids": sample_ids,
        "masks": torch.cat(masks, dim=0),
        "logits": torch.cat(logits_parts, dim=0),
        "representations": torch.cat(representations, dim=0),
        "per_image_results": per_image_results,
        "metrics": aggregate.compute(),
    }


def validate_segmentation_prediction_export(output_dir: str | Path) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(output_dir / "predictions.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    errors = []
    required = {
        "sample_id",
        "label",
        "logits",
        "probabilities",
        "prediction",
        "correctness",
        "confidence",
        "predictive_entropy",
        "valid_mask",
    }
    missing = sorted(required - set(arrays))
    if missing:
        return {"valid": False, "errors": [f"missing arrays: {missing}"]}
    logits = arrays["logits"]
    probabilities = arrays["probabilities"]
    labels = arrays["label"]
    prediction = arrays["prediction"]
    valid = arrays["valid_mask"].astype(bool)
    if logits.ndim != 4 or probabilities.shape != logits.shape:
        errors.append("logits/probabilities do not share [N,C,H,W] shape")
    if labels.shape != logits.shape[:1] + logits.shape[2:]:
        errors.append("label shape does not match logits")
    if len(arrays["sample_id"]) != logits.shape[0] or len(set(arrays["sample_id"].astype(str))) != logits.shape[0]:
        errors.append("sample_id count or uniqueness is invalid")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        errors.append("logits/probabilities contain NaN or Inf")
    torch_probabilities = torch.from_numpy(logits).softmax(dim=1).numpy()
    if not np.allclose(probabilities, torch_probabilities, atol=1.0e-6, rtol=0.0):
        errors.append("probabilities do not equal softmax(logits)")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6, rtol=0.0):
        errors.append("probabilities do not sum to one")
    expected_prediction = probabilities.argmax(axis=1)
    if not np.array_equal(prediction, expected_prediction):
        errors.append("prediction does not equal probability argmax")
    expected_correctness = (prediction == labels) & valid
    if not np.array_equal(arrays["correctness"].astype(bool), expected_correctness):
        errors.append("correctness does not respect label and ignore mask")
    expected_confidence = probabilities.max(axis=1)
    if not np.allclose(arrays["confidence"], expected_confidence, atol=1.0e-6, rtol=0.0):
        errors.append("confidence is inconsistent")
    safe = np.clip(probabilities.astype(np.float64), np.finfo(np.float64).tiny, 1.0)
    expected_entropy = -(probabilities.astype(np.float64) * np.log(safe)).sum(axis=1)
    if not np.allclose(arrays["predictive_entropy"], expected_entropy, atol=1.0e-5, rtol=0.0):
        errors.append("predictive entropy is inconsistent")
    ignore_index = manifest.get("ignore_index")
    expected_valid = np.ones_like(labels, dtype=bool) if ignore_index is None else labels != int(ignore_index)
    if not np.array_equal(valid, expected_valid):
        errors.append("valid mask is inconsistent with ignore_index")
    representation_path = output_dir / "representations.npz"
    representation_shape = None
    if manifest.get("representations", {}).get("available"):
        if not representation_path.is_file():
            errors.append("manifest declares representations but representations.npz is missing")
        else:
            with np.load(representation_path, allow_pickle=False) as archive:
                representation_ids = archive["sample_id"]
                representation = archive["representation"]
            representation_shape = list(representation.shape)
            if representation.ndim != 2 or representation.shape[0] != logits.shape[0]:
                errors.append("representations do not have [N,D] shape")
            if not np.array_equal(representation_ids.astype(str), arrays["sample_id"].astype(str)):
                errors.append("representation sample IDs are not aligned")
            if not np.isfinite(representation).all():
                errors.append("representations contain NaN or Inf")
            if representation_shape != manifest["representations"].get("shape"):
                errors.append("representation shape disagrees with manifest")
    per_image_path = output_dir / "per_image_metrics.csv"
    if manifest.get("per_image_metrics") and not per_image_path.is_file():
        errors.append("manifest declares per-image metrics but CSV is missing")
    return {
        "valid": not errors,
        "errors": errors,
        "sample_count": int(logits.shape[0]) if logits.ndim else 0,
        "valid_pixels": int(valid.sum()),
        "ignored_pixels": int((~valid).sum()),
        "finite": not any("NaN or Inf" in error for error in errors),
        "representation_shape": representation_shape,
    }


def move_segmentation_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {**batch, "image": batch["image"].to(device), "mask": batch["mask"].to(device)}


def train_segmentation_epoch(
    model: FoundationSegmentationModel,
    loader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ignore_index: Optional[int],
    limit_batches: Optional[int] = None,
    audit_gradients: bool = False,
) -> Dict[str, Any]:
    model.train()
    total_loss = 0.0
    batches = 0
    samples = 0
    gradient_audit: Optional[Dict[str, Any]] = None
    for batch_index, raw_batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        batch = move_segmentation_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["image"])
        loss = segmentation_loss(logits, batch["mask"], ignore_index)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite segmentation loss")
        loss.backward()
        if audit_gradients and gradient_audit is None:
            groups: Dict[str, Dict[str, Any]] = {}
            for group_name, module in (("backbone", model.backbone), ("head", model.head)):
                trainable = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
                gradients = [(name, parameter.grad) for name, parameter in trainable if parameter.grad is not None]
                finite = all(bool(torch.isfinite(gradient).all()) for _, gradient in gradients)
                maximum = max((float(gradient.detach().abs().max().item()) for _, gradient in gradients), default=0.0)
                groups[group_name] = {
                    "trainable_parameter_tensors": len(trainable),
                    "gradient_parameter_tensors": len(gradients),
                    "all_trainable_parameters_have_gradients": len(trainable) == len(gradients),
                    "all_gradients_finite": finite,
                    "maximum_absolute_gradient": maximum,
                    "any_nonzero_gradient": maximum > 0.0,
                }
            gradient_audit = {"batch_index": batch_index, "groups": groups}
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
        samples += int(batch["image"].shape[0])
    if not batches:
        raise ValueError("Segmentation training loader produced no batches")
    result: Dict[str, Any] = {"loss": total_loss / batches, "batches": batches, "samples": samples}
    if audit_gradients:
        result["gradient_audit"] = gradient_audit
    return result


@torch.no_grad()
def evaluate_segmentation(
    model: FoundationSegmentationModel,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    class_names: Sequence[str],
    ignore_index: Optional[int],
    n_bins: int = 15,
    foreground_class_index: Optional[int] = None,
    boundary_radius: int = 1,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    accumulator = SegmentationMetricAccumulator(
        len(class_names), class_names, ignore_index, n_bins, foreground_class_index, boundary_radius
    )
    batches = 0
    for batch_index, raw_batch in enumerate(loader):
        if limit_batches is not None and batch_index >= limit_batches:
            break
        batch = move_segmentation_batch(raw_batch, device)
        logits = model(batch["image"])
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite segmentation logits")
        accumulator.update(logits, batch["mask"])
        batches += 1
    if not batches:
        raise ValueError("Segmentation evaluation loader produced no batches")
    return accumulator.compute()
