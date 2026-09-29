from __future__ import annotations

"""Shared downstream-head/decoder MC Dropout mode controls."""

from typing import Any

import torch.nn as nn


_STOCHASTIC_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)
_BATCHNORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def designate_mc_dropout(module: nn.Module, placement: str) -> nn.Module:
    if not isinstance(module, _STOCHASTIC_TYPES):
        raise TypeError("Only standard PyTorch dropout modules can be designated for MC inference")
    if not 0.0 < float(module.p) < 1.0:
        raise ValueError("Designated MC Dropout must have 0 < p < 1")
    module._mc_dropout_designated = True
    module._mc_dropout_placement = str(placement)
    return module


def designated_mc_dropout_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name or "<root>", module)
        for name, module in model.named_modules()
        if bool(getattr(module, "_mc_dropout_designated", False))
    ]


def activate_downstream_mc_dropout(model: nn.Module) -> dict[str, Any]:
    """Enable only explicitly designated downstream dropout while all other layers stay in eval mode."""

    model.eval()
    designated = designated_mc_dropout_modules(model)
    if len(designated) != 1:
        raise RuntimeError(f"Expected exactly one designated MC Dropout module, found {len(designated)}")
    for _, module in designated:
        module.train()

    batchnorm_training = [
        name or "<root>" for name, module in model.named_modules()
        if isinstance(module, _BATCHNORM_TYPES) and module.training
    ]
    non_designated_training = [
        name or "<root>" for name, module in model.named_modules()
        if isinstance(module, _STOCHASTIC_TYPES)
        and module.training
        and not bool(getattr(module, "_mc_dropout_designated", False))
    ]
    backbone_stochastic_training = [
        name for name, module in model.named_modules()
        if name.startswith("backbone.") and isinstance(module, _STOCHASTIC_TYPES) and module.training
    ]
    if batchnorm_training:
        raise RuntimeError(f"BatchNorm unexpectedly active during MC inference: {batchnorm_training}")
    if non_designated_training or backbone_stochastic_training:
        raise RuntimeError(
            "Only the designated downstream dropout may be active during MC inference: "
            f"non_designated={non_designated_training}, backbone={backbone_stochastic_training}"
        )
    return {
        "model_training": model.training,
        "designated_modules": [
            {
                "path": name,
                "class": type(module).__name__,
                "p": float(module.p),
                "placement": str(getattr(module, "_mc_dropout_placement")),
                "training": module.training,
            }
            for name, module in designated
        ],
        "batchnorm_training_modules": batchnorm_training,
        "non_designated_stochastic_training_modules": non_designated_training,
        "backbone_stochastic_training_modules": backbone_stochastic_training,
    }
