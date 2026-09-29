from __future__ import annotations

"""Pinned GEO-Bench-2 adapters shared by classification and segmentation tasks."""

from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


GEOBENCH2_COMMIT = "fd9d0b664e6fb0faba54636bdff4906634debd4b"

SENTINEL2_WAVELENGTHS_NM = {
    "B01": 443.0,
    "B02": 490.0,
    "B03": 560.0,
    "B04": 665.0,
    "B05": 705.0,
    "B06": 740.0,
    "B07": 783.0,
    "B08": 842.0,
    "B8A": 865.0,
    "B09": 945.0,
    "B10": 1375.0,
    "B11": 1610.0,
    "B12": 2190.0,
}

TREESATAI_BANDS = (
    "B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"
)
TREESATAI_WAVELENGTHS_NM = tuple(SENTINEL2_WAVELENGTHS_NM[band] for band in TREESATAI_BANDS)

CLOUDSEN12_BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
CLOUDSEN12_WAVELENGTHS_NM = tuple(SENTINEL2_WAVELENGTHS_NM[band] for band in CLOUDSEN12_BANDS)

SPACENET7_BANDS = ("red", "green", "blue")
SPACENET7_WAVELENGTHS_NM = (665.0, 560.0, 490.0)


def _resize_image(image: torch.Tensor, image_size: int) -> torch.Tensor:
    if image.shape[-2:] == (image_size, image_size):
        return image
    leading = image.shape[:-3]
    flat = image.reshape(-1, *image.shape[-3:])
    resized = F.interpolate(flat, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.reshape(*leading, *resized.shape[-3:])


def _resize_mask(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    if mask.shape[-2:] == (image_size, image_size):
        return mask.long()
    resized = F.interpolate(mask[None, None].float(), size=(image_size, image_size), mode="nearest")
    return resized[0, 0].long()


class GeoBenchTreeSatAITemporalDataset(Dataset):
    """TreeSatAI adapter exposing a common [time, channel, height, width] contract.

    The pinned official artifact contains a single static Sentinel-2 image per sample.
    It records paths to external HDF5 time series but does not package those HDF5 files.
    Therefore the official artifact is represented honestly as T=1; no raw-release data
    are substituted. The model wrappers still execute the exact shared-encoder/mean-feature
    temporal path, so a future official artifact with packaged timestamps can extend T.
    """

    def __init__(self, root: Path, split: str, image_size: int = 224, download: bool = False) -> None:
        from geobench_v2.datasets import GeoBenchTreeSatAI

        if split not in {"train", "val", "validation", "test"}:
            raise ValueError(f"Unsupported TreeSatAI split: {split}")
        self.dataset = GeoBenchTreeSatAI(
            root=root,
            split=split,
            band_order={"s2": TREESATAI_BANDS},
            include_ts=False,
            download=download,
        )
        self.split = "validation" if split == "val" else split
        self.image_size = int(image_size)
        self.class_names = tuple(GeoBenchTreeSatAI.classes)
        self.multilabel = bool(GeoBenchTreeSatAI.multilabel)
        metadata = self.dataset.data_df
        self.sample_ids = metadata["tortilla:id"].astype(str).tolist()
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError(f"Duplicate TreeSatAI sample IDs detected in {self.split}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample["image_s2"].float()
        if image.ndim != 3 or image.shape[0] != len(TREESATAI_BANDS):
            raise ValueError(f"Unexpected TreeSatAI Sentinel-2 shape: {tuple(image.shape)}")
        image = _resize_image(image, self.image_size).unsqueeze(0)
        label = sample["label"].float()
        if label.shape != (len(self.class_names),):
            raise ValueError(f"Unexpected TreeSatAI multilabel shape: {tuple(label.shape)}")
        if not torch.isfinite(image).all() or not torch.isfinite(label).all():
            raise FloatingPointError(f"Non-finite TreeSatAI sample {self.sample_ids[index]}")
        row = self.dataset.data_df.iloc[index]
        return {
            "image": image,
            "label": label,
            "sample_id": self.sample_ids[index],
            "split": self.split,
            "temporal_mask": torch.ones(1, dtype=torch.bool),
            "timestamp_year": torch.tensor([int(float(row["stac:time_start"]))], dtype=torch.int16),
            "temporal_source": "official_static_s2_single_timestamp",
        }


class GeoBenchSegmentationDataset(Dataset):
    """Common CloudSEN12/SpaceNet7 segmentation adapter."""

    def __init__(
        self,
        name: Literal["cloudsen12", "spacenet7"],
        root: Path,
        split: str,
        image_size: int = 224,
        download: bool = False,
    ) -> None:
        from geobench_v2.datasets import GeoBenchCloudSen12, GeoBenchSpaceNet7

        self.name = name
        self.split = "validation" if split == "val" else split
        self.image_size = int(image_size)
        if name == "cloudsen12":
            self.bands: Sequence[str] = CLOUDSEN12_BANDS
            self.wavelengths_nm = CLOUDSEN12_WAVELENGTHS_NM
            self.class_names = tuple(GeoBenchCloudSen12.classes)
            self.ignore_index = None
            self.dataset = GeoBenchCloudSen12(
                root=root, split=split, band_order=self.bands, download=download
            )
        elif name == "spacenet7":
            self.bands = SPACENET7_BANDS
            self.wavelengths_nm = SPACENET7_WAVELENGTHS_NM
            self.class_names = ("background", "building")
            self.ignore_index = 255
            self.dataset = GeoBenchSpaceNet7(
                root=root, split=split, band_order=list(self.bands), download=download
            )
        else:
            raise ValueError(f"Unsupported GEO-Bench-2 segmentation dataset: {name}")
        metadata = self.dataset.data_df
        self.sample_ids = metadata["tortilla:id"].astype(str).tolist()
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError(f"Duplicate {name} sample IDs detected in {self.split}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = sample["image"].float()
        mask = sample["mask"].long()
        if self.name == "cloudsen12":
            # GeoBenchV2 0.9 computes this dictionary but accidentally returns the
            # unnormalized one. Apply the instantiated official normalizer explicitly.
            image = self.dataset.data_normalizer({"image": image})["image"]
            valid_values = {0, 1, 2, 3}
        else:
            # Official v0.9 emits 1=no-building and 2=building after adding one to
            # the generated binary mask. Convert to the thesis binary contract and
            # preserve a possible official value 0 as ignore (none observed so far).
            converted = torch.full_like(mask, self.ignore_index)
            converted[mask == 1] = 0
            converted[mask == 2] = 1
            mask = converted
            valid_values = {0, 1, self.ignore_index}
        observed = set(int(value) for value in torch.unique(mask).tolist())
        if not observed <= valid_values:
            raise ValueError(f"Invalid {self.name} mask values {sorted(observed)} for {self.sample_ids[index]}")
        if image.ndim != 3 or image.shape[0] != len(self.bands):
            raise ValueError(f"Unexpected {self.name} image shape: {tuple(image.shape)}")
        if image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"Unsynchronized {self.name} image/mask for {self.sample_ids[index]}")
        image = _resize_image(image, self.image_size)
        mask = _resize_mask(mask, self.image_size)
        if not torch.isfinite(image).all():
            raise FloatingPointError(f"Non-finite {self.name} image {self.sample_ids[index]}")
        row = self.dataset.data_df.iloc[index]
        output: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "sample_id": self.sample_ids[index],
            "split": self.split,
        }
        for key in ("roi_id", "s2_id", "aoi", "patch_id", "stac:time_start"):
            if key in row:
                output[key] = str(row[key])
        return output
