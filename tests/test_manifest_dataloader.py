import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from scripts.run_experiments import EuroSATClassificationTransform, ManifestEuroSATDataset


class ManifestDataLoaderTests(unittest.TestCase):
    def test_dataset_reads_manifest_record_and_preserves_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "images" / "AnnualCrop_1.tif"
            image_path.parent.mkdir(parents=True)
            image = np.ones((13, 64, 64), dtype=np.uint16)
            with rasterio.open(
                image_path,
                "w",
                driver="GTiff",
                width=64,
                height=64,
                count=13,
                dtype=image.dtype,
                crs="EPSG:32631",
                transform=from_origin(0, 640, 10, 10),
            ) as dataset:
                dataset.write(image)
            record = {
                "sample_id": "AnnualCrop_1",
                "dataset_index": "0",
                "file_path": "images/AnnualCrop_1.tif",
                "class_label": "AnnualCrop",
                "class_index": "0",
                "split": "calibration",
            }
            dataset = ManifestEuroSATDataset(
                root,
                [record],
                EuroSATClassificationTransform(image_size=32, normalize=False, input_bands="rgb"),
            )
            sample = dataset[0]
            self.assertEqual(sample["sample_id"], "AnnualCrop_1")
            self.assertEqual(sample["file_path"], record["file_path"])
            self.assertEqual(sample["split"], "calibration")
            self.assertEqual(tuple(sample["image"].shape), (3, 32, 32))
            self.assertEqual(int(sample["label"]), 0)

    def test_explicit_channelwise_normalization_is_used(self):
        transform = EuroSATClassificationTransform(
            image_size=2,
            normalize=True,
            input_bands="rgb",
            normalization_mean=[4.0, 3.0, 2.0],
            normalization_std=[2.0, 2.0, 2.0],
        )
        image = torch.stack(
            [
                torch.ones(2, 2),
                torch.full((2, 2), 2.0),
                torch.full((2, 2), 3.0),
                torch.full((2, 2), 4.0),
            ]
        )
        transformed = transform({"image": image, "label": 0})["image"]
        self.assertTrue(torch.equal(transformed, torch.zeros_like(transformed)))


if __name__ == "__main__":
    unittest.main()
