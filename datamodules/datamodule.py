import torch
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT
from torchgeo.transforms import AugmentationSequential
import pytorch_lightning as pl

class EuroSATDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.transforms = AugmentationSequential(
            torch.nn.Identity(),  # 可后续加 augmentation
            data_keys=["image"],
        )

    def setup(self, stage=None):
        self.train_dataset = EuroSAT(
            root=self.root,
            split="train",
            transforms=self.transforms,
            download=True,
        )
        self.val_dataset = EuroSAT(
            root=self.root,
            split="val",
            transforms=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
