import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class EOClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.classifier = nn.Linear(
            backbone.out_features, num_classes
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.val_acc(logits.softmax(dim=-1), y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

        return {"logits": logits, "labels": y}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
