import yaml
import pytorch_lightning as pl
from datamodules.eurosat_dm import EuroSATDataModule
from models.foundation import EOClassifier

def main(cfg):
    dm = EuroSATDataModule(
        root=cfg["data"]["root"],
        batch_size=cfg["training"]["batch_size"],
    )

    backbone = load_foundation_model(cfg["model"]["name"])
    model = EOClassifier(
        backbone=backbone,
        num_classes=cfg["model"]["num_classes"],
        lr=cfg["training"]["lr"],
    )

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="gpu",
    )

    trainer.fit(model, dm)

if __name__ == "__main__":
    with open("configs/eurosat_dofa.yaml") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
