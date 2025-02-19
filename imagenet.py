import os
import torch
import numpy as np
import typer

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

import lightning as L  
from transforms import image_transforms
from utils import replace_batch_norm_layers, set_random_seeds, get_optimizer_and_scheduler
from losses import AdaptiveGroupNormLoss


class LitResNet18(L.LightningModule):
    def __init__(
        self,
        norm: str,
        lam: float,
        compression_factor: int,
        dataset: str,
        optim_name: str,
        epochs: int,
        use_groups: bool,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(weights=None)
        if norm != "bn":
            # If use_groups is True, we assume a different replacement scheme.
            cf = None if use_groups else compression_factor
            replace_batch_norm_layers(self.model, norm, compression_factor=cf)

        if norm == "agn":
            self.criterion = AdaptiveGroupNormLoss(model=self.model, lam=lam)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer_and_scheduler(
            self.model, self.hparams.optim_name, self.hparams.epochs, dataset=self.hparams.dataset
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def main(
    seeds: int = 1,
    norm: str = "agn",
    root: str = "./data",
    batch_size: int = 256,
    optim_name: str = "sgd",
    lam: float = 1e-2,  # set to zero for normal cross entropy
    compression_factor: int = 16,
    epochs: int = 100,
    use_groups: bool = False,  # use 32 groups at every layer if true
    gpus: int = 1  # adjust this for your multi-GPU run (e.g., gpus=2 for 2 GPUs)
):
    assert norm in ["agn", "gn", "ln", "in", "bn"], "Norm method not recognized."
    train_dataset = datasets.ImageNet(
        root=root,
        split="train",
        transform=image_transforms["imagenet"]["train"],
    )
    val_dataset = datasets.ImageNet(
        root=root,
        split="val",
        transform=image_transforms["imagenet"]["test"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,  # Larger batch size for validation
        shuffle=False,
        num_workers=4,
    )

    # Loop over seeds if needed.
    for seed in range(seeds):
        set_random_seeds(seed)
        model = LitResNet18(
            norm=norm,
            lam=lam,
            compression_factor=compression_factor,
            dataset="imagenet",
            optim_name=optim_name,
            epochs=epochs,
            use_groups=use_groups,
        )
        trainer = L.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=gpus,  # set to number of GPUs you want to use
            log_every_n_steps=50,
        )
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    typer.run(main)