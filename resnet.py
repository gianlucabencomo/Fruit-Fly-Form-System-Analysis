import typer
from tqdm.auto import tqdm

import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.optim.lr_scheduler import CosineAnnealingLR

from transforms import image_transforms
from utils import *
from normalization import *
from losses import AdaptiveGroupNormLoss


def train(dataloader, model, criterion, optimizer, scheduler, epochs, device):
    model.train()
    print("Starting Training...")
    epoch_losses = []
    for epoch in range(epochs):
        losses = []
        for X, y in tqdm(
            dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"
        ):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)
            losses.append(loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        epoch_losses.append(np.mean(losses))
        print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.3f}")
    return epoch_losses


def test(dataloader, model, criterion, device):
    model.eval()
    correct, losses = 0, []
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        losses.append(criterion(pred, y).detach().cpu().numpy())
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return correct / len(dataloader.dataset), np.mean(losses)


def main(
    seed: int = 0,
    root: str = "./data",
    dataset: str = "cifar100",
    batch_size: int = 256,
    alpha: float = 1e-3,
    lam: float = 1e-2, # set to zero for normal cross entropy
    epochs: int = 10,
):
    device = get_device()

    if dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            transform=image_transforms[dataset]["train"],
        )

        test_dataset = datasets.CIFAR100(
            root=root,
            train=False,
            transform=image_transforms[dataset]["test"],
        )
    elif dataset == "imagenet":
        train_dataset = datasets.ImageNet(
            root=root,
            split="train",
            transform=image_transforms[dataset]["train"],
        )

        test_dataset = datasets.ImageNet(
            root=root,
            split="val",
            transform=image_transforms[dataset]["test"],
        )
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, prefetch_factor=2)

    ms = ["bn", "agn"]
    for m in ms:
        set_random_seeds(seed)
        model = models.resnet18(weights=None).to(device)
        if m == "agn":
            replace_batch_norm_layers(model, AdaptiveGroupNorm)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=alpha, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        if m == "agn":
            criterion = AdaptiveGroupNormLoss(model=model, lam=lam)
        else:
            criterion = nn.CrossEntropyLoss()
        train_losses = train(train_loader, model, criterion, optimizer, scheduler, epochs, device)
        acc, test_loss = test(test_loader, model, criterion, device)

        print(f"\n{m} → Test Accuracy: {acc:.4f}, Test Loss: {test_loss:.4f}\n")
        save_results(model, train_losses, acc, test_loss, f"./{dataset}", m)

if __name__ == "__main__":
    typer.run(main)
