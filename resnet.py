import typer
from tqdm.auto import tqdm

import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets

from transforms import image_transforms
from utils import *
from normalization import *

def train(dataloader, model, criterion, optimizer, epochs, device):
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
    batch_size: int = 64,
    alpha: float = 1e-3,
    epochs: int = 10,
):
    device = get_device()

    train_dataset = datasets.CIFAR100(
        root=root,
        train=True,
        transform=image_transforms[dataset]["train"],
        download=False,
    )

    test_dataset = datasets.CIFAR100(
        root=root,
        train=False,
        transform=image_transforms[dataset]["test"],
        download=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for i in range(2):
        set_random_seeds(seed)
        model = models.resnet18(weights=None).to(device)
        if i == 0:
            replace_batch_norm_layers(model, CorrelatedGroupNorm)
        model = model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=alpha)
        criterion = nn.CrossEntropyLoss()
        _ = train(
            train_loader, model, criterion, optimizer, epochs, device
        )
        acc, test_loss = test(test_loader, model, criterion, device)

        print(
            f"\n{i} → Test Accuracy: {acc:.4f}, Test Loss: {test_loss:.4f}\n"
        )


if __name__ == "__main__":
    typer.run(main)
