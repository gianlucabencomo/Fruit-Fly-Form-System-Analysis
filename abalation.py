import typer
from tqdm.auto import tqdm

import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from transforms import image_transforms
from utils import *

import matplotlib.pyplot as plt
from models import CNN

import torch
import matplotlib.pyplot as plt

from normalization import *
from losses import AdaptiveGroupNormLoss

WIDTHS = {
    1: [32],
    2: [32, 64],
    3: [32, 64, 128],
    4: [32, 64, 128, 256],
}


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
    seeds: int = 3,
    root: str = "./data",
    dataset: str = "cifar100",
    n_layers: int = 4,
    batch_size: int = 64,
    alpha: float = 1e-3,
    epochs: int = 20,
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

    lams = [0.0, 1e-1, 1e-2, 1e-3, 1e-4]
    norms = [
        [
            partial(AdaptiveGroupNorm, 2),
            partial(AdaptiveGroupNorm, 4),
            partial(AdaptiveGroupNorm, 8),
            partial(AdaptiveGroupNorm, 16),
        ],
        [
            partial(AdaptiveGroupNorm, 4),
            partial(AdaptiveGroupNorm, 8),
            partial(AdaptiveGroupNorm, 16),
            partial(AdaptiveGroupNorm, 64),
        ],
        [
            partial(AdaptiveGroupNorm, 8),
            partial(AdaptiveGroupNorm, 16),
            partial(AdaptiveGroupNorm, 32),
            partial(AdaptiveGroupNorm, 64),
        ],
        [
            partial(AdaptiveGroupNorm, 16),
            partial(AdaptiveGroupNorm, 32),
            partial(AdaptiveGroupNorm, 64),
            partial(AdaptiveGroupNorm, 128),
        ]
    ]

    from itertools import product

    all_train_losses = {i: [] for i in list(product(range(len(norms)), lams))}
    all_accs = {i: [] for i in list(product(range(len(norms)), lams))}
    all_test_losses = {i: [] for i in list(product(range(len(norms)), lams))}

    print("\n=== Running Experiments Across Seeds ===\n")

    seeds = [i for i in range(0, seeds)]
    for seed in seeds:
        print(f"\n--- Running for Seed {seed} ---\n")

        for i, lam in list(product(range(len(norms)), lams)):
            set_random_seeds(seed)
            print(f"Running {i} with seed {seed}...")
            norm_layer = norms[i]
            norm_layer = norm_layer[:n_layers]
            model = CNN(
                num_layers=n_layers, width=WIDTHS[n_layers], norm_layers=norm_layer
            ).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=alpha)
            criterion = AdaptiveGroupNormLoss(model=model, lam=lam)

            train_loss = train(
                train_loader, model, criterion, optimizer, epochs, device
            )
            acc, test_loss = test(test_loader, model, criterion, device)

            all_train_losses[(i, lam)].append(train_loss)
            all_accs[(i, lam)].append(acc)
            all_test_losses[(i, lam)].append(test_loss)

            print(
                f"\n{(i, lam)} → Seed {seed}: Test Accuracy: {acc:.4f}, Test Loss: {test_loss:.4f}\n"
            )

    print("\n=== Final Results Across Seeds ===")
    for i in list(product(range(len(norms)), lams)):
        avg_acc = np.mean(all_accs[i])
        avg_loss = np.mean(all_test_losses[i])
        print(
            f"{i}: Avg Test Accuracy = {avg_acc:.4f}, Avg Test Loss = {avg_loss:.4f}"
        )


if __name__ == "__main__":
    typer.run(main)
