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
from models import CNN
from normalization import *
from losses import AdaptiveGroupNormLoss

WIDTHS = {
    1: [32],
    2: [32, 64],
    3: [32, 64, 128],
    4: [32, 64, 128, 256],
}


def train(train_loader, test_loader, model, criterion, optimizer, epochs, device):
    model.train()
    print("Starting Training...")
    train_res, test_res = [], []
    for epoch in range(epochs):
        for X, y in tqdm(
            train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}"
        ):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_res.append(test(train_loader, model, nn.CrossEntropyLoss(), device))
        test_res.append(test(test_loader, model, nn.CrossEntropyLoss(), device))
        print(f"Epoch {epoch+1}: Loss = {train_res[-1][1]:.3f}")
    return train_res, test_res


def test(dataloader, model, criterion, device):
    model.eval()
    with torch.no_grad():
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
    n_layers: int = 4,
    batch_size: int = 64,
    alpha: float = 1e-3,
    weight_decay: float = 0.01,
    dropout: float = 0.0,
    lam: float = 1e-3,  # set to zero for normal cross entropy
    epochs: int = 40,
):
    device = get_device()

    train_dataset = datasets.CIFAR100(
        root=root,
        train=True,
        transform=image_transforms["cifar100"]["train"],
        download=False,
    )

    test_dataset = datasets.CIFAR100(
        root=root,
        train=False,
        transform=image_transforms["cifar100"]["test"],
        download=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),  # Only use pin_memory if GPU is available
        num_workers=8 if torch.cuda.is_available() else 2,
        prefetch_factor=2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),  # Only use pin_memory if GPU is available
        num_workers=8 if torch.cuda.is_available() else 2,
        prefetch_factor=2,
    )

    norms = ["agn", "identity", "bn", "ln", "gn", "in"]

    train_results = {norm: [] for norm in norms}
    test_results = {norm: [] for norm in norms}

    seeds = [i for i in range(0, seeds)]
    for seed in seeds:
        for norm in norms:
            set_random_seeds(seed)
            print(f"Running {norm} with seed {seed}...")
            norm_layer = get_norm_layer(norm)
            if norm in ["agn", "gn", "agn2"]:
                norm_layer = norm_layer[:n_layers]
            model = CNN(
                num_layers=n_layers, width=WIDTHS[n_layers], norm_layers=norm_layer, dropout=dropout
            ).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=alpha, weight_decay=weight_decay)
            if norm == "agn":
                criterion = AdaptiveGroupNormLoss(model=model, lam=lam)
            else:
                criterion = nn.CrossEntropyLoss()

            train_res, test_res = train(
                train_loader, test_loader, model, criterion, optimizer, epochs, device
            )

            train_results[norm].append(train_res)
            test_results[norm].append(test_res)

            print(
                f"\n{norm} → Seed {seed}: Test Accuracy: {test_res[-1][0]:.4f}, Test Loss: {test_res[-1][1]:.4f}\n"
            )

    # TODO: Add print + save functionality


if __name__ == "__main__":
    typer.run(main)
