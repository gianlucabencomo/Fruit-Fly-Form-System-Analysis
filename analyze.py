import typer
from tqdm.auto import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn.functional as F

from transforms import image_transforms
from utils import *

from models import CNN
from normalization import AdaptiveGroupNorm
from losses import AdaptiveGroupNormLoss

import numpy as np
import matplotlib.pyplot as plt

WIDTHS = {
    1: [32],
    2: [32, 64],
    3: [32, 64, 128],
    4: [32, 64, 128, 256],
}


def update_q_v(model, Qs, Vs):
    q, v = {}, {}
    for name, module in model.named_modules():
        if isinstance(module, (AdaptiveGroupNorm)):
            q[name] = module.Q.detach().cpu()
            v[name] = module.V.detach().cpu()
    Qs.append(q)
    Vs.append(v)


def train(dataloader, model, criterion, optimizer, epochs, device):
    model.train()
    print("Starting Training...")
    epoch_losses, Qs, Vs = [], [], []
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
        update_q_v(model, Qs, Vs)
        print(f"Epoch {epoch+1}: Loss = {np.mean(losses):.3f}")
    return epoch_losses, Qs, Vs


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
    n_layers: int = 4,
    batch_size: int = 64,
    alpha: float = 1e-3,
    lam: float = 1e-3,  # set to zero for normal cross entropy
    epochs: int = 10,
):
    device = get_device()
    set_random_seeds(seed)

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

    set_random_seeds(seed)
    norm_layers = [
        partial(AdaptiveGroupNorm, 16),
        partial(AdaptiveGroupNorm, 32),
        partial(AdaptiveGroupNorm, 64),
        partial(AdaptiveGroupNorm, 128),
    ]
    norm_layers = norm_layers[:n_layers]
    model = CNN(
        num_layers=n_layers, width=WIDTHS[n_layers], norm_layers=norm_layers
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=alpha)
    criterion = AdaptiveGroupNormLoss(model=model, lam=lam)

    train_loss, Qs, Vs = train(
        train_loader, model, criterion, optimizer, epochs, device
    )
    acc, test_loss = test(test_loader, model, criterion, device)

    print(
        f"\nagn → Seed {seed}: Test Accuracy: {acc:.4f}, Test Loss: {test_loss:.4f}\n"
    )

    for i in range(len(Qs)):
        for key in Qs[i].keys():
            Qs[i][key] = F.softmax(Qs[i][key], dim=0)

    # for last epoch only
    i = len(Qs) - 1
    for j, key in enumerate(Qs[i].keys()):  # layers
        data = Qs[i][key].numpy()
        groups = []
        for k in range(data.shape[1]):  # groupings
            sorted_data = np.flip(np.sort(data[:, k]))
            y = 6 / len(sorted_data)
            sorted_data = sorted_data[:10]  # first 10 values
            groups.append(sorted_data)

        num_groups = len(groups)
        cols = min(8, num_groups)
        rows = (num_groups + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 1.5 * rows))
        fig.suptitle(f"Epoch {i}, Layer {key}")

        # Ensure axes is always a 2D array for consistent indexing
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if cols == 1:
            axes = np.expand_dims(axes, axis=1)

        for idx, sorted_data in enumerate(groups):
            row, col = divmod(idx, cols)
            axes[row, col].bar(range(len(sorted_data)), sorted_data)
            axes[row, col].axhline(y=y, color="r", linestyle="--")
            axes[row, col].set_title(f"Group {idx}")

        # Hide unused subplots
        for idx in range(num_groups, rows * cols):
            row, col = divmod(idx, cols)
            fig.delaxes(axes[row, col])

        plt.tight_layout()
        plt.show()
    #         x = np.arange(data.shape[0])
    #         width = 0.1

    #         for k in range(data.shape[1]):
    #             ax.bar(x + k * width, data[:, k], width, alpha=0.7, label=f"Output {k+1}")

    #         ax.set_title(f"Epoch {i+1} - {key}")
    #         ax.set_xlabel("Feature Index")
    #         ax.set_ylabel("Softmax Value")

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    typer.run(main)
