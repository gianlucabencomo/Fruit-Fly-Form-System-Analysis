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

import warnings
warnings.filterwarnings("ignore")

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
    lam: float = 1e-1,  # set to zero for normal cross entropy
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),  # Only use pin_memory if GPU is available
        num_workers=8 if torch.cuda.is_available() else 0,
        prefetch_factor=2 if torch.cuda.is_available() else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),  # Only use pin_memory if GPU is available
        num_workers=8 if torch.cuda.is_available() else 0,
        prefetch_factor=2 if torch.cuda.is_available() else None,
    )

    set_random_seeds(seed)
    norm_layers = [
            # Compression factor: 4
            partial(AdaptiveGroupNorm, 8),
            partial(AdaptiveGroupNorm, 16),
            partial(AdaptiveGroupNorm, 32),
            partial(AdaptiveGroupNorm, 64),
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
            Vs[i][key] = F.softmax(Vs[i][key], dim=1)

    # for last epoch only
    last_epoch = len(Qs) - 1
    for i, layer_key in enumerate(Qs[last_epoch].keys()):
        # 'data' has shape: (num_features, num_groups)
        data = Qs[last_epoch][layer_key].numpy()
        num_groups = data.shape[1]
        cols = min(5, num_groups)
        rows = (num_groups + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        for group_idx in range(num_groups):
            ax = axes[group_idx]
            p = data[:, group_idx]
            H = -np.sum(p * np.log(p + 1e-8))
            # Sort the probabilities in descending order and select the top 10.
            sorted_indices = np.argsort(p)[::-1]
            top_indices = sorted_indices[:10]
            top_p = p[top_indices]
            # Convert indices to string labels.
            top_labels = [str(idx) for idx in top_indices]
            # Compute a threshold similar to the second snippet:
            threshold = 10 * (1 / len(p))
            # Create the bar plot.
            ax.bar(top_labels, top_p, alpha=0.7, edgecolor="black")
            ax.axhline(y=threshold, alpha=0.7, ls="--", c="red")
            ax.set_title(f"Gr{group_idx}", fontsize=12)
            ax.set_ylabel("In Percentage")
            ax.set_xticklabels(top_labels, rotation=60)
            # Annotate with entropy.
            ax.text(
                0.95,
                0.95,
                f"H[p] = {H:.3f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="black"),
            )

        # Remove any unused subplots.
        for j in range(num_groups, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"layer{i+1}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    typer.run(main)
