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

WIDTHS = {
    1: [32],
    2: [32, 64],
    3: [32, 64, 128],
    4: [32, 64, 128, 256],
}


def plot_training_losses(
    norms, seeds, all_train_losses, epochs, save_path="./training_loss_plot.png"
):
    plt.figure(figsize=(10, 5))

    # Define unique colors for each normalization method
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # Define different line styles for different seeds
    linestyles = ["-", "--", "-.", ":", (0, (3, 5, 1, 5))]

    for norm_idx, norm in enumerate(norms):
        color = colors[norm_idx % len(colors)]  # Assign a color to each norm

        for seed_idx, seed in enumerate(seeds):
            linestyle = linestyles[
                seed_idx % len(linestyles)
            ]  # Assign a unique line style per seed

            plt.plot(
                range(1, epochs + 1),
                all_train_losses[norm][seed_idx],
                label=f"{norm}"
                if seed_idx == 0
                else None,  # Label only the first seed per norm
                color=color,
                linestyle=linestyle,
                alpha=0.7,  # Transparency for clarity
            )

    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Normalization Method Across Seeds")
    plt.legend()
    plt.grid(True)

    # Save the plot instead of showing it
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    print(f"Plot saved as {save_path}")


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
    seeds: int = 5,
    root: str = "./data",
    dataset: str = "cifar100",
    n_layers: int = 1,
    batch_size: int = 64,
    alpha: float = 1e-3,
    epochs: int = 10,
    n_groups: int = 4,
    threshold: float = 0.5,
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

    norms = [
        "batch_norm",
        "adaptive_norm",
        "group_norm",
        "instance_norm",
        "layer_norm",
    ]

    all_train_losses = {norm: [] for norm in norms}
    all_accs = {norm: [] for norm in norms}
    all_test_losses = {norm: [] for norm in norms}

    print("\n=== Running Experiments Across Seeds ===\n")

    seeds = [i for i in range(0, seeds)]
    for seed in seeds:
        print(f"\n--- Running for Seed {seed} ---\n")

        for norm in norms:
            set_random_seeds(seed)
            print(f"Running {norm} with seed {seed}...")
            norm_layer = get_norm_layer(norm, n_groups, threshold)
            model = CNN(
                num_layers=n_layers, width=WIDTHS[n_layers], norm_layer=norm_layer
            ).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=alpha)
            criterion = nn.CrossEntropyLoss()

            train_loss = train(
                train_loader, model, criterion, optimizer, epochs, device
            )
            acc, test_loss = test(test_loader, model, criterion, device)

            all_train_losses[norm].append(train_loss)
            all_accs[norm].append(acc)
            all_test_losses[norm].append(test_loss)

            print(
                f"\n{norm} → Seed {seed}: Test Accuracy: {acc:.4f}, Test Loss: {test_loss:.4f}\n"
            )

    print("\n=== Final Results Across Seeds ===")
    for norm in norms:
        avg_acc = np.mean(all_accs[norm])
        avg_loss = np.mean(all_test_losses[norm])
        print(
            f"{norm}: Avg Test Accuracy = {avg_acc:.4f}, Avg Test Loss = {avg_loss:.4f}"
        )

    plot_training_losses(
        norms,
        seeds,
        all_train_losses,
        epochs,
        save_path=f"./{epochs}_{n_layers}_{n_groups}.png",
    )


if __name__ == "__main__":
    typer.run(main)
