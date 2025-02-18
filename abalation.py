import os
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


def train(train_loader, test_loader, model, criterion, optimizer, scheduler, epochs, device, clip: bool = False):
    train_res, test_res = [], []
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train() # set back to train after test sets to eval
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        train_res.append(test(train_loader, model, nn.CrossEntropyLoss(), device))
        test_res.append(test(test_loader, model, nn.CrossEntropyLoss(), device))
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
    epochs: int = 200,
    optim_name: str = "sgd",
    batch_size: int = 64,
    dropout: float = 0.0,
    clip: bool = False
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
        num_workers=8 if torch.cuda.is_available() else 0,
        prefetch_factor=2 if torch.cuda.is_available() else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256, # Larger batch sizes for testing
        shuffle=False,
        pin_memory=torch.cuda.is_available(),  # Only use pin_memory if GPU is available
        num_workers=8 if torch.cuda.is_available() else 0,
        prefetch_factor=2 if torch.cuda.is_available() else None,
    )

    lams = [0.0, 1e-1, 1e-2]
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
            partial(AdaptiveGroupNorm, 32),
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

    train_results = {i: [] for i in list(product(range(len(norms)), lams))}
    test_results = {i: [] for i in list(product(range(len(norms)), lams))}

    seeds = [i for i in range(0, seeds)]
    for seed in seeds:
        for i, lam in list(product(range(len(norms)), lams)):
            print(f"Running {i} with seed {seed}, optimizer {optim_name}, batch size {batch_size}, lam {lam}, clip {clip}, and dropout {dropout}.")
            set_random_seeds(seed)
            norm_layer = norms[i]
            norm_layer = norm_layer[:n_layers]
            model = CNN(
                num_layers=n_layers, width=WIDTHS[n_layers], norm_layers=norm_layer, dropout=dropout
            ).to(device)
            optimizer, scheduler = get_optimizer_and_scheduler(model, optim_name, epochs)
            criterion = AdaptiveGroupNormLoss(model=model, lam=lam)

            train_res, test_res = train(
                train_loader, test_loader, model, criterion, optimizer, scheduler, epochs, device, clip
            )

            train_results[(i, lam)].append(train_res)
            test_results[(i, lam)].append(test_res)

            print(
                f"\n{(i, lam)} → Seed {seed}: Test Accuracy: {test_res[-1][0]:.4f}, Test Loss: {test_res[-1][1]:.4f}\n"
            )

    os.makedirs("results", exist_ok=True)
    np.savez(
        os.path.join("results", f"ablation_{batch_size}_{str(1) if dropout != 0.0 else str(0)}_{optim_name}_{epochs}_{clip}.npz"),
        train_results=train_results,
        test_results=test_results,
    )

    print("\n=== Final Results Across Seeds ===")
    for i, lam in list(product(range(len(norms)), lams)):
        # Extract final epoch metrics from each seed run
        final_test_accs = [results[-1][0] for results in test_results[(i, lam)]]
        final_test_losses = [results[-1][1] for results in test_results[(i, lam)]]
        final_train_accs = [results[-1][0] for results in train_results[(i, lam)]]
        final_train_losses = [results[-1][1] for results in train_results[(i, lam)]]
        
        mean_test_acc = np.mean(final_test_accs)
        mean_test_loss = np.mean(final_test_losses)
        mean_train_acc = np.mean(final_train_accs)
        mean_train_loss = np.mean(final_train_losses)
        
        print(f"{(i, lam)}:")
        print(f"  Final Test Accuracy = {mean_test_acc:.4f}, Final Test Loss = {mean_test_loss:.4f}")
        print(f"  Final Train Accuracy = {mean_train_acc:.4f}, Final Train Loss = {mean_train_loss:.4f}")


if __name__ == "__main__":
    typer.run(main)
