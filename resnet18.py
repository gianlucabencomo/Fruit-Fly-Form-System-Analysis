import os
import typer
from tqdm.auto import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets

from transforms import image_transforms
from utils import *
from losses import AdaptiveGroupNormLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")


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
    seeds: int = 1,
    root: str = "./data",
    dataset: str = "cifar100",
    batch_size: int = 256,
    optim_name: str = "sgd",
    lam: float = 1e-2,  # set to zero for normal cross entropy
    compression_factor: int = 16,
    epochs: int = 100,
    use_groups: bool = False # use 32 groups at every layer if true
):
    device = get_device()
    if use_groups:
        compression_factor = None

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

    norms = ["ln", "gn", "agn", "in", "bn"]

    train_results = {norm: [] for norm in norms}
    test_results = {norm: [] for norm in norms}

    seeds = [i for i in range(0, seeds)]
    for seed in seeds:
        for norm in norms:
            set_random_seeds(seed)
            print(f"Running {norm} with seed {seed}, optimizer {optim_name}, batch size {batch_size}, lam {lam}.")
            model = models.resnet18(weights=None).to(device)
            if norm != "bn":
                replace_batch_norm_layers(model, norm, compression_factor=compression_factor)
            model = model.to(device)
            optimizer, scheduler = get_optimizer_and_scheduler(model, optim_name, epochs, dataset=dataset)
            if norm == "agn":
                criterion = AdaptiveGroupNormLoss(model=model, lam=lam)
            else:
                criterion = nn.CrossEntropyLoss()
            
            train_res, test_res = train(
                train_loader, test_loader, model, criterion, optimizer, scheduler, epochs, device
            )

            train_results[norm].append(train_res)
            test_results[norm].append(test_res)

            print(
                f"\n{norm} → Seed {seed}: Test Accuracy: {test_res[-1][0]:.4f}, Test Loss: {test_res[-1][1]:.4f}\n"
            )

    os.makedirs("results", exist_ok=True)
    lam_str = f"{lam:.0e}" if lam != 0.0 else "no_lam"
    np.savez(
        os.path.join("results", f"resnet_{batch_size}_{optim_name}_{epochs}_{lam_str}.npz"),
        train_results=train_results,
        test_results=test_results,
    )

    print("\nMean over seeds for each normalization method:")
    for norm in norms:
        # Extract final epoch metrics from each seed run
        final_test_accs = [results[-1][0] for results in test_results[norm]]
        final_test_losses = [results[-1][1] for results in test_results[norm]]
        final_train_accs = [results[-1][0] for results in train_results[norm]]
        final_train_losses = [results[-1][1] for results in train_results[norm]]
        
        mean_test_acc = np.mean(final_test_accs)
        mean_test_loss = np.mean(final_test_losses)
        mean_train_acc = np.mean(final_train_accs)
        mean_train_loss = np.mean(final_train_losses)
        
        print(f"{norm}:")
        print(f"  Final Test Accuracy = {mean_test_acc:.4f}, Final Test Loss = {mean_test_loss:.4f}")
        print(f"  Final Train Accuracy = {mean_train_acc:.4f}, Final Train Loss = {mean_train_loss:.4f}")



if __name__ == "__main__":
    typer.run(main)
