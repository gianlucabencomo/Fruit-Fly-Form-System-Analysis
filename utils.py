import torch
import torch.nn as nn
import numpy as np
import random
from functools import partial


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_random_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer_and_scheduler(model, optimizer: str, epochs: int = 200, warmup_epochs: int = 5, dataset: str = "cifar100"):
    """Optimizers and schedulers for CIFAR-100"""
    if optimizer not in ["sgd", "adamw"]:
        raise NotImplementedError(f"Optimizer '{optimizer}' is not implemented. Choose 'sgd' or 'adamw'.")
    assert epochs > warmup_epochs, "Total epochs must be greater than number of warm-up epochs (5)."
    epochs = epochs - warmup_epochs
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,   
            weight_decay=1e-4,
        )
        milestones = [25, 55, 85] if dataset == "imagenet" else [int(0.5 * epochs), int(0.75 * epochs)]
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1         
        )

    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=1e-2
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-6
        )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-2,  # Start at 1% of base LR
        end_factor=1.0,  # Linearly increase to 100% base LR
        total_iters=warmup_epochs
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )

    return optimizer, scheduler



def get_norm_layer(norm: str, use_local: bool = True, n_groups: list = [8, 16, 32, 64]):
    # import inside function to avoid circular import error
    from normalization import (
        BatchNorm2d,
        LayerNorm2d,
        GroupNorm,
        InstanceNorm2d,
        AdaptiveGroupNorm,
        LocalContextNorm,
    )

    if norm == "in":
        norm_layer = InstanceNorm2d if use_local else nn.InstanceNorm2d
    elif norm == "bn":
        norm_layer = BatchNorm2d if use_local else nn.BatchNorm2d
    elif norm == "gn":
        if use_local:
            norm_layer = [partial(GroupNorm, G) for G in n_groups]
        else:
            norm_layer = [partial(nn.GroupNorm, G) for G in n_groups]
    elif norm == "ln":
        norm_layer = LayerNorm2d
    elif norm == "agn":
        norm_layer = [partial(AdaptiveGroupNorm, G) for G in n_groups]
    elif norm == "lcn":
        norm_layer = LocalContextNorm
    elif norm == "identity":
        norm_layer = None
    else:
        raise NotImplementedError

    return norm_layer


def replace_batch_norm_layers(model, norm: str, compression_factor: int = None, n_groups: int = 32):
    # import inside function to avoid circular import error
    from normalization import GroupNorm, AdaptiveGroupNorm, LayerNorm2d, InstanceNorm2d
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
            num_channels = module.num_features
            if norm in ["gn", "agn"]:
                if compression_factor is not None:
                    n_groups = num_channels // compression_factor
                f = AdaptiveGroupNorm if norm == "agn" else GroupNorm
                setattr(model, name, f(n_groups, num_channels))
            else:
                f = LayerNorm2d if norm == "ln" else InstanceNorm2d
                setattr(model, name, f(num_channels))
        else:
            replace_batch_norm_layers(module, norm, compression_factor, n_groups)
