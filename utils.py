import torch
import torch.nn as nn
import numpy as np
import random
from functools import partial

import os
import json


def save_results(model, train_losses, test_acc, test_loss, save_dir, model_name):
    """Saves model weights and training results."""
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # Save training losses and test results
    results = {
        "train_losses": train_losses,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
    }
    results_path = os.path.join(save_dir, f"{model_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Training results saved to {results_path}")


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


def get_norm_layer(norm: str, use_local: bool = True, n_groups: list = [4, 8, 16, 32]):
    # import inside function to avoid circular import error
    from normalization import (
        BatchNorm2d,
        LayerNorm2d,
        GroupNorm,
        InstanceNorm2d,
        AdaptiveGroupNorm,
        AdaptiveGroupNorm2,
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
    elif norm == "agn2":
        norm_layer = [partial(AdaptiveGroupNorm2, G) for G in n_groups]
    elif norm == "lcn":
        norm_layer = LocalContextNorm
    elif norm == "identity":
        norm_layer = None
    else:
        raise NotImplementedError

    return norm_layer


def replace_batch_norm_layers(model, custom_norm_fn, compression_factor: int = 2):
    # import inside function to avoid circular import error
    from normalization import (
        BatchNorm2d,
        LayerNorm2d,
        GroupNorm,
        InstanceNorm2d,
        AdaptiveGroupNorm,
        LocalContextNorm,
    )

    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
            num_channels = module.num_features
            if custom_norm_fn in [GroupNorm, AdaptiveGroupNorm]:
                n_groups = num_channels // compression_factor
                setattr(model, name, custom_norm_fn(n_groups, num_channels))
            else:
                setattr(model, name, custom_norm_fn(num_channels))
        else:
            replace_batch_norm_layers(module, custom_norm_fn)
