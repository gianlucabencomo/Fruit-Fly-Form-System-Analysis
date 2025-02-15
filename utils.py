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


def get_norm_layer(
    norm: str, n_groups: int, use_local: bool = True, threshold: float = 0.5
):
    # import inside function to avoid circular import error
    from normalization import (
        BatchNorm2d,
        LayerNorm2d,
        GroupNorm,
        InstanceNorm2d,
        AdaptiveGroupNorm,
        LocalContextNorm
    )
    if norm == "in":
        norm_layer = InstanceNorm2d if use_local else nn.InstanceNorm2d
    elif norm == "bn":
        norm_layer = BatchNorm2d if use_local else nn.BatchNorm2d
    elif norm == "gn":
        norm_layer = (
            partial(GroupNorm, n_groups)
            if use_local
            else partial(nn.GroupNorm, n_groups)
        )
    elif norm == "ln":
        norm_layer = LayerNorm2d
    elif norm == "agn":
        norm_layer = partial(AdaptiveGroupNorm, n_groups)
    elif norm == "lcn":
        norm_layer = LocalContextNorm
    else:
        raise NotImplementedError

    return norm_layer


def replace_batch_norm_layers(model, custom_norm_fn):
    # import inside function to avoid circular import error
    from normalization import (
        BatchNorm2d,
        LayerNorm2d,
        GroupNorm,
        InstanceNorm2d,
        AdaptiveGroupNorm,
        LocalContextNorm
    )
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
            num_channels = module.num_features
            if custom_norm_fn in [GroupNorm, AdaptiveGroupNorm]:
                n_groups = num_channels // 4  # compression factor
                setattr(model, name, custom_norm_fn(n_groups, num_channels))
            else:
                setattr(model, name, custom_norm_fn(num_channels))
        else:
            replace_batch_norm_layers(module, custom_norm_fn)
