import torch
import torch.nn as nn
import numpy as np
import random
from functools import partial

from normalization import BatchNorm2d, LayerNorm2d, GroupNorm, InstanceNorm2d, CorrelatedGroupNorm, DeCorrelatedGroupNorm, NegativeCorrelatedGroupNorm, PositiveCorrelatedGroupNorm


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

def get_norm_layer(norm: str, n_groups: int, use_local: bool = True):
    if norm == "instance_norm":
        norm_layer = InstanceNorm2d if use_local else nn.InstanceNorm2d
    elif norm == "batch_norm":
        norm_layer = BatchNorm2d if use_local else nn.BatchNorm2d
    elif norm == "group_norm":
        norm_layer = partial(GroupNorm, n_groups) if use_local else partial(nn.GroupNorm, n_groups)
    elif norm == "layer_norm":
         norm_layer = LayerNorm2d
    elif norm == "correlated_group":
         norm_layer = partial(CorrelatedGroupNorm, n_groups)
    elif norm == "de_correlated_group":
         norm_layer = partial(DeCorrelatedGroupNorm, n_groups)
    elif norm == "pos_correlated_group":
         norm_layer = partial(PositiveCorrelatedGroupNorm, n_groups)
    elif norm == "neg_correlated_group":
         norm_layer = partial(NegativeCorrelatedGroupNorm, n_groups)
    else:
        raise NotImplementedError
    
    return norm_layer