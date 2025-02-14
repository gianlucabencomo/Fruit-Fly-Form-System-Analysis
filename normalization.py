import typer

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeCorrelatedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-05, momentum: float = 0.1, affine=True):
        super().__init__()
        assert num_features % num_groups == 0, "Number of groups must be divisble by number of features."
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer("cos_sim_mean", torch.zeros(num_features, num_features))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            x_flat = x.view(B, C, -1)
            x_norm = F.normalize(x_flat, p=2, dim=2)
            cos_sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
            cos_sim_mean = cos_sim.mean(dim=0)
        else:
            cos_sim_mean = self.cos_sim_mean
        _, top_inds = torch.topk(torch.abs(cos_sim_mean), k=self.num_features // self.num_groups, dim=0, largest=False)
        group = [x]
        for i in range(1, self.num_features // self.num_groups):
            group.append(x[:, top_inds[i], :, :])
        x_groups = torch.stack(group, dim=2)
        mean = x_groups.mean(dim=(2, 3, 4), keepdim=True)
        var = x_groups.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        mean = mean.squeeze(2)
        var = var.squeeze(2)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        x_hat = x_hat.view(B, C, H, W)

        # update buffer
        if self.training:
            self.cos_sim_mean.data.mul_(1 - self.momentum).add_(self.momentum * cos_sim_mean)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return x_hat

class NegativeCorrelatedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-05, momentum: float = 0.1, affine=True):
        super().__init__()
        assert num_features % num_groups == 0, "Number of groups must be divisble by number of features."
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer("cos_sim_mean", torch.zeros(num_features, num_features))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            x_flat = x.view(B, C, -1)
            x_norm = F.normalize(x_flat, p=2, dim=2)
            cos_sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
            cos_sim_mean = cos_sim.mean(dim=0)
        else:
            cos_sim_mean = self.cos_sim_mean
        _, top_inds = torch.topk(cos_sim_mean, k=self.num_features // self.num_groups, dim=0, largest=False)
        group = [x]
        for i in range(1, self.num_features // self.num_groups):
            group.append(x[:, top_inds[i], :, :])
        x_groups = torch.stack(group, dim=2)
        mean = x_groups.mean(dim=(2, 3, 4), keepdim=True)
        var = x_groups.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        mean = mean.squeeze(2)
        var = var.squeeze(2)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        x_hat = x_hat.view(B, C, H, W)

        # update buffer
        if self.training:
            self.cos_sim_mean.data.mul_(1 - self.momentum).add_(self.momentum * cos_sim_mean)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return x_hat

class PositiveCorrelatedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-05, momentum: float = 0.1, affine=True):
        super().__init__()
        assert num_features % num_groups == 0, "Number of groups must be divisble by number of features."
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer("cos_sim_mean", torch.zeros(num_features, num_features))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            x_flat = x.view(B, C, -1)
            x_norm = F.normalize(x_flat, p=2, dim=2)
            cos_sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
            cos_sim_mean = cos_sim.mean(dim=0)
        else:
            cos_sim_mean = self.cos_sim_mean
        _, top_inds = torch.topk(cos_sim_mean, k=self.num_features // self.num_groups, dim=0)
        group = [x]
        for i in range(1, self.num_features // self.num_groups):
            group.append(x[:, top_inds[i], :, :])
        x_groups = torch.stack(group, dim=2)
        mean = x_groups.mean(dim=(2, 3, 4), keepdim=True)
        var = x_groups.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        mean = mean.squeeze(2)
        var = var.squeeze(2)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        x_hat = x_hat.view(B, C, H, W)

        # update buffer
        if self.training:
            self.cos_sim_mean.data.mul_(1 - self.momentum).add_(self.momentum * cos_sim_mean)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return x_hat

class CorrelatedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-05, momentum: float = 0.1, affine=True):
        super().__init__()
        assert num_features % num_groups == 0, "Number of groups must be divisble by number of features."
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer("cos_sim_mean", torch.zeros(num_features, num_features))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            x_flat = x.view(B, C, -1)
            x_norm = F.normalize(x_flat, p=2, dim=2)
            cos_sim = torch.bmm(x_norm, x_norm.transpose(1, 2))
            cos_sim_mean = cos_sim.mean(dim=0)
        else:
            cos_sim_mean = self.cos_sim_mean
        _, top_inds = torch.topk(torch.abs(cos_sim_mean), k=self.num_features // self.num_groups, dim=0)
        group = [x]
        for i in range(1, self.num_features // self.num_groups):
            group.append(x[:, top_inds[i], :, :])
        x_groups = torch.stack(group, dim=2)
        mean = x_groups.mean(dim=(2, 3, 4), keepdim=True)
        var = x_groups.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        mean = mean.squeeze(2)
        var = var.squeeze(2)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        x_hat = x_hat.view(B, C, H, W)

        # update buffer
        if self.training:
            self.cos_sim_mean.data.mul_(1 - self.momentum).add_(self.momentum * cos_sim_mean)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return x_hat

class LayerNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)

        return x_hat


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-05, affine=True):
        super().__init__()
        assert num_features % num_groups == 0, "Number of features must be divisible by the number of groups."
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = x.view(B, self.num_groups, C // self.num_groups, H, W) # B, G, C // G, H, W
        mean = x_groups.mean(dim=(2, 3, 4), keepdim=True)
        var = x_groups.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        x_hat = (x_groups - mean) / torch.sqrt(var + self.eps)

        x_hat = x_hat.view(B, C, H, W)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)
        return x_hat

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)

        return x_hat

class BatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            x_hat = (x - mean) / torch.sqrt(var + self.eps)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)

        return x_hat

class LocalBatchNorm2d(nn.Module):
    def __init__(self, height, width, n_channels, kernel_size: int = 3, stride: int = 1, eps: float = 1e-5, affine=True):
        super().__init__()
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = kernel_size // 2
        self.eps = eps
        self.momentum = 0.1
        self.affine = affine

        self.register_buffer("running_mean", torch.zeros(n_channels, height, width))
        self.register_buffer("running_var", torch.ones(n_channels, height, width))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(n_channels, height, width))
            self.beta = nn.Parameter(torch.zeros(n_channels, height, width))
        else:
            self.gamma = None
            self.beta = None

    def forward(self, x):
        # B, C, H, W
        if self.training:
            x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='constant', value=0)
            patches = F.unfold(x_pad, kernel_size=self.kernel_size, dilation=1, padding=self.padding, stride=self.stride)
            B, _, L = patches.shape
            H, W, P =  self.height + 2 * self.padding, self.width + 2 * self.padding, self.padding
            patches = patches.view(B, self.n_channels, self.kernel_size[0] * self.kernel_size[1], L)
            mean = patches.mean(dim=(0, 2), keepdim=True).view(1, self.n_channels, H, W)[:, :, P:-P, P:-P]
            var = patches.var(dim=(0, 2), unbiased=False, keepdim=True).view(1, self.n_channels, H, W)[:, :, P:-P, P:-P]

            x_hat = (x - mean) / torch.sqrt(var + self.eps)

            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.unsqueeze(0)
            var = self.running_var.unsqueeze(0)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_hat = self.gamma.unsqueeze(0) * x_hat + self.beta.unsqueeze(0)

        return x_hat
    
def test_batch_norm(n_samples: int = 100, tol: float = 1e-5):
    res = []
    for _ in range(n_samples):
        B, C, H, W = torch.randint(low=1, high=64, size=(4,))
        torch_batch_norm = nn.BatchNorm2d(num_features=C, eps=1e-05, momentum=0.1, affine=True)
        custom_batch_norm = BatchNorm2d(num_features=C)
        x = torch.randn(B, C, H, W)
        torch_bn_output = torch_batch_norm(x)
        custom_bn_output = custom_batch_norm(x)
        res.append(torch.allclose(torch_bn_output, custom_bn_output, atol=tol))
    print(f"Batch Norm Passed = {sum(res) ==  len(res)} ({sum(res)} / {len(res)})")

def test_instance_norm(n_samples: int = 100, tol: float = 1e-5):
    res = []
    for _ in range(n_samples):
        B, C, H, W = torch.randint(low=1, high=64, size=(4,))
        torch_instance_norm = nn.InstanceNorm2d(num_features=C, eps=1e-05, momentum=0.1, affine=True)
        custom_instance_norm = InstanceNorm2d(num_features=C)
        x = torch.randn(B, C, H, W)
        torch_in_output = torch_instance_norm(x)
        custom_in_output = custom_instance_norm(x)
        res.append(torch.allclose(torch_in_output, custom_in_output, atol=tol))
    print(f"Instance Norm Passed = {sum(res) ==  len(res)} ({sum(res)} / {len(res)})")

def test_group_norm(n_samples: int = 100, tol: float = 1e-5):
    res = []
    for _ in range(n_samples):
        B, H, W = torch.randint(low=1, high=64, size=(3,))
        G = torch.randint(low=2, high=8, size=(1,)).item()
        C = G * 4
        torch_group_norm = nn.GroupNorm(num_groups=G, num_channels=C)
        custom_group_norm = GroupNorm(num_groups=G, num_features=C)
        x = torch.randn(B, C, H, W)
        torch_gn_output = torch_group_norm(x)
        custom_gn_output = custom_group_norm(x)
        res.append(torch.allclose(torch_gn_output, custom_gn_output, atol=tol))
    print(f"Group Norm Passed = {sum(res) ==  len(res)} ({sum(res)} / {len(res)})")

def test_layer_norm(n_samples: int = 100, tol: float = 1e-5):
    res = []
    for _ in range(n_samples):
        B, C, H, W = torch.randint(low=1, high=64, size=(4,))
        torch_layer_norm = nn.LayerNorm([C, H, W], eps=1e-05, elementwise_affine=True)
        custom_layer_norm = LayerNorm2d(num_features=C)

        x = torch.randn(B, C, H, W)
        torch_ln_output = torch_layer_norm(x)  # PyTorch LayerNorm
        custom_ln_output = custom_layer_norm(x)  # Custom LayerNorm2d

        res.append(torch.allclose(torch_ln_output, custom_ln_output, atol=tol))

    print(f"LayerNorm Passed = {sum(res) == len(res)} ({sum(res)} / {len(res)})")

def unit_tests(channels: int = 3, height: int = 5, width: int = 5, kernel_size: int = 3, stride: int = 1):
    test_batch_norm()
    test_instance_norm()
    test_group_norm()
    test_layer_norm()
    exit()

    local_norm = LocalBatchNorm2d(height=height, width=width, n_channels=channels,
                            kernel_size=kernel_size, stride=stride)
    
    output = local_norm(x)
    print("Input shape: ", x.shape)
    print("Output shape: ", output.shape)


if __name__ == "__main__":
    typer.run(unit_tests)