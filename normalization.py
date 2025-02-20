import typer

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import get_device


class LocalContextNorm(nn.Module):
    """Code adapted from https://github.com/anthonymlortiz/lcn."""

    def __init__(
        self, num_features, channels_per_group=2, window_size=(16, 16), eps=1e-5
    ):
        super(LocalContextNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.channels_per_group = channels_per_group
        self.eps = eps
        self.window_size = window_size
        self.device = get_device()

    def forward(self, x):
        B, C, H, W = x.size()
        G = C // self.channels_per_group
        assert C % self.channels_per_group == 0
        if self.window_size[0] < H and self.window_size[1] < W:
            # Build integral image
            x_squared = x**2
            integral_img = x.cumsum(dim=2).cumsum(dim=3)
            integral_img_sq = x_squared.cumsum(dim=2).cumsum(dim=3)
            # Dilation
            d = (1, self.window_size[0], self.window_size[1])
            integral_img = torch.unsqueeze(integral_img, dim=1)
            integral_img_sq = torch.unsqueeze(integral_img_sq, dim=1)
            kernel = torch.tensor([[[[[1.0, -1.0], [-1.0, 1.0]]]]]).to(self.device)
            c_kernel = torch.ones((1, 1, self.channels_per_group, 1, 1)).to(self.device)
            with torch.no_grad():
                # Dilated conv
                sums = F.conv3d(integral_img, kernel, stride=[1, 1, 1], dilation=d)
                sums = F.conv3d(sums, c_kernel, stride=[self.channels_per_group, 1, 1])
                squares = F.conv3d(
                    integral_img_sq, kernel, stride=[1, 1, 1], dilation=d
                )
                squares = F.conv3d(
                    squares, c_kernel, stride=[self.channels_per_group, 1, 1]
                )
            n = self.window_size[0] * self.window_size[1] * self.channels_per_group
            means = torch.squeeze(sums / n, dim=1)
            var = torch.squeeze((1.0 / n * (squares - sums * sums / n)), dim=1)
            _, _, h, w = means.size()
            pad2d = (
                int(math.floor((W - w) / 2)),
                int(math.ceil((W - w) / 2)),
                int(math.floor((H - h) / 2)),
                int(math.ceil((H - h) / 2)),
            )
            padded_means = F.pad(means, pad2d, "replicate")
            padded_vars = F.pad(var, pad2d, "replicate") + self.eps
            for i in range(G):
                x[
                    :,
                    i * self.channels_per_group : i * self.channels_per_group
                    + self.channels_per_group,
                    :,
                    :,
                ] = (
                    x[
                        :,
                        i * self.channels_per_group : i * self.channels_per_group
                        + self.channels_per_group,
                        :,
                        :,
                    ]
                    - torch.unsqueeze(padded_means[:, i, :, :], dim=1).to(self.device)
                ) / (
                    (torch.unsqueeze(padded_vars[:, i, :, :], dim=1)).to(self.device)
                ).sqrt()
            del integral_img
            del integral_img_sq
        else:
            x = x.view(B, G, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)
            x = (x - mean) / (var + self.eps).sqrt()
            x = x.view(B, C, H, W)

        return x * self.gamma + self.beta


class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, num_features, eps=1e-05, affine=False):
        super().__init__()
        assert (
            num_features % num_groups == 0
        ), "Number of features must be divisible by the number of groups."
        self.num_groups = num_groups
        self.num_features = num_features
        self.eps = eps
        self.k = num_features // num_groups
        self.affine = affine

        self.Q = nn.Parameter(torch.randn(num_features, num_groups))
        self.V = nn.Parameter(torch.randn(num_groups, num_features))

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.gamma = None
            self.beta = None
            
        self.u = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        A = F.softmax(self.Q, dim=0)
        V = F.softmax(self.V, dim=1)
        u = 1 + (self.k - 1) * torch.sigmoid(self.u)
        M = torch.matmul(A, V) * u

        x_view = x.permute(0, 2, 3, 1).contiguous()

        # compute first and second moments
        x_1 = torch.matmul(x_view, M)
        x_2 = torch.matmul(x_view**2, M)

        # global average in transformed space
        mean = x_1.mean(dim=(1, 2), keepdim=True).permute(0, 3, 1, 2).contiguous()
        x_2 = x_2.mean(dim=(1, 2), keepdim=True).permute(0, 3, 1, 2).contiguous()
        var = torch.clamp(x_2 - mean**2, min=self.eps)

        # ! NEW
        self.reconstructed = x_1.permute(0, 3, 1, 2)
        self.original = x
        
        # normalize in original space
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # not neccessary but let's see if it does something
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
        assert (
            num_features % num_groups == 0
        ), "Number of features must be divisible by the number of groups."
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
        x_groups = x.view(
            B, self.num_groups, C // self.num_groups, H, W
        )  # B, G, C // G, H, W
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

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.squeeze()
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_hat = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)

        return x_hat


class LocalBatchNorm2d(nn.Module):
    def __init__(
        self,
        height,
        width,
        n_channels,
        kernel_size: int = 3,
        stride: int = 1,
        eps: float = 1e-5,
        affine=True,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.n_channels = n_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
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
            x_pad = F.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
                mode="constant",
                value=0,
            )
            patches = F.unfold(
                x_pad,
                kernel_size=self.kernel_size,
                dilation=1,
                padding=self.padding,
                stride=self.stride,
            )
            B, _, L = patches.shape
            H, W, P = (
                self.height + 2 * self.padding,
                self.width + 2 * self.padding,
                self.padding,
            )
            patches = patches.view(
                B, self.n_channels, self.kernel_size[0] * self.kernel_size[1], L
            )
            mean = patches.mean(dim=(0, 2), keepdim=True).view(
                1, self.n_channels, H, W
            )[:, :, P:-P, P:-P]
            var = patches.var(dim=(0, 2), unbiased=False, keepdim=True).view(
                1, self.n_channels, H, W
            )[:, :, P:-P, P:-P]

            x_hat = (x - mean) / torch.sqrt(var + self.eps)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.squeeze()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var.squeeze()
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
        torch_batch_norm = nn.BatchNorm2d(
            num_features=C, eps=1e-05, momentum=0.1, affine=True
        )
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
        torch_instance_norm = nn.InstanceNorm2d(
            num_features=C, eps=1e-05, momentum=0.1, affine=True
        )
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


def test_adaptive_group_norm(n_samples: int = 100, tol: float = 1e-5):
    res = []
    for _ in range(n_samples):
        B, H, W = torch.randint(low=1, high=64, size=(3,))
        G = torch.randint(low=2, high=8, size=(1,)).item()
        C = G * 4
        layer_norm = LayerNorm2d(num_features=C)
        agn = AdaptiveGroupNorm(num_groups=G, num_features=C)
        x = torch.randn(B, C, H, W) * 10
        torch_gn_output = layer_norm(x)
        custom_gn_output = agn(x)
        res.append(torch.allclose(torch_gn_output, custom_gn_output, atol=tol))
    print(
        f"Dynamic Group Norm Passed = {sum(res) ==  len(res)} ({sum(res)} / {len(res)})"
    )


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


def unit_tests(
    channels: int = 3,
    height: int = 5,
    width: int = 5,
    kernel_size: int = 3,
    stride: int = 1,
):
    test_adaptive_group_norm()
    exit()
    test_batch_norm()
    test_instance_norm()
    test_group_norm()
    test_layer_norm()


if __name__ == "__main__":
    typer.run(unit_tests)
