"""
KAN (Kolmogorov-Arnold Network) layers implementation
Based on learnable activation functions using splines
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KANLinear(nn.Module):
    """
    KAN Linear layer with learnable activation functions
    Uses B-spline basis functions for flexible non-linear transformations
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create grid points for spline basis
        h = (2.0) / grid_size
        grid = torch.linspace(-1, 1, grid_size + 1)
        self.register_buffer('grid', grid)

        # Learnable spline coefficients for each input-output pair
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.1
        )

        # Optional linear transformation (base weight)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)

    def b_splines(self, x):
        """
        Compute B-spline basis functions
        x: input tensor [..., in_features]
        returns: basis functions [..., in_features, grid_size + spline_order]
        """
        # Clamp x to grid range
        x = torch.clamp(x, -1, 1)

        # Find the relevant grid interval
        grid = self.grid
        # Compute basis functions (simplified cubic B-spline)
        bases = []
        for i in range(len(grid) - 1):
            # Normalized position in interval
            mask = (x >= grid[i]) & (x < grid[i + 1])
            basis = mask.float()
            bases.append(basis)

        # Pad to match grid_size + spline_order
        while len(bases) < self.grid_size + self.spline_order:
            bases.append(torch.zeros_like(bases[0]))

        return torch.stack(bases, dim=-1)

    def forward(self, x):
        """
        Forward pass
        x: [..., in_features]
        returns: [..., out_features]
        """
        batch_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_features)

        # Normalize input to [-1, 1] range
        x_normalized = torch.tanh(x_flat)

        # Compute spline basis
        basis = self.b_splines(x_normalized)  # [batch, in_features, grid_size + spline_order]

        # Apply spline transformation
        # [batch, in_features, n_basis] * [out_features, in_features, n_basis]
        spline_output = torch.einsum('bin,oin->bo', basis, self.spline_weight)

        # Add base linear transformation
        base_output = F.linear(x_flat, self.base_weight)

        output = spline_output + base_output
        return output.reshape(*batch_shape, self.out_features)


class KAN(nn.Module):
    """
    Multi-layer KAN network
    """
    def __init__(self, layers_dims, grid_size=5, spline_order=3):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(len(layers_dims) - 1):
            self.layers.append(
                KANLinear(layers_dims[i], layers_dims[i+1], grid_size, spline_order)
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpatialKAN2D(nn.Module):
    """
    2D Spatial KAN layer for processing feature maps
    Applies KAN transformations across spatial dimensions
    """
    def __init__(self, in_channels, out_channels, grid_size=5):
        super(SpatialKAN2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Use 1x1 convolution as base
        self.base_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # KAN transformation per channel
        self.kan_transform = KANLinear(in_channels, out_channels, grid_size=grid_size)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Base convolution
        base = self.base_conv(x)

        # Permute for KAN: [B, C, H, W] -> [B, H, W, C] -> [B*H*W, C]
        x_permuted = x.permute(0, 2, 3, 1).reshape(-1, C)

        # Apply KAN transformation
        kan_out = self.kan_transform(x_permuted)  # [B*H*W, out_channels]

        # Reshape back: [B*H*W, out_channels] -> [B, H, W, out_channels] -> [B, out_channels, H, W]
        kan_out = kan_out.reshape(B, H, W, self.out_channels).permute(0, 3, 1, 2)

        return base + kan_out


class AdaptiveKANBlock(nn.Module):
    """
    Adaptive KAN block with skip connection and normalization
    """
    def __init__(self, channels, grid_size=5):
        super(AdaptiveKANBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.kan = SpatialKAN2D(channels, channels, grid_size)
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.kan(out)
        out = self.norm2(out)
        return out + identity
