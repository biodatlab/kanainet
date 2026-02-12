"""
KAN-based specialized modules for endoscopic image segmentation
- KAN-IMM: Illumination Modulation Module
- KAN-BAM: Boundary Attention Module
- KAN Texture Pathway
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .kan_layers import KANLinear, SpatialKAN2D, AdaptiveKANBlock


class KANIlluminationModulationModule(nn.Module):
    """
    KAN-IMM: Illumination Modulation Module
    Adaptively modulates features based on local illumination statistics
    """
    def __init__(self, channels, reduction=16):
        super(KANIlluminationModulationModule, self).__init__()
        self.channels = channels

        # Global pooling to get image statistics
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Local statistics extraction
        self.local_stats_conv = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True)
        )

        # KAN-based modulation function
        # Input: local stats + global stats + variance
        self.kan_modulator = KANLinear(
            in_features=channels // reduction + 3,  # local + (mean, std, brightness)
            out_features=channels,
            grid_size=7
        )

        self.sigmoid = nn.Sigmoid()

    def compute_illumination_stats(self, x):
        """
        Compute illumination statistics: mean, std, brightness
        """
        B, C, H, W = x.shape

        # Global mean and std
        mean = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        std = x.std(dim=[2, 3], keepdim=True)    # [B, C, 1, 1]

        # Brightness (average across channels)
        brightness = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        brightness_global = brightness.mean(dim=[2, 3], keepdim=True)  # [B, 1, 1, 1]

        return mean, std, brightness_global

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Compute illumination statistics
        mean, std, brightness = self.compute_illumination_stats(x)

        # Local feature statistics
        local_stats = self.local_stats_conv(x)  # [B, C//reduction, H, W]
        local_stats_pooled = self.global_pool(local_stats).squeeze(-1).squeeze(-1)  # [B, C//reduction]

        # Global statistics
        mean_flat = mean.squeeze(-1).squeeze(-1).mean(dim=1, keepdim=True)  # [B, 1]
        std_flat = std.squeeze(-1).squeeze(-1).mean(dim=1, keepdim=True)    # [B, 1]
        brightness_flat = brightness.squeeze(-1).squeeze(-1)                 # [B, 1]

        # Concatenate all statistics
        stats = torch.cat([local_stats_pooled, mean_flat, std_flat, brightness_flat], dim=1)  # [B, C//reduction + 3]

        # KAN-based adaptive modulation
        modulation = self.kan_modulator(stats)  # [B, C]
        modulation = self.sigmoid(modulation).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Apply modulation
        return x * modulation


class KANBoundaryAttentionModule(nn.Module):
    """
    KAN-BAM: Boundary Attention Module
    Enhances boundary detection through learned non-linear edge functions
    """
    def __init__(self, channels, num_scales=3):
        super(KANBoundaryAttentionModule, self).__init__()
        self.channels = channels
        self.num_scales = num_scales

        # Multi-scale edge detection
        self.edge_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=2*i+3, padding=i+1, groups=channels)
            for i in range(num_scales)
        ])

        # KAN-based edge enhancement
        self.kan_edge_enhance = SpatialKAN2D(
            in_channels=channels * (num_scales + 1),  # original + multi-scale edges
            out_channels=channels,
            grid_size=7
        )

        # Attention generation
        self.attention_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        # Multi-scale edge features
        edge_features = [x]
        for edge_conv in self.edge_convs:
            edge = edge_conv(x)
            edge_features.append(edge)

        # Concatenate multi-scale features
        multi_scale = torch.cat(edge_features, dim=1)  # [B, C*(num_scales+1), H, W]

        # KAN-based edge enhancement
        enhanced = self.kan_edge_enhance(multi_scale)  # [B, C, H, W]

        # Generate boundary attention map
        attention = self.attention_conv(enhanced)  # [B, 1, H, W]

        # Apply attention
        return x * (1 + attention), attention


class KANTexturePathway(nn.Module):
    """
    KAN Texture Pathway
    Processes texture-specific features for different tissue types
    """
    def __init__(self, in_channels, out_channels, num_texture_filters=8):
        super(KANTexturePathway, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Texture extraction (Gabor-like filters with different orientations)
        self.texture_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // num_texture_filters,
                     kernel_size=7, padding=3, groups=1)
            for _ in range(num_texture_filters)
        ])

        # KAN-based texture discrimination
        self.kan_texture = SpatialKAN2D(
            in_channels=out_channels,
            out_channels=out_channels,
            grid_size=5
        )

        # Normalization
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        # Extract texture features
        texture_features = []
        for texture_conv in self.texture_convs:
            feat = texture_conv(x)
            texture_features.append(feat)

        # Concatenate texture features
        textures = torch.cat(texture_features, dim=1)  # [B, out_channels, H, W]

        # KAN-based texture processing
        texture_enhanced = self.kan_texture(textures)
        texture_enhanced = self.norm(texture_enhanced)

        return texture_enhanced


class KANFusionModule(nn.Module):
    """
    KAN-based fusion module for combining different pathways
    """
    def __init__(self, channels_list, out_channels):
        super(KANFusionModule, self).__init__()
        total_channels = sum(channels_list)

        # KAN-based fusion
        self.fusion_conv = nn.Conv2d(total_channels, out_channels, 1)
        self.kan_fusion = SpatialKAN2D(out_channels, out_channels, grid_size=5)

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feature_list):
        """
        feature_list: list of [B, C_i, H, W]
        """
        # Ensure all features have same spatial size
        target_size = feature_list[0].shape[2:]
        resized_features = []
        for feat in feature_list:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(feat)

        # Concatenate
        fused = torch.cat(resized_features, dim=1)

        # Fusion
        fused = self.fusion_conv(fused)
        fused = self.kan_fusion(fused)
        fused = self.norm(fused)
        fused = self.relu(fused)

        return fused
