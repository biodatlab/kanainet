"""
KAN-Enhanced Adaptive Context Network (KAN-ACNet)
Main architecture combining all KAN modules for endoscopic image segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .kan_modules import (
    KANIlluminationModulationModule,
    KANBoundaryAttentionModule,
    KANTexturePathway,
    KANFusionModule
)


class EncoderBlock(nn.Module):
    """
    Encoder block with KAN-IMM
    """
    def __init__(self, in_channels, out_channels, use_kan_imm=True):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # KAN Illumination Modulation Module
        self.use_kan_imm = use_kan_imm
        if use_kan_imm:
            self.kan_imm = KANIlluminationModulationModule(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.use_kan_imm:
            x = self.kan_imm(x)

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with KAN-BAM
    """
    def __init__(self, in_channels, skip_channels, out_channels, use_kan_bam=True):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)

        self.conv1 = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # KAN Boundary Attention Module
        self.use_kan_bam = use_kan_bam
        if use_kan_bam:
            self.kan_bam = KANBoundaryAttentionModule(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)

        # Match dimensions if needed
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        if self.use_kan_bam:
            x, attention = self.kan_bam(x)
            return x, attention
        return x, None


class KANACNet(nn.Module):
    """
    KAN-Enhanced Adaptive Context Network
    Main architecture for endoscopic image segmentation
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_channels=64,
        use_pretrained_backbone=True,
        use_texture_pathway=True,
        backbone='resnet34',
        encoder_kan_blocks=None,
        decoder_kan_blocks=None
    ):
        super(KANACNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_texture_pathway = use_texture_pathway

        # Ablation study: specify which KAN blocks to use
        # encoder_kan_blocks: list of encoder stages to include KAN-IMM (1-4)
        # decoder_kan_blocks: list of decoder stages to include KAN-BAM (1-4)
        self.encoder_kan_blocks = encoder_kan_blocks if encoder_kan_blocks is not None else [1, 2, 3, 4]
        self.decoder_kan_blocks = decoder_kan_blocks if decoder_kan_blocks is not None else [1, 2, 3, 4]

        self.backbone_type = backbone

        # Encoder backbone
        # encoder_channels = [init_conv_out, e1_out, e2_out, e3_out, e4_out]
        # For ConvNext: init_conv = stem+stage1 (96ch H/4),
        #   encoder1 = downsample+stage2 (192ch H/8),
        #   encoder2 = downsample+stage3 (384ch H/16),
        #   encoder3 = downsample only   (768ch H/32),
        #   encoder4 = stage4            (768ch H/32)
        _CONVNEXT_CHANNELS = {
            'convnext_tiny':  [96,  192, 384, 768,  768],
            'convnext_small': [96,  192, 384, 768,  768],
            'convnext_base':  [128, 256, 512, 1024, 1024],
            'convnext_large': [192, 384, 768, 1536, 1536],
        }

        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=use_pretrained_backbone)
            self.encoder_channels = [64, 64, 128, 256, 512]
            self._init_resnet_encoder(resnet)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=use_pretrained_backbone)
            self.encoder_channels = [64, 256, 512, 1024, 2048]
            self._init_resnet_encoder(resnet)
        elif backbone in _CONVNEXT_CHANNELS:
            self.encoder_channels = _CONVNEXT_CHANNELS[backbone]
            weights = 'DEFAULT' if use_pretrained_backbone else None
            convnext = getattr(models, backbone)(weights=weights)
            feats = convnext.features
            # feats[0:2]: stem + stage-1 → C0 ch at H/4
            self.init_conv = nn.Sequential(feats[0], feats[1])
            # feats[2:4]: downsample + stage-2 → C1 ch at H/8
            self.encoder1 = nn.Sequential(feats[2], feats[3])
            # feats[4:6]: downsample + stage-3 → C2 ch at H/16
            self.encoder2 = nn.Sequential(feats[4], feats[5])
            # feats[6]: downsample only → C3 ch at H/32  (wrapped to keep key depth consistent)
            self.encoder3 = nn.Sequential(feats[6])
            # feats[7]: stage-4 → C4 ch at H/32 (same spatial as encoder3)
            self.encoder4 = nn.Sequential(feats[7])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. "
                             f"Supported: resnet34, resnet50, "
                             f"{', '.join(_CONVNEXT_CHANNELS)}")

        self.kan_imm1 = KANIlluminationModulationModule(self.encoder_channels[1]) if 1 in self.encoder_kan_blocks else None
        self.kan_imm2 = KANIlluminationModulationModule(self.encoder_channels[2]) if 2 in self.encoder_kan_blocks else None
        self.kan_imm3 = KANIlluminationModulationModule(self.encoder_channels[3]) if 3 in self.encoder_kan_blocks else None
        self.kan_imm4 = KANIlluminationModulationModule(self.encoder_channels[4]) if 4 in self.encoder_kan_blocks else None

        # Texture Pathway
        if use_texture_pathway:
            self.texture_path1 = KANTexturePathway(self.encoder_channels[1], self.encoder_channels[1])
            self.texture_path2 = KANTexturePathway(self.encoder_channels[2], self.encoder_channels[2])
            self.texture_path3 = KANTexturePathway(self.encoder_channels[3], self.encoder_channels[3])

            # Fusion modules
            self.fusion1 = KANFusionModule([self.encoder_channels[1], self.encoder_channels[1]], self.encoder_channels[1])
            self.fusion2 = KANFusionModule([self.encoder_channels[2], self.encoder_channels[2]], self.encoder_channels[2])
            self.fusion3 = KANFusionModule([self.encoder_channels[3], self.encoder_channels[3]], self.encoder_channels[3])

        # Bridge
        self.bridge = nn.Sequential(
            nn.Conv2d(self.encoder_channels[4], self.encoder_channels[4], 3, padding=1),
            nn.BatchNorm2d(self.encoder_channels[4]),
            nn.ReLU(inplace=True)
        )

        # Decoder with KAN-BAM (conditionally based on ablation config)
        self.decoder4 = DecoderBlock(self.encoder_channels[4], self.encoder_channels[3], self.encoder_channels[3],
                                    use_kan_bam=4 in self.decoder_kan_blocks)
        self.decoder3 = DecoderBlock(self.encoder_channels[3], self.encoder_channels[2], self.encoder_channels[2],
                                    use_kan_bam=3 in self.decoder_kan_blocks)
        self.decoder2 = DecoderBlock(self.encoder_channels[2], self.encoder_channels[1], self.encoder_channels[1],
                                    use_kan_bam=2 in self.decoder_kan_blocks)
        self.decoder1 = DecoderBlock(self.encoder_channels[1], self.encoder_channels[0], base_channels,
                                    use_kan_bam=1 in self.decoder_kan_blocks)

        # Final segmentation head
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1)
        )

        # Deep supervision outputs (optional, for training)
        self.ds_conv4 = nn.Conv2d(self.encoder_channels[3], num_classes, 1)
        self.ds_conv3 = nn.Conv2d(self.encoder_channels[2], num_classes, 1)
        self.ds_conv2 = nn.Conv2d(self.encoder_channels[1], num_classes, 1)

    def _init_resnet_encoder(self, resnet):
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1 = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x, return_attention=False):
        """
        Forward pass
        x: [B, 3, H, W]
        returns: segmentation mask [B, num_classes, H, W]
        """
        input_size = x.shape[2:]

        # Initial convolution
        x0 = self.init_conv(x)  # [B, 64, H/2, W/2]

        # Encoder path with KAN-IMM (conditionally applied)
        e1 = self.encoder1(x0)  # [B, 64/256, H/4, W/4]
        if self.kan_imm1 is not None:
            e1 = self.kan_imm1(e1)

        e2 = self.encoder2(e1)  # [B, 128/512, H/8, W/8]
        if self.kan_imm2 is not None:
            e2 = self.kan_imm2(e2)

        e3 = self.encoder3(e2)  # [B, 256/1024, H/16, W/16]
        if self.kan_imm3 is not None:
            e3 = self.kan_imm3(e3)

        e4 = self.encoder4(e3)  # [B, 512/2048, H/32, W/32]
        if self.kan_imm4 is not None:
            e4 = self.kan_imm4(e4)

        # Texture pathway
        if self.use_texture_pathway:
            t1 = self.texture_path1(e1)
            e1_fused = self.fusion1([e1, t1])

            t2 = self.texture_path2(e2)
            e2_fused = self.fusion2([e2, t2])

            t3 = self.texture_path3(e3)
            e3_fused = self.fusion3([e3, t3])
        else:
            e1_fused, e2_fused, e3_fused = e1, e2, e3

        # Bridge
        bridge = self.bridge(e4)

        # Decoder path with KAN-BAM
        d4, att4 = self.decoder4(bridge, e3_fused)
        d3, att3 = self.decoder3(d4, e2_fused)
        d2, att2 = self.decoder2(d3, e1_fused)
        d1, att1 = self.decoder1(d2, x0)

        # Final output
        out = self.final_conv(d1)
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)

        if self.training:
            # Deep supervision outputs
            ds4 = F.interpolate(self.ds_conv4(d4), size=input_size, mode='bilinear', align_corners=False)
            ds3 = F.interpolate(self.ds_conv3(d3), size=input_size, mode='bilinear', align_corners=False)
            ds2 = F.interpolate(self.ds_conv2(d2), size=input_size, mode='bilinear', align_corners=False)
            return out, [ds4, ds3, ds2]

        if return_attention:
            return out, [att1, att2, att3, att4]

        return out


def build_kan_acnet(config):
    """
    Build KAN-ACNet model from config
    """
    return KANACNet(
        in_channels=config.get('in_channels', 3),
        num_classes=config.get('num_classes', 1),
        base_channels=config.get('base_channels', 64),
        use_pretrained_backbone=config.get('use_pretrained_backbone', True),
        use_texture_pathway=config.get('use_texture_pathway', True),
        backbone=config.get('backbone', 'resnet34')
    )
