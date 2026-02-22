"""
KAN-Enhanced Adaptive Context Network (KAN-ACNet)
Main architecture + built-in inference wrapper

Quick test:
    from kan_acnet import KANACNet, visualize

    kan  = KANACNet("model.pth")
    mask = kan("test.jpg")
    visualize("test.jpg", mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models

# ── try to import KAN modules (graceful fallback for standalone testing) ──────
try:
    from .kan_modules import (
        KANIlluminationModulationModule,
        KANBoundaryAttentionModule,
        KANTexturePathway,
        KANFusionModule,
    )
except ImportError:
    try:
        from kan_modules import (
            KANIlluminationModulationModule,
            KANBoundaryAttentionModule,
            KANTexturePathway,
            KANFusionModule,
        )
    except ImportError:
        raise ImportError(
            "Cannot find kan_modules.py. "
            "Make sure it lives in the same folder as kan_acnet.py."
        )

# ── ImageNet normalisation ────────────────────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.224, 0.225, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helpers  (no extra file needed)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess(image_path: str, size: int = 512):
    """Load an image and return a normalised (1,3,H,W) tensor + original PIL image."""
    img     = Image.open(image_path).convert("RGB")
    resized = img.resize((size, size), Image.BILINEAR)
    arr     = (np.array(resized, dtype=np.float32) / 255.0 - _MEAN) / _STD
    tensor  = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor, img          # tensor,  original-size PIL image


def _postprocess(logits: torch.Tensor,
                 original_size: tuple,
                 threshold: float = 0.5) -> np.ndarray:
    """Convert raw logits → binary uint8 mask (0/255) at original resolution."""
    prob = torch.sigmoid(logits)
    prob = F.interpolate(
        prob,
        size=(original_size[1], original_size[0]),   # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    return (prob.squeeze().cpu().numpy() >= threshold).astype(np.uint8) * 255


def visualize(image_path: str,
              mask: np.ndarray,
              alpha: float = 0.45,
              color: tuple = (0.0, 1.0, 0.4)) -> None:
    """
    Overlay the predicted mask on the original image and display it.

    Args:
        image_path : path to the original image
        mask       : uint8 numpy array (0/255) returned by KANACNet.__call__
        alpha      : mask overlay transparency  (0 = invisible, 1 = opaque)
        color      : RGB tuple for the overlay colour  (default: bright green)
    """
    img  = np.array(Image.open(image_path).convert("RGB"))
    mask_bool = mask > 0

    overlay = img.copy().astype(np.float32)
    for c, val in enumerate(color):
        overlay[..., c] = np.where(
            mask_bool,
            overlay[..., c] * (1 - alpha) + val * 255 * alpha,
            overlay[..., c],
        )
    overlay = overlay.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(img);                axes[0].set_title("Original Image")
    axes[1].imshow(mask, cmap="gray");  axes[1].set_title("Predicted Mask")
    axes[2].imshow(overlay);            axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Architecture blocks
# ─────────────────────────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Encoder block with optional KAN-IMM."""
    def __init__(self, in_channels, out_channels, use_kan_imm=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
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
    """Decoder block with optional KAN-BAM."""
    def __init__(self, in_channels, skip_channels, out_channels, use_kan_bam=True):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1  = nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1    = nn.BatchNorm2d(out_channels)
        self.conv2  = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2    = nn.BatchNorm2d(out_channels)
        self.relu   = nn.ReLU(inplace=True)
        self.use_kan_bam = use_kan_bam
        if use_kan_bam:
            self.kan_bam = KANBoundaryAttentionModule(out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_kan_bam:
            x, attention = self.kan_bam(x)
            return x, attention
        return x, None


# ─────────────────────────────────────────────────────────────────────────────
# Main model  (supports KANACNet("model.pth") and kan("image.jpg"))
# ─────────────────────────────────────────────────────────────────────────────

class KANACNet(nn.Module):
    """
    KAN-Enhanced Adaptive Context Network.

    Can be instantiated in two ways:

        # --- standard PyTorch way ---
        model = KANACNet(backbone="resnet34")
        model.load_state_dict(torch.load("model.pth"))

        # --- quick-test shortcut (pass checkpoint path as first arg) ---
        kan  = KANACNet("model.pth")
        mask = kan("test.jpg")          # returns numpy uint8 mask
        visualize("test.jpg", mask)
    """

    # ── constructor ───────────────────────────────────────────────────────────
    def __init__(
        self,
        checkpoint: str | None = None,   # path to .pth  OR  None
        *,
        in_channels: int  = 3,
        num_classes: int  = 1,
        base_channels: int = 64,
        use_pretrained_backbone: bool = True,
        use_texture_pathway: bool = True,
        backbone: str = "resnet34",
        encoder_kan_blocks=None,
        decoder_kan_blocks=None,
        input_size: int = 512,
        threshold: float = 0.5,
        device: str | None = None,
    ):
        super().__init__()

        # ── inference settings (only used when calling with an image path) ───
        self._input_size = input_size
        self._threshold  = threshold
        self._device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ── ablation config ──────────────────────────────────────────────────
        self.encoder_kan_blocks = encoder_kan_blocks if encoder_kan_blocks is not None else [1, 2, 3, 4]
        self.decoder_kan_blocks = decoder_kan_blocks if decoder_kan_blocks is not None else [1, 2, 3, 4]
        self.use_texture_pathway = use_texture_pathway
        self.num_classes = num_classes
        self.backbone_type = backbone

        # ── encoder backbone ─────────────────────────────────────────────────
        _CONVNEXT_CHANNELS = {
            "convnext_tiny":  [96,  192, 384, 768,  768],
            "convnext_small": [96,  192, 384, 768,  768],
            "convnext_base":  [128, 256, 512, 1024, 1024],
            "convnext_large": [192, 384, 768, 1536, 1536],
        }

        if backbone == "resnet34":
            resnet = models.resnet34(pretrained=use_pretrained_backbone)
            self.encoder_channels = [64, 64, 128, 256, 512]
            self._init_resnet_encoder(resnet)
        elif backbone == "resnet50":
            resnet = models.resnet50(pretrained=use_pretrained_backbone)
            self.encoder_channels = [64, 256, 512, 1024, 2048]
            self._init_resnet_encoder(resnet)
        elif backbone in _CONVNEXT_CHANNELS:
            self.encoder_channels = _CONVNEXT_CHANNELS[backbone]
            weights  = "DEFAULT" if use_pretrained_backbone else None
            convnext = getattr(models, backbone)(weights=weights)
            feats    = convnext.features
            self.init_conv = nn.Sequential(feats[0], feats[1])
            self.encoder1  = nn.Sequential(feats[2], feats[3])
            self.encoder2  = nn.Sequential(feats[4], feats[5])
            self.encoder3  = nn.Sequential(feats[6])
            self.encoder4  = nn.Sequential(feats[7])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # ── KAN-IMM after each encoder stage ─────────────────────────────────
        self.kan_imm1 = KANIlluminationModulationModule(self.encoder_channels[1]) if 1 in self.encoder_kan_blocks else None
        self.kan_imm2 = KANIlluminationModulationModule(self.encoder_channels[2]) if 2 in self.encoder_kan_blocks else None
        self.kan_imm3 = KANIlluminationModulationModule(self.encoder_channels[3]) if 3 in self.encoder_kan_blocks else None
        self.kan_imm4 = KANIlluminationModulationModule(self.encoder_channels[4]) if 4 in self.encoder_kan_blocks else None

        # ── Texture pathway ───────────────────────────────────────────────────
        if use_texture_pathway:
            self.texture_path1 = KANTexturePathway(self.encoder_channels[1], self.encoder_channels[1])
            self.texture_path2 = KANTexturePathway(self.encoder_channels[2], self.encoder_channels[2])
            self.texture_path3 = KANTexturePathway(self.encoder_channels[3], self.encoder_channels[3])
            self.fusion1 = KANFusionModule([self.encoder_channels[1], self.encoder_channels[1]], self.encoder_channels[1])
            self.fusion2 = KANFusionModule([self.encoder_channels[2], self.encoder_channels[2]], self.encoder_channels[2])
            self.fusion3 = KANFusionModule([self.encoder_channels[3], self.encoder_channels[3]], self.encoder_channels[3])

        # ── Bridge ────────────────────────────────────────────────────────────
        self.bridge = nn.Sequential(
            nn.Conv2d(self.encoder_channels[4], self.encoder_channels[4], 3, padding=1),
            nn.BatchNorm2d(self.encoder_channels[4]),
            nn.ReLU(inplace=True),
        )

        # ── Decoder with KAN-BAM ──────────────────────────────────────────────
        self.decoder4 = DecoderBlock(self.encoder_channels[4], self.encoder_channels[3], self.encoder_channels[3], use_kan_bam=4 in self.decoder_kan_blocks)
        self.decoder3 = DecoderBlock(self.encoder_channels[3], self.encoder_channels[2], self.encoder_channels[2], use_kan_bam=3 in self.decoder_kan_blocks)
        self.decoder2 = DecoderBlock(self.encoder_channels[2], self.encoder_channels[1], self.encoder_channels[1], use_kan_bam=2 in self.decoder_kan_blocks)
        self.decoder1 = DecoderBlock(self.encoder_channels[1], self.encoder_channels[0], base_channels,             use_kan_bam=1 in self.decoder_kan_blocks)

        # ── Segmentation head ─────────────────────────────────────────────────
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, num_classes, 1),
        )

        # ── Deep supervision heads ────────────────────────────────────────────
        self.ds_conv4 = nn.Conv2d(self.encoder_channels[3], num_classes, 1)
        self.ds_conv3 = nn.Conv2d(self.encoder_channels[2], num_classes, 1)
        self.ds_conv2 = nn.Conv2d(self.encoder_channels[1], num_classes, 1)

        # ── Load checkpoint if a path was supplied ────────────────────────────
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location=self._device)
            # support raw state-dict or {"model": state_dict} / {"state_dict": …}
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self.load_state_dict(state)
            print(f"✓ Loaded weights from '{checkpoint}'")

        self.to(self._device)
        self.eval()

    # ── ResNet encoder init ───────────────────────────────────────────────────
    def _init_resnet_encoder(self, resnet):
        self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.encoder1  = nn.Sequential(resnet.maxpool, resnet.layer1)
        self.encoder2  = resnet.layer2
        self.encoder3  = resnet.layer3
        self.encoder4  = resnet.layer4

    # ── __call__ overload: accepts image path OR tensor ───────────────────────
    def __call__(self, x, **kwargs):
        """
        Accepts either:
          - a file path (str)  → runs full preprocessing + returns numpy mask
          - a torch.Tensor     → standard nn.Module forward (returns logits)
        """
        if isinstance(x, str):
            return self._predict_from_path(x)
        return super().__call__(x, **kwargs)    # normal nn.Module path

    def _predict_from_path(self, image_path: str) -> np.ndarray:
        """Full pipeline: image path → uint8 numpy mask."""
        tensor, original_img = _preprocess(image_path, size=self._input_size)
        tensor = tensor.to(self._device)
        with torch.no_grad():
            logits = self.forward(tensor)
        return _postprocess(logits, original_img.size, threshold=self._threshold)

    # ── standard forward ──────────────────────────────────────────────────────
    def forward(self, x, return_attention=False):
        input_size = x.shape[2:]

        x0 = self.init_conv(x)

        e1 = self.encoder1(x0)
        if self.kan_imm1: e1 = self.kan_imm1(e1)

        e2 = self.encoder2(e1)
        if self.kan_imm2: e2 = self.kan_imm2(e2)

        e3 = self.encoder3(e2)
        if self.kan_imm3: e3 = self.kan_imm3(e3)

        e4 = self.encoder4(e3)
        if self.kan_imm4: e4 = self.kan_imm4(e4)

        if self.use_texture_pathway:
            e1 = self.fusion1([e1, self.texture_path1(e1)])
            e2 = self.fusion2([e2, self.texture_path2(e2)])
            e3 = self.fusion3([e3, self.texture_path3(e3)])

        bridge = self.bridge(e4)

        d4, att4 = self.decoder4(bridge, e3)
        d3, att3 = self.decoder3(d4,     e2)
        d2, att2 = self.decoder2(d3,     e1)
        d1, att1 = self.decoder1(d2,     x0)

        out = self.final_conv(d1)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        if self.training:
            ds4 = F.interpolate(self.ds_conv4(d4), size=input_size, mode="bilinear", align_corners=False)
            ds3 = F.interpolate(self.ds_conv3(d3), size=input_size, mode="bilinear", align_corners=False)
            ds2 = F.interpolate(self.ds_conv2(d2), size=input_size, mode="bilinear", align_corners=False)
            return out, [ds4, ds3, ds2]

        if return_attention:
            return out, [att1, att2, att3, att4]

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Config-based builder helper
# ─────────────────────────────────────────────────────────────────────────────

def build_kan_acnet(config: dict) -> KANACNet:
    """Build KANACNet from a config dictionary (no checkpoint loading)."""
    return KANACNet(
        checkpoint              = config.get("checkpoint"),
        in_channels             = config.get("in_channels", 3),
        num_classes             = config.get("num_classes", 1),
        base_channels           = config.get("base_channels", 64),
        use_pretrained_backbone = config.get("use_pretrained_backbone", True),
        use_texture_pathway     = config.get("use_texture_pathway", True),
        backbone                = config.get("backbone", "resnet34"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test  (run file directly: python kan_acnet.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        checkpoint_path, image_path = sys.argv[1], sys.argv[2]
        kan  = KANACNet(checkpoint_path)
        mask = kan(image_path)
        visualize(image_path, mask)
    else:
        print("Usage: python kan_acnet.py model.pth test.jpg")
