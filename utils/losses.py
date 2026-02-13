"""
Loss functions for medical image segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W] (logits or probs)
        target: [B, 1, H, W] (binary 0/1)
        """
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class IoULoss(nn.Module):
    """
    IoU (Jaccard) Loss
    """
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection

        iou = (intersection + self.smooth) / (union + self.smooth)

        return 1 - iou


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W] (logits)
        target: [B, 1, H, W] (binary 0/1)
        """
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_prob = torch.sigmoid(pred)

        # Focal term
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        focal_loss = focal_weight * bce

        return focal_loss.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary Loss - emphasizes boundary pixels
    """
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, pred, target):
        """
        pred: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        # Compute boundary using morphological operations
        kernel = torch.ones(1, 1, 3, 3).to(target.device)

        # Erosion
        target_eroded = F.conv2d(target, kernel, padding=1)
        target_eroded = (target_eroded == 9).float()  # All neighbors are 1

        # Boundary = original - eroded
        boundary = target - target_eroded

        # Weighted BCE on boundary pixels
        pred_sigmoid = torch.sigmoid(pred)
        boundary_loss = F.binary_cross_entropy(
            pred_sigmoid * boundary,
            target * boundary,
            reduction='sum'
        ) / (boundary.sum() + 1e-8)

        return boundary_loss

class DiceFocalLoss(nn.Module):
    """
    Simple two-loss approach: Dice + Focal
    Proven effective for imbalanced segmentation
    """
    def __init__(self, dice_weight=1.0, focal_weight=0.5, alpha=0.25, gamma=2.0):
        super(DiceFocalLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss for multi-scale outputs
    """
    def __init__(self, base_loss, weights=[1.0, 0.5, 0.25, 0.125]):
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, outputs, target):
        """
        outputs: list of [B, 1, H, W] predictions at different scales
        target: [B, 1, H, W]
        """
        if not isinstance(outputs, (list, tuple)):
            return self.base_loss(outputs, target)

        total_loss = 0.0
        for i, (output, weight) in enumerate(zip(outputs, self.weights)):
            total_loss += weight * self.base_loss(output, target)

        return total_loss