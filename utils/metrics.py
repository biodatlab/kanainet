"""
Evaluation metrics for segmentation

Includes standard metrics:
- mIoU (mean Intersection over Union)
- mDice (mean Dice score)
- S-measure (Sα) - Structure measure
- Weighted F-measure (Fβ)
- MAE (Mean Absolute Error)
"""
import torch
import numpy as np
from scipy import ndimage


class SegmentationMetrics:
    """
    Compute segmentation metrics: mIoU, mDice, S-measure, weighted F-measure, MAE, HD95, ASD
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        # For per-image metrics
        self.dice_scores = []
        self.iou_scores = []
        self.s_measure_scores = []
        self.weighted_f_scores = []
        self.mae_scores = []
        self.hd95_scores = []
        self.asd_scores = []
        # Store predictions and targets for detailed metrics
        self.all_preds = []
        self.all_targets = []

    def update(self, pred, target):
        """
        Update metrics with batch predictions
        pred: [B, 1, H, W] (logits or probs)
        target: [B, 1, H, W] (binary 0/1)
        """
        pred = torch.sigmoid(pred) if pred.max() > 1 else pred
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()

        pred_binary = (pred > self.threshold).float()
        pred_binary_np = pred_binary.cpu().numpy()

        # Update global TP, FP, TN, FN
        pred_flat = pred_binary_np.reshape(-1)
        target_flat = target_np.reshape(-1)

        self.tp += np.sum((pred_flat == 1) & (target_flat == 1))
        self.fp += np.sum((pred_flat == 1) & (target_flat == 0))
        self.tn += np.sum((pred_flat == 0) & (target_flat == 0))
        self.fn += np.sum((pred_flat == 0) & (target_flat == 1))

        # Compute per-image metrics for mean calculation
        batch_size = pred.shape[0]
        for i in range(batch_size):
            pred_i = pred_np[i, 0]  # [H, W]
            target_i = target_np[i, 0]  # [H, W]
            pred_binary_i = pred_binary_np[i, 0]  # [H, W]

            # Dice score
            dice = self._compute_dice(pred_binary_i, target_i)
            self.dice_scores.append(dice)

            # IoU score
            iou = self._compute_iou(pred_binary_i, target_i)
            self.iou_scores.append(iou)

            # S-measure
            s_measure = self._compute_s_measure(pred_i, target_i)
            self.s_measure_scores.append(s_measure)

            # Weighted F-measure
            weighted_f = self._compute_weighted_f_measure(pred_binary_i, target_i)
            self.weighted_f_scores.append(weighted_f)

            # MAE (use binary predictions for consistency)
            mae = self._compute_mae(pred_binary_i, target_i)
            self.mae_scores.append(mae)

            # HD95 and ASD (use binary masks)
            hd95, asd = self._compute_hd95_asd(pred_binary_i, target_i)
            self.hd95_scores.append(hd95)
            self.asd_scores.append(asd)

    def _compute_dice(self, pred, target, smooth=1e-7):
        """Compute Dice coefficient for a single image"""
        intersection = np.sum(pred * target)
        return (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

    def _compute_iou(self, pred, target, smooth=1e-7):
        """Compute IoU for a single image"""
        intersection = np.sum(pred * target)
        union = np.sum(pred) + np.sum(target) - intersection
        return (intersection + smooth) / (union + smooth)

    def _compute_mae(self, pred, target):
        """Compute Mean Absolute Error"""
        return np.mean(np.abs(pred - target))

    def _compute_hd95_asd(self, pred, target):
        """
        Compute HD95 (95th percentile Hausdorff Distance) and ASD (Average Surface Distance).
        Uses boundary extraction via morphological erosion and distance transforms.
        Returns (hd95, asd) in pixels. Returns (0, 0) if both masks are empty,
        or (max_dim, max_dim) if only one mask is empty.
        """
        pred_bool = pred.astype(bool)
        target_bool = target.astype(bool)

        # Handle edge cases
        if not np.any(pred_bool) and not np.any(target_bool):
            return 0.0, 0.0
        if not np.any(pred_bool) or not np.any(target_bool):
            # Penalize: use image diagonal as max distance
            h, w = pred.shape
            max_dist = np.sqrt(h**2 + w**2)
            return max_dist, max_dist

        # Extract boundaries using erosion
        from scipy.ndimage import binary_erosion
        struct = ndimage.generate_binary_structure(2, 1)
        pred_boundary = pred_bool ^ binary_erosion(pred_bool, structure=struct)
        target_boundary = target_bool ^ binary_erosion(target_bool, structure=struct)

        # If erosion removed everything (single-pixel regions), use the mask itself
        if not np.any(pred_boundary):
            pred_boundary = pred_bool
        if not np.any(target_boundary):
            target_boundary = target_bool

        # Distance transform from each boundary
        dt_pred = ndimage.distance_transform_edt(~pred_boundary)
        dt_target = ndimage.distance_transform_edt(~target_boundary)

        # Surface distances: distance from each target boundary pixel to nearest pred boundary pixel, and vice versa
        dist_target_to_pred = dt_pred[target_boundary]
        dist_pred_to_target = dt_target[pred_boundary]

        all_distances = np.concatenate([dist_target_to_pred, dist_pred_to_target])

        hd95 = np.percentile(all_distances, 95)
        asd = np.mean(all_distances)

        return float(hd95), float(asd)

    def _compute_s_measure(self, pred, target, alpha=0.5):
        """
        Compute S-measure (Structure measure)
        Reference: Fan et al. "Structure-measure: A New Way to Evaluate Foreground Maps" (ICCV 2017)

        S = alpha * S_o + (1 - alpha) * S_r
        where S_o is object-aware structural similarity and S_r is region-aware structural similarity
        """
        target_mean = np.mean(target)

        # If target is empty or full, handle edge cases
        if target_mean == 0:  # No foreground
            # S-measure for empty GT is based on how much pred is also empty
            return 1.0 - np.mean(pred)
        elif target_mean == 1:  # All foreground
            return np.mean(pred)

        # Object-aware structural similarity
        s_object = self._s_object(pred, target)

        # Region-aware structural similarity
        s_region = self._s_region(pred, target)

        # Combined S-measure
        s_measure = alpha * s_object + (1 - alpha) * s_region

        return s_measure

    def _s_object(self, pred, target):
        """Object-aware structural similarity"""
        # Foreground similarity
        pred_fg = pred * target
        target_fg = target

        if np.sum(target_fg) == 0:
            o_fg = 0
        else:
            o_fg = 2 * np.mean(pred_fg) / (np.mean(pred_fg) + np.mean(target_fg) + 1e-7)

        # Background similarity
        pred_bg = (1 - pred) * (1 - target)
        target_bg = 1 - target

        if np.sum(target_bg) == 0:
            o_bg = 0
        else:
            o_bg = 2 * np.mean(pred_bg) / (np.mean(pred_bg) + np.mean(target_bg) + 1e-7)

        # Weighted combination
        u = np.mean(target)
        return u * o_fg + (1 - u) * o_bg

    def _s_region(self, pred, target):
        """Region-aware structural similarity using SSIM-like approach"""
        x, y = self._centroid(target)
        h, w = target.shape

        # Divide into 4 regions based on centroid
        if x == 0 or y == 0 or x == w or y == h:
            return self._ssim(pred, target)

        # Four quadrants
        gt1 = target[:y, :x]
        gt2 = target[:y, x:]
        gt3 = target[y:, :x]
        gt4 = target[y:, x:]

        pred1 = pred[:y, :x]
        pred2 = pred[:y, x:]
        pred3 = pred[y:, :x]
        pred4 = pred[y:, x:]

        # Compute SSIM for each region
        w1 = gt1.size / target.size
        w2 = gt2.size / target.size
        w3 = gt3.size / target.size
        w4 = gt4.size / target.size

        s1 = self._ssim(pred1, gt1) if gt1.size > 0 else 0
        s2 = self._ssim(pred2, gt2) if gt2.size > 0 else 0
        s3 = self._ssim(pred3, gt3) if gt3.size > 0 else 0
        s4 = self._ssim(pred4, gt4) if gt4.size > 0 else 0

        return w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4

    def _centroid(self, target):
        """Compute centroid of the target mask"""
        if np.sum(target) == 0:
            h, w = target.shape
            return w // 2, h // 2

        rows = np.any(target, axis=1)
        cols = np.any(target, axis=0)

        if not np.any(rows) or not np.any(cols):
            h, w = target.shape
            return w // 2, h // 2

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return (x_min + x_max) // 2, (y_min + y_max) // 2

    def _ssim(self, pred, target, eps=1e-7):
        """Simplified structural similarity"""
        pred_mean = np.mean(pred)
        target_mean = np.mean(target)

        pred_std = np.std(pred)
        target_std = np.std(target)

        cov = np.mean((pred - pred_mean) * (target - target_mean))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        ssim = ((2 * pred_mean * target_mean + c1) * (2 * cov + c2)) / \
               ((pred_mean ** 2 + target_mean ** 2 + c1) * (pred_std ** 2 + target_std ** 2 + c2) + eps)

        return np.clip(ssim, 0, 1)

    def _compute_weighted_f_measure(self, pred, target, beta_sq=1.0):
        """
        Compute weighted F-measure (Fβ^w)
        Reference: Margolin et al. "How to Evaluate Foreground Maps" (CVPR 2014)

        Uses distance-based weighting where pixels closer to the boundary are weighted more
        """
        if np.sum(target) == 0:
            if np.sum(pred) == 0:
                return 1.0
            else:
                return 0.0

        if np.sum(pred) == 0:
            return 0.0

        # Compute distance transform for weighting
        target_bool = target.astype(bool)
        dist = ndimage.distance_transform_edt(~target_bool)
        dist = 1.0 / (1.0 + dist)  # Weight: closer to boundary = higher weight

        # Normalize weights
        weights = dist / (np.sum(dist) + 1e-7)

        # Weighted precision and recall
        tp_w = np.sum(weights * pred * target)
        fp_w = np.sum(weights * pred * (1 - target))
        fn_w = np.sum(weights * (1 - pred) * target)

        precision_w = tp_w / (tp_w + fp_w + 1e-7)
        recall_w = tp_w / (tp_w + fn_w + 1e-7)

        # Weighted F-measure
        f_measure_w = ((1 + beta_sq) * precision_w * recall_w) / \
                      (beta_sq * precision_w + recall_w + 1e-7)

        return f_measure_w

    def get_metrics(self):
        """
        Compute all metrics
        Returns both global metrics and mean per-image metrics
        """
        epsilon = 1e-7

        # Global pixel-level metrics
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + epsilon)
        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou_global = self.tp / (self.tp + self.fp + self.fn + epsilon)
        specificity = self.tn / (self.tn + self.fp + epsilon)

        # Mean per-image metrics
        m_dice = np.mean(self.dice_scores) if self.dice_scores else 0.0
        m_iou = np.mean(self.iou_scores) if self.iou_scores else 0.0
        s_measure = np.mean(self.s_measure_scores) if self.s_measure_scores else 0.0
        weighted_f = np.mean(self.weighted_f_scores) if self.weighted_f_scores else 0.0
        mae = np.mean(self.mae_scores) if self.mae_scores else 0.0
        hd95 = np.mean(self.hd95_scores) if self.hd95_scores else 0.0
        asd = np.mean(self.asd_scores) if self.asd_scores else 0.0

        return {
            # Legacy metrics (for backwards compatibility)
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            # Primary metrics (new standard metrics)
            'dice': m_dice,           # mDice (mean Dice)
            'iou': m_iou,             # mIoU (mean IoU)
            'm_dice': m_dice,         # Explicit mDice
            'm_iou': m_iou,           # Explicit mIoU
            's_measure': s_measure,   # S-measure (Sα)
            'weighted_f': weighted_f, # Weighted F-measure (Fβ^w)
            'mae': mae,               # Mean Absolute Error
            'hd95': hd95,             # 95th percentile Hausdorff Distance
            'asd': asd,               # Average Surface Distance
            # Global metrics (computed over all pixels)
            'global_dice': f1,
            'global_iou': iou_global,
        }

    def get_dice(self):
        """Get mean Dice coefficient"""
        if self.dice_scores:
            return np.mean(self.dice_scores)
        epsilon = 1e-7
        return 2 * self.tp / (2 * self.tp + self.fp + self.fn + epsilon)

    def get_iou(self):
        """Get mean IoU"""
        if self.iou_scores:
            return np.mean(self.iou_scores)
        epsilon = 1e-7
        return self.tp / (self.tp + self.fp + self.fn + epsilon)


def dice_coefficient(pred, target, threshold=0.5, smooth=1.0):
    """
    Compute Dice coefficient for a single prediction
    """
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()


def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    Compute IoU score for a single prediction
    """
    pred = torch.sigmoid(pred) if pred.max() > 1 else pred
    pred = (pred > threshold).float()

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def hausdorff_distance(pred, target, threshold=0.5):
    """
    Compute Hausdorff distance (simplified version)
    """
    try:
        from scipy.spatial.distance import directed_hausdorff

        pred = torch.sigmoid(pred) if pred.max() > 1 else pred
        pred = (pred > threshold).float()

        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()

        # Get boundary points
        pred_points = np.argwhere(pred_np > 0)
        target_points = np.argwhere(target_np > 0)

        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')

        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]

        return max(hd1, hd2)
    except ImportError:
        return None


def calculate_metrics(pred, target, smooth=1e-7):
    """
    Calculate basic metrics for threshold tuning (used by train.py)
    pred: [B, 1, H, W] binary predictions
    target: [B, 1, H, W] binary targets
    """
    pred_np = pred.cpu().numpy().reshape(-1)
    target_np = target.cpu().numpy().reshape(-1)

    intersection = np.sum(pred_np * target_np)
    dice = (2. * intersection + smooth) / (np.sum(pred_np) + np.sum(target_np) + smooth)
    iou = (intersection + smooth) / (np.sum(pred_np) + np.sum(target_np) - intersection + smooth)

    return {'dice': dice, 'iou': iou}
