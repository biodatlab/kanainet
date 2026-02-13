"""
Inference script for evaluating KAN-ACNet on unseen external validation dataset
Evaluates with: mIoU, mDice, Sα (S-measure), Fβ^w (Weighted F-measure), MAE, HD95, ASD

Supports single checkpoint or ablation study evaluation.
Uses unseen external validation data from ./data_unseen
"""

import os
import argparse
import torch
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from config import Config
from models.kan_acnet import KANACNet
from dataset.kvasir_dataset import get_val_transform
from utils.metrics import SegmentationMetrics
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Default paths
DEFAULT_ABLATION_RESULTS = './ablation_threshold_miccai.json'
DEFAULT_CHECKPOINTS_DIR = './checkpoints/convnext_ablation_miccai'
DEFAULT_SINGLE_CHECKPOINT = '/home/badboy-005/Desktop/uab-projects/KAN_models/checkpoints/convnext_ablation_miccai_0902_focal_loss/symmetric_4_blocks/best_model.pth'
DEFAULT_DATA_ROOT = './data_unseen'


class ExternalValidationDataset(Dataset):
    """
    Dataset for loading external validation images.
    Expects 'images' and 'masks' subdirectories.
    """
    def __init__(
        self,
        dataset_dir,
        image_size=(256, 256),
        transform=None
    ):
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.mask_dir = os.path.join(dataset_dir, 'masks')
        self.image_size = image_size
        self.transform = transform
        self.dataset_name = os.path.basename(dataset_dir)

        # Verify directories exist
        if not os.path.exists(self.image_dir):
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not os.path.exists(self.mask_dir):
            raise ValueError(f"Mask directory not found: {self.mask_dir}")

        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'))
        ])

        print(f"  {self.dataset_name}: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files)

    def _get_mask_name(self, img_name):
        """
        Get corresponding mask filename for an image.
        Handles different naming conventions:
        - kvasir-seg: same name (image.jpg -> image.jpg)
        - polypsgen: data_C1_images_* -> data_C1_masks_*_mask
        """
        base_name, ext = os.path.splitext(img_name)

        # List of possible mask naming patterns
        mask_candidates = [
            # Same name (kvasir-seg style)
            img_name,
            base_name + '.png',
            base_name + '.jpg',
            # PolypGen pattern: replace 'images' with 'masks' and add '_mask' suffix
            base_name.replace('_images_', '_masks_') + '_mask' + ext,
            base_name.replace('_images_', '_masks_') + '_mask.png',
            base_name.replace('_images_', '_masks_') + '_mask.jpg',
            # General _mask suffix
            base_name + '_mask' + ext,
            base_name + '_mask.png',
            base_name + '_mask.jpg',
        ]

        for mask_name in mask_candidates:
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                return mask_name

        return None

    def __getitem__(self, idx):
        import cv2

        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find and load mask
        mask_name = self._get_mask_name(img_name)
        if mask_name is None:
            raise ValueError(f"Could not find mask for: {img_name}")

        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        # Binarize mask
        mask = (mask > 127).astype(np.uint8)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.float()
        else:
            image = cv2.resize(image, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask, f"{self.dataset_name}/{img_name}"


def get_external_validation_datasets(data_root='./data_unseen'):
    """
    Find datasets with images and masks folders.
    Returns list of dataset directories.

    Handles two cases:
    1. data_root contains subdirectories with images/masks (e.g., data_unseen/CVC-ColonDB)
    2. data_root itself has images/masks folders (e.g., data_unseen/data_C6)
    """
    data_path = Path(data_root)
    datasets = []

    # Case 1: Check if data_root itself has images and masks folders
    if (data_path / 'images').exists() and (data_path / 'masks').exists():
        datasets.append(str(data_path))
        print(f"Found dataset directly in: {data_path}")
        return datasets

    # Case 2: Search for subdirectories with images and masks folders
    for item in sorted(data_path.iterdir()):
        if item.is_dir():
            images_dir = item / 'images'
            masks_dir = item / 'masks'
            if images_dir.exists() and masks_dir.exists():
                datasets.append(str(item))

    return datasets


def create_dataloaders(
    data_root='./data_unseen',
    batch_size=8,
    image_size=(256, 256),
    num_workers=4
):
    """
    Create dataloaders for all external validation datasets.
    Returns combined loader and individual loaders per dataset.
    """
    transform = get_val_transform(image_size)
    dataset_dirs = get_external_validation_datasets(data_root)

    if not dataset_dirs:
        raise ValueError(f"No datasets with images/masks folders found in {data_root}")

    print(f"\nFound {len(dataset_dirs)} external validation datasets:")

    all_datasets = []
    individual_loaders = {}

    for dataset_dir in dataset_dirs:
        ds = ExternalValidationDataset(
            dataset_dir=dataset_dir,
            image_size=image_size,
            transform=transform
        )
        all_datasets.append(ds)

        # Create individual loader for per-dataset evaluation
        individual_loaders[ds.dataset_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    # Combined dataset
    combined_dataset = ConcatDataset(all_datasets)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\nTotal images: {len(combined_dataset)}")

    return combined_loader, individual_loaders


def load_all_configurations(ablation_results_path, checkpoints_dir, include_baseline=True):
    """
    Load all configurations from ablation study results.
    Returns list of all model configurations with their best models.
    """
    print(f"\nLoading ablation results from: {ablation_results_path}")

    with open(ablation_results_path, 'r') as f:
        results = json.load(f)

    if not results:
        raise ValueError("No results found in ablation results file")

    print(f"\nFound {len(results)} trained configurations")

    all_configs = []

    # Check if baseline_no_kan is already in ablation results
    baseline_in_results = any(r.get('config_name') == 'baseline_no_kan' for r in results)

    # Add baseline_no_kan if requested, exists, and not already in results
    if include_baseline and not baseline_in_results:
        baseline_checkpoint_path = os.path.join(checkpoints_dir, 'baseline_no_kan', 'best_model.pth')
        if os.path.exists(baseline_checkpoint_path):
            import torch
            # Load baseline checkpoint to get metrics
            baseline_ckpt = torch.load(baseline_checkpoint_path, map_location='cpu', weights_only=False)
            baseline_metrics = baseline_ckpt.get('metrics', {})

            baseline_config = {
                'checkpoint_path': baseline_checkpoint_path,
                'config_name': 'baseline_no_kan',
                'description': 'Baseline ACNet without KAN blocks',
                'best_threshold': 0.4,  # Default threshold for baseline
                'encoder_kan_blocks': [],
                'decoder_kan_blocks': [],
                'val_metrics': {
                    'dice': float(baseline_metrics.get('dice', 0)) if baseline_metrics.get('dice') is not None else None,
                    'iou': float(baseline_metrics.get('iou', 0)) if baseline_metrics.get('iou') is not None else None,
                    's_measure': float(baseline_metrics.get('s_measure', 0)) if baseline_metrics.get('s_measure') is not None else None,
                    'weighted_f': float(baseline_metrics.get('weighted_f', 0)) if baseline_metrics.get('weighted_f') is not None else None,
                    'mae': float(baseline_metrics.get('mae', 0)) if baseline_metrics.get('mae') is not None else None,
                },
                'test_metrics': {
                    'dice': float(baseline_metrics.get('dice', 0)) if baseline_metrics.get('dice') is not None else None,
                    'iou': float(baseline_metrics.get('iou', 0)) if baseline_metrics.get('iou') is not None else None,
                    's_measure': float(baseline_metrics.get('s_measure', 0)) if baseline_metrics.get('s_measure') is not None else None,
                    'weighted_f': float(baseline_metrics.get('weighted_f', 0)) if baseline_metrics.get('weighted_f') is not None else None,
                    'mae': float(baseline_metrics.get('mae', 0)) if baseline_metrics.get('mae') is not None else None,
                }
            }
            all_configs.append(baseline_config)
            print(f"Added baseline_no_kan configuration")
        else:
            print(f"Warning: Baseline checkpoint not found at {baseline_checkpoint_path}")

    for result in results:
        # Construct checkpoint path
        checkpoint_path = os.path.join(checkpoints_dir, result['config_name'], 'best_model.pth')

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found for {result['config_name']}: {checkpoint_path}")
            continue

        config_info = {
            'checkpoint_path': checkpoint_path,
            'config_name': result['config_name'],
            'description': result.get('description', ''),
            'best_threshold': result.get('best_threshold', 0.4),
            'encoder_kan_blocks': result.get('encoder_kan_blocks', []),
            'decoder_kan_blocks': result.get('decoder_kan_blocks', []),
            'val_metrics': {
                'dice': result.get('val_dice'),
                'iou': result.get('val_iou'),
                's_measure': result.get('val_s_measure'),
                'weighted_f': result.get('val_weighted_f'),
                'mae': result.get('val_mae'),
            },
            'test_metrics': {
                'dice': result.get('test_dice'),
                'iou': result.get('test_iou'),
                's_measure': result.get('test_s_measure'),
                'weighted_f': result.get('test_weighted_f'),
                'mae': result.get('test_mae'),
            }
        }
        all_configs.append(config_info)

    print(f"Loaded {len(all_configs)} valid configurations")
    return all_configs


def infer_kan_blocks_from_state_dict(state_dict):
    """
    Infer encoder and decoder KAN blocks from state_dict keys.
    Returns (encoder_kan_blocks, decoder_kan_blocks)
    """
    # Check for encoder KAN blocks (kan_imm{i}.kan_modulator.spline_weight)
    encoder_kan_blocks = []
    for i in range(1, 5):
        if f'kan_imm{i}.kan_modulator.spline_weight' in state_dict:
            encoder_kan_blocks.append(i)

    # Check for decoder KAN blocks
    decoder_kan_blocks = []
    for i in range(1, 5):
        if f'decoder{i}.kan_bam.kan_edge_enhance.kan_transform.spline_weight' in state_dict:
            decoder_kan_blocks.append(i)

    return encoder_kan_blocks, decoder_kan_blocks


def load_model(checkpoint_path, device, encoder_kan_blocks=None, decoder_kan_blocks=None):
    """
    Load a trained KAN-ACNet model from checkpoint.
    """
    print(f"\nLoading model from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use provided blocks or try to get from checkpoint
    if encoder_kan_blocks is None:
        encoder_kan_blocks = checkpoint.get('encoder_kan_blocks')
    if decoder_kan_blocks is None:
        decoder_kan_blocks = checkpoint.get('decoder_kan_blocks')

    # If still None, infer from state_dict
    if encoder_kan_blocks is None or decoder_kan_blocks is None:
        inferred_enc, inferred_dec = infer_kan_blocks_from_state_dict(checkpoint['model_state_dict'])
        if encoder_kan_blocks is None:
            encoder_kan_blocks = inferred_enc
        if decoder_kan_blocks is None:
            decoder_kan_blocks = inferred_dec

    model = KANACNet(
        in_channels=Config.in_channels,
        num_classes=Config.num_classes,
        base_channels=Config.base_channels,
        use_pretrained_backbone=Config.use_pretrained_backbone,
        use_texture_pathway=Config.use_texture_pathway,
        backbone=Config.backbone,
        encoder_kan_blocks=encoder_kan_blocks,
        decoder_kan_blocks=decoder_kan_blocks
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")
    print(f"Encoder KAN blocks: {encoder_kan_blocks}")
    print(f"Decoder KAN blocks: {decoder_kan_blocks}")

    return model


def evaluate_dataset(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on a dataset and return metrics.
    """
    model.eval()
    meter = SegmentationMetrics(threshold=threshold)

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            masks = batch[1].to(device)

            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            meter.update(outputs, masks)

    return meter.get_metrics()


def print_metrics_table(results, title="Evaluation Results"):
    """Print metrics in a formatted table."""
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)
    print(f"{'Dataset':<25} {'mDice':>10} {'mIoU':>10} {'Sα':>10} {'Fβw':>10} {'MAE':>10} {'HD95':>10} {'ASD':>10}")
    print("-" * 130)

    for dataset_name, metrics in results.items():
        print(f"{dataset_name:<25} {metrics['dice']:>10.4f} {metrics['iou']:>10.4f} "
              f"{metrics['s_measure']:>10.4f} {metrics['weighted_f']:>10.4f} {metrics['mae']:>10.4f} "
              f"{metrics.get('hd95', 0):>10.2f} {metrics.get('asd', 0):>10.2f}")

    print("-" * 130)


def main():
    parser = argparse.ArgumentParser(description='Inference on unseen external validation dataset with all KAN-ACNet configurations')
    parser.add_argument('--ablation_results', type=str, default=DEFAULT_ABLATION_RESULTS,
                        help='Path to ablation results JSON file')
    parser.add_argument('--checkpoints_dir', type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help='Directory containing model checkpoints')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_SINGLE_CHECKPOINT,
                        help='Path to single checkpoint file (overrides ablation_results)')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory containing unseen external validation datasets')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for inference')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (auto-generated if not specified)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Override threshold (default: use from checkpoint or 0.5)')
    parser.add_argument('--include_baseline', action='store_true', default=True,
                        help='Include baseline_no_kan model in evaluation')
    parser.add_argument('--no_baseline', dest='include_baseline', action='store_false',
                        help='Exclude baseline_no_kan model from evaluation')
    args = parser.parse_args()

    # Check if single checkpoint mode
    if args.checkpoint and os.path.exists(args.checkpoint):
        device = torch.device(Config.device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        print(f"\nSingle checkpoint mode: {args.checkpoint}")

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        encoder_blocks = checkpoint.get('encoder_kan_blocks')
        decoder_blocks = checkpoint.get('decoder_kan_blocks')
        checkpoint_metrics = checkpoint.get('metrics', {})
        threshold = args.threshold if args.threshold is not None else checkpoint_metrics.get('threshold', 0.4)

        # Infer blocks from state_dict if not in checkpoint
        if encoder_blocks is None or decoder_blocks is None:
            inferred_enc, inferred_dec = infer_kan_blocks_from_state_dict(checkpoint['model_state_dict'])
            if encoder_blocks is None:
                encoder_blocks = inferred_enc
            if decoder_blocks is None:
                decoder_blocks = inferred_dec

        # Build model
        model = KANACNet(
            in_channels=Config.in_channels,
            num_classes=Config.num_classes,
            base_channels=Config.base_channels,
            use_pretrained_backbone=Config.use_pretrained_backbone,
            use_texture_pathway=Config.use_texture_pathway,
            backbone=Config.backbone,
            encoder_kan_blocks=encoder_blocks,
            decoder_kan_blocks=decoder_blocks
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded: {total_params:,} parameters")
        print(f"Encoder KAN blocks: {encoder_blocks}")
        print(f"Decoder KAN blocks: {decoder_blocks}")
        print(f"Using threshold: {threshold}")

        # Create dataloaders
        combined_loader, individual_loaders = create_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            image_size=Config.image_size,
            num_workers=Config.num_workers
        )

        # Generate output filename
        if args.output is None:
            checkpoint_dir = os.path.dirname(os.path.dirname(args.checkpoint))
            checkpoint_name = os.path.basename(checkpoint_dir)
            config_name = os.path.basename(os.path.dirname(args.checkpoint))
            os.makedirs('./external_validation_result', exist_ok=True)
            args.output = f'./external_validation_result/{checkpoint_name}_{config_name}_unseen.json'

        # Evaluate
        print(f"\nEvaluating on external validation dataset (threshold={threshold:.2f})")
        per_dataset_results = {}
        for dataset_name, loader in individual_loaders.items():
            metrics = evaluate_dataset(model, loader, device, threshold=threshold)
            per_dataset_results[dataset_name] = {
                'dice': float(metrics['dice']),
                'iou': float(metrics['iou']),
                's_measure': float(metrics['s_measure']),
                'weighted_f': float(metrics['weighted_f']),
                'mae': float(metrics['mae']),
                'hd95': float(metrics['hd95']),
                'asd': float(metrics['asd']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'specificity': float(metrics['specificity']),
                'num_images': len(loader.dataset)
            }

        combined_metrics = evaluate_dataset(model, combined_loader, device, threshold=threshold)

        # Build results
        result = {
            'checkpoint': args.checkpoint,
            'encoder_kan_blocks': encoder_blocks,
            'decoder_kan_blocks': decoder_blocks,
            'threshold': threshold,
            'data_root': args.data_root,
            'timestamp': datetime.now().isoformat(),
            'external_validation': {
                'per_dataset': per_dataset_results,
                'combined': {
                    'dice': float(combined_metrics['dice']),
                    'iou': float(combined_metrics['iou']),
                    's_measure': float(combined_metrics['s_measure']),
                    'weighted_f': float(combined_metrics['weighted_f']),
                    'mae': float(combined_metrics['mae']),
                    'hd95': float(combined_metrics['hd95']),
                    'asd': float(combined_metrics['asd']),
                    'precision': float(combined_metrics['precision']),
                    'recall': float(combined_metrics['recall']),
                    'specificity': float(combined_metrics['specificity']),
                    'num_images': len(combined_loader.dataset)
                }
            }
        }

        # Print results
        print_metrics_table(per_dataset_results, "Per-Dataset Results")
        print(f"\nCombined Results:")
        print(f"  mDice:          {combined_metrics['dice']:.4f}")
        print(f"  mIoU:           {combined_metrics['iou']:.4f}")
        print(f"  S-measure (Sα): {combined_metrics['s_measure']:.4f}")
        print(f"  Weighted-F (Fβw): {combined_metrics['weighted_f']:.4f}")
        print(f"  MAE:            {combined_metrics['mae']:.4f}")
        print(f"  HD95:           {combined_metrics['hd95']:.2f} pixels")
        print(f"  ASD:            {combined_metrics['asd']:.2f} pixels")

        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to: {args.output}")
        return

    # Original ablation mode
    device = torch.device(Config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load all configurations from ablation results
    all_configs = load_all_configurations(
        args.ablation_results,
        args.checkpoints_dir,
        include_baseline=args.include_baseline
    )

    # Create dataloaders (same for all models)
    combined_loader, individual_loaders = create_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=Config.image_size,
        num_workers=Config.num_workers
    )

    # Results for all configurations
    all_results = {
        'data_root': args.data_root,
        'timestamp': datetime.now().isoformat(),
        'configurations': []
    }

    # Evaluate each configuration
    for idx, config_info in enumerate(all_configs, 1):
        print("\n" + "=" * 100)
        print(f"Evaluating Configuration {idx}/{len(all_configs)}: {config_info['config_name']}")
        print(f"Description: {config_info['description']}")
        print("=" * 100)

        # Load model for this configuration
        model = load_model(
            config_info['checkpoint_path'],
            device,
            config_info['encoder_kan_blocks'],
            config_info['decoder_kan_blocks']
        )

        threshold = config_info['best_threshold']
        print(f"Using threshold from training: {threshold}")

        # Print original test set performance
        print(f"\nOriginal test set performance:")
        print(f"  mDice: {config_info['test_metrics']['dice']:.4f}")
        print(f"  mIoU:  {config_info['test_metrics']['iou']:.4f}")
        print(f"  Sα:    {config_info['test_metrics']['s_measure']:.4f}")
        print(f"  Fβw:   {config_info['test_metrics']['weighted_f']:.4f}")
        print(f"  MAE:   {config_info['test_metrics']['mae']:.4f}")

        # Evaluate on external validation data
        print(f"\nEvaluating on unseen external validation dataset (threshold={threshold:.2f})")

        # Evaluate per-dataset
        per_dataset_results = {}
        for dataset_name, loader in individual_loaders.items():
            metrics = evaluate_dataset(model, loader, device, threshold=threshold)
            per_dataset_results[dataset_name] = {
                'dice': float(metrics['dice']),
                'iou': float(metrics['iou']),
                's_measure': float(metrics['s_measure']),
                'weighted_f': float(metrics['weighted_f']),
                'mae': float(metrics['mae']),
                'hd95': float(metrics['hd95']),
                'asd': float(metrics['asd']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'specificity': float(metrics['specificity']),
                'num_images': len(loader.dataset)
            }

        # Evaluate combined
        combined_metrics = evaluate_dataset(model, combined_loader, device, threshold=threshold)

        # Store results for this configuration
        config_results = {
            'config_name': config_info['config_name'],
            'description': config_info['description'],
            'encoder_kan_blocks': config_info['encoder_kan_blocks'],
            'decoder_kan_blocks': config_info['decoder_kan_blocks'],
            'threshold': threshold,
            'original_test_metrics': config_info['test_metrics'],
            'original_val_metrics': config_info['val_metrics'],
            'external_validation': {
                'per_dataset': per_dataset_results,
                'combined': {
                    'dice': float(combined_metrics['dice']),
                    'iou': float(combined_metrics['iou']),
                    's_measure': float(combined_metrics['s_measure']),
                    'weighted_f': float(combined_metrics['weighted_f']),
                    'mae': float(combined_metrics['mae']),
                    'hd95': float(combined_metrics['hd95']),
                    'asd': float(combined_metrics['asd']),
                    'precision': float(combined_metrics['precision']),
                    'recall': float(combined_metrics['recall']),
                    'specificity': float(combined_metrics['specificity']),
                    'num_images': len(combined_loader.dataset)
                }
            }
        }

        all_results['configurations'].append(config_results)

        # Print results for this configuration
        print_metrics_table(per_dataset_results, f"Per-Dataset Results - {config_info['config_name']}")

        print(f"\nCombined Results - {config_info['config_name']}:")
        print(f"  mDice:          {combined_metrics['dice']:.4f}")
        print(f"  mIoU:           {combined_metrics['iou']:.4f}")
        print(f"  S-measure (Sα): {combined_metrics['s_measure']:.4f}")
        print(f"  Weighted-F (Fβw): {combined_metrics['weighted_f']:.4f}")
        print(f"  MAE:            {combined_metrics['mae']:.4f}")
        print(f"  HD95:           {combined_metrics['hd95']:.2f} pixels")
        print(f"  ASD:            {combined_metrics['asd']:.2f} pixels")

        # Clean up model
        del model
        torch.cuda.empty_cache()

    # Print summary comparing all configurations
    print("\n" + "=" * 130)
    print("SUMMARY: Unseen External Validation Results for All Configurations")
    print("=" * 130)
    print(f"{'Configuration':<40} {'mDice':>10} {'mIoU':>10} {'Sα':>10} {'Fβw':>10} {'MAE':>10} {'HD95':>10} {'ASD':>10}")
    print("-" * 130)

    for config_result in all_results['configurations']:
        metrics = config_result['external_validation']['combined']
        print(f"{config_result['config_name']:<40} {metrics['dice']:>10.4f} {metrics['iou']:>10.4f} "
              f"{metrics['s_measure']:>10.4f} {metrics['weighted_f']:>10.4f} {metrics['mae']:>10.4f} "
              f"{metrics['hd95']:>10.2f} {metrics['asd']:>10.2f}")

    print("=" * 130)

    # Find best configuration on external validation
    best_config = max(all_results['configurations'],
                     key=lambda x: x['external_validation']['combined']['dice'])

    print(f"\nBest configuration on unseen external validation: {best_config['config_name']}")
    print(f"  External mDice: {best_config['external_validation']['combined']['dice']:.4f}")
    print(f"  Original test mDice: {best_config['original_test_metrics']['dice']:.4f}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
