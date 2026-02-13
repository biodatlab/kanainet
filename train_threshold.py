"""
Training + Threshold tuning script for KAN-ACNet on Kvasir-SEG dataset
Ablation Study: Experimenting with different numbers of KAN blocks in encoders and decoders
Includes threshold optimization after training each model.
"""

import os
import time
import torch
import torch.optim as optim
import random
import numpy as np
import json
from datetime import datetime

from config import Config
from models.kan_acnet import KANACNet
from dataset.kvasir_dataset import create_data_loaders
from utils.losses import DiceFocalLoss, DeepSupervisionLoss
from utils.trainer import Trainer
from utils.metrics import SegmentationMetrics


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Optimizer / Scheduler
# ============================================================
_BACKBONE_MODULES = {'init_conv', 'encoder1', 'encoder2', 'encoder3', 'encoder4'}


def get_param_groups(model, config):
    """
    Build parameter groups for the optimizer.

    For ConvNext backbones this enforces two best-practices:
      1. No weight-decay on 1-D parameters (bias, LayerNorm weight/bias).
      2. A 10× lower learning-rate for the pretrained backbone so the
         newly-added KAN decoder does not destabilise the pre-trained weights.

    For ResNet backbones a flat parameter list is returned (no change to
    existing behaviour).
    """
    if not getattr(config, 'backbone', '').startswith('convnext'):
        return model.parameters()

    backbone_lr = config.learning_rate * 0.1

    decay_backbone, no_decay_backbone = [], []
    decay_head, no_decay_head = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        top = name.split('.')[0]
        is_backbone = top in _BACKBONE_MODULES
        # 1-D params (bias, LN weight, LN bias) → no weight-decay
        no_wd = param.ndim == 1
        if is_backbone:
            (no_decay_backbone if no_wd else decay_backbone).append(param)
        else:
            (no_decay_head if no_wd else decay_head).append(param)

    return [
        {'params': decay_backbone,    'lr': backbone_lr,          'weight_decay': config.weight_decay},
        {'params': no_decay_backbone, 'lr': backbone_lr,          'weight_decay': 0.0},
        {'params': decay_head,        'lr': config.learning_rate, 'weight_decay': config.weight_decay},
        {'params': no_decay_head,     'lr': config.learning_rate, 'weight_decay': 0.0},
    ]


def build_optimizer(model, config):
    params = get_param_groups(model, config)
    if config.optimizer == 'adam':
        return optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adamw':
        return optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'sgd':
        return optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def build_scheduler(optimizer, config, num_epochs):
    if config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
    elif config.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif config.scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    return None


# ============================================================
# Ablation configurations
# ============================================================
def generate_ablation_configs():
    """Generate all ablation configurations for the study."""
    configs = []

    # Baseline
    configs.append({
        'name': 'baseline_no_kan',
        'encoder_kan_blocks': [],
        'decoder_kan_blocks': [],
        'description': 'Baseline without any KAN blocks'
    })

    # Encoder only
    for n in range(1, 5):
        configs.append({
            'name': f'encoder_only_{n}_blocks',
            'encoder_kan_blocks': list(range(1, n + 1)),
            'decoder_kan_blocks': [],
            'description': f'Only {n} encoder KAN blocks'
        })

    # Decoder only
    for n in range(1, 5):
        configs.append({
            'name': f'decoder_only_{n}_blocks',
            'encoder_kan_blocks': [],
            'decoder_kan_blocks': list(range(1, n + 1)),
            'description': f'Only {n} decoder KAN blocks'
        })

    # Symmetric
    for n in range(1, 5):
        configs.append({
            'name': f'symmetric_{n}_blocks',
            'encoder_kan_blocks': list(range(1, n + 1)),
            'decoder_kan_blocks': list(range(1, n + 1)),
            'description': f'{n} KAN blocks in encoder and decoder'
        })

    # Asymmetric combinations
    asymmetric = [
        ('enc1_dec2', [1], [1, 2]),
        ('enc2_dec1', [1, 2], [1]),
        ('enc3_dec1', [1, 2, 3], [1]),
        ('enc3_dec2', [1, 2, 3], [1, 2]),
        ('enc4_dec2', [1, 2, 3, 4], [1, 2]),
        ('enc4_dec3', [1, 2, 3, 4], [1, 2, 3]),
    ]
    for name, enc, dec in asymmetric:
        configs.append({
            'name': name,
            'encoder_kan_blocks': enc,
            'decoder_kan_blocks': dec,
            'description': f'Encoder {enc} / Decoder {dec}'
        })

    # Full model
    configs.append({
        'name': 'full_model_all_blocks',
        'encoder_kan_blocks': [1, 2, 3, 4],
        'decoder_kan_blocks': [1, 2, 3, 4],
        'description': 'Full model with all KAN blocks'
    })

    return configs


# ============================================================
# Threshold tuning
# ============================================================
def find_optimal_threshold(model, val_loader, device, thresholds=None):
    """
    Find the optimal threshold for binary segmentation.
    Reports: mIoU, mDice, S-measure (Sα), Weighted F-measure (Fβ^w), MAE
    """
    if thresholds is None:
        thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    model.eval()
    print("\n" + "-" * 100)
    print("Finding optimal threshold with full metrics")
    print("-" * 100)
    print(f"{'Thr':<6} {'mDice':<10} {'mIoU':<10} {'S-meas':<10} {'W-F':<10} {'MAE':<10} {'Prec':<10} {'Recall':<10}")
    print("-" * 100)

    best_threshold = 0.5
    best_dice = 0.0
    best_metrics = None
    threshold_results = []

    with torch.no_grad():
        for threshold in thresholds:
            meter = SegmentationMetrics(threshold=threshold)

            for batch in val_loader:
                images = batch[0].to(device)
                masks = batch[1].to(device)

                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                meter.update(outputs, masks)

            metrics = meter.get_metrics()

            threshold_results.append({
                'threshold': threshold,
                'dice': metrics['dice'],
                'iou': metrics['iou'],
                's_measure': metrics['s_measure'],
                'weighted_f': metrics['weighted_f'],
                'mae': metrics['mae'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'specificity': metrics['specificity']
            })

            print(
                f"{threshold:<6.2f} {metrics['dice']:<10.4f} {metrics['iou']:<10.4f} "
                f"{metrics['s_measure']:<10.4f} {metrics['weighted_f']:<10.4f} {metrics['mae']:<10.4f} "
                f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f}"
            )

            if metrics['dice'] > best_dice:
                best_dice = metrics['dice']
                best_threshold = threshold
                best_metrics = metrics

    print("-" * 100)
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Best mDice: {best_dice:.4f}, mIoU: {best_metrics['iou']:.4f}")
    print(f"S-measure: {best_metrics['s_measure']:.4f}, Weighted-F: {best_metrics['weighted_f']:.4f}, MAE: {best_metrics['mae']:.4f}")
    print("-" * 100)

    return {
        'best_threshold': best_threshold,
        'best_dice': best_dice,
        'best_iou': best_metrics['iou'],
        'best_s_measure': best_metrics['s_measure'],
        'best_weighted_f': best_metrics['weighted_f'],
        'best_mae': best_metrics['mae'],
        'all_thresholds': threshold_results
    }


# ============================================================
# Train + Evaluate one configuration
# ============================================================
def train_and_evaluate_config(ablation_config, device, train_loader, val_loader, save_base_dir, log_base_dir):
    """Train a model and then perform threshold tuning."""

    print("\n" + "=" * 80)
    print(f"Training: {ablation_config['name']}")
    print(ablation_config['description'])
    print("=" * 80)

    # Build model
    model = KANACNet(
        in_channels=Config.in_channels,
        num_classes=Config.num_classes,
        base_channels=Config.base_channels,
        use_pretrained_backbone=Config.use_pretrained_backbone,
        use_texture_pathway=Config.use_texture_pathway,
        backbone=Config.backbone,
        encoder_kan_blocks=ablation_config['encoder_kan_blocks'],
        decoder_kan_blocks=ablation_config['decoder_kan_blocks']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Loss function: Dice + Focal
    base_loss = DiceFocalLoss(
        dice_weight=Config.dice_weight,
        focal_weight=Config.focal_weight,
    )
    criterion = DeepSupervisionLoss(base_loss) if Config.use_deep_supervision else base_loss

    # Optimizer and scheduler
    optimizer = build_optimizer(model, Config)
    scheduler = build_scheduler(optimizer, Config, Config.num_epochs)

    # Directories
    save_dir = os.path.join(save_base_dir, ablation_config['name'])
    log_dir = os.path.join(log_base_dir, ablation_config['name'])

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        log_dir=log_dir,
        use_deep_supervision=Config.use_deep_supervision
    )

    # Train
    start_time = time.time()
    trainer.train(Config.num_epochs, Config.early_stopping_patience)
    training_time = time.time() - start_time

    # Load best checkpoint for threshold tuning
    best_checkpoint_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_checkpoint_path):
        print(f"\nLoading best checkpoint for threshold tuning: {best_checkpoint_path}")
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

    # Find optimal threshold
    threshold_results = find_optimal_threshold(model, val_loader, device)

    # Final inference with optimal threshold
    print("\n" + "=" * 80)
    print("FINAL VALIDATION INFERENCE (with optimal threshold)")
    print("=" * 80)
    final_meter = SegmentationMetrics(threshold=threshold_results['best_threshold'])

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(device)
            masks = batch[1].to(device)
            outputs = model(images)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            final_meter.update(outputs, masks)

    final_metrics = final_meter.get_metrics()

    print("\nFinal Validation Results (Optimal Threshold):")
    print("-" * 60)
    print(f"  mDice:        {final_metrics['dice']:.4f}")
    print(f"  mIoU:         {final_metrics['iou']:.4f}")
    print(f"  S-measure:    {final_metrics['s_measure']:.4f}")
    print(f"  Weighted-F:   {final_metrics['weighted_f']:.4f}")
    print(f"  MAE:          {final_metrics['mae']:.4f}")
    print("-" * 60)

    # Convert threshold tuning results to native Python types for JSON serialization
    threshold_tuning_serializable = []
    for th_result in threshold_results['all_thresholds']:
        threshold_tuning_serializable.append({
            k: float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v
            for k, v in th_result.items()
        })

    # Collect results (convert numpy types to Python native types for JSON serialization)
    result = {
        'config_name': ablation_config['name'],
        'description': ablation_config['description'],
        'encoder_kan_blocks': ablation_config['encoder_kan_blocks'],
        'decoder_kan_blocks': ablation_config['decoder_kan_blocks'],
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'training_best_dice': float(trainer.best_dice),
        'training_best_iou': float(trainer.best_iou),
        'best_threshold': float(threshold_results['best_threshold']),
        # Metrics with optimal threshold
        'dice_with_optimal_threshold': float(threshold_results['best_dice']),
        'iou_with_optimal_threshold': float(threshold_results['best_iou']),
        's_measure_with_optimal_threshold': float(threshold_results['best_s_measure']),
        'weighted_f_with_optimal_threshold': float(threshold_results['best_weighted_f']),
        'mae_with_optimal_threshold': float(threshold_results['best_mae']),
        # Final validation metrics
        'final_m_dice': float(final_metrics['dice']),
        'final_m_iou': float(final_metrics['iou']),
        'final_s_measure': float(final_metrics['s_measure']),
        'final_weighted_f': float(final_metrics['weighted_f']),
        'final_mae': float(final_metrics['mae']),
        'final_precision': float(final_metrics['precision']),
        'final_recall': float(final_metrics['recall']),
        'final_specificity': float(final_metrics['specificity']),
        'threshold_tuning_results': threshold_tuning_serializable,
        'training_time_seconds': float(training_time)
    }

    return result


# ============================================================
# Main
# ============================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train KAN-ACNet models with threshold tuning')
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints/ablation_with_threshold',
                        help='Base directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str,
                        default='./logs/ablation_with_threshold',
                        help='Base directory for training logs')
    parser.add_argument('--output', type=str,
                        default='./ablation_results_with_threshold.json',
                        help='Output JSON file for results')
    args = parser.parse_args()

    Config.print_config()
    set_seed(Config.seed)

    device = torch.device(Config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Save directory: {args.save_dir}")
    print(f"Log directory: {args.log_dir}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        root_dir=Config.data_root,
        batch_size=Config.batch_size,
        image_size=Config.image_size,
        val_split=Config.val_split,
        num_workers=Config.num_workers,
        seed=Config.seed
    )

    ablation_configs = generate_ablation_configs()
    results = []

    print(f"\nTotal configurations to train: {len(ablation_configs)}")
    print("=" * 80)

    for i, cfg in enumerate(ablation_configs, 1):
        print(f"\n{'#'*80}")
        print(f"# Experiment {i}/{len(ablation_configs)}: {cfg['name']}")
        print(f"{'#'*80}")

        result = train_and_evaluate_config(
            cfg, device, train_loader, val_loader,
            args.save_dir, args.log_dir
        )
        results.append(result)

        # Save intermediate results after each model
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nIntermediate results saved to: {args.output}")

    # Print final summary
    print("\n" + "=" * 120)
    print("ABLATION STUDY SUMMARY (with Threshold Tuning)")
    print("Metrics: mDice, mIoU, S-measure (Sα), Weighted F-measure (Fβ^w), MAE")
    print("=" * 120)

    print(f"\n{'Model':<25} {'Params':>10} {'Thr':>6} {'mDice':>8} {'mIoU':>8} {'S-meas':>8} {'W-F':>8} {'MAE':>8}")
    print("-" * 120)
    for r in sorted(results, key=lambda x: x.get('final_m_dice', x['dice_with_optimal_threshold']), reverse=True):
        print(f"{r['config_name']:<25} {r['total_params']:>10,} {r['best_threshold']:>6.2f} "
              f"{r.get('final_m_dice', r['dice_with_optimal_threshold']):>8.4f} "
              f"{r.get('final_m_iou', r['iou_with_optimal_threshold']):>8.4f} "
              f"{r.get('final_s_measure', 0):>8.4f} "
              f"{r.get('final_weighted_f', 0):>8.4f} "
              f"{r.get('final_mae', 0):>8.4f}")

    print("-" * 120)
    print(f"\nFinal results saved to: {args.output}")
    print("Ablation study with threshold tuning completed.")


if __name__ == '__main__':
    main()
