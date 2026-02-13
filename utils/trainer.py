"""
Training utilities and trainer class
"""
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from .metrics import SegmentationMetrics, dice_coefficient, iou_score
from .losses import DiceFocalLoss, DeepSupervisionLoss


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device='cuda',
        save_dir='./checkpoints',
        log_dir='./logs',
        use_deep_supervision=True,
        use_amp=True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.use_deep_supervision = use_deep_supervision
        self.use_amp = use_amp and (device == 'cuda' or str(device).startswith('cuda'))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)

        # Best metrics
        self.best_dice = 0.0
        self.best_iou = 0.0

        # Move model to device
        self.model.to(device)

    def train_epoch(self, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        train_loss = 0.0
        train_metrics = SegmentationMetrics()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            with torch.autocast(device_type='cuda', enabled=self.use_amp):
                if self.use_deep_supervision and self.model.training:
                    outputs, ds_outputs = self.model(images)
                    all_outputs = [outputs] + ds_outputs
                    loss = self.criterion(all_outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update metrics
            train_loss += loss.item()
            train_metrics.update(outputs.detach(), masks.detach())

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': train_metrics.get_dice()
            })

        # Average loss
        avg_loss = train_loss / len(self.train_loader)
        metrics = train_metrics.get_metrics()

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, epoch):
        """
        Validate the model
        """
        self.model.eval()
        val_loss = 0.0
        val_metrics = SegmentationMetrics()

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]')
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = self.criterion(outputs, masks)

            # Update metrics
            val_loss += loss.item()
            val_metrics.update(outputs, masks)

            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'dice': val_metrics.get_dice()
            })

        # Average loss
        avg_loss = val_loss / len(self.val_loader)
        metrics = val_metrics.get_metrics()

        return avg_loss, metrics

    def train(self, num_epochs, early_stopping_patience=None, start_epoch=1):
        """
        Full training loop
        """
        print(f"Starting training for {num_epochs} epochs (from epoch {start_epoch})...")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print("=" * 60)

        best_epoch = 0
        patience_counter = 0

        for epoch in range(start_epoch, num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self.validate(epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Epoch time
            epoch_time = time.time() - epoch_start_time

            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('mDice/train', train_metrics['dice'], epoch)
            self.writer.add_scalar('mDice/val', val_metrics['dice'], epoch)
            self.writer.add_scalar('mIoU/train', train_metrics['iou'], epoch)
            self.writer.add_scalar('mIoU/val', val_metrics['iou'], epoch)
            self.writer.add_scalar('S-measure/val', val_metrics.get('s_measure', 0), epoch)
            self.writer.add_scalar('Weighted-F/val', val_metrics.get('weighted_f', 0), epoch)
            self.writer.add_scalar('MAE/val', val_metrics.get('mae', 0), epoch)
            self.writer.add_scalar('LR', current_lr, epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train mDice: {train_metrics['dice']:.4f} | Val mDice: {val_metrics['dice']:.4f}")
            print(f"  Train mIoU:  {train_metrics['iou']:.4f} | Val mIoU:  {val_metrics['iou']:.4f}")
            print(f"  Val S-measure: {val_metrics.get('s_measure', 0):.4f} | Val Weighted-F: {val_metrics.get('weighted_f', 0):.4f} | Val MAE: {val_metrics.get('mae', 0):.4f}")
            print(f"  LR: {current_lr:.6f}")

            # Save best model
            if val_metrics['dice'] > self.best_dice:
                self.best_dice = val_metrics['dice']
                self.best_iou = val_metrics['iou']
                best_epoch = epoch
                patience_counter = 0

                self.save_checkpoint(
                    epoch,
                    val_loss,
                    val_metrics,
                    is_best=True
                )
                print(f"  ✓ Best model saved! (Dice: {self.best_dice:.4f})")
            else:
                patience_counter += 1

            # Save regular checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, val_loss, val_metrics, is_best=False)

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best Dice: {self.best_dice:.4f} at epoch {best_epoch}")
                break

            print("=" * 60)

        print("\nTraining completed!")
        print(f"Best mDice: {self.best_dice:.4f} | Best mIoU: {self.best_iou:.4f}")
        print(f"Best epoch: {best_epoch}")

        self.writer.close()

    @torch.no_grad()
    def final_validation_inference(self, threshold=0.5):
        """
        Perform final inference on validation set with the best model
        Reports all metrics: mIoU, mDice, S-measure, Weighted F-measure, MAE
        """
        self.model.eval()
        final_metrics = SegmentationMetrics(threshold=threshold)

        print("\n" + "=" * 60)
        print("FINAL VALIDATION INFERENCE")
        print(f"Threshold: {threshold}")
        print("=" * 60)

        pbar = tqdm(self.val_loader, desc='Final Validation')
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            final_metrics.update(outputs, masks)

        metrics = final_metrics.get_metrics()

        print("\nFinal Validation Results:")
        print("-" * 60)
        print(f"  mDice:        {metrics['dice']:.4f}")
        print(f"  mIoU:         {metrics['iou']:.4f}")
        print(f"  S-measure:    {metrics['s_measure']:.4f}")
        print(f"  Weighted-F:   {metrics['weighted_f']:.4f}")
        print(f"  MAE:          {metrics['mae']:.4f}")
        print("-" * 60)
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  Specificity:  {metrics['specificity']:.4f}")
        print("=" * 60)

        return metrics

    def save_checkpoint(self, epoch, loss, metrics, is_best=False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'dice': metrics['dice'],
            'iou': metrics['iou'],
            'metrics': metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.save_dir, 'best_model.pth')
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')

        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore best metrics so early stopping / best-model logic works
        if 'dice' in checkpoint:
            self.best_dice = checkpoint['dice']
        if 'iou' in checkpoint:
            self.best_iou = checkpoint['iou']

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Dice: {checkpoint['dice']:.4f}")
        print(f"  IoU: {checkpoint['iou']:.4f}")

        return checkpoint['epoch']
