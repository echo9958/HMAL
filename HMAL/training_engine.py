"""
Training Engine for RGB-IR Pedestrian Detection.
Implements end-to-end training pipeline with advanced optimization strategies.
"""

import argparse
import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Import custom modules
from models.hmal import HMALDetector
from core.pedestrian_detector import PedestrianDetector
from core.data_preprocessing import create_pedestrian_dataloader
from utils.loss import ComputeLoss
from core.evaluation_metrics import PedestrianDetectionMetrics
from utils.torch_utils import ModelEMA, select_device

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrainingCoordinator:
    """
    Coordinates the entire training process including data loading, optimization,
    validation, and checkpointing.
    """

    def __init__(self, config: Dict, device: torch.device):
        """
        Initialize training coordinator.

        Args:
            config: Configuration dictionary
            device: PyTorch device
        """
        self.config = config
        self.device = device
        self.start_epoch = 0
        self.best_metric = 0.0

        # Setup directories
        self.setup_directories()

        # Initialize model
        self.model = self.build_model()

        # Initialize optimizer and scheduler
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()

        # Initialize EMA for stable training
        self.ema = ModelEMA(self.model) if config.get('use_ema', True) else None

        # Initialize loss function
        self.criterion = self.build_criterion()

        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=config.get('use_amp', True))

        # Initialize metrics
        self.train_metrics = PedestrianDetectionMetrics(num_classes=config['num_classes'])
        self.val_metrics = PedestrianDetectionMetrics(num_classes=config['num_classes'])

        # Initialize tensorboard
        self.writer = SummaryWriter(log_dir=self.output_dir / 'logs')

        # Training state
        self.global_step = 0
        self.epoch_losses = []

    def setup_directories(self):
        """Create necessary directories for outputs."""
        self.output_dir = Path(self.config['output_dir'])
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.visualization_dir = self.output_dir / 'visualizations'

        for directory in [self.output_dir, self.checkpoint_dir, self.visualization_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

    def build_model(self) -> nn.Module:
        """Build detection model based on configuration."""
        model_type = self.config.get('model_type', 'hmal')

        if model_type == 'hmal':
            model = HMALDetector(
                nc=self.config['num_classes'],
                anchors=self.config.get('anchors')
            )
        elif model_type == 'pedestrian':
            model = PedestrianDetector(
                num_classes=self.config['num_classes'],
                fusion_channels=self.config.get('fusion_channels', 256)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model.to(self.device)

        # Load pretrained weights if specified
        if self.config.get('pretrained_weights'):
            self.load_pretrained_weights(model, self.config['pretrained_weights'])

        # Enable DataParallel for multi-GPU
        if torch.cuda.device_count() > 1 and self.config.get('multi_gpu', True):
            logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
            model = nn.DataParallel(model)

        return model

    def build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with parameter groups."""
        # Separate parameters into groups for different learning rates
        param_groups = self.create_parameter_groups()

        optimizer_type = self.config.get('optimizer', 'adamw').lower()

        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=self.config['learning_rate'],
                betas=(self.config.get('momentum', 0.9), 0.999),
                weight_decay=self.config.get('weight_decay', 0.0001)
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=self.config['learning_rate'],
                momentum=self.config.get('momentum', 0.9),
                weight_decay=self.config.get('weight_decay', 0.0001),
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def create_parameter_groups(self) -> list:
        """Create parameter groups for different learning rates."""
        backbone_params = []
        head_params = []
        bn_params = []

        model = self.model.module if hasattr(self.model, 'module') else self.model

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if 'bn' in name.lower() or 'batch_norm' in name.lower():
                bn_params.append(param)
            elif 'detect' in name.lower() or 'head' in name.lower():
                head_params.append(param)
            else:
                backbone_params.append(param)

        param_groups = [
            {'params': backbone_params, 'lr': self.config['learning_rate']},
            {'params': head_params, 'lr': self.config['learning_rate'] * 2.0},
            {'params': bn_params, 'lr': self.config['learning_rate'], 'weight_decay': 0.0}
        ]

        logger.info(f"Parameter groups: Backbone={len(backbone_params)}, "
                   f"Head={len(head_params)}, BN={len(bn_params)}")

        return param_groups

    def build_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Build learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine').lower()

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['num_epochs'],
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.get('lr_milestones', [60, 80]),
                gamma=self.config.get('lr_gamma', 0.1)
            )
        elif scheduler_type == 'onecycle':
            scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['learning_rate'],
                total_steps=self.config['num_epochs'],
                pct_start=0.3,
                anneal_strategy='cos'
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return scheduler

    def build_criterion(self) -> nn.Module:
        """Build loss function."""
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return ComputeLoss(model)

    def load_pretrained_weights(self, model: nn.Module, weights_path: str):
        """Load pretrained weights into model."""
        if not os.path.exists(weights_path):
            logger.warning(f"Pretrained weights not found: {weights_path}")
            return

        checkpoint = torch.load(weights_path, map_location=self.device)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Handle DataParallel wrapper
        if hasattr(model, 'module'):
            model = model.module

        # Load state dict with strict=False to allow partial loading
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded pretrained weights from: {weights_path}")

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']}")

        for batch_idx, (images, targets, paths, _) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device)

            # Split RGB and IR channels
            rgb_images = images[:, :3, :, :]
            ir_images = images[:, 3:, :, :]

            # Normalize images
            rgb_images = rgb_images.float() / 255.0
            ir_images = ir_images.float() / 255.0

            # Forward pass with mixed precision
            with autocast(enabled=self.config.get('use_amp', True)):
                predictions = self.model(rgb_images, ir_images)
                loss, loss_components = self.criterion(predictions, targets)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Update EMA
            if self.ema is not None:
                self.ema.update(self.model)

            # Accumulate loss
            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # Log to tensorboard
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for key, value in loss_components.items():
                    self.writer.add_scalar(f'train/{key}', value, self.global_step)

            self.global_step += 1

        avg_loss = epoch_loss / num_batches

        return {
            'loss': avg_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

    def validate(self, val_loader, epoch: int) -> Dict[str, float]:
        """
        Validate model on validation set.

        Args:
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()

        val_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for images, targets, paths, _ in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device)

                # Split RGB and IR
                rgb_images = images[:, :3, :, :].float() / 255.0
                ir_images = images[:, 3:, :, :].float() / 255.0

                # Forward pass
                if self.ema is not None:
                    predictions = self.ema.ema(rgb_images, ir_images)
                else:
                    predictions = self.model(rgb_images, ir_images)

                # Compute loss
                loss, _ = self.criterion(predictions, targets)
                val_loss += loss.item()

                # Accumulate predictions for metrics
                # (Implementation depends on model output format)

        avg_val_loss = val_loss / num_batches

        # Compute metrics
        metrics = self.val_metrics.evaluate()
        metrics['val_loss'] = avg_val_loss

        # Log metrics
        for key, value in metrics.items():
            self.writer.add_scalar(f'val/{key}', value, epoch)

        logger.info(f"Validation - Loss: {avg_val_loss:.4f}, mAP: {metrics.get('mAP', 0):.4f}")

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.ema.state_dict()

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with mAP: {metrics.get('mAP', 0):.4f}")

        # Save periodic checkpoint
        if epoch % self.config.get('save_interval', 10) == 0:
            periodic_path = self.checkpoint_dir / f'epoch_{epoch}.pt'
            torch.save(checkpoint, periodic_path)

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Configuration: {self.config}")

        # Load data
        train_loader = create_pedestrian_dataloader(
            rgb_root=self.config['train_rgb_path'],
            thermal_root=self.config['train_ir_path'],
            annotation_file=self.config['train_annotations'],
            batch_size=self.config['batch_size'],
            img_size=(self.config['img_size'], self.config['img_size']),
            augment=True,
            num_workers=self.config.get('num_workers', 8)
        )

        val_loader = create_pedestrian_dataloader(
            rgb_root=self.config['val_rgb_path'],
            thermal_root=self.config['val_ir_path'],
            annotation_file=self.config['val_annotations'],
            batch_size=self.config['batch_size'] * 2,
            img_size=(self.config['img_size'], self.config['img_size']),
            augment=False,
            num_workers=self.config.get('num_workers', 8),
            shuffle=False
        )

        # Training loop
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Train one epoch
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader, epoch)

            # Update learning rate
            self.scheduler.step()

            # Check if best model
            current_metric = val_metrics.get('mAP', 0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Log epoch summary
            logger.info(f"Epoch {epoch} Summary:")
            logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Val mAP: {val_metrics.get('mAP', 0):.4f}")
            logger.info(f"  Best mAP: {self.best_metric:.4f}")

        logger.info("Training completed!")
        self.writer.close()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RGB-IR Pedestrian Detector')

    # Data arguments
    parser.add_argument('--train-rgb', type=str, required=True, help='Path to training RGB images')
    parser.add_argument('--train-ir', type=str, required=True, help='Path to training IR images')
    parser.add_argument('--val-rgb', type=str, required=True, help='Path to validation RGB images')
    parser.add_argument('--val-ir', type=str, required=True, help='Path to validation IR images')
    parser.add_argument('--train-ann', type=str, required=True, help='Training annotations file')
    parser.add_argument('--val-ann', type=str, required=True, help='Validation annotations file')

    # Model arguments
    parser.add_argument('--model', type=str, default='hmal', choices=['hmal', 'pedestrian'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=1, help='Number of classes')
    parser.add_argument('--pretrained', type=str, default='', help='Path to pretrained weights')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'onecycle'])

    # System arguments
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of dataloader workers')
    parser.add_argument('--output-dir', type=str, default='./runs/train', help='Output directory')
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use-ema', action='store_true', help='Use exponential moving average')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Create configuration dictionary
    config = {
        'train_rgb_path': args.train_rgb,
        'train_ir_path': args.train_ir,
        'val_rgb_path': args.val_rgb,
        'val_ir_path': args.val_ir,
        'train_annotations': args.train_ann,
        'val_annotations': args.val_ann,
        'model_type': args.model,
        'num_classes': args.num_classes,
        'pretrained_weights': args.pretrained,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
        'num_workers': args.num_workers,
        'output_dir': args.output_dir,
        'use_amp': args.use_amp,
        'use_ema': args.use_ema,
        'grad_clip': 10.0,
        'log_interval': 10,
        'save_interval': 10
    }

    # Setup device
    device = select_device(args.device)

    # Initialize training coordinator
    coordinator = TrainingCoordinator(config, device)

    # Start training
    coordinator.train()


if __name__ == '__main__':
    main()
