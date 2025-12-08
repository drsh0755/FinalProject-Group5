"""
Training script for Temporal Fusion Transformer with AWS A10 GPU optimization.
Includes mixed precision training, DDG-DA integration, and comprehensive logging.
"""
import sys
import os
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
import logging
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import yaml

from models.tft import TFTClassifier, count_parameters
from models.model_wrapper import TFTModelWrapper

from data import (
    StockDataset,
    create_train_val_test_loaders,
    DataLoaderConfig,
    prepare_data_for_dataset,
    train_val_test_split
)
from ddg_da import DDGDADistributionPredictor, DDGDADataAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and compute training/validation metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
        self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """Compute aggregate metrics."""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        # Accuracy
        accuracy = (preds == targets).mean()

        # Precision, Recall, F1
        tp = ((preds == 1) & (targets == 1)).sum()
        fp = ((preds == 1) & (targets == 0)).sum()
        fn = ((preds == 0) & (targets == 1)).sum()
        tn = ((preds == 0) & (targets == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Average loss
        avg_loss = np.mean(self.losses) if self.losses else 0.0

        # Class distribution
        class_distribution = {
            'class_0': (targets == 0).sum() / len(targets) if len(targets) > 0 else 0.0,
            'class_1': (targets == 1).sum() / len(targets) if len(targets) > 0 else 0.0
        }

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'class_distribution': class_distribution
        }


class TFTTrainer:
    """Trainer for Temporal Fusion Transformer with GPU optimization."""

    def __init__(
        self,
        model: TFTClassifier,
        train_loader,
        val_loader,
        test_loader,
        config: Dict,
        device: str = None,
        experiment_name: str = None,
        use_ddg_da: bool = False,
        ddg_da_predictor: Optional[DDGDADistributionPredictor] = None
    ):
        """
        Initialize trainer.

        Args:
            model: TFT model
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            test_loader: Test DataLoader
            config: Training configuration
            device: Device to train on
            experiment_name: Name for logging/checkpoints
            use_ddg_da: Whether to use DDG-DA adaptation
            ddg_da_predictor: DDG-DA predictor instance
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.use_ddg_da = use_ddg_da
        self.ddg_da_predictor = ddg_da_predictor
        self.ddg_da_adapter = None

        if use_ddg_da and ddg_da_predictor:
            self.ddg_da_adapter = DDGDADataAdapter(ddg_da_predictor)

        # Experiment setup
        self.experiment_name = experiment_name or f"tft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = os.path.join(config.get('checkpoint_dir', 'checkpoints'), self.experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Logging
        self.log_dir = os.path.join(config.get('log_dir', 'logs'), self.experiment_name)
        self.writer = SummaryWriter(self.log_dir)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        class_weights = config.get('class_weights', None)
        if class_weights:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Mixed precision training (for A10 GPU)
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)

        # Training state
        self.current_epoch = 0
        self.best_val_metric = -float('inf')
        self.patience_counter = 0

        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        logger.info(f"Trainer initialized for experiment: {self.experiment_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"DDG-DA enabled: {self.use_ddg_da}")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")

        # Enable cudnn optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from config."""
        optimizer_type = self.config.get('optimizer', 'adamw').lower()
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        logger.info(f"Optimizer: {optimizer_type}, LR: {lr}, Weight decay: {weight_decay}")
        return optimizer

    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_type = self.config.get('scheduler', 'cosine').lower()
        epochs = self.config.get('epochs', 100)

        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('scheduler_step_size', 10),
                gamma=self.config.get('scheduler_gamma', 0.5)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None

        logger.info(f"Scheduler: {scheduler_type}")
        return scheduler

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')

        for batch_idx, (features, targets) in enumerate(pbar):
            # Move to device
            features = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in features.items()
            }
            targets = targets.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            with autocast(enabled=self.use_amp):
                outputs = self.model(features)
                logits = outputs['logits']
                loss = self.criterion(logits, targets)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metrics
            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1)
                self.train_metrics.update(preds, targets, loss.item())

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        metrics = self.train_metrics.compute()
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')

        for features, targets in pbar:
            features = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in features.items()
            }
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(features)
                logits = outputs['logits']
                loss = self.criterion(logits, targets)

            preds = torch.argmax(logits, dim=-1)
            self.val_metrics.update(preds, targets, loss.item())

            pbar.set_postfix({'loss': loss.item()})

        metrics = self.val_metrics.compute()
        return metrics

    def save_checkpoint(self, is_best: bool = False, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{self.current_epoch}.pt'

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_metric = checkpoint['best_val_metric']

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")

    def log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics to tensorboard and console."""
        epoch = self.current_epoch

        # Tensorboard
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Train/{key}', value, epoch)

        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'Val/{key}', value, epoch)

        # Learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Learning_rate', current_lr, epoch)

        # Console
        logger.info(f"\nEpoch {epoch+1} Results:")
        logger.info(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}, "
                   f"F1: {train_metrics['f1']:.4f}")
        logger.info(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}")
        logger.info(f"  LR: {current_lr:.6f}")

    def train(self):
        """Main training loop."""
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 15)
        metric_to_monitor = self.config.get('metric_to_monitor', 'f1')

        logger.info(f"\nStarting training for {epochs} epochs")
        logger.info(f"Monitoring metric: {metric_to_monitor}")
        logger.info(f"Early stopping patience: {patience}")

        # DDG-DA adaptation (if enabled)
        if self.use_ddg_da and self.ddg_da_adapter:
            logger.info("\nApplying DDG-DA adaptation...")
            try:
                adapted_loader = self.ddg_da_adapter.get_adapted_dataloader(
                    self.train_loader.dataset,
                    batch_size=self.train_loader.batch_size,
                    num_workers=self.train_loader.num_workers
                )
                self.train_loader = adapted_loader
                logger.info("DDG-DA adaptation applied successfully")
            except Exception as e:
                logger.error(f"DDG-DA adaptation failed: {e}")
                logger.info("Continuing with original data loader")

        start_time = time.time()

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Log
            self.log_metrics(train_metrics, val_metrics)

            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics[metric_to_monitor])
                else:
                    self.scheduler.step()

            # Check for improvement
            current_metric = val_metrics[metric_to_monitor]
            is_best = current_metric > self.best_val_metric

            if is_best:
                self.best_val_metric = current_metric
                self.patience_counter = 0
                logger.info(f"✓ New best {metric_to_monitor}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(f"  No improvement ({self.patience_counter}/{patience})")

            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        logger.info(f"Best {metric_to_monitor}: {self.best_val_metric:.4f}")

        # Test evaluation
        logger.info("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader)
        self.log_test_metrics(test_metrics)

        # Save final model wrapper
        self.save_model_wrapper()

        self.writer.close()

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        """Evaluate on test set."""
        self.model.eval()
        metrics_tracker = MetricsTracker()

        all_probs = []

        for features, targets in tqdm(loader, desc='Evaluating'):
            features = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in features.items()
            }
            targets = targets.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(features)
                logits = outputs['logits']
                loss = self.criterion(logits, targets)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            metrics_tracker.update(preds, targets, loss.item())
            all_probs.append(probs.cpu().numpy())

        metrics = metrics_tracker.compute()

        # Compute AUC if possible
        try:
            from sklearn.metrics import roc_auc_score
            all_probs = np.vstack(all_probs)
            targets_np = np.array(metrics_tracker.targets)
            auc = roc_auc_score(targets_np, all_probs[:, 1])
            metrics['auc'] = auc
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")

        return metrics

    def log_test_metrics(self, test_metrics: Dict):
        """Log test metrics."""
        logger.info("\nTest Set Results:")
        logger.info(f"  Loss: {test_metrics['loss']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        if 'auc' in test_metrics:
            logger.info(f"  AUC: {test_metrics['auc']:.4f}")

        # Save test metrics
        metrics_path = os.path.join(self.checkpoint_dir, 'test_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics saved to {metrics_path}")

    def save_model_wrapper(self):
        """Save model wrapper for inference."""
        wrapper_dir = os.path.join(self.checkpoint_dir, 'model_wrapper')

        # Get feature columns from dataset
        sample_data = next(iter(self.train_loader))
        features, _ = sample_data

        # Create wrapper
        wrapper = TFTModelWrapper(
            model=self.model,
            feature_cols=self.train_loader.dataset.feature_cols,
            ticker_to_id=self.train_loader.dataset.ticker_to_id
        )

        # Save
        wrapper.save(wrapper_dir)

        logger.info(f"Model wrapper saved to {wrapper_dir}")


def load_config(config_path: str) -> Dict:
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train TFT model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data', type=str, required=True, help='Path to processed data CSV')
    parser.add_argument('--use-ddg-da', action='store_true', help='Use DDG-DA adaptation')
    parser.add_argument('--ddg-da-model', type=str, help='Path to DDG-DA predictor')
    parser.add_argument('--experiment-name', type=str, help='Experiment name for logging')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Loading data from {args.data}")

    # Load data
    df = pd.read_csv(args.data)
    prepared_df = prepare_data_for_dataset(df)

    # Split data
    train_df, val_df, test_df = train_val_test_split(prepared_df)

    # Get feature columns
    exclude_cols = ['Date', 'Ticker', 'target']
    feature_cols = [col for col in prepared_df.columns if col not in exclude_cols]
    num_features = len(feature_cols)
    num_tickers = prepared_df['Ticker'].nunique()

    logger.info(f"Using {num_features} features")

    # Create dataloaders
    loaders_config = DataLoaderConfig(
        batch_size=config.get('batch_size', 32),
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        train_df, val_df, test_df, loaders_config
    )

    # Load DDG-DA if specified
    ddg_da_predictor = None
    if args.use_ddg_da and args.ddg_da_model:
        logger.info(f"Loading DDG-DA predictor from {args.ddg_da_model}")
        ddg_da_predictor = DDGDADistributionPredictor.load(args.ddg_da_model)

    # Initialize model - REMOVED sequence_length parameter
    model = TFTClassifier(
        num_features=num_features,
        hidden_dim=config.get('hidden_dim', 64),
        num_heads=config.get('num_heads', 2),
        lstm_layers=config.get('lstm_layers', 1),
        num_tickers=num_tickers,
        dropout=config.get('dropout', 0.3)
    )

    # Create trainer
    trainer = TFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        experiment_name=args.experiment_name,
        use_ddg_da=args.use_ddg_da,
        ddg_da_predictor=ddg_da_predictor
    )

    # Train
    trainer.train()

    logger.info("Training complete!")

    return 0


if __name__ == "__main__":
    main()
