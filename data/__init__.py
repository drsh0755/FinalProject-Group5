"""
Data module for stock direction forecasting.
Includes PyTorch datasets and dataloaders optimized for A10 GPU.
"""

from .dataset import (
    StockDataset,
    MultiHorizonStockDataset,
    collate_fn,
    prepare_data_for_dataset,
    train_val_test_split
)

from .dataloaders import (
    DataLoaderConfig,
    create_dataloader,
    create_train_val_test_loaders,
    create_weighted_sampler,
    estimate_optimal_batch_size,
    benchmark_dataloader
)

__all__ = [
    'StockDataset',
    'MultiHorizonStockDataset',
    'collate_fn',
    'prepare_data_for_dataset',
    'train_val_test_split',
    'DataLoaderConfig',
    'create_dataloader',
    'create_train_val_test_loaders',
    'create_weighted_sampler',
    'estimate_optimal_batch_size',
    'benchmark_dataloader'
]
