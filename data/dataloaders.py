"""
DataLoader utilities optimized for AWS EC2 G5 instances with A10 GPU.
Includes GPU-friendly settings: pin_memory, optimized num_workers, etc.
"""

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import logging
import os
import multiprocessing as mp
from typing import Optional, Tuple
import pandas as pd

from .dataset import StockDataset, collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoaderConfig:
    """Configuration for DataLoader creation optimized for A10 GPU."""

    def __init__(
            self,
            batch_size: int = 64,
            num_workers: Optional[int] = None,
            pin_memory: bool = True,
            prefetch_factor: int = 2,
            persistent_workers: bool = True,
            use_weighted_sampling: bool = False
    ):
        """
        Initialize DataLoader configuration.

        Args:
            batch_size: Batch size for training/inference
            num_workers: Number of worker processes (auto-detect if None)
            pin_memory: Enable pinned memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch per worker
            persistent_workers: Keep workers alive between epochs
            use_weighted_sampling: Balance classes with weighted sampling
        """
        self.batch_size = batch_size

        # Auto-detect optimal num_workers based on CPU count
        if num_workers is None:
            cpu_count = mp.cpu_count()
            # For EC2 G5 instances: typically 4-48 vCPUs depending on instance size
            # Use 4-8 workers to balance I/O without overwhelming CPU
            num_workers = min(8, max(2, cpu_count // 2))

        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers and num_workers > 0
        self.use_weighted_sampling = use_weighted_sampling

        logger.info(f"DataLoader config: batch_size={batch_size}, num_workers={num_workers}, "
                    f"pin_memory={self.pin_memory}, persistent_workers={self.persistent_workers}")


def create_weighted_sampler(dataset: StockDataset) -> WeightedRandomSampler:
    """
    Create a weighted sampler to balance class distribution.

    Args:
        dataset: StockDataset instance

    Returns:
        WeightedRandomSampler
    """
    logger.info("Creating weighted sampler for class balancing...")

    # Get all targets
    targets = []
    for i in range(len(dataset)):
        _, target = dataset[i]
        targets.append(target.item())

    targets = np.array(targets)

    # Compute class weights
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts

    # Assign weight to each sample
    sample_weights = class_weights[targets]

    logger.info(f"Class counts: {class_counts}")
    logger.info(f"Class weights: {class_weights}")

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def create_dataloader(
        dataset: StockDataset,
        config: DataLoaderConfig,
        shuffle: bool = True,
        drop_last: bool = False
) -> DataLoader:
    """
    Create a DataLoader with specified configuration.

    Args:
        dataset: StockDataset instance
        config: DataLoaderConfig instance
        shuffle: Whether to shuffle data (ignored if using weighted sampler)
        drop_last: Whether to drop incomplete last batch

    Returns:
        DataLoader
    """
    # Determine sampler
    sampler = None
    if config.use_weighted_sampling:
        sampler = create_weighted_sampler(dataset)
        shuffle = False  # Mutually exclusive with sampler

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=config.persistent_workers,
        collate_fn=collate_fn,
        drop_last=drop_last
    )

    logger.info(f"Created DataLoader with {len(dataset)} samples, "
                f"{len(loader)} batches per epoch")

    return loader


def create_train_val_test_loaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: DataLoaderConfig
):
    """
    Create train, validation, and test dataloaders.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        config: DataLoader configuration

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get feature columns
    exclude_cols = ['Date', 'Ticker', 'target']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    # Create ticker mapping
    all_tickers = sorted(train_df['Ticker'].unique())
    ticker_to_id = {ticker: idx for idx, ticker in enumerate(all_tickers)}

    # Create datasets
    train_dataset = StockDataset(
        df=train_df,
        feature_cols=feature_cols,
        ticker_to_id=ticker_to_id,
        sequence_length=60,
        horizon=1
    )

    val_dataset = StockDataset(
        df=val_df,
        feature_cols=feature_cols,
        ticker_to_id=ticker_to_id,
        sequence_length=60,
        horizon=1
    )

    test_dataset = StockDataset(
        df=test_df,
        feature_cols=feature_cols,
        ticker_to_id=ticker_to_id,
        sequence_length=60,
        horizon=1
    )

    logger.info(f"DataLoader config: batch_size={config.batch_size}, num_workers={config.num_workers}, "
                f"pin_memory={config.pin_memory}, persistent_workers={config.persistent_workers}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers > 0 else False,
        drop_last=False
    )

    logger.info(f"Created train/val/test loaders")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val: {len(val_loader)} batches")
    logger.info(f"  Test: {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def estimate_optimal_batch_size(
        dataset: StockDataset,
        model: torch.nn.Module,
        device: torch.device,
        max_batch_size: int = 512,
        memory_fraction: float = 0.8
) -> int:
    """
    Estimate optimal batch size that fits in GPU memory.

    Args:
        dataset: Dataset to use for testing
        model: Model to test with
        device: Device to test on
        max_batch_size: Maximum batch size to try
        memory_fraction: Fraction of GPU memory to use

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using default batch size of 32")
        return 32

    logger.info("Estimating optimal batch size...")

    model = model.to(device)
    model.eval()

    # Binary search for max batch size
    low, high = 1, max_batch_size
    optimal_batch_size = low

    while low <= high:
        batch_size = (low + high) // 2

        try:
            # Create test batch
            config = DataLoaderConfig(batch_size=batch_size, num_workers=0)
            loader = create_dataloader(dataset, config, shuffle=False)

            # Try forward pass
            features, targets = next(iter(loader))
            features = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in features.items()}

            with torch.no_grad():
                _ = model(features)

            # Success - try larger batch
            optimal_batch_size = batch_size
            low = batch_size + 1

            # Clear cache
            del features, targets
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                # OOM - try smaller batch
                high = batch_size - 1
                torch.cuda.empty_cache()
            else:
                raise e

    # Apply safety margin
    optimal_batch_size = int(optimal_batch_size * memory_fraction)

    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def benchmark_dataloader(loader: DataLoader, num_batches: int = 100) -> float:
    """
    Benchmark DataLoader throughput.

    Args:
        loader: DataLoader to benchmark
        num_batches: Number of batches to iterate

    Returns:
        Average time per batch in seconds
    """
    import time

    logger.info(f"Benchmarking DataLoader ({num_batches} batches)...")

    times = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        start = time.time()
        # Simulate some work
        features, targets = batch
        _ = features['encoder_cont'].shape
        end = time.time()

        times.append(end - start)

    avg_time = np.mean(times)
    throughput = loader.batch_size / avg_time

    logger.info(f"Avg time per batch: {avg_time * 1000:.2f}ms")
    logger.info(f"Throughput: {throughput:.1f} samples/sec")

    return avg_time


if __name__ == "__main__":
    # Example usage
    from .dataset import StockDataset, prepare_data_for_dataset, train_val_test_split
    import pandas as pd

    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']

    data_list = []
    for ticker in tickers:
        n = len(dates)
        df_ticker = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Close': np.cumsum(np.random.randn(n)) + 100,
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n),
        })
        data_list.append(df_ticker)

    sample_data = pd.concat(data_list, ignore_index=True)
    prepared_data = prepare_data_for_dataset(sample_data)
    train_df, val_df, test_df = train_val_test_split(prepared_data)

    # Create datasets
    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    train_dataset = StockDataset(train_df, feature_cols=feature_cols)
    val_dataset = StockDataset(val_df, feature_cols=feature_cols)
    test_dataset = StockDataset(test_df, feature_cols=feature_cols)

    # Create loaders
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        train_dataset, val_dataset, test_dataset
    )

    # Test iteration
    for batch_idx, (features, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Encoder shape: {features['encoder_cont'].shape}")
        print(f"  Targets shape: {targets.shape}")
        if batch_idx >= 2:
            break

    # Benchmark
    benchmark_dataloader(train_loader, num_batches=50)
