"""
Test script to validate the complete data pipeline.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from data.dataset import StockDataset, prepare_data_for_dataset, train_val_test_split
from data.dataloaders import create_train_val_test_loaders, DataLoaderConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_pipeline():
    """Test the complete data pipeline."""

    logger.info("=" * 60)
    logger.info("Testing Stock Direction Forecasting Data Pipeline")
    logger.info("=" * 60)

    # 1. Create synthetic data
    logger.info("\n1. Creating synthetic data...")
    dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']

    data_list = []
    for ticker in tickers:
        n = len(dates)

        # Generate realistic-looking price data with trend
        trend = np.linspace(0, 50, n)
        noise = np.cumsum(np.random.randn(n) * 2)
        close = 100 + trend + noise

        df_ticker = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Close': close,
            'Volume': np.random.randint(1e6, 1e8, n),
            'return_1d': np.random.randn(n) * 0.02,
            'return_5d': np.random.randn(n) * 0.05,
            'sma_20': close,
            'rsi': np.random.uniform(30, 70, n),
            'volatility_10d': np.random.uniform(0.01, 0.03, n),
            'sentiment_score': np.random.uniform(-0.5, 0.5, n),
        })
        data_list.append(df_ticker)

    sample_data = pd.concat(data_list, ignore_index=True)
    logger.info(f"Created data: {sample_data.shape}")

    # 2. Prepare data
    logger.info("\n2. Preparing data for dataset...")
    prepared_data = prepare_data_for_dataset(sample_data, min_periods=100)

    # 3. Split data
    logger.info("\n3. Splitting into train/val/test...")
    train_df, val_df, test_df = train_val_test_split(prepared_data)

    # 4. Create datasets
    logger.info("\n4. Creating PyTorch datasets...")
    feature_cols = ['return_1d', 'return_5d', 'sma_20', 'rsi',
                    'volatility_10d', 'sentiment_score']

    train_dataset = StockDataset(
        train_df,
        sequence_length=60,
        horizon=1,
        feature_cols=feature_cols
    )

    val_dataset = StockDataset(
        val_df,
        sequence_length=60,
        horizon=1,
        feature_cols=feature_cols
    )

    test_dataset = StockDataset(
        test_df,
        sequence_length=60,
        horizon=1,
        feature_cols=feature_cols
    )

    # 5. Create dataloaders
    logger.info("\n5. Creating DataLoaders...")
    train_config = DataLoaderConfig(batch_size=32, use_weighted_sampling=True)
    eval_config = DataLoaderConfig(batch_size=64, use_weighted_sampling=False)

    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        train_dataset, val_dataset, test_dataset,
        train_config, eval_config
    )

    # 6. Test iteration
    logger.info("\n6. Testing batch iteration...")
    for batch_idx, (features, targets) in enumerate(train_loader):
        logger.info(f"\nBatch {batch_idx + 1}:")
        logger.info(f"  Encoder features shape: {features['encoder_cont'].shape}")
        logger.info(f"  Ticker IDs shape: {features['ticker_id'].shape}")
        logger.info(f"  Targets shape: {targets.shape}")
        logger.info(f"  Target distribution: {torch.bincount(targets)}")

        # Check for NaN
        assert not torch.isnan(features['encoder_cont']).any(), "NaN in features!"
        assert not torch.isnan(targets.float()).any(), "NaN in targets!"

        if batch_idx >= 2:
            break

    # 7. GPU transfer test (if available)
    if torch.cuda.is_available():
        logger.info("\n7. Testing GPU transfer...")
        device = torch.device('cuda')

        features, targets = next(iter(train_loader))
        features_gpu = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in features.items()
        }
        targets_gpu = targets.to(device)

        logger.info(f"  Features transferred to {device}")
        logger.info(f"  Encoder on GPU: {features_gpu['encoder_cont'].device}")
        logger.info(f"  Targets on GPU: {targets_gpu.device}")

    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_data_pipeline()
