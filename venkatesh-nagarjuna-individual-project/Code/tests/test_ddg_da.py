"""
Test script for DDG-DA meta-learning components.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from ddg_da import DDGDADistributionPredictor, DDGDADataAdapter
from data.dataset import StockDataset, prepare_data_for_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_ddg_da_pipeline():
    """Test complete DDG-DA pipeline."""

    logger.info("=" * 60)
    logger.info("Testing DDG-DA Meta-Learning Pipeline")
    logger.info("=" * 60)

    # Create synthetic data with regime shifts
    logger.info("\n1. Creating synthetic data with regime shifts...")
    dates = pd.date_range('2019-01-01', '2024-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

    data_list = []
    for ticker in tickers:
        n = len(dates)

        # Create different volatility regimes
        volatility_regimes = np.concatenate([
            np.ones(n // 3) * 0.01,  # Low vol regime
            np.ones(n // 3) * 0.03,  # High vol regime
            np.ones(n - 2 * (n // 3)) * 0.015  # Medium vol regime
        ])

        returns = np.random.randn(n) * volatility_regimes
        close = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Close': close,
            'Volume': np.random.randint(1e6, 1e8, n),
            'sentiment_score': np.random.uniform(-0.5, 0.5, n),
            'rsi': np.random.uniform(30, 70, n),
            'volatility_10d': volatility_regimes + np.random.randn(n) * 0.001
        })
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)
    logger.info(f"Data shape: {data.shape}")

    # Train DDG-DA predictor
    logger.info("\n2. Training DDG-DA regime predictor...")
    predictor = DDGDADistributionPredictor(
        regime_window_size=20,
        regime_duration=60,
        history_length=5
    )

    predictor.train(data, epochs=30, batch_size=16, lr=0.001)

    # Predict future regime
    logger.info("\n3. Predicting future regime...")
    future_regime = predictor.predict_future_regime()
    logger.info(f"Predicted regime shape: {future_regime.shape}")
    logger.info(f"Predicted regime (first 10 dims): {future_regime[:10]}")

    # Create dataset
    logger.info("\n4. Creating dataset for adaptation...")
    prepared_data = prepare_data_for_dataset(data, min_periods=100)
    feature_cols = ['sentiment_score', 'rsi', 'volatility_10d']

    dataset = StockDataset(
        prepared_data,
        sequence_length=60,
        feature_cols=feature_cols
    )

    # Create adapter
    logger.info("\n5. Creating DDG-DA data adapter...")
    adapter = DDGDADataAdapter(
        predictor,
        similarity_metric='rbf',
        temperature=2.0
    )

    # Get adapted dataloader
    logger.info("\n6. Creating adapted DataLoader...")
    adapted_loader = adapter.get_adapted_dataloader(
        dataset,
        batch_size=32,
        num_workers=2
    )

    # Test iteration
    logger.info("\n7. Testing adapted data iteration...")
    for i, (features, targets) in enumerate(adapted_loader):
        logger.info(f"Batch {i + 1}:")
        logger.info(f"  Features shape: {features['encoder_cont'].shape}")
        logger.info(f"  Targets shape: {targets.shape}")
        if i >= 2:
            break

    # Save predictor
    logger.info("\n8. Saving predictor...")
    predictor.save('ddg_da_predictor.pkl')

    logger.info("\n" + "=" * 60)
    logger.info("✓ DDG-DA pipeline test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_ddg_da_pipeline()
