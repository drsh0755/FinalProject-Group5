"""
Train DDG-DA regime predictor with proper error handling.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from ddg_da import DDGDADistributionPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)

    # Load data
    logger.info("Loading data...")
    data = pd.read_csv('data/processed/features_with_sentiment.csv')
    data['Date'] = pd.to_datetime(data['Date'])

    logger.info(f"Data shape: {data.shape}")

    # Data validation
    logger.info("Validating data...")

    # Check for NaN
    nan_count = data.isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values, filling with 0")
        data = data.fillna(0)

    # Check for Inf
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(data[numeric_cols]).sum().sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} Inf values, replacing with 0")
        data = data.replace([np.inf, -np.inf], 0)

    # Check for extremely large values
    for col in numeric_cols:
        if col in ['Date', 'Ticker']:
            continue
        max_val = data[col].abs().max()
        if max_val > 1e6:
            logger.warning(f"Column {col} has very large values (max={max_val}), clipping")
            data[col] = data[col].clip(-1e6, 1e6)

    logger.info("Data validation complete")

    # Train predictor
    logger.info("Training DDG-DA predictor...")

    predictor = DDGDADistributionPredictor(
        regime_window_size=20,
        regime_duration=60,
        history_length=5
    )

    try:
        predictor.train(
            data,
            epochs=50,
            batch_size=16,
            lr=0.0001  # Lower learning rate for stability
        )

        # Save
        predictor.save('checkpoints/ddg_da_predictor.pkl')
        logger.info("✓ DDG-DA predictor saved successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
