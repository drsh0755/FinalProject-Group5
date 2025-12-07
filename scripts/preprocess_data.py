"""
Preprocess downloaded data for model training.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import prepare_data_for_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for training')
    parser.add_argument('--input', type=str, required=True, help='Input CSV with features')
    parser.add_argument('--output', type=str, required=True, help='Output CSV for training')
    parser.add_argument('--min-periods', type=int, default=60, help='Min periods per ticker')

    args = parser.parse_args()

    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    df['Date'] = pd.to_datetime(df['Date'])

    logger.info(f"Loaded {len(df)} rows, {df['Ticker'].nunique()} tickers")

    # Prepare for training
    logger.info("Preparing dataset...")
    df = prepare_data_for_dataset(df, min_periods=args.min_periods)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Saved to {output_path}")
    logger.info(f"Final shape: {df.shape}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
