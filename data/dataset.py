"""
PyTorch Dataset for stock price time series with multi-ticker support.
Handles sliding windows, feature preparation, and label generation.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """PyTorch Dataset for stock sequences."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ticker_to_id: Dict[str, int],
        sequence_length: int = 60,
        horizon: int = 1
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names
            ticker_to_id: Mapping from ticker symbols to IDs
            sequence_length: Length of input sequences
            horizon: Prediction horizon
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.ticker_to_id = ticker_to_id
        self.sequence_length = sequence_length
        self.horizon = horizon

        # Create sequences
        self.sequences = self._create_sequences()

        logger.info(f"Dataset initialized with {len(self.sequences)} sequences")
        logger.info(f"Sequence length: {sequence_length}, Horizon: {horizon}")
        logger.info(f"Features: {len(feature_cols)}, Tickers: {len(ticker_to_id)}")

    def _create_sequences(self) -> List[pd.DataFrame]:
        """Create sequences from data."""
        sequences = []

        for ticker in self.df['Ticker'].unique():
            ticker_data = self.df[self.df['Ticker'] == ticker].sort_values('Date')

            for i in range(len(ticker_data) - self.sequence_length):
                seq = ticker_data.iloc[i:i + self.sequence_length].copy()
                if len(seq) == self.sequence_length:
                    sequences.append(seq)

        return sequences

    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get a single sequence."""
        sequence = self.sequences[idx]

        # Get features
        encoder_features = sequence[self.feature_cols].values
        encoder_features = pd.DataFrame(encoder_features).apply(pd.to_numeric, errors='coerce').values
        encoder_features = np.nan_to_num(encoder_features, nan=0.0, posinf=0.0, neginf=0.0)
        encoder_features = encoder_features.astype(np.float32)

        # Get ticker ID
        ticker = sequence['Ticker'].iloc[0]
        ticker_id = self.ticker_to_id[ticker]

        # Get target (last value in sequence)
        target = int(sequence['target'].iloc[-1])

        # Create feature dict
        features = {
            'encoder_cont': torch.FloatTensor(encoder_features),
            'ticker_id': torch.LongTensor([ticker_id])
        }

        target = torch.LongTensor([target])[0]

        return features, target



class MultiHorizonStockDataset(StockDataset):
    """Extended dataset supporting multiple forecast horizons."""

    def __init__(
            self,
            data: pd.DataFrame,
            sequence_length: int = 60,
            horizons: List[int] = [1, 5, 10],
            **kwargs
    ):
        """
        Initialize multi-horizon dataset.

        Args:
            data: DataFrame with features and target
            sequence_length: Number of historical days
            horizons: List of forecast horizons
            **kwargs: Additional arguments passed to StockDataset
        """
        self.horizons = horizons
        super().__init__(data, sequence_length, horizon=max(horizons), **kwargs)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get a single sequence with multiple horizon targets.

        Returns:
            Tuple of (features_dict, targets_dict)
            targets_dict maps horizon to target
        """
        seq_info = self.sequences[idx]
        ticker_data = seq_info['data']

        # Get features (same as base class)
        start_idx = seq_info['start_idx']
        end_idx = seq_info['end_idx']

        encoder_features = ticker_data.iloc[start_idx:end_idx][self.feature_cols].values
        encoder_features = torch.FloatTensor(encoder_features)
        encoder_features = torch.nan_to_num(encoder_features, nan=0.0)

        if self.static_feature_cols:
            static_features = ticker_data.iloc[start_idx][self.static_feature_cols].values
            static_features = torch.FloatTensor(static_features)
            static_features = torch.nan_to_num(static_features, nan=0.0)
        else:
            static_features = torch.FloatTensor([])

        ticker_id = torch.LongTensor([seq_info['ticker_id']])

        features = {
            'encoder_cont': encoder_features,
            'ticker_id': ticker_id,
            'static_features': static_features,
            'ticker': seq_info['ticker']
        }

        # Multiple horizon targets
        targets = {}
        current_close = ticker_data.iloc[end_idx - 1]['Close']

        for h in self.horizons:
            target_idx = end_idx + h - 1
            if target_idx < len(ticker_data):
                future_close = ticker_data.iloc[target_idx]['Close']
                target = 1 if future_close > current_close else 0
                targets[f'horizon_{h}'] = torch.LongTensor([int(target)])
            else:
                targets[f'horizon_{h}'] = torch.LongTensor([-1])  # Invalid

        return features, targets


def collate_fn(batch: List[Tuple]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Custom collate function for batching sequences.

    Args:
        batch: List of (features, target) tuples

    Returns:
        Batched features and targets
    """
    features_list, targets_list = zip(*batch)

    # Stack encoder features
    encoder_cont = torch.stack([f['encoder_cont'] for f in features_list])
    ticker_ids = torch.stack([f['ticker_id'] for f in features_list])

    # Stack static features if present
    if features_list[0]['static_features'].numel() > 0:
        static_features = torch.stack([f['static_features'] for f in features_list])
    else:
        static_features = torch.FloatTensor([])

    # Stack targets
    targets = torch.stack(targets_list).squeeze(1)

    batched_features = {
        'encoder_cont': encoder_cont,
        'ticker_id': ticker_ids,
        'static_features': static_features,
        'tickers': [f['ticker'] for f in features_list]  # Keep as list
    }

    return batched_features, targets


def prepare_data_for_dataset(
        df: pd.DataFrame,
        min_periods: int = 60,
        target_horizon: int = 1
) -> pd.DataFrame:
    """
    Prepare data for StockDataset.

    Args:
        df: Raw data with features
        min_periods: Minimum periods required per ticker
        target_horizon: Days ahead to predict

    Returns:
        Prepared DataFrame
    """
    logger.info("Preparing data for dataset...")

    # Standardize column names
    if 'ticker' in df.columns and 'Ticker' not in df.columns:
        df = df.rename(columns={'ticker': 'Ticker'})
    if 'date' in df.columns and 'Date' not in df.columns:
        df = df.rename(columns={'date': 'Date'})

    # Drop duplicate columns if they exist
    if 'ticker' in df.columns and 'Ticker' in df.columns:
        df = df.drop(columns=['ticker'])
    if 'date' in df.columns and 'Date' in df.columns:
        df = df.drop(columns=['date'])

    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)

    # Sort by ticker and date
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Convert features to numeric
    logger.info("Converting features to numeric...")
    exclude_cols = ['Date', 'Ticker', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    logger.info("Feature conversion complete")

    # Create target if not exists
    if 'target' not in df.columns:
        logger.info(f"Creating target with {target_horizon}-day horizon...")

        # Create target without using apply (avoids include_groups issue)
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        # Calculate future returns per ticker
        df['future_return'] = np.nan
        df['target'] = np.nan

        for ticker in df['Ticker'].unique():
            mask = df['Ticker'] == ticker
            ticker_data = df.loc[mask, 'Close']
            future_return = ticker_data.pct_change(target_horizon).shift(-target_horizon)
            df.loc[mask, 'future_return'] = future_return
            df.loc[mask, 'target'] = (future_return > 0).astype(float)

        # Drop rows with NaN target
        df = df.dropna(subset=['target']).reset_index(drop=True)
        df['target'] = df['target'].astype(int)

    # Filter tickers with sufficient data
    ticker_counts = df.groupby('Ticker').size()
    valid_tickers = ticker_counts[ticker_counts >= min_periods].index
    df = df[df['Ticker'].isin(valid_tickers)].reset_index(drop=True)

    # Drop the temporary future_return column
    if 'future_return' in df.columns:
        df = df.drop(columns=['future_return'])

    logger.info(f"Data prepared: {len(df)} rows, {len(valid_tickers)} tickers")

    if 'target' in df.columns:
        class_dist = df['target'].value_counts(normalize=True)
        logger.info(f"Class distribution:\n{class_dist}")

    return df


def train_val_test_split(
        data: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        date_col: str = 'Date'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test by time (no data leakage).

    Args:
        data: DataFrame with temporal data
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        date_col: Date column name

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    data = data.sort_values(date_col).reset_index(drop=True)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Date ranges:")
    logger.info(f"  Train: {train_df[date_col].min()} to {train_df[date_col].max()}")
    logger.info(f"  Val: {val_df[date_col].min()} to {val_df[date_col].max()}")
    logger.info(f"  Test: {test_df[date_col].min()} to {test_df[date_col].max()}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']

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

    # Prepare data
    prepared_data = prepare_data_for_dataset(sample_data)

    # Split data
    train_df, val_df, test_df = train_val_test_split(prepared_data)

    # Create dataset
    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    train_dataset = StockDataset(
        train_df,
        sequence_length=60,
        horizon=1,
        feature_cols=feature_cols
    )

    print(f"Dataset size: {len(train_dataset)}")

    # Test getting a sample
    features, target = train_dataset[0]
    print(f"Encoder features shape: {features['encoder_cont'].shape}")
    print(f"Ticker ID: {features['ticker_id']}")
    print(f"Target: {target}")
