"""
Technical indicators and feature engineering for stock data.
Includes returns, volatility, moving averages, RSI, MACD, Bollinger Bands.
"""

import pandas as pd
import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Compute technical indicators for stock price data."""

    @staticmethod
    def add_returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Add return features for multiple periods.

        Args:
            df: DataFrame with 'Close' prices
            periods: List of periods for computing returns

        Returns:
            DataFrame with added return columns
        """
        df = df.copy()
        for period in periods:
            df[f'return_{period}d'] = df.groupby('Ticker')['Close'].pct_change(period)
        return df

    @staticmethod
    def add_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50, 200]) -> pd.DataFrame:
        """
        Add simple and exponential moving averages.

        Args:
            df: DataFrame with 'Close' prices
            windows: List of window sizes

        Returns:
            DataFrame with added MA columns
        """
        df = df.copy()
        for window in windows:
            df[f'sma_{window}'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window).mean())
            df[f'ema_{window}'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=window).mean())
            df[f'close_to_sma_{window}'] = df['Close'] / df[f'sma_{window}'] - 1
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add rolling volatility features.

        Args:
            df: DataFrame with return columns
            windows: List of window sizes

        Returns:
            DataFrame with added volatility columns
        """
        df = df.copy()
        if 'return_1d' not in df.columns:
            df['return_1d'] = df.groupby('Ticker')['Close'].pct_change()

        for window in windows:
            df[f'volatility_{window}d'] = df.groupby('Ticker')['return_1d'].transform(
                lambda x: x.rolling(window).std()
            )
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add RSI indicator."""

        def compute_rsi(group):
            """Compute RSI for a group."""
            close = group['Close'].values
            delta = np.diff(close)

            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)

            avg_gain = np.zeros(len(close))
            avg_loss = np.zeros(len(close))

            if len(gain) >= period:
                avg_gain[period] = np.mean(gain[:period])
                avg_loss[period] = np.mean(loss[:period])

                for i in range(period + 1, len(close)):
                    avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
                    avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

            rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
            rsi = 100 - (100 / (1 + rs))

            # Return Series with same index as group
            return pd.Series(rsi, index=group.index)

        # Apply per ticker and concatenate results
        rsi_series = df.groupby('Ticker', group_keys=False)['Close'].apply(
            lambda x: compute_rsi(df.loc[x.index])
        )

        df['rsi'] = rsi_series

        return df

    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).

        Args:
            df: DataFrame with 'Close' prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with MACD, signal, and histogram columns
        """
        df = df.copy()

        def compute_macd(group):
            exp1 = group['Close'].ewm(span=fast, adjust=False).mean()
            exp2 = group['Close'].ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            return pd.DataFrame({'macd': macd, 'macd_signal': signal_line, 'macd_hist': histogram})

        macd_df = df.groupby('Ticker', group_keys=False).apply(compute_macd)
        df = pd.concat([df, macd_df], axis=1)
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands.

        Args:
            df: DataFrame with 'Close' prices
            window: Moving average window
            num_std: Number of standard deviations

        Returns:
            DataFrame with Bollinger Band columns
        """
        df = df.copy()

        def compute_bb(group):
            sma = group['Close'].rolling(window).mean()
            std = group['Close'].rolling(window).std()
            upper = sma + (std * num_std)
            lower = sma - (std * num_std)
            bb_width = (upper - lower) / sma
            bb_position = (group['Close'] - lower) / (upper - lower)
            return pd.DataFrame({
                'bb_upper': upper,
                'bb_middle': sma,
                'bb_lower': lower,
                'bb_width': bb_width,
                'bb_position': bb_position
            })

        bb_df = df.groupby('Ticker', group_keys=False).apply(compute_bb)
        df = pd.concat([df, bb_df], axis=1)
        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Add volume-based features.

        Args:
            df: DataFrame with 'Volume' column
            windows: List of window sizes

        Returns:
            DataFrame with volume feature columns
        """
        df = df.copy()

        for window in windows:
            df[f'volume_sma_{window}'] = df.groupby('Ticker')['Volume'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'volume_ratio_{window}'] = df['Volume'] / df[f'volume_sma_{window}']

        return df

    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators at once.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with all technical indicators
        """
        logger.info("Computing all technical indicators...")

        df = TechnicalIndicators.add_returns(df)
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_volatility(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_volume_features(df)

        # Add calendar features
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['day_of_week'] = df['Date'].dt.dayofweek
            df['day_of_month'] = df['Date'].dt.day
            df['month'] = df['Date'].dt.month
            df['quarter'] = df['Date'].dt.quarter

        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df

    @staticmethod
    def normalize_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:
        """
        Normalize features per ticker using z-score normalization.

        Args:
            df: DataFrame with features
            exclude_cols: Columns to exclude from normalization

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()
        exclude_cols = exclude_cols or ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        logger.info(f"Normalizing {len(cols_to_normalize)} features per ticker")

        for col in cols_to_normalize:
            df[col] = df.groupby('Ticker')[col].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )

        return df


if __name__ == "__main__":
    # Example usage
    from data_sources.yfinance_loader import YFinanceLoader

    loader = YFinanceLoader(["AAPL", "MSFT"])
    df = loader.combine_all_tickers()

    # Add all indicators
    df_with_features = TechnicalIndicators.add_all_indicators(df)
    print(f"Features: {df_with_features.columns.tolist()}")
    print(df_with_features.head())
