"""
Create technical indicators from 2-year data
Optimized for faster training: sequence_length=30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ta


def load_stock_data(filepath):
    """Load yfinance CSV with proper cleaning"""
    # Read CSV, skip both ticker row (row 1) and empty date row (row 2)
    df = pd.read_csv(filepath, skiprows=[1, 2])

    print(f"  Initial shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  First row: {df.iloc[0].tolist()}")

    # The first column is 'Price' but actually contains dates
    # Rename it to 'Date'
    df = df.rename(columns={'Price': 'Date'})

    # Set date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Clean column names to lowercase
    df.columns = df.columns.str.lower().str.strip()

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN in close
    df = df.dropna(subset=['close'])

    # Sort by date
    df = df.sort_index()

    print(f"  Cleaned shape: {df.shape}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    return df


def create_technical_indicators(df):
    """Create 39 technical indicators"""

    # Get price data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # Price-based features
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))

    # Moving averages
    for window in [5, 10, 20, 50]:
        df[f'sma_{window}'] = close.rolling(window).mean()
        df[f'ema_{window}'] = close.ewm(span=window, adjust=False).mean()

    # Momentum indicators
    df['rsi'] = ta.momentum.rsi(close, window=14)
    df['stoch'] = ta.momentum.stoch(high, low, close, window=14)
    df['williams_r'] = ta.momentum.williams_r(high, low, close, lbp=14)

    # MACD
    macd = ta.trend.MACD(close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close, window=20)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = bollinger.bollinger_wband()

    # Volume indicators
    df['volume_sma_20'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / df['volume_sma_20']
    df['obv'] = ta.volume.on_balance_volume(close, volume)

    # Volatility
    df['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
    for window in [5, 10, 20]:
        df[f'volatility_{window}'] = close.pct_change().rolling(window).std()

    # Trend indicators
    df['adx'] = ta.trend.adx(high, low, close, window=14)
    df['cci'] = ta.trend.cci(high, low, close, window=20)

    # Price position
    df['price_position'] = (close - low) / (high - low + 1e-10)

    # Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = close.shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

    return df


def main():
    print("\n" + "=" * 60)
    print("CREATING FEATURES FROM 2-YEAR DATA")
    print("=" * 60 + "\n")

    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load QQQ 2-year data
    print("Loading 2-year QQQ data...")
    qqq_file = data_dir / 'QQQ_2year.csv'

    if not qqq_file.exists():
        print(f"✗ File not found: {qqq_file}")
        print("  Please run download_2year_data.py first")
        return

    df = load_stock_data(qqq_file)
    print(f"\n✓ Loaded {len(df)} days of data")

    # Create features
    print("\nCreating technical indicators...")
    df_features = create_technical_indicators(df)

    # Drop NaN rows
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    final_rows = len(df_features)

    print(f"✓ Created {len(df_features.columns)} features")
    print(f"  Dropped {initial_rows - final_rows} rows with NaN")
    print(f"  Final dataset: {final_rows} days")

    # With sequence_length=30, calculate sequences
    sequence_length = 30
    usable_sequences = final_rows - sequence_length
    print(f"\n✓ Usable sequences (seq_len=30): {usable_sequences}")

    # Save
    output_file = output_dir / 'qqq_features_2year.csv'
    df_features.to_csv(output_file)
    print(f"\n✓ Saved: {output_file}")

    # Show sample
    print("\nSample features (first 3 rows, selected columns):")
    sample_cols = ['close', 'returns', 'sma_20', 'rsi', 'macd']
    if all(col in df_features.columns for col in sample_cols):
        print(df_features[sample_cols].head(3).to_string())

    print("\n" + "=" * 60)
    print("✓ FEATURE CREATION COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()