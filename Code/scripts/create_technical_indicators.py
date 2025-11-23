#!/usr/bin/env python3
"""
Create technical indicators for LSTM model training
Generates features: MA, EMA, RSI, MACD, Bollinger Bands, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_technical_indicators(df):
    """
    Calculate technical indicators for stock data
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    print("Calculating technical indicators...")
    
    data = df.copy()
    
    # ============================================================
    # 1. MOVING AVERAGES
    # ============================================================
    print("  - Moving averages...")
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # ============================================================
    # 2. MACD (Moving Average Convergence Divergence)
    # ============================================================
    print("  - MACD...")
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # ============================================================
    # 3. RSI (Relative Strength Index)
    # ============================================================
    print("  - RSI...")
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # ============================================================
    # 4. BOLLINGER BANDS
    # ============================================================
    print("  - Bollinger Bands...")
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # ============================================================
    # 5. MOMENTUM INDICATORS
    # ============================================================
    print("  - Momentum indicators...")
    data['Momentum'] = data['Close'] - data['Close'].shift(10)
    data['ROC'] = ((data['Close'] - data['Close'].shift(10)) / data['Close'].shift(10)) * 100
    
    # ============================================================
    # 6. VOLATILITY
    # ============================================================
    print("  - Volatility...")
    data['Daily_Return'] = data['Close'].pct_change()
    data['Volatility_10'] = data['Daily_Return'].rolling(window=10).std()
    data['Volatility_20'] = data['Daily_Return'].rolling(window=20).std()
    data['Volatility_50'] = data['Daily_Return'].rolling(window=50).std()
    
    # ============================================================
    # 7. VOLUME INDICATORS
    # ============================================================
    print("  - Volume indicators...")
    data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']
    
    # ============================================================
    # 8. PRICE CHANGES
    # ============================================================
    print("  - Price changes...")
    data['Price_Change'] = data['Close'].diff()
    data['Price_Change_Pct'] = data['Close'].pct_change() * 100
    data['High_Low_Pct'] = ((data['High'] - data['Low']) / data['Close']) * 100
    data['Open_Close_Pct'] = ((data['Close'] - data['Open']) / data['Open']) * 100
    
    # ============================================================
    # 9. LAG FEATURES
    # ============================================================
    print("  - Lag features...")
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag2'] = data['Close'].shift(2)
    data['Close_Lag3'] = data['Close'].shift(3)
    data['Close_Lag5'] = data['Close'].shift(5)
    
    print(f"✓ Created {len(data.columns) - len(df.columns)} new features")
    
    return data

def main():
    """Main function"""
    
    print("\n" + "="*60)
    print("TECHNICAL INDICATORS GENERATION")
    print("="*60)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    code_dir = script_dir.parent
    raw_data_dir = code_dir / 'data' / 'raw'
    processed_data_dir = code_dir / 'data' / 'processed'
    
    # Create processed directory
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Ticker to process
    ticker = 'SPY'
    input_file = raw_data_dir / f'{ticker}_historical.csv'
    output_file = processed_data_dir / f'{ticker}_with_indicators.csv'
    
    print(f"\nInput: {input_file}")
    print(f"Output: {output_file}")
    
    # Load data
    print("\nLoading data...")
    if not input_file.exists():
        print(f"✗ Error: {input_file} not found!")
        sys.exit(1)
    
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} rows")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Original features: {list(df.columns)}")
    
    # Calculate indicators
    print()
    df_indicators = calculate_technical_indicators(df)
    
    # Remove rows with NaN (from rolling calculations)
    print("\nCleaning data...")
    rows_before = len(df_indicators)
    df_clean = df_indicators.dropna()
    rows_after = len(df_clean)
    print(f"  Removed {rows_before - rows_after} rows with NaN values")
    print(f"  Final dataset: {rows_after} rows, {len(df_clean.columns)} features")
    
    # Save
    df_clean.to_csv(output_file)
    print(f"\n✓ Saved to: {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    print(f"Total features: {len(df_clean.columns)}")
    print("\nFeature categories:")
    print("  - Original OHLCV: 7")
    print("  - Moving Averages: 7")
    print("  - MACD: 3")
    print("  - RSI: 1")
    print("  - Bollinger Bands: 5")
    print("  - Momentum: 2")
    print("  - Volatility: 4")
    print("  - Volume: 2")
    print("  - Price Changes: 4")
    print("  - Lag Features: 4")
    
    print("\nSample data (last 3 rows):")
    print(df_clean[['Close', 'SMA_20', 'RSI', 'MACD', 'BB_Width']].tail(3))
    
    print("\n" + "="*60)
    print("✓ Technical indicators generation complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
