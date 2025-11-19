"""
Create technical indicators for stock prediction
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PRIMARY_TICKER

def calculate_technical_indicators(df):
    """Calculate common technical indicators"""
    print("Calculating technical indicators...")
    
    # Make a copy
    data = df.copy()
    
    # 1. Simple Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    
    # 2. Exponential Moving Averages
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    
    # 3. MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # 4. RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
    
    # 6. Volume indicators
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    # 7. Price changes
    data['Daily_Return'] = data['Close'].pct_change()
    data['Price_Change'] = data['Close'].diff()
    
    # 8. Volatility
    data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
    
    print(f"✓ Created {len(data.columns) - len(df.columns)} new features")
    
    return data

def main():
    """Main function"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING - TECHNICAL INDICATORS")
    print("="*60)
    
    # Load raw data
    input_file = RAW_DATA_DIR / f'{PRIMARY_TICKER}_historical.csv'
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    print(f"Original shape: {df.shape}")
    
    # Calculate indicators
    df_features = calculate_technical_indicators(df)
    
    # Remove NaN rows (from rolling calculations)
    df_clean = df_features.dropna()
    print(f"After removing NaN: {df_clean.shape}")
    print(f"Removed {len(df_features) - len(df_clean)} rows with NaN values")
    
    # Save processed data
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / f'{PRIMARY_TICKER}_features.csv'
    df_clean.to_csv(output_file)
    print(f"\n✓ Saved processed data to: {output_file}")
    
    # Display summary
    print(f"\nFeature Summary:")
    print(f"{'='*60}")
    print(f"Total features: {len(df_clean.columns)}")
    print(f"\nFeature list:")
    for i, col in enumerate(df_clean.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print(f"\n{'='*60}")
    print(f"Sample data (last 3 rows):")
    print(df_clean[['Close', 'SMA_20', 'RSI', 'MACD', 'BB_Width']].tail(3))
    
    print(f"\n{'='*60}")
    print("✓ Feature engineering complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
