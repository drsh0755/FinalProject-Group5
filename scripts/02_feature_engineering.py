# scripts/02_feature_engineering.py
"""
Feature engineering: compute technical indicators and merge market data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DataConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils import compute_technical_indicators

data_config = DataConfig()

def engineer_features():
    """Compute technical indicators for the target stock."""
    symbol = data_config.symbol
    
    price_path = RAW_DATA_DIR / "prices" / f"{symbol}.csv"
    print(f"Loading price data from {price_path}...")
    
    df = pd.read_csv(price_path, index_col="Date", parse_dates=True)
    df.columns = [col.lower() for col in df.columns]
    
    print(f"Computing technical indicators for {symbol}...")
    df = compute_technical_indicators(df)
    
    return df


def merge_with_market_data(price_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price data with market indices."""
    symbol = data_config.symbol
    indices = data_config.market_indices
    
    df = price_df.copy()
    df = df.reset_index().rename(columns={"Date": "date"})
    
    indices_dir = RAW_DATA_DIR / "market_indices"
    
    for idx in indices:
        safe_name = idx.replace("^", "")
        idx_path = indices_dir / f"{safe_name}.csv"
        
        if not idx_path.exists():
            print(f"  Warning: {idx_path} not found, skipping {idx}")
            continue
        
        print(f"  Merging {idx}...")
        idx_df = pd.read_csv(idx_path, index_col="Date", parse_dates=True)
        idx_df.columns = [col.lower() for col in idx_df.columns]
        idx_df = idx_df[["close"]].rename(columns={"close": f"{safe_name}_close"})
        idx_df[f"{safe_name}_return"] = idx_df[f"{safe_name}_close"].pct_change()
        
        # Merge
        idx_df = idx_df.reset_index().rename(columns={"Date": "date"})
        df = pd.merge(df, idx_df[["date", f"{safe_name}_return"]], on="date", how="left")
    
    df = df.dropna()
    return df


def main():
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    
    # Engineer features
    price_df = engineer_features()
    
    # Merge with market data
    print(f"Merging with market indices...")
    merged_df = merge_with_market_data(price_df)
    
    # Save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    symbol = data_config.symbol
    output_path = PROCESSED_DATA_DIR / f"{symbol}_features_merged.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved merged features to {output_path}")
    print(f"  Shape: {merged_df.shape}")
    print(f"  Columns: {list(merged_df.columns)}")


if __name__ == "__main__":
    main()
