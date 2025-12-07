# scripts/05_build_sentiment_features.py
"""
Build sentiment features from news data using NLP (e.g., VADER, FinBERT).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from config import DataConfig, RAW_DATA_DIR, PROCESSED_DATA_DIR

data_config = DataConfig()

def build_daily_sentiment_features():
    """
    Load news data and compute daily sentiment statistics.
    Returns a DataFrame with columns: [date, sentiment_mean, sentiment_std, sentiment_count]
    """
    
    symbol = data_config.symbol
    
    # Attempt to load news data (from Kaggle, Alpha Vantage, etc.)
    news_csv_path = data_config.news_csv_path
    
    if not news_csv_path.exists():
        print(f"Warning: News data not found at {news_csv_path}")
        print("Creating dummy sentiment features for demonstration...")
        
        # Create dummy sentiment features for demonstration
        price_path = RAW_DATA_DIR / "prices" / f"{symbol}.csv"
        df = pd.read_csv(price_path)
        dates = pd.to_datetime(df["Date"])
        
        sentiment_df = pd.DataFrame({
            "date": dates,
            "sentiment_mean": np.random.normal(0.0, 0.2, len(dates)),
            "sentiment_std": np.abs(np.random.normal(0.15, 0.05, len(dates))),
            "sentiment_count": np.random.randint(1, 50, len(dates)),
        })
    else:
        print(f"Loading news data from {news_csv_path}...")
        news_df = pd.read_csv(news_csv_path)
        
        # Expected columns: date, sentiment, text, symbol, etc.
        # Filter for the target symbol
        if "symbol" in news_df.columns:
            news_df = news_df[news_df["symbol"] == symbol]
        
        if "date" not in news_df.columns:
            news_df["date"] = pd.to_datetime(news_df["date"])
        else:
            news_df["date"] = pd.to_datetime(news_df["date"])
        
        # Compute daily sentiment statistics
        sentiment_df = news_df.groupby("date").agg({
            "sentiment": ["mean", "std", "count"]
        }).reset_index()
        
        sentiment_df.columns = ["date", "sentiment_mean", "sentiment_std", "sentiment_count"]
        sentiment_df["sentiment_std"] = sentiment_df["sentiment_std"].fillna(0)
    
    return sentiment_df


def merge_sentiment_with_features():
    """Merge sentiment features with engineering features."""
    symbol = data_config.symbol
    
    print(f"Building sentiment features for {symbol}...")
    sentiment_df = build_daily_sentiment_features()
    
    # Load merged features
    features_path = PROCESSED_DATA_DIR / f"{symbol}_features_merged.csv"
    print(f"Loading features from {features_path}...")
    features_df = pd.read_csv(features_path, parse_dates=["date"])
    
    # Merge
    print("Merging sentiment with features...")
    merged_df = pd.merge(features_df, sentiment_df, on="date", how="left")
    
    # Fill NaN sentiment values with 0
    merged_df["sentiment_mean"] = merged_df["sentiment_mean"].fillna(0.0)
    merged_df["sentiment_std"] = merged_df["sentiment_std"].fillna(0.0)
    merged_df["sentiment_count"] = merged_df["sentiment_count"].fillna(0.0)
    
    # Save
    output_path = PROCESSED_DATA_DIR / f"{symbol}_features_with_sentiment.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"✓ Saved features with sentiment to {output_path}")
    print(f"  Shape: {merged_df.shape}")
    
    return merged_df


def main():
    print("=" * 60)
    print("Building Sentiment Features")
    print("=" * 60)
    
    merge_sentiment_with_features()
    print("\n✓ Sentiment feature engineering complete!")


if __name__ == "__main__":
    main()
