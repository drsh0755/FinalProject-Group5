"""
Align news sentiment with stock trading days
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("\n" + "=" * 60)
    print("ALIGNING SENTIMENT WITH STOCK DATA")
    print("=" * 60 + "\n")

    data_dir = Path(__file__).parent.parent / 'data' / 'processed'

    # Load stock features
    print("Loading stock features...")
    stock_file = data_dir / 'spy_features_2year.csv'
    stock_df = pd.read_csv(stock_file, index_col=0, parse_dates=True)
    print(f"✓ Stock data: {len(stock_df)} days")
    print(f"  Date range: {stock_df.index[0]} to {stock_df.index[-1]}")

    # Load sentiment
    print("\nLoading sentiment data...")
    sentiment_file = data_dir / 'news_sentiment.csv'
    sentiment_df = pd.read_csv(sentiment_file, parse_dates=['date'])
    print(f"✓ Sentiment data: {len(sentiment_df)} articles")

    if 'date' in sentiment_df.columns:
        print(f"  Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")

    # Aggregate sentiment by day
    print("\nAggregating daily sentiment...")
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

    daily_sentiment = sentiment_df.groupby('date').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'positive': 'mean',
        'negative': 'mean',
        'neutral': 'mean'
    }).reset_index()

    daily_sentiment.columns = [
        'date',
        'sentiment_mean', 'sentiment_std', 'article_count',
        'positive_ratio', 'negative_ratio', 'neutral_ratio'
    ]

    # Fill NaN std with 0 (single article days)
    daily_sentiment['sentiment_std'].fillna(0, inplace=True)

    print(f"✓ Daily sentiment: {len(daily_sentiment)} days")

    # Merge with stock data
    print("\nMerging sentiment with stock data...")
    stock_df['date'] = stock_df.index.date

    merged_df = stock_df.merge(
        daily_sentiment,
        on='date',
        how='left'
    )

    merged_df.set_index(stock_df.index, inplace=True)
    merged_df.drop('date', axis=1, inplace=True)

    # Fill missing sentiment days
    # Strategy: Forward fill (use previous day's sentiment)
    sentiment_cols = [
        'sentiment_mean', 'sentiment_std', 'article_count',
        'positive_ratio', 'negative_ratio', 'neutral_ratio'
    ]

    for col in sentiment_cols:
        merged_df[col].fillna(method='ffill', inplace=True)
        merged_df[col].fillna(0, inplace=True)  # First days

    print(f"✓ Merged dataset: {len(merged_df)} days")
    print(f"  Total features: {len(merged_df.columns)}")

    # Check coverage
    days_with_sentiment = (merged_df['article_count'] > 0).sum()
    coverage = (days_with_sentiment / len(merged_df)) * 100
    print(f"  Days with news: {days_with_sentiment} ({coverage:.1f}%)")

    # Save
    output_file = data_dir / 'spy_features_with_sentiment.csv'
    merged_df.to_csv(output_file)
    print(f"\n✓ Saved: {output_file}")

    # Summary
    print("\nFeature Summary:")
    print(f"  Original features: {len(stock_df.columns)}")
    print(f"  Sentiment features: {len(sentiment_cols)}")
    print(f"  Total features: {len(merged_df.columns)}")

    print("\n" + "=" * 60)
    print("✓ ALIGNMENT COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()