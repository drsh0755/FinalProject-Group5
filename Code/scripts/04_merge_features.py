#!/usr/bin/env python3
import pandas as pd
import numpy as np

print("📊 Merging sentiment data with technical features...")

# Load news
news = pd.read_csv('Code/data/processed/alphavantage_news_2024_2025.csv')
news['date'] = pd.to_datetime(news['date'], format='%Y%m%d')

print(f"   Loaded {len(news)} articles")
print(f"   Date range: {news['date'].min().date()} → {news['date'].max().date()}")

# Group by date and calculate sentiment metrics
sentiment_daily = news.groupby(news['date'].dt.date).agg({
    'sentiment_score': ['mean', 'median', 'std', 'min', 'max', 'count']
}).reset_index()

sentiment_daily.columns = ['date', 'sentiment_mean', 'sentiment_median', 'sentiment_std',
                           'sentiment_min', 'sentiment_max', 'article_count']

# Calculate positive ratio
sentiment_daily['positive_ratio'] = news.groupby(news['date'].dt.date)['sentiment_score'].apply(
    lambda x: (x > 0).sum() / len(x)
).reset_index(drop=True)

# Convert date format for merging
sentiment_daily['date'] = pd.to_datetime(sentiment_daily['date'])

# Load stock features
features = pd.read_csv('Code/data/processed/spy_features_2year_updated.csv')
features['Date'] = pd.to_datetime(features['Date'])

print(f"   Loaded {len(features)} stock records")

# Merge on date
merged = features.merge(sentiment_daily, left_on='Date', right_on='date', how='left')

# Forward fill sentiment features for days with no news (using new syntax)
merged['sentiment_mean'] = merged['sentiment_mean'].ffill()
merged['sentiment_median'] = merged['sentiment_median'].ffill()
merged['sentiment_std'] = merged['sentiment_std'].fillna(0)
merged['sentiment_min'] = merged['sentiment_min'].ffill()
merged['sentiment_max'] = merged['sentiment_max'].ffill()
merged['article_count'] = merged['article_count'].fillna(0)
merged['positive_ratio'] = merged['positive_ratio'].ffill()

# Drop the duplicate date column
merged = merged.drop('date', axis=1)

# Verify sentiment features
print(f"\n   Sentiment features statistics:")
print(f"   sentiment_mean: min={merged['sentiment_mean'].min():.3f}, max={merged['sentiment_mean'].max():.3f}")
print(f"   sentiment_std: min={merged['sentiment_std'].min():.3f}, max={merged['sentiment_std'].max():.3f}")
print(f"   article_count: min={merged['article_count'].min():.0f}, max={merged['article_count'].max():.0f}")
print(f"   positive_ratio: min={merged['positive_ratio'].min():.3f}, max={merged['positive_ratio'].max():.3f}")

# Save
merged.to_csv('Code/data/processed/spy_features_with_sentiment.csv', index=False)

print(f"\n✅ Sentiment features merged!")
print(f"   Rows: {len(merged)}")
print(f"   Columns: {len(merged.columns)}")
print(f"   Saved to: Code/data/processed/spy_features_with_sentiment.csv")
