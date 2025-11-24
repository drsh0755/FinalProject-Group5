import pandas as pd
import numpy as np

print("=" * 60)
print("ALIGNING ALPHA VANTAGE SENTIMENT WITH STOCK DATA")
print("=" * 60)

# Load sentiment data
sentiment = pd.read_csv('data/raw/news/alphavantage_news_with_sentiment.csv')
sentiment['date'] = pd.to_datetime(sentiment['date'])

print(f"\n✓ Sentiment: {len(sentiment)} articles")
print(f"  Date range: {sentiment['date'].min()} to {sentiment['date'].max()}")

# Load stock data
stock = pd.read_csv('data/processed/spy_features_2year.csv')
stock['Date'] = pd.to_datetime(stock['Date'])

print(f"\n✓ Stock data: {len(stock)} days")
print(f"  Date range: {stock['Date'].min()} to {stock['Date'].max()}")

# Aggregate sentiment by day
daily_sentiment = sentiment.groupby(sentiment['date'].dt.date).agg({
    'sentiment_score': ['mean', 'std', 'count'],
    'sentiment_label': lambda x: (x == 'Bullish').sum() / len(x)  # % positive
}).reset_index()

daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 
                           'article_count', 'positive_ratio']
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

print(f"\n✓ Aggregated to {len(daily_sentiment)} days with sentiment")

# Merge with stock data
merged = stock.merge(
    daily_sentiment,
    left_on=stock['Date'].dt.date,
    right_on=daily_sentiment['date'].dt.date,
    how='left'
)

# Forward fill missing days
sentiment_cols = ['sentiment_mean', 'sentiment_std', 'article_count', 'positive_ratio']
for col in sentiment_cols:
    merged[col] = merged[col].fillna(method='ffill').fillna(0)

# Remove the extra date column
merged = merged.drop('key_0', axis=1, errors='ignore')
merged = merged.drop('date', axis=1, errors='ignore')

# Save
output_file = 'data/processed/spy_features_with_alphavantage_sentiment.csv'
merged.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("✓ ALIGNMENT COMPLETE!")
print("=" * 60)
print(f"Total days: {len(merged)}")
print(f"Days with sentiment: {(merged['article_count'] > 0).sum()}")
print(f"Coverage: {(merged['article_count'] > 0).sum() / len(merged) * 100:.1f}%")
print(f"Saved to: {output_file}")

print(f"\n📊 Sentiment Statistics:")
print(f"  Mean: {merged['sentiment_mean'].mean():.4f}")
print(f"  Std: {merged['sentiment_std'].mean():.4f}")
print(f"  Avg articles/day: {merged['article_count'].mean():.1f}")
