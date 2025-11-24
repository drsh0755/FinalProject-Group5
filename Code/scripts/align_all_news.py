import pandas as pd
import numpy as np

print("=" * 60)
print("ALIGNING ALL NEWS WITH STOCK DATA")
print("=" * 60)

# Load all news
news = pd.read_csv('data/raw/news/all_alphavantage_news.csv')
news['date'] = pd.to_datetime(news['date'])
print(f"✓ News: {len(news):,} articles")
print(f"  Date range: {news['date'].min()} to {news['date'].max()}")

# Load stock data
stock = pd.read_csv('data/processed/spy_features_2year.csv')
stock['Date'] = pd.to_datetime(stock['Date'])
print(f"✓ Stock: {len(stock)} days")

# Aggregate daily sentiment
daily = news.groupby(news['date'].dt.date).agg({
    'sentiment_score': ['mean', 'std', 'count'],
    'sentiment_label': lambda x: (x.isin(['Bullish', 'Somewhat-Bullish'])).sum() / len(x)
}).reset_index()

daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count', 'positive_ratio']
daily['date'] = pd.to_datetime(daily['date'])
print(f"✓ Daily aggregated: {len(daily)} days")

# Merge
merged = stock.merge(daily, left_on=stock['Date'].dt.date, right_on=daily['date'].dt.date, how='left')

# Forward fill
sent_cols = ['sentiment_mean', 'sentiment_std', 'article_count', 'positive_ratio']
for col in sent_cols:
    merged[col] = merged[col].ffill().fillna(0)

merged = merged.drop(['key_0', 'date'], axis=1, errors='ignore')

# Save
output = 'data/processed/spy_features_with_full_sentiment.csv'
merged.to_csv(output, index=False)

print(f"\n✓ Merged: {len(merged)} days")
print(f"✓ Days with news: {(merged['article_count'] > 0).sum()} ({(merged['article_count'] > 0).sum()/len(merged)*100:.1f}%)")
print(f"✓ Saved: {output}")
