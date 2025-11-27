"""
Check if sentiment helps during high-volatility periods
"""

import pandas as pd
import numpy as np

df = pd.read_csv('Code/data/processed/spy_features_with_full_sentiment.csv')

# Calculate volatility
df['returns'] = df['close'].pct_change()
df['volatility_5d'] = df['returns'].rolling(5).std()

# Split into low/high volatility periods
median_vol = df['volatility_5d'].median()
df['high_vol'] = df['volatility_5d'] > median_vol

print("SENTIMENT CORRELATION ANALYSIS")
print("=" * 60)
print()

# Correlation in different regimes
print("Sentiment vs Next-Day Returns:")
df['next_return'] = df['returns'].shift(-1)

low_vol_corr = df[~df['high_vol']]['sentiment_mean'].corr(df[~df['high_vol']]['next_return'])
high_vol_corr = df[df['high_vol']]['sentiment_mean'].corr(df[df['high_vol']]['next_return'])

print(f"  Low volatility periods:  {low_vol_corr:.4f}")
print(f"  High volatility periods: {high_vol_corr:.4f}")
print()

if abs(high_vol_corr) > abs(low_vol_corr):
    print("✅ Sentiment is MORE useful during volatile periods!")
else:
    print("⚠️  Sentiment doesn't show clear pattern")

# Days with extreme sentiment
extreme_negative = df['sentiment_mean'] < -0.5
extreme_positive = df['sentiment_mean'] > 0.5

print()
print(f"Days with extreme negative sentiment: {extreme_negative.sum()}")
print(f"Days with extreme positive sentiment: {extreme_positive.sum()}")
print(f"Days with neutral sentiment: {(~extreme_negative & ~extreme_positive).sum()}")
print()

if extreme_negative.sum() < 10:
    print("⚠️  ISSUE: Very few days with extreme sentiment!")
    print("   Model had little signal to learn from.")
