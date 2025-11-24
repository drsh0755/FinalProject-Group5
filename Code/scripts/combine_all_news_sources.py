import pandas as pd
import os
from pathlib import Path
from datetime import datetime

print("=" * 60)
print("COMBINING ALL NEWS SOURCES")
print("=" * 60)

all_news = []

# 1. AlphaVantage Historical
try:
    av = pd.read_csv('data/raw/news/alphavantage_historical_news.csv')
    av['source_type'] = 'AlphaVantage'
    all_news.append(av[['date', 'headline', 'sentiment_score', 'sentiment_label', 'source_type']])
    print(f"✓ AlphaVantage: {len(av):,} articles")
except Exception as e:
    print(f"✗ AlphaVantage: {str(e)}")

# 2. AlphaVantage Recent
try:
    av_recent = pd.read_csv('data/raw/news/alphavantage_news_with_sentiment.csv')
    av_recent['source_type'] = 'AlphaVantage-Recent'
    all_news.append(av_recent[['date', 'headline', 'sentiment_score', 'sentiment_label', 'source_type']])
    print(f"✓ AlphaVantage Recent: {len(av_recent):,} articles")
except Exception as e:
    print(f"✗ AlphaVantage Recent: {str(e)}")

# 3. Kaggle
kaggle_dirs = ['data/raw/news/kaggle_2024_2025', 'data/raw/news/kaggle_combined']
for kaggle_dir in kaggle_dirs:
    if os.path.exists(kaggle_dir):
        for csv_file in Path(kaggle_dir).glob('*.csv'):
            try:
                print(f"\nProcessing: {csv_file.name}")
                df = pd.read_csv(csv_file, low_memory=False, nrows=10)  # Preview first
                print(f"  Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"  Error: {str(e)}")

# Combine
if all_news:
    combined = pd.concat(all_news, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce')
    combined = combined.dropna(subset=['date'])
    combined = combined.drop_duplicates(subset=['headline', 'date'])
    
    # Filter to stock period
    combined = combined[(combined['date'] >= '2024-02-02') & (combined['date'] <= '2025-11-19')]
    combined = combined.sort_values('date')
    
    output_file = 'data/raw/news/all_news_combined.csv'
    combined.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("✓ COMBINED")
    print("=" * 60)
    print(f"Total: {len(combined):,} articles")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Days covered: {combined['date'].dt.date.nunique()}")
    print(f"Coverage: {(combined['date'].dt.date.nunique() / 657) * 100:.1f}%")
    print(f"Saved: {output_file}")
else:
    print("\n✗ No data to combine!")
