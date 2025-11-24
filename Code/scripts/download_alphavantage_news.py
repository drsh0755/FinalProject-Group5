import requests
import pandas as pd
from datetime import datetime
import time

# Add your API key here
API_KEY = 'GR9K9CDZ4SK596YU'

def get_news(tickers, limit=1000):
    """Get news with sentiment from Alpha Vantage"""
    url = "https://www.alphavantage.co/query"
    
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': tickers,
        'apikey': API_KEY,
        'limit': limit
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'Error Message' in data:
            print(f"  ✗ Error: {data['Error Message']}")
            return []
        
        if 'Note' in data:
            print(f"  ⚠ Rate limit hit")
            return []
        
        return data.get('feed', [])
    
    return []

print("=" * 60)
print("ALPHA VANTAGE NEWS WITH SENTIMENT")
print("=" * 60)

# Ticker groups (max 3-4 tickers per group works best)
ticker_groups = [
    'SPY',
    'QQQ',
    'AAPL',
    'MSFT',
    'GOOGL',
    'AMZN',
    'TSLA',
    'NVDA',
]

all_articles = []

for i, ticker in enumerate(ticker_groups):
    print(f"\n[{i+1}/{len(ticker_groups)}] Fetching {ticker}...")
    
    articles = get_news(ticker, limit=200)  # 200 per ticker
    
    for article in articles:
        try:
            # Parse date
            date_str = article.get('time_published', '')
            if len(date_str) >= 8:
                date = datetime.strptime(date_str[:8], '%Y%m%d')
            else:
                continue
            
            # Extract sentiment (ALREADY PROVIDED!)
            sentiment_score = float(article.get('overall_sentiment_score', 0))
            sentiment_label = article.get('overall_sentiment_label', 'neutral')
            
            all_articles.append({
                'date': date,
                'headline': article.get('title', ''),
                'summary': article.get('summary', ''),
                'source': article.get('source', 'Unknown'),
                'sentiment_score': sentiment_score,  # ← Already calculated!
                'sentiment_label': sentiment_label,   # ← Already calculated!
                'ticker': ticker
            })
        
        except Exception as e:
            continue
    
    print(f"  ✓ {len(articles)} articles")
    
    # Rate limit: 5 calls/minute for free tier
    if (i + 1) % 5 == 0:
        print("  ⏸ Waiting 60s (rate limit)...")
        time.sleep(60)
    else:
        time.sleep(12)

# Save
df = pd.DataFrame(all_articles)

if len(df) > 0:
    df = df.drop_duplicates(subset=['headline', 'date'])
    df = df.sort_values('date')
    
    output_file = 'data/raw/news/alphavantage_news_with_sentiment.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("✓ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"Total articles: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Saved to: {output_file}")
    print(f"\n📊 Sentiment Distribution:")
    print(f"  Mean: {df['sentiment_score'].mean():.4f}")
    print(f"  Positive: {(df['sentiment_label'] == 'Bullish').sum()}")
    print(f"  Negative: {(df['sentiment_label'] == 'Bearish').sum()}")
    print(f"  Neutral: {(df['sentiment_label'] == 'Neutral').sum()}")
    print("\n✨ Sentiment already calculated - NO FinBERT needed!")
else:
    print("\n✗ No articles downloaded")
    print("Check your API key!")
