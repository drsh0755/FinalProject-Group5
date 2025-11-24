import requests
import pandas as pd
from datetime import datetime, timedelta
import time

API_KEY = 'GR9K9CDZ4SK596YU'

def get_news_for_period(ticker, from_date, to_date, limit=1000):
    """Get news for a specific time period"""
    url = "https://www.alphavantage.co/query"
    
    # Format: YYYYMMDDTHHMM
    time_from = from_date.strftime('%Y%m%dT0000')
    time_to = to_date.strftime('%Y%m%dT2359')
    
    params = {
        'function': 'NEWS_SENTIMENT',
        'tickers': ticker,
        'time_from': time_from,
        'time_to': time_to,
        'apikey': API_KEY,
        'limit': limit,
        'sort': 'EARLIEST'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'Error Message' in data:
            print(f"  ✗ Error: {data['Error Message']}")
            return []
        
        if 'Note' in data:
            print(f"  ⚠ Rate limit")
            return []
        
        return data.get('feed', [])
    
    return []

print("=" * 60)
print("DOWNLOADING HISTORICAL NEWS (FEB 2024 - NOV 2025)")
print("=" * 60)

# Target period: Feb 2024 to Nov 2025
start_date = datetime(2024, 2, 1)
end_date = datetime(2025, 11, 24)

# Tickers that gave us data before
tickers = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'GOOGL']

all_articles = []
total_calls = 0

# Download in 3-month chunks for each ticker
for ticker in tickers:
    print(f"\n{'='*60}")
    print(f"Ticker: {ticker}")
    print(f"{'='*60}")
    
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=90), end_date)
        
        print(f"  Period: {current_start.date()} to {current_end.date()}")
        
        articles = get_news_for_period(ticker, current_start, current_end)
        
        for article in articles:
            try:
                date_str = article.get('time_published', '')
                if len(date_str) >= 8:
                    date = datetime.strptime(date_str[:8], '%Y%m%d')
                else:
                    continue
                
                sentiment_score = float(article.get('overall_sentiment_score', 0))
                sentiment_label = article.get('overall_sentiment_label', 'neutral')
                
                all_articles.append({
                    'date': date,
                    'headline': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'source': article.get('source', 'Unknown'),
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label,
                    'ticker': ticker
                })
            except:
                continue
        
        print(f"    Found: {len(articles)} articles")
        total_calls += 1
        
        current_start = current_end
        
        # Rate limiting: 5 calls per minute
        if total_calls % 5 == 0:
            print(f"    ⏸ Rate limit pause (used {total_calls} calls)...")
            time.sleep(60)
        else:
            time.sleep(12)
        
        # Free tier limit: 500 calls per day
        if total_calls >= 100:  # Safety limit
            print(f"\n⚠ Reached 100 API calls - stopping to preserve quota")
            print(f"   (Alpha Vantage free tier: 500 calls/day)")
            break
    
    if total_calls >= 100:
        break

# Save results
df = pd.DataFrame(all_articles)

if len(df) > 0:
    df = df.drop_duplicates(subset=['headline', 'date'])
    df = df.sort_values('date')
    
    output_file = 'data/raw/news/alphavantage_historical_news.csv'
    df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"✓ Total articles: {len(df)}")
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"✓ API calls used: {total_calls}/500")
    print(f"✓ Saved to: {output_file}")
    
    # Calculate coverage
    stock_start = datetime(2024, 2, 2)
    stock_end = datetime(2025, 11, 19)
    stock_days = (stock_end - stock_start).days
    
    news_days = df['date'].nunique()
    coverage = (news_days / stock_days) * 100
    
    print(f"\n📊 Coverage:")
    print(f"  Stock period: {stock_days} days")
    print(f"  Days with news: {news_days}")
    print(f"  Coverage: {coverage:.1f}%")
    
    print(f"\n📊 Sentiment:")
    print(f"  Mean: {df['sentiment_score'].mean():.4f}")
    print(df['sentiment_label'].value_counts())
else:
    print("\n✗ No articles downloaded")

print("\n" + "=" * 60)
