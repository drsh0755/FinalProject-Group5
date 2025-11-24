import yfinance as yf
import pandas as pd
from datetime import datetime
import time

def get_all_news(ticker, max_results=1000):
    """Get all available news for a ticker"""
    print(f"Fetching news for {ticker}...")
    
    stock = yf.Ticker(ticker)
    news = stock.news
    
    articles = []
    for item in news:
        try:
            articles.append({
                'date': datetime.fromtimestamp(item['providerPublishTime']),
                'headline': item['title'],
                'publisher': item.get('publisher', 'Unknown'),
                'link': item.get('link', ''),
                'ticker': ticker
            })
        except Exception as e:
            continue
    
    return articles

# List of tickers to get news for
tickers = [
    'SPY', 'QQQ', 'DIA',  # Major ETFs
    '^VIX', '^TNX',  # Volatility and Treasury
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Big tech
    'JPM', 'BAC', 'GS',  # Financials
    'XLE', 'XLF', 'XLK'  # Sector ETFs
]

print("=" * 60)
print("DOWNLOADING YAHOO FINANCE NEWS")
print("=" * 60)

all_articles = []
for ticker in tickers:
    try:
        articles = get_all_news(ticker)
        all_articles.extend(articles)
        print(f"✓ {ticker}: {len(articles)} articles")
        time.sleep(2)  # Be nice to Yahoo
    except Exception as e:
        print(f"✗ {ticker}: {str(e)}")

# Create DataFrame
df = pd.DataFrame(all_articles)

# Remove duplicates
df = df.drop_duplicates(subset=['headline', 'date'])
df = df.sort_values('date')

# Save
output_file = 'data/raw/news/yahoo_news_2023_2025.csv'
df.to_csv(output_file, index=False)

print("=" * 60)
print(f"✓ Total unique articles: {len(df)}")
if len(df) > 0:
    print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
print(f"✓ Saved to: {output_file}")
print("=" * 60)
