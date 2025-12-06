#!/usr/bin/env python3
"""Download 2 years of stock data"""
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
data_dir = Path(__file__).parent.parent / 'data' / 'raw'
data_dir.mkdir(parents=True, exist_ok=True)

# Date range: 2 years back from today
end_date = datetime.now()
start_date = end_date - timedelta(days=730)

tickers = ['QQQ', 'SPY', 'DIA', '^VIX', '^TNX']

print(f"\nDownloading 2 years of data: {start_date.date()} to {end_date.date()}\n")

for ticker in tickers:
    print(f"Downloading {ticker}...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if not df.empty:
            filename = data_dir / f'{ticker}_2year.csv'
            df.to_csv(filename)
            print(f"  ✓ Saved: {filename} ({len(df)} days)")
        else:
            print(f"  ✗ No data returned for {ticker}")

    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {e}")

print("\n✓ Download complete!")
