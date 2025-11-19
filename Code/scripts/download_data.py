"""
Download historical stock market data for training
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import *

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    print(f"\n{'='*60}")
    print(f"Downloading {ticker}...")
    print(f"{'='*60}")
    
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"⚠ No data returned for {ticker}")
            return None
        
        # Save to CSV
        filename = f"{ticker.replace('^', '')}_historical.csv"
        filepath = RAW_DATA_DIR / filename
        df.to_csv(filepath)
        
        print(f"✓ Downloaded {len(df)} days of data")
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Saved to: {filepath}")
        
        # Display sample
        print(f"\nSample data (last 3 days):")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(3))
        
        return df
        
    except Exception as e:
        print(f"✗ Error downloading {ticker}: {e}")
        return None

def main():
    """Main function to download all required data"""
    print("\n" + "="*60)
    print("STOCK MARKET DATA DOWNLOAD")
    print("="*60)
    print(f"Date range: {TRAIN_START} to {TRAIN_END}")
    print(f"Target directory: {RAW_DATA_DIR}")
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Track successful downloads
    successful = []
    failed = []
    
    # Download primary ticker
    print(f"\n{'='*60}")
    print(f"PRIMARY TICKER")
    print(f"{'='*60}")
    df = download_stock_data(PRIMARY_TICKER, TRAIN_START, TRAIN_END)
    if df is not None:
        successful.append(PRIMARY_TICKER)
    else:
        failed.append(PRIMARY_TICKER)
    
    # Download market indices
    print(f"\n{'='*60}")
    print(f"MARKET INDICES")
    print(f"{'='*60}")
    for ticker in MARKET_TICKERS:
        if ticker != PRIMARY_TICKER:  # Skip if already downloaded
            df = download_stock_data(ticker, TRAIN_START, TRAIN_END)
            if df is not None:
                successful.append(ticker)
            else:
                failed.append(ticker)
    
    # Download context indicators
    print(f"\n{'='*60}")
    print(f"CONTEXT INDICATORS")
    print(f"{'='*60}")
    for ticker in CONTEXT_TICKERS:
        df = download_stock_data(ticker, TRAIN_START, TRAIN_END)
        if df is not None:
            successful.append(ticker)
        else:
            failed.append(ticker)
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successful: {len(successful)} tickers")
    for ticker in successful:
        print(f"  - {ticker}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)} tickers")
        for ticker in failed:
            print(f"  - {ticker}")
    
    print(f"\n{'='*60}")
    print("Data download complete!")
    print(f"Files saved to: {RAW_DATA_DIR}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
