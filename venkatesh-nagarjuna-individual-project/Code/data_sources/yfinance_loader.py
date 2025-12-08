"""
Historical market data loader using yfinance.
Handles corporate actions and missing data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YFinanceLoader:
    """Load and preprocess historical stock data from Yahoo Finance."""

    def __init__(self, tickers: List[str], start_date: str = None, end_date: str = None):
        """
        Initialize the loader.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in 'YYYY-MM-DD' format (default: 5 years ago)
            end_date: End date in 'YYYY-MM-DD' format (default: today)
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
        self.data = {}

    def download_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download historical OHLCV data for all tickers.

        Returns:
            Dictionary mapping ticker to DataFrame with OHLCV data
        """
        logger.info(f"Downloading data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}")

        for ticker in self.tickers:
            try:
                logger.info(f"Fetching {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)

                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    continue

                # Reset index to make Date a column
                df = df.reset_index()
                df['Ticker'] = ticker

                # Rename columns for consistency
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                              'Dividends', 'Stock Splits', 'Ticker']

                # Handle missing data with forward fill, then backward fill
                df = df.sort_values('Date')
                df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[
                    ['Open', 'High', 'Low', 'Close', 'Volume']].ffill().bfill()

                # Remove rows with remaining NaNs
                df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

                self.data[ticker] = df
                logger.info(f"Successfully loaded {len(df)} rows for {ticker}")

            except Exception as e:
                logger.error(f"Error downloading data for {ticker}: {e}")
                continue

        return self.data

    def get_fundamentals(self, ticker: str) -> Dict:
        """
        Fetch basic fundamental data as static features.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary of fundamental metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            fundamentals = {
                'market_cap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'beta': info.get('beta', 1.0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
            }

            return fundamentals
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {ticker}: {e}")
            return {}

    def combine_all_tickers(self) -> pd.DataFrame:
        """
        Combine all ticker data into a single DataFrame.

        Returns:
            DataFrame with all tickers stacked
        """
        if not self.data:
            self.download_data()

        all_data = pd.concat([df for df in self.data.values()], axis=0, ignore_index=True)
        all_data = all_data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        logger.info(f"Combined data shape: {all_data.shape}")
        return all_data

    def save_data(self, filepath: str):
        """Save downloaded data to CSV."""
        combined = self.combine_all_tickers()

        # Remove timezone if present
        if 'Date' in combined.columns:
            combined['Date'] = pd.to_datetime(combined['Date']).dt.tz_localize(None)

        combined.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
    loader = YFinanceLoader(tickers)
    data = loader.download_data()
    loader.save_data("data/raw/historical_data.csv")
