"""
Unified market data provider that abstracts yfinance and Alpha Vantage.
Handles caching and data source selection.
"""

import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pickle
import os

from .yfinance_loader import YFinanceLoader
from .alphavantage_loader import AlphaVantageLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketDataProvider:
    """Unified interface for market data from multiple sources."""

    def __init__(self, tickers: List[str], alphavantage_key: Optional[str] = None,
                 cache_dir: str = 'data/cache'):
        """
        Initialize the provider.

        Args:
            tickers: List of stock ticker symbols
            alphavantage_key: Alpha Vantage API key (optional)
            cache_dir: Directory for caching data
        """
        self.tickers = tickers
        self.yf_loader = YFinanceLoader(tickers)
        self.av_loader = AlphaVantageLoader(alphavantage_key) if alphavantage_key else None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self._historical_cache = {}
        self._quote_cache = {}
        self._cache_timestamp = {}

    def get_historical_data(self, start_date: str = None, end_date: str = None,
                            use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical data for all tickers.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with historical OHLCV data
        """
        cache_key = f"historical_{start_date}_{end_date}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if use_cache and os.path.exists(cache_file):
            cache_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
            if cache_age < 1:  # Cache valid for 1 day
                logger.info(f"Loading historical data from cache ({cache_file})")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)

        # Fetch fresh data
        logger.info("Fetching fresh historical data...")
        self.yf_loader.start_date = start_date
        self.yf_loader.end_date = end_date
        data = self.yf_loader.combine_all_tickers()

        # Save to cache
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Cached historical data to {cache_file}")

        return data

    def get_latest_quotes(self, use_cache: bool = True, cache_timeout: int = 60) -> pd.DataFrame:
        """
        Get latest quotes for all tickers.

        Args:
            use_cache: Whether to use cached quotes
            cache_timeout: Cache timeout in seconds

        Returns:
            DataFrame with latest quote data
        """
        cache_key = 'latest_quotes'

        if use_cache and cache_key in self._quote_cache:
            cache_age = (datetime.now() - self._cache_timestamp.get(cache_key, datetime.min)).total_seconds()
            if cache_age < cache_timeout:
                logger.info(f"Using cached quotes (age: {cache_age:.1f}s)")
                return self._quote_cache[cache_key]

        # Fetch fresh quotes
        if self.av_loader:
            logger.info("Fetching latest quotes from Alpha Vantage...")
            quotes = self.av_loader.get_multiple_quotes(self.tickers)
        else:
            logger.info("Fetching latest quotes from yfinance...")
            quotes = self._get_yfinance_quotes()

        # Cache the result
        self._quote_cache[cache_key] = quotes
        self._cache_timestamp[cache_key] = datetime.now()

        return quotes

    def _get_yfinance_quotes(self) -> pd.DataFrame:
        """Fallback method to get latest quotes using yfinance."""
        quotes = []
        for ticker in self.tickers:
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info

                quote = {
                    'ticker': ticker,
                    'timestamp': datetime.now().isoformat(),
                    'close': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'volume': info.get('volume', 0),
                    'change_percent': info.get('regularMarketChangePercent', 0)
                }
                quotes.append(quote)
            except Exception as e:
                logger.error(f"Error fetching yfinance quote for {ticker}: {e}")
                continue

        return pd.DataFrame(quotes)

    def get_fundamentals(self, ticker: str) -> Dict:
        """Get fundamental data for a ticker."""
        return self.yf_loader.get_fundamentals(ticker)

    def clear_cache(self):
        """Clear all cached data."""
        self._historical_cache.clear()
        self._quote_cache.clear()
        self._cache_timestamp.clear()
        logger.info("Cache cleared")


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
    provider = MarketDataProvider(tickers)

    # Get historical data
    historical = provider.get_historical_data()
    print(f"Historical data shape: {historical.shape}")

    # Get latest quotes
    quotes = provider.get_latest_quotes()
    print(f"Latest quotes:\n{quotes}")
