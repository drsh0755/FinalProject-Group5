"""
Real-time and intraday data loader using Alpha Vantage API.
Includes rate limiting and exponential backoff.
"""

import requests
import pandas as pd
import time
import logging
from typing import Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlphaVantageLoader:
    """Fetch real-time and intraday stock data from Alpha Vantage."""

    BASE_URL = "https://www.alphavantage.co/query"
    RATE_LIMIT_DELAY = 12  # seconds between calls (free tier: 5 calls/min)

    def __init__(self, api_key: str):
        """
        Initialize the loader with API key.

        Args:
            api_key: Alpha Vantage API key
        """
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")
        self.api_key = api_key
        self.last_call_time = 0

    def _rate_limit(self):
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_call_time = time.time()

    def _make_request(self, params: Dict, max_retries: int = 3) -> Dict:
        """
        Make API request with exponential backoff retry logic.

        Args:
            params: Query parameters
            max_retries: Maximum number of retry attempts

        Returns:
            JSON response as dictionary
        """
        self._rate_limit()
        params['apikey'] = self.api_key

        for attempt in range(max_retries):
            try:
                response = requests.get(self.BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                # Check for API error messages
                if 'Error Message' in data:
                    raise ValueError(f"API Error: {data['Error Message']}")
                if 'Note' in data:
                    logger.warning(f"API Note: {data['Note']}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt * 60  # exponential backoff
                        logger.info(f"Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                        continue

                return data

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError("Max retries exceeded")

    def get_intraday(self, ticker: str, interval: str = '5min') -> pd.DataFrame:
        """
        Fetch intraday time series data.

        Args:
            ticker: Stock ticker symbol
            interval: Time interval (1min, 5min, 15min, 30min, 60min)

        Returns:
            DataFrame with intraday OHLCV data
        """
        logger.info(f"Fetching intraday data for {ticker} ({interval})")

        params = {
            'function': 'TIME_SERIES_INTRADAY',
            'symbol': ticker,
            'interval': interval,
            'outputsize': 'full'
        }

        data = self._make_request(params)

        time_series_key = f'Time Series ({interval})'
        if time_series_key not in data:
            logger.error(f"Unexpected response format: {list(data.keys())}")
            return pd.DataFrame()

        df = pd.DataFrame.from_dict(data[time_series_key], orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Rename columns
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.astype(float)
        df['Ticker'] = ticker
        df = df.reset_index().rename(columns={'index': 'Timestamp'})

        logger.info(f"Retrieved {len(df)} intraday bars for {ticker}")
        return df

    def get_quote(self, ticker: str) -> Dict:
        """
        Fetch latest quote/bar for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with latest quote data
        """
        logger.info(f"Fetching latest quote for {ticker}")

        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': ticker
        }

        data = self._make_request(params)

        if 'Global Quote' not in data:
            logger.error(f"Unexpected response format: {list(data.keys())}")
            return {}

        quote = data['Global Quote']

        # Parse and return relevant fields
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'open': float(quote.get('02. open', 0)),
            'high': float(quote.get('03. high', 0)),
            'low': float(quote.get('04. low', 0)),
            'close': float(quote.get('05. price', 0)),
            'volume': float(quote.get('06. volume', 0)),
            'latest_trading_day': quote.get('07. latest trading day', ''),
            'change': float(quote.get('09. change', 0)),
            'change_percent': quote.get('10. change percent', '').rstrip('%')
        }

    def get_multiple_quotes(self, tickers: list) -> pd.DataFrame:
        """
        Fetch latest quotes for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            DataFrame with latest quote data for all tickers
        """
        quotes = []
        for ticker in tickers:
            try:
                quote = self.get_quote(ticker)
                if quote:
                    quotes.append(quote)
            except Exception as e:
                logger.error(f"Error fetching quote for {ticker}: {e}")
                continue

        return pd.DataFrame(quotes)


if __name__ == "__main__":
    # Example usage (requires API key)
    import os

    api_key = os.getenv('ALPHAVANTAGE_API_KEY', 'demo')
    loader = AlphaVantageLoader(api_key)

    # Test quote fetch
    quote = loader.get_quote('AAPL')
    print(quote)
