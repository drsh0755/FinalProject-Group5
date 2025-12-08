"""
Download historical data with technical indicators and sentiment from Alpha Vantage.
One-stop script to create a complete dataset ready for training.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sources.market_data_provider import MarketDataProvider
from features.technical_indicators import TechnicalIndicators
from features.sentiment_processing import SentimentProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_news_sentiment(tickers, start_date, end_date, api_key):
    """
    Fetch news sentiment from Alpha Vantage News API.

    Args:
        tickers: List of ticker symbols
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        api_key: Alpha Vantage API key

    Returns:
        DataFrame with news sentiment aggregated daily
    """
    import requests
    import time

    all_news = []

    for idx, ticker in enumerate(tickers):
        logger.info(f"Fetching news for {ticker} ({idx+1}/{len(tickers)})...")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key,
            "limit": 1000,
            "time_from": start_date.replace("-", "") + "T0000",
            "time_to": end_date.replace("-", "") + "T2359",
            "sort": "LATEST"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            # Check for API errors
            if "Note" in data:
                logger.warning(f"API limit reached: {data['Note']}")
                logger.info("Sleeping for 60 seconds...")
                time.sleep(60)
                continue

            if "feed" not in data:
                logger.warning(f"No news data for {ticker}")
                continue

            for article in data["feed"]:
                # Parse timestamp
                time_published = article.get("time_published", "")
                if len(time_published) >= 8:
                    date_str = time_published[:8]
                    date = pd.to_datetime(date_str, format="%Y%m%d")
                else:
                    continue

                # Get ticker-specific sentiment
                ticker_sentiment = None
                for ticker_data in article.get("ticker_sentiment", []):
                    if ticker_data.get("ticker") == ticker:
                        ticker_sentiment = float(ticker_data.get("ticker_sentiment_score", 0))
                        break

                if ticker_sentiment is None:
                    ticker_sentiment = float(article.get("overall_sentiment_score", 0))

                all_news.append({
                    "ticker": ticker,
                    "date": date,
                    "sentiment_score": ticker_sentiment,
                })

            logger.info(f"Fetched {len(data['feed'])} articles for {ticker}")

            # Rate limiting: Alpha Vantage free tier = 25 requests/day
            if idx < len(tickers) - 1:
                time.sleep(12)  # ~5 requests per minute

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            continue

    if not all_news:
        logger.warning("No news articles fetched")
        return pd.DataFrame()

    news_df = pd.DataFrame(all_news)
    logger.info(f"Total news articles: {len(news_df)}")

    return news_df


def main():
    parser = argparse.ArgumentParser(
        description='Download stock data with technical indicators and sentiment'
    )
    parser.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        help='Ticker symbols to download'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=(datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD), default: 4 years ago'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/historical_data.csv',
        help='Output file path'
    )
    parser.add_argument(
        '--alphavantage-key',
        type=str,
        help='Alpha Vantage API key (or set ALPHAVANTAGE_API_KEY env var)'
    )
    parser.add_argument(
        '--skip-sentiment',
        action='store_true',
        help='Skip sentiment fetching (faster, but no sentiment features)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.alphavantage_key or os.getenv('ALPHAVANTAGE_API_KEY')

    logger.info("="*80)
    logger.info("STOCK DATA DOWNLOAD WITH TECHNICAL + SENTIMENT FEATURES")
    logger.info("="*80)
    logger.info(f"Tickers: {args.tickers}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output: {args.output}")
    logger.info("="*80)

    # Step 1: Download price data
    logger.info("\n[1/4] Downloading historical price data...")
    provider = MarketDataProvider(tickers=args.tickers)

    try:
        df = provider.get_historical_data(
            start_date=args.start_date,
            end_date=args.end_date
        )
    except Exception as e:
        logger.error(f"Failed to download price data: {e}")
        return 1

    if df.empty:
        logger.error("No price data downloaded")
        return 1

    logger.info(f"✓ Downloaded {len(df)} rows for {df['Ticker'].nunique()} tickers")

    # Step 2: Add technical indicators
    logger.info("\n[2/4] Computing technical indicators...")

    try:
        df = TechnicalIndicators.add_all_indicators(df)
        logger.info(f"✓ Added technical indicators, total columns: {len(df.columns)}")
    except Exception as e:
        logger.error(f"Failed to compute technical indicators: {e}")
        return 1

    # Step 3: Fetch and add sentiment
    if not args.skip_sentiment:
        if not api_key:
            logger.warning("No Alpha Vantage API key provided. Skipping sentiment.")
            logger.info("Set ALPHAVANTAGE_API_KEY environment variable or use --alphavantage-key")
            args.skip_sentiment = True

    if not args.skip_sentiment:
        logger.info("\n[3/4] Fetching news sentiment from Alpha Vantage...")
        logger.info("This may take several minutes due to API rate limits...")

        try:
            news_df = fetch_news_sentiment(
                tickers=args.tickers,
                start_date=args.start_date,
                end_date=args.end_date,
                api_key=api_key
            )

            if not news_df.empty:
                # Process sentiment
                sentiment_processor = SentimentProcessor()
                daily_sentiment = sentiment_processor.aggregate_daily_sentiment(news_df)

                # Rename to match
                daily_sentiment = daily_sentiment.rename(columns={
                    'ticker': 'Ticker',
                    'date': 'Date'
                })

                # Get sentiment columns
                sentiment_cols = [c for c in daily_sentiment.columns if c not in ['Ticker', 'Date']]

                # Merge
                df = df.merge(
                    daily_sentiment[['Ticker', 'Date'] + sentiment_cols],
                    on=['Ticker', 'Date'],
                    how='left'
                )

                # Fill missing with 0
                df[sentiment_cols] = df[sentiment_cols].fillna(0)

                logger.info(f"✓ Added {len(sentiment_cols)} sentiment features")
            else:
                logger.warning("No sentiment data available, adding zero features")
                args.skip_sentiment = True

        except Exception as e:
            logger.error(f"Sentiment fetching failed: {e}")
            logger.warning("Continuing without sentiment features")
            args.skip_sentiment = True

    # Add zero sentiment if skipped
    if args.skip_sentiment:
        logger.info("\n[3/4] Adding zero sentiment features...")
        sentiment_cols = [
            'sentiment_score_mean', 'sentiment_score_median', 'sentiment_score_std',
            'sentiment_score_min', 'sentiment_score_max', 'news_count',
            'sentiment_positive_mean', 'sentiment_negative_mean', 'sentiment_neutral_mean',
            'sentiment_score_mean_lag_1', 'sentiment_positive_mean_lag_1',
            'sentiment_negative_mean_lag_1', 'sentiment_score_mean_lag_3',
            'sentiment_positive_mean_lag_3', 'sentiment_negative_mean_lag_3',
            'sentiment_score_mean_lag_5', 'sentiment_positive_mean_lag_5',
            'sentiment_negative_mean_lag_5', 'sentiment_score_mean_lag_7',
            'sentiment_positive_mean_lag_7', 'sentiment_negative_mean_lag_7'
        ]
        for col in sentiment_cols:
            df[col] = 0.0
        logger.info(f"✓ Added {len(sentiment_cols)} zero sentiment features")

    # Step 4: Save
    logger.info("\n[4/4] Saving to disk...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"✓ Final shape: {df.shape}")
    logger.info(f"✓ Columns: {list(df.columns)[:10]}...")

    logger.info("\n" + "="*80)
    logger.info("✓ DOWNLOAD COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nNext steps:")
    logger.info(f"1. Process data:")
    logger.info(f"   python scripts/preprocess_data.py \\")
    logger.info(f"       --input {args.output} \\")
    logger.info(f"       --output data/processed/features_with_sentiment.csv")
    logger.info(f"\n2. Train model:")
    logger.info(f"   python training/train_tft.py \\")
    logger.info(f"       --config training/train_config.yaml \\")
    logger.info(f"       --data data/processed/features_with_sentiment.csv")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
