#!/usr/bin/env python3
"""
SMART NEWS FETCHER - Auto Date Range Coverage
Automatically fetches enough news to cover from Feb 2024 to today
Intelligently checks current file and fetches only missing dates
"""

import os
import sys
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
import requests


class SmartNewsAPIFetcher:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('ALPHA_VANTAGE_KEY')

        if not self.api_key:
            print("❌ ERROR: ALPHA_VANTAGE_KEY not found!")
            print("   export ALPHA_VANTAGE_KEY='your_api_key'")
            sys.exit(1)

        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12

    def get_latest_date_in_file(self, file_path):
        """Get the latest date from existing file."""
        if not os.path.exists(file_path):
            print(f"ℹ️  No existing file found")
            return None

        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None

            # Parse dates - they're stored as YYYYMMDD
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            latest_date = df['date'].max()
            earliest_date = df['date'].min()

            if pd.isna(latest_date):
                return None

            print(f"📂 Found existing file with {len(df)} articles")
            print(f"   Date range: {earliest_date.date()} → {latest_date.date()}")
            print(f"   Days covered: {df['date'].nunique()}")

            return latest_date
        except Exception as e:
            print(f"⚠️  Error reading file: {e}")
            return None

    def get_required_pages(self, target_start_date):
        """
        Calculate how many pages are needed to cover from target_start_date to today.
        Rough estimate: ~100 articles per page, assuming ~5-10 articles per day
        So to go back N days, we need roughly N/8 pages (conservative estimate)
        """
        today = datetime.now().date()
        days_to_cover = (today - target_start_date).days

        # Conservative estimate: fetch more pages to ensure coverage
        # Assuming ~7 articles per day average
        estimated_pages = max(20, int(days_to_cover / 5))

        print(f"📅 Target date range: {target_start_date} → {today}")
        print(f"   Days to cover: {days_to_cover}")
        print(f"   Estimated pages needed: {estimated_pages} (conservative)")

        return estimated_pages

    def fetch_news(self, pages, ticker='SPY'):
        """Fetch news from Alpha Vantage API."""
        all_articles = []

        print(f"\n📰 Fetching {pages} pages of news for {ticker}...")
        print(f"   (This will take ~{pages * 12} seconds due to API rate limits)")

        for page in range(1, pages + 1):
            print(f"   Page {page}/{pages}...", end='', flush=True)

            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': ticker,
                'limit': 100,
                'sort': 'LATEST',
                'page': page,
                'apikey': self.api_key
            }

            try:
                response = requests.get(self.base_url, params=params, timeout=10)
                data = response.json()

                if 'feed' in data:
                    articles = data['feed']
                    print(f" ✓ {len(articles)} articles")

                    for article in articles:
                        # Extract date from format like '20251130T020649'
                        pub_time_full = article['time_published']
                        pub_date = pub_time_full[:8]  # Extract YYYYMMDD only

                        sentiment_score = float(article.get('overall_sentiment_score', 0))

                        all_articles.append({
                            'date': pub_date,
                            'title': article['title'],
                            'sentiment_score': sentiment_score,
                            'relevance': float(article.get('relevance_score', 0)),
                            'url': article.get('url', '')
                        })
                else:
                    print(f" ⚠️  No articles")

            except Exception as e:
                print(f" ❌ Error: {e}")

            if page < pages:
                time.sleep(self.rate_limit_delay)

        print(f"\n✅ Total articles fetched: {len(all_articles)}")
        return all_articles

    def filter_by_date(self, articles, start_date):
        """Filter articles to keep only those from start_date onwards."""
        filtered = []

        for article in articles:
            try:
                article_date = pd.to_datetime(article['date'], format='%Y%m%d')
                if article_date.date() >= start_date:
                    filtered.append(article)
            except:
                continue

        return filtered

    def save_articles(self, articles, output_file):
        """Save articles, merging with existing file if present."""
        if not articles:
            print("❌ No articles to save!")
            return False

        # Convert to dataframe
        new_df = pd.DataFrame(articles)

        # Parse dates - they should be YYYYMMDD format now
        new_df['date'] = pd.to_datetime(new_df['date'], format='%Y%m%d', errors='coerce')
        new_df = new_df.dropna(subset=['date'])

        print(f"\n📂 Processing {len(new_df)} articles...")

        # Load existing if present
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            existing_df['date'] = pd.to_datetime(existing_df['date'], format='%Y%m%d', errors='coerce')

            # Combine
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # Remove duplicates
            before = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=['date', 'title'], keep='first')
            after = len(combined_df)

            print(f"   Merged {len(existing_df)} existing + {len(new_df)} new")
            print(f"   Removed {before - after} duplicates")
        else:
            combined_df = new_df
            print(f"   Creating new file with {len(new_df)} articles")

        # Sort and save
        combined_df = combined_df.sort_values('date').reset_index(drop=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        # Convert date back to YYYYMMDD format for storage
        combined_df['date'] = combined_df['date'].dt.strftime('%Y%m%d')
        combined_df.to_csv(output_file, index=False)

        # Read back for stats
        stats_df = pd.read_csv(output_file)
        stats_df['date'] = pd.to_datetime(stats_df['date'], format='%Y%m%d')

        print(f"\n✅ Saved to: {output_file}")
        print(f"   Total articles: {len(stats_df)}")
        print(f"   Date range: {stats_df['date'].min().date()} → {stats_df['date'].max().date()}")
        print(f"   Days covered: {stats_df['date'].nunique()}")
        print(f"   Sentiment mean: {stats_df['sentiment_score'].mean():.3f}")
        print(f"   Sentiment std: {stats_df['sentiment_score'].std():.4f}")

        return True


def main():
    parser = argparse.ArgumentParser(description='Smart news fetcher with auto date range coverage')

    parser.add_argument('--output', type=str, default='Code/data/processed/alphavantage_news_2024_2025.csv')
    parser.add_argument('--start-date', type=str, default='2024-02-02', help='Target start date (default: 2024-02-02)')
    parser.add_argument('--force-full', action='store_true', help='Overwrite existing file and fetch fresh')
    parser.add_argument('--ticker', type=str, default='SPY')
    parser.add_argument('--pages', type=int, help='Override automatic page calculation')

    args = parser.parse_args()

    print("=" * 70)
    print("SMART NEWS FETCHER - AUTO DATE RANGE COVERAGE")
    print("=" * 70)

    fetcher = SmartNewsAPIFetcher()

    target_start_date = pd.to_datetime(args.start_date).date()

    # Check existing file unless force-full
    if not args.force_full:
        print("\n🔍 Checking existing file...")
        latest_date = fetcher.get_latest_date_in_file(args.output)

        if latest_date and latest_date.date() >= (datetime.now().date() - timedelta(days=1)):
            print(f"\n✅ File is up-to-date! (Latest: {latest_date.date()})")
            print("No new articles to fetch.")
            return

    # Calculate pages needed
    print()
    if args.pages:
        pages_to_fetch = args.pages
        print(f"🔧 Using specified pages: {pages_to_fetch}")
    else:
        pages_to_fetch = fetcher.get_required_pages(target_start_date)

    # Fetch articles
    articles = fetcher.fetch_news(pages=pages_to_fetch, ticker=args.ticker)

    if not articles:
        print("❌ No articles fetched!")
        sys.exit(1)

    # Filter by start date
    print(f"\n🔍 Filtering articles from {target_start_date} onwards...")
    filtered_articles = fetcher.filter_by_date(articles, target_start_date)
    print(f"   ✓ {len(filtered_articles)} articles in target date range")

    if not filtered_articles:
        print("⚠️  No articles in target date range!")
        return

    # Save articles
    fetcher.save_articles(filtered_articles, args.output)

    print("\n" + "=" * 70)
    print("✅ NEWS FETCHING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    main()