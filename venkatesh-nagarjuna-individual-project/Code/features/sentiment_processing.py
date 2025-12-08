"""
News sentiment processing using transformer-based models.
Aggregates sentiment to daily per-ticker features.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentProcessor:
    """Process news sentiment using FinBERT or similar transformer models."""

    def __init__(self, model_name: str = 'ProsusAI/finbert', device: str = None):
        """
        Initialize sentiment processor.

        Args:
            model_name: Hugging Face model name
            device: Device to run model on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading sentiment model: {model_name} on {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None

    def predict_sentiment(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Predict sentiment for a list of texts.

        Args:
            texts: List of news headlines/articles
            batch_size: Batch size for inference

        Returns:
            List of dictionaries with sentiment scores
        """
        if not self.model:
            logger.warning("Model not loaded, returning neutral sentiment")
            return [{'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'score': 0.0}] * len(texts)

        results = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                # Predict
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()

                # Process results
                for prob in probs:
                    # Assuming FinBERT outputs: [positive, negative, neutral]
                    sentiment = {
                        'positive': float(prob[0]),
                        'negative': float(prob[1]),
                        'neutral': float(prob[2]),
                        'score': float(prob[0] - prob[1])  # Compound score
                    }
                    results.append(sentiment)

        return results

    def process_news_data(self, news_df: pd.DataFrame,
                          text_col: str = 'headline',
                          ticker_col: str = 'ticker',
                          date_col: str = 'date') -> pd.DataFrame:
        """
        Process news DataFrame and add sentiment scores.

        Args:
            news_df: DataFrame with news articles
            text_col: Column name for text content
            ticker_col: Column name for ticker symbols
            date_col: Column name for dates

        Returns:
            DataFrame with added sentiment columns
        """
        logger.info(f"Processing sentiment for {len(news_df)} news items")

        if text_col not in news_df.columns:
            logger.error(f"Text column '{text_col}' not found")
            return news_df

        # Get sentiment predictions
        sentiments = self.predict_sentiment(news_df[text_col].fillna('').tolist())

        # Add to DataFrame
        news_df['sentiment_positive'] = [s['positive'] for s in sentiments]
        news_df['sentiment_negative'] = [s['negative'] for s in sentiments]
        news_df['sentiment_neutral'] = [s['neutral'] for s in sentiments]
        news_df['sentiment_score'] = [s['score'] for s in sentiments]

        logger.info("Sentiment processing complete")
        return news_df

    def aggregate_daily_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate news sentiment to daily features.

        Args:
            news_df: DataFrame with columns [ticker, date, sentiment_score]

        Returns:
            DataFrame with daily sentiment features per ticker
        """
        logger.info("Aggregating sentiment to daily features")

        if news_df.empty:
            return pd.DataFrame()

        # Ensure date is datetime
        news_df['date'] = pd.to_datetime(news_df['date'])

        # Classify sentiment
        news_df['sentiment_positive'] = (news_df['sentiment_score'] > 0.15).astype(int)
        news_df['sentiment_negative'] = (news_df['sentiment_score'] < -0.15).astype(int)
        news_df['sentiment_neutral'] = ((news_df['sentiment_score'] >= -0.15) &
                                        (news_df['sentiment_score'] <= 0.15)).astype(int)

        # Daily aggregation
        daily_agg = news_df.groupby(['ticker', 'date']).agg({
            'sentiment_score': ['mean', 'median', 'std', 'min', 'max'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'title': 'count'  # News count
        }).reset_index()

        # Flatten column names
        daily_agg.columns = ['ticker', 'date',
                             'sentiment_score_mean', 'sentiment_score_median',
                             'sentiment_score_std', 'sentiment_score_min', 'sentiment_score_max',
                             'sentiment_positive_mean', 'sentiment_negative_mean',
                             'sentiment_neutral_mean', 'news_count']

        # Fill NaN std with 0
        daily_agg['sentiment_score_std'] = daily_agg['sentiment_score_std'].fillna(0)

        # Add lagged features (1, 3, 5, 7 days)
        for lag in [1, 3, 5, 7]:
            for col in ['sentiment_score_mean', 'sentiment_positive_mean', 'sentiment_negative_mean']:
                daily_agg[f'{col}_lag_{lag}'] = daily_agg.groupby('ticker')[col].shift(lag)

        # Fill NaN from lagging
        daily_agg = daily_agg.fillna(0)

        logger.info(f"Daily sentiment shape: {daily_agg.shape}")

        return daily_agg

    @staticmethod
    def create_mock_news_data(tickers: List[str], start_date: str, end_date: str,
                              news_per_day: int = 5) -> pd.DataFrame:
        """
        Create mock news data for testing (when real news API is not available).

        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            news_per_day: Average number of news items per day per ticker

        Returns:
            DataFrame with mock news data
        """
        logger.info("Creating mock news data for testing")

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        news_items = []
        for ticker in tickers:
            for date in date_range:
                n_items = np.random.poisson(news_per_day)
                for _ in range(n_items):
                    # Generate random sentiment
                    sentiment_type = np.random.choice(['positive', 'negative', 'neutral'],
                                                      p=[0.4, 0.3, 0.3])

                    if sentiment_type == 'positive':
                        headline = f"{ticker} shows strong performance and growth potential"
                        score = np.random.uniform(0.3, 1.0)
                    elif sentiment_type == 'negative':
                        headline = f"{ticker} faces challenges amid market uncertainty"
                        score = np.random.uniform(-1.0, -0.3)
                    else:
                        headline = f"{ticker} announces quarterly results"
                        score = np.random.uniform(-0.2, 0.2)

                    news_items.append({
                        'ticker': ticker,
                        'date': date,
                        'headline': headline,
                        'sentiment_score': score,
                        'sentiment_positive': max(0, score),
                        'sentiment_negative': max(0, -score),
                        'sentiment_neutral': 1 - abs(score)
                    })

        news_df = pd.DataFrame(news_items)
        logger.info(f"Created {len(news_df)} mock news items")
        return news_df


if __name__ == "__main__":
    # Example usage with mock data
    tickers = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

    # Create mock news data
    news_df = SentimentProcessor.create_mock_news_data(
        tickers,
        start_date='2023-01-01',
        end_date='2023-12-31'
    )

    # Process sentiment (if model is available)
    processor = SentimentProcessor()
    news_df = processor.process_news_data(news_df)

    # Aggregate to daily features
    daily_sentiment = processor.aggregate_daily_sentiment(news_df)
    print(daily_sentiment.head())
