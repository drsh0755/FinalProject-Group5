"""
Real-Time Live Prediction System - WITH COMPREHENSIVE LOGGING
Fetches latest data and news, makes predictions for tomorrow

"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import requests
from transformers import BertTokenizer, BertForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm.model import StockLSTM
from utils.logger import setup_logger, log_section, log_dict, log_dataframe_info

# Exact feature order (46 features)
EXPECTED_FEATURES = [
    'high', 'low', 'open', 'volume',
    'returns', 'log_returns',
    'sma_5', 'ema_5', 'sma_10', 'ema_10', 'sma_20', 'ema_20', 'sma_50', 'ema_50',
    'rsi', 'stoch', 'williams_r',
    'macd', 'macd_signal', 'macd_diff',
    'bb_high', 'bb_low', 'bb_mid', 'bb_width',
    'volume_sma_20', 'volume_ratio', 'obv',
    'atr',
    'volatility_5', 'volatility_10', 'volatility_20',
    'adx', 'cci', 'price_position',
    'close_lag_1', 'returns_lag_1',
    'close_lag_2', 'returns_lag_2',
    'close_lag_3', 'returns_lag_3',
    'close_lag_5', 'returns_lag_5',
    'sentiment_mean', 'sentiment_std', 'article_count', 'positive_ratio'
]


class RealtimePredictor:
    def __init__(self, model_path, results_path, logger):
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        log_section(self.logger, "SYSTEM INITIALIZATION")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")

        # Load config
        self.logger.info(f"Loading configuration from {results_path}")
        with open(results_path, 'r') as f:
            self.config = json.load(f)
        log_dict(self.logger, self.config, "Model Configuration")

        # Load model
        self.logger.info(f"Loading trained model from {model_path}")
        self.model = StockLSTM(
            input_size=46,
            hidden_size=self.config.get('hidden_size', 128),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model loaded successfully")
        self.logger.info(f"  Total parameters: {param_count:,}")
        self.logger.info(
            f"  Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        self.sequence_length = self.config.get('sequence_length', 30)

        # Load FinBERT
        self.logger.info("Loading FinBERT for sentiment analysis...")
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.sentiment_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.sentiment_model.eval()
        self.logger.info("FinBERT loaded successfully")

        self.av_api_key = "GR9K9CDZ4SK596YU"

        self.logger.info("✅ System initialization complete")

    def fetch_latest_stock_data(self, symbol='SPY', days=60):
        log_section(self.logger, f"FETCHING STOCK DATA: {symbol}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        self.logger.info(f"Requesting {days} days of data")
        self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                self.logger.error(f"No data returned for {symbol}")
                raise ValueError(f"No data returned for {symbol}")

            df = df.reset_index()
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={'date': 'Date'})

            self.logger.info(f"✅ Successfully fetched {len(df)} days of data")
            self.logger.info(f"Latest date: {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
            self.logger.info(f"Latest close: ${df['close'].iloc[-1]:.2f}")
            self.logger.info(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            self.logger.info(f"Average volume: {df['volume'].mean():,.0f}")

            return df

        except Exception as e:
            self.logger.error(f"Error fetching stock data: {str(e)}", exc_info=True)
            raise

    def fetch_latest_news(self, tickers):
        log_section(self.logger, "FETCHING NEWS DATA")
        self.logger.info(f"Target companies: {', '.join(tickers)}")

        all_articles = []
        successful_fetches = 0
        failed_fetches = 0

        for ticker in tickers:
            try:
                self.logger.info(f"Fetching news for {ticker}...")
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ticker,
                    'apikey': self.av_api_key,
                    'limit': 50
                }

                response = requests.get(url, params=params, timeout=10)
                data = response.json()

                if 'feed' in data:
                    for item in data['feed']:
                        all_articles.append({
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'time_published': item.get('time_published', ''),
                            'ticker': ticker
                        })
                    self.logger.info(f"  ✅ {ticker}: {len(data['feed'])} articles")
                    successful_fetches += 1
                else:
                    self.logger.warning(f"  ⚠️  {ticker}: No feed in response")
                    failed_fetches += 1

            except Exception as e:
                self.logger.error(f"  ❌ {ticker}: {str(e)}")
                failed_fetches += 1
                continue

        self.logger.info(f"")
        self.logger.info(f"News fetch summary:")
        self.logger.info(f"  Successful: {successful_fetches}/{len(tickers)}")
        self.logger.info(f"  Failed: {failed_fetches}/{len(tickers)}")
        self.logger.info(f"  Total articles: {len(all_articles)}")

        return all_articles

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        positive = predictions[0][0].item()
        negative = predictions[0][1].item()

        return positive - negative

    def process_news_sentiment(self, articles):
        log_section(self.logger, "SENTIMENT ANALYSIS")

        if not articles:
            self.logger.warning("⚠️  No articles to process, using neutral sentiment")
            return {
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'article_count': 0,
                'positive_ratio': 0.5
            }

        self.logger.info(f"Processing {len(articles)} articles with FinBERT")

        sentiments = []
        start_time = datetime.now()

        for i, article in enumerate(articles):
            text = f"{article['title']}. {article['summary']}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)

            if (i + 1) % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (i + 1) / elapsed
                remaining = (len(articles) - i - 1) / rate
                self.logger.info(
                    f"  Progress: {i + 1}/{len(articles)} ({rate:.1f} articles/sec, ~{remaining:.0f}s remaining)")

        sentiments = np.array(sentiments)

        sentiment_features = {
            'sentiment_mean': sentiments.mean(),
            'sentiment_std': sentiments.std(),
            'article_count': len(sentiments),
            'positive_ratio': (sentiments > 0).sum() / len(sentiments)
        }

        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.info(f"")
        self.logger.info(f"✅ Sentiment analysis complete in {processing_time:.1f} seconds")
        log_dict(self.logger, sentiment_features, "Sentiment Features")

        # Log distribution
        positive_count = (sentiments > 0).sum()
        negative_count = (sentiments < 0).sum()
        neutral_count = (sentiments == 0).sum()
        self.logger.info(f"Sentiment distribution:")
        self.logger.info(f"  Positive: {positive_count} ({positive_count / len(sentiments) * 100:.1f}%)")
        self.logger.info(f"  Negative: {negative_count} ({negative_count / len(sentiments) * 100:.1f}%)")
        self.logger.info(f"  Neutral: {neutral_count} ({neutral_count / len(sentiments) * 100:.1f}%)")

        return sentiment_features

    def compute_technical_indicators(self, df):
        log_section(self.logger, "COMPUTING TECHNICAL INDICATORS")

        initial_rows = len(df)
        self.logger.info(f"Input data: {initial_rows} rows")

        # Basic features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_high'] = df['bb_mid'] + (bb_std * 2)
        df['bb_low'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = df['bb_high'] - df['bb_low']

        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # Volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()

        # ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr = true_range
        atr_14 = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(14).mean()

        # CCI
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

        # Price position
        df['price_position'] = (df['close'] - df['low'].rolling(14).min()) / \
                               (df['high'].rolling(14).max() - df['low'].rolling(14).min())

        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

        valid_rows = len(df.dropna())
        self.logger.info(f"✅ Computed 42 technical indicators")
        self.logger.info(f"Valid rows after indicators: {valid_rows}/{initial_rows}")

        return df

    def prepare_features(self, df, sentiment_features):
        log_section(self.logger, "FEATURE PREPARATION")

        # Add sentiment
        for key, value in sentiment_features.items():
            df[key] = value

        self.logger.info(f"Total columns in dataframe: {len(df.columns)}")

        # Select features
        features_df = df[EXPECTED_FEATURES].iloc[-self.sequence_length:]

        self.logger.info(f"Selected {len(EXPECTED_FEATURES)} features for prediction")
        self.logger.info(f"Sequence length: {self.sequence_length} days")

        # Check for missing values
        nan_count = features_df.isnull().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"⚠️  Found {nan_count} NaN values, filling...")
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            self.logger.info(f"✅ NaN values filled")

        # Normalize
        feature_mean = features_df.mean()
        feature_std = features_df.std()
        features_normalized = (features_df - feature_mean) / (feature_std + 1e-8)

        self.logger.info(f"Feature normalization:")
        self.logger.info(f"  Shape: {features_normalized.shape}")
        self.logger.info(f"  Expected: (30, 46)")
        self.logger.info(f"  Match: {features_normalized.shape == (30, 46)}")

        # Target normalization
        self.target_mean = df['close'].mean()
        self.target_std = df['close'].std()
        self.logger.info(f"Target normalization:")
        self.logger.info(f"  Mean: ${self.target_mean:.2f}")
        self.logger.info(f"  Std:  ${self.target_std:.2f}")

        return features_normalized.values

    def predict_tomorrow(self, features, current_price):
        log_section(self.logger, "MAKING PREDICTION")

        self.logger.info("Running model inference...")
        self.logger.info(f"Input shape: {features.shape}")
        self.logger.info(f"Current price: ${current_price:.2f}")

        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            self.logger.info(f"Tensor shape: {x.shape}")

            pred_normalized = self.model(x).cpu().item()
            self.logger.info(f"Normalized prediction: {pred_normalized:.4f}")

        # Denormalize
        pred_price = pred_normalized * self.target_std + self.target_mean
        change = pred_price - current_price
        change_pct = (change / current_price) * 100

        self.logger.info(f"")
        self.logger.info(f"✅ Prediction complete:")
        self.logger.info(f"  Predicted price: ${pred_price:.2f}")
        self.logger.info(f"  Absolute change: ${change:+.2f}")
        self.logger.info(f"  Percentage change: {change_pct:+.2f}%")
        self.logger.info(f"  Direction: {'↑ UP' if change > 0 else '↓ DOWN'}")

        return pred_price, change, change_pct

    def run_realtime_prediction(self):
        pipeline_start = datetime.now()

        log_section(self.logger, "REAL-TIME PREDICTION PIPELINE START")
        self.logger.info(f"Pipeline started at: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"")

        try:
            # Fetch data
            df = self.fetch_latest_stock_data('SPY', days=60)
            current_price = df['close'].iloc[-1]
            current_date = df['Date'].iloc[-1]

            # Fetch news
            articles = self.fetch_latest_news(['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'GOOGL'])
            sentiment_features = self.process_news_sentiment(articles)

            # Compute indicators
            df = self.compute_technical_indicators(df)

            # Prepare features
            features = self.prepare_features(df, sentiment_features)

            # Predict
            pred_price, change, change_pct = self.predict_tomorrow(features, current_price)

            # Calculate next date
            next_date = current_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            # Display results
            log_section(self.logger, "FINAL PREDICTION RESULTS")
            self.logger.info(f"Current Date: {current_date.strftime('%Y-%m-%d %A')}")
            self.logger.info(f"Current SPY Price: ${current_price:.2f}")
            self.logger.info(f"")
            self.logger.info(f"News Sentiment Summary:")
            self.logger.info(f"  Mean: {sentiment_features['sentiment_mean']:.3f}")
            self.logger.info(f"  Std:  {sentiment_features['sentiment_std']:.3f}")
            self.logger.info(f"  Articles: {sentiment_features['article_count']}")
            self.logger.info(f"  Positive ratio: {sentiment_features['positive_ratio']:.1%}")
            self.logger.info(f"")
            self.logger.info(f"PREDICTION FOR {next_date.strftime('%Y-%m-%d %A')}:")
            self.logger.info(f"  Predicted Price: ${pred_price:.2f}")
            direction = "↑" if change > 0 else "↓"
            self.logger.info(f"  Expected Change: {direction} ${abs(change):.2f} ({change_pct:+.2f}%)")

            # Save prediction
            prediction_record = {
                'current_date': current_date.strftime('%Y-%m-%d'),
                'prediction_date': next_date.strftime('%Y-%m-%d'),
                'current_price': float(current_price),
                'predicted_price': float(pred_price),
                'predicted_change': float(change),
                'predicted_change_pct': float(change_pct),
                'sentiment_mean': float(sentiment_features['sentiment_mean']),
                'article_count': int(sentiment_features['article_count']),
                'timestamp': datetime.now().isoformat()
            }

            predictions_file = Path('Exhibition/realtime_predictions.jsonl')
            predictions_file.parent.mkdir(exist_ok=True)
            with open(predictions_file, 'a') as f:
                f.write(json.dumps(prediction_record) + '\n')

            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()

            log_section(self.logger, "PIPELINE COMPLETE")
            self.logger.info(f"Prediction saved to: {predictions_file}")
            self.logger.info(f"Pipeline duration: {duration:.1f} seconds")
            self.logger.info(f"Completed at: {pipeline_end.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"")
            self.logger.info(f"⏰ Check back tomorrow to verify accuracy!")

            return prediction_record

        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    # Setup logger
    logger, log_file = setup_logger('realtime_prediction', log_dir='Exhibition/logs')

    log_section(logger, "REAL-TIME STOCK PREDICTION SYSTEM")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"")

    model_path = Path('Code/models/checkpoints/lstm_with_sentiment_best.pth')
    results_path = Path('Code/results/lstm_with_sentiment_results.json')

    if not model_path.exists():
        logger.error(f"❌ Model not found: {model_path}")
        return

    if not results_path.exists():
        logger.error(f"❌ Results not found: {results_path}")
        return

    try:
        predictor = RealtimePredictor(model_path, results_path, logger)
        prediction = predictor.run_realtime_prediction()

        logger.info("")
        logger.info("✅ SUCCESS: Prediction pipeline completed successfully!")
        logger.info(f"📊 Log file saved: {log_file}")

    except Exception as e:
        logger.error(f"❌ FAILURE: Pipeline encountered an error", exc_info=True)
        raise


if __name__ == "__main__":
    main()