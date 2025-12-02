#!/usr/bin/env python3
"""
Real-Time Live Prediction System - FIXED VERSION
Uses Code/models/ directory (43 features ONLY - CORRECTED)

Fetches latest data and news, makes predictions for tomorrow
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# MODEL ARCHITECTURE (same as training)
# ============================================================================

class ImprovedLSTMModel(nn.Module):
    """Improved LSTM with batch normalization and regularization"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


# ============================================================================
# FEATURE NAMES (EXACTLY 43 features - VERIFIED FROM YOUR CSV)
# ============================================================================

FEATURE_COLUMNS = [
    'Open',
    'High',
    'Low',
    'Volume',
    'SMA_5',
    'SMA_10',
    'SMA_20',
    'SMA_50',
    'EMA_5',
    'EMA_10',
    'EMA_20',
    'MACD',
    'MACD_Signal',
    'MACD_Hist',
    'RSI',
    'BB_Middle',
    'BB_Upper',
    'BB_Lower',
    'BB_Width',
    'BB_Position',
    'Momentum',
    'ROC',
    'Daily_Return',
    'Volatility_10',
    'Volatility_20',
    'Volatility_50',
    'Volume_SMA_20',
    'Volume_Ratio',
    'Price_Change',
    'Price_Change_Pct',
    'High_Low_Pct',
    'Open_Close_Pct',
    'Close_Lag1',
    'Close_Lag2',
    'Close_Lag3',
    'Close_Lag5',
    'sentiment_mean',
    'sentiment_median',
    'sentiment_std',
    'sentiment_min',
    'sentiment_max',
    'article_count',
    'positive_ratio',
]

# VERIFY COUNT
assert len(FEATURE_COLUMNS) == 43, f"ERROR: Expected 43 features, got {len(FEATURE_COLUMNS)}"
print(f"✓ Verified: {len(FEATURE_COLUMNS)} features (43 confirmed)")


# ============================================================================
# REALTIME PREDICTOR CLASS
# ============================================================================

class RealtimePredictor:
    def __init__(self, model_path, results_path):
        print("\n" + "=" * 80)
        print("🚀 INITIALIZING REALTIME PREDICTION SYSTEM")
        print("=" * 80)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\n✓ Device: {self.device}")

        # Load results config
        print(f"✓ Loading config from {results_path}")
        with open(results_path, 'r') as f:
            self.config = json.load(f)

        # Load model
        print(f"✓ Loading model from {model_path}")
        self.model = ImprovedLSTMModel(
            input_size=43,  # EXACTLY 43 features
            hidden_size=64,
            num_layers=2,
            dropout=0.3
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.sequence_length = 30
        self.av_api_key = "GR9K9CDZ4SK596YU"

        print(f"✓ Model loaded successfully")
        print(f"✓ System ready for predictions\n")

    def fetch_latest_stock_data(self, symbol='SPY', days=60):
        """Fetch latest stock data from Yahoo Finance"""
        print(f"\n📊 FETCHING STOCK DATA: {symbol}")
        print("-" * 80)

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            df = df.reset_index()
            df.columns = [col.capitalize() if col.lower() != 'date' else 'Date' for col in df.columns]

            print(f"✓ Fetched {len(df)} days of data")
            print(f"  Latest date: {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
            print(f"  Latest close: ${df['Close'].iloc[-1]:.2f}")

            return df

        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    def fetch_latest_news(self, tickers=['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA']):
        """Fetch latest news from Alpha Vantage"""
        print(f"\n📰 FETCHING NEWS DATA")
        print("-" * 80)

        all_articles = []
        for ticker in tickers:
            try:
                url = "https://www.alphavantage.co/query"
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
                            'ticker': ticker
                        })
                    print(f"  ✓ {ticker}: {len(data['feed'])} articles")

            except Exception as e:
                print(f"  ⚠️  {ticker}: {str(e)}")

        print(f"✓ Total articles: {len(all_articles)}")
        return all_articles

    def compute_sentiment(self, articles):
        """Compute sentiment features"""
        if not articles:
            return {
                'sentiment_mean': 0.0,
                'sentiment_median': 0.0,
                'sentiment_std': 0.0,
                'sentiment_min': 0.0,
                'sentiment_max': 0.0,
                'article_count': 0,
                'positive_ratio': 0.5
            }

        positive_words = ['growth', 'gain', 'bull', 'strong', 'surge', 'rally',
                          'positive', 'upgrade', 'profit', 'beat']
        negative_words = ['loss', 'bear', 'weak', 'down', 'decline', 'fall',
                          'negative', 'downgrade', 'miss', 'risk']

        sentiments = []
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('summary', '')).lower()

            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)

            if positive_count + negative_count == 0:
                sentiment = 0.0
            else:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)

            sentiments.append(sentiment)

        sentiments = np.array(sentiments)
        positive_ratio = np.sum(sentiments > 0) / len(sentiments) if len(sentiments) > 0 else 0.5

        return {
            'sentiment_mean': float(np.mean(sentiments)),
            'sentiment_median': float(np.median(sentiments)),
            'sentiment_std': float(np.std(sentiments)),
            'sentiment_min': float(np.min(sentiments)),
            'sentiment_max': float(np.max(sentiments)),
            'article_count': len(articles),
            'positive_ratio': float(positive_ratio)
        }

    def compute_technical_indicators(self, df):
        """Compute technical indicators matching training data"""
        print(f"\n📈 COMPUTING TECHNICAL INDICATORS")
        print("-" * 80)

        # Simple Moving Averages
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()

        # Exponential Moving Averages
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_10'] = df['Close'].ewm(span=10).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + 1e-10)

        # Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)

        # Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

        # Daily Return
        df['Daily_Return'] = df['Close'].pct_change()

        # Volatility
        df['Volatility_10'] = df['Daily_Return'].rolling(10).std()
        df['Volatility_20'] = df['Daily_Return'].rolling(20).std()
        df['Volatility_50'] = df['Daily_Return'].rolling(50).std()

        # Volume indicators
        df['Volume_SMA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-10)

        # Price changes
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Change_Pct'] = (df['Price_Change'] / df['Open']) * 100
        df['High_Low_Pct'] = ((df['High'] - df['Low']) / df['Low']) * 100
        df['Open_Close_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100

        # Lagged features
        df['Close_Lag1'] = df['Close'].shift(1)
        df['Close_Lag2'] = df['Close'].shift(2)
        df['Close_Lag3'] = df['Close'].shift(3)
        df['Close_Lag5'] = df['Close'].shift(5)

        print(f"✓ Computed technical indicators")
        return df

    def prepare_features(self, df, sentiment_features):
        """Prepare features for model"""
        print(f"\n⚙️  PREPARING FEATURES")
        print("-" * 80)

        # Add sentiment columns
        for key, value in sentiment_features.items():
            df[key] = value

        # Select ONLY the 43 features
        features_df = df[FEATURE_COLUMNS].iloc[-self.sequence_length:]

        # Handle NaN
        nan_count = features_df.isnull().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values, filling...")
            features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Normalize
        feature_mean = features_df.mean()
        feature_std = features_df.std()
        features_normalized = (features_df - feature_mean) / (feature_std + 1e-8)

        # Store for denormalization
        self.target_mean = df['Close'].mean()
        self.target_std = df['Close'].std()

        print(f"✓ Features prepared: {features_normalized.shape}")

        return features_normalized.values

    def predict_next_day(self, features, current_price):
        """Make prediction for next day"""
        print(f"\n🎯 MAKING PREDICTION")
        print("-" * 80)

        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            pred_normalized = self.model(x).cpu().item()

        pred_price = pred_normalized * self.target_std + self.target_mean
        change = pred_price - current_price
        change_pct = (change / current_price) * 100

        print(f"✓ Current price: ${current_price:.2f}")
        print(f"✓ Predicted price: ${pred_price:.2f}")
        print(f"✓ Expected change: ${change:+.2f} ({change_pct:+.2f}%)")

        return pred_price, change, change_pct

    def run(self):
        """Run complete prediction pipeline"""
        try:
            # Fetch data
            df = self.fetch_latest_stock_data('SPY', days=60)
            current_price = df['Close'].iloc[-1]
            current_date = df['Date'].iloc[-1]

            # Fetch news
            articles = self.fetch_latest_news()
            sentiment_features = self.compute_sentiment(articles)

            # Compute indicators
            df = self.compute_technical_indicators(df)

            # Prepare features
            features = self.prepare_features(df, sentiment_features)

            # Predict
            pred_price, change, change_pct = self.predict_next_day(features, current_price)

            # Next trading day
            next_date = current_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            # Save prediction
            prediction_record = {
                'current_date': current_date.strftime('%Y-%m-%d'),
                'prediction_date': next_date.strftime('%Y-%m-%d'),
                'current_price': float(current_price),
                'predicted_price': float(pred_price),
                'predicted_change': float(change),
                'predicted_change_pct': float(change_pct),
                'timestamp': datetime.now().isoformat()
            }

            predictions_file = Path('Exhibition/realtime_predictions.jsonl')
            predictions_file.parent.mkdir(exist_ok=True)

            with open(predictions_file, 'a') as f:
                f.write(json.dumps(prediction_record) + '\n')

            print("\n" + "=" * 80)
            print("📈 PREDICTION COMPLETE")
            print("=" * 80)
            print(f"\nCurrent: ${current_price:.2f} ({current_date.strftime('%Y-%m-%d')})")
            print(f"Predicted: ${pred_price:.2f} ({next_date.strftime('%Y-%m-%d')})")
            print(f"Change: {change_pct:+.2f}%")
            print(f"\nSaved to: {predictions_file}")
            print("=" * 80 + "\n")

            return prediction_record

        except Exception as e:
            print(f"\n❌ Error: {e}")
            raise


# ============================================================================
# MAIN
# ============================================================================

def main():
    model_path = Path('Code/models/lstm_model_sentiment.pt')
    results_path = Path('Code/results/lstm_sentiment_results.json')

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        return False

    predictor = RealtimePredictor(model_path, results_path)
    predictor.run()
    return True


if __name__ == "__main__":
    main()
