#!/usr/bin/env python3
"""
Unified Prediction Script - QQQ Version with Correct Features
FIXED: Uses actual QQQ feature columns (50 features)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import argparse
import warnings

warnings.filterwarnings('ignore')


class ImprovedLSTMModel(nn.Module):
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


# QQQ FEATURE COLUMNS - These match your actual data file
FEATURE_COLUMNS = [
    'close',
    'high',
    'low',
    'open',
    'volume',
    'returns',
    'log_returns',
    'sma_5',
    'ema_5',
    'sma_10',
    'ema_10',
    'sma_20',
    'ema_20',
    'sma_50',
    'ema_50',
    'rsi',
    'stoch',
    'williams_r',
    'macd',
    'macd_signal',
    'macd_diff',
    'bb_high',
    'bb_low',
    'bb_mid',
    'bb_width',
    'volume_sma_20',
    'volume_ratio',
    'obv',
    'atr',
    'volatility_5',
    'volatility_10',
    'volatility_20',
    'adx',
    'cci',
    'price_position',
    'close_lag_1',
    'returns_lag_1',
    'close_lag_2',
    'returns_lag_2',
    'close_lag_3',
    'returns_lag_3',
    'close_lag_5',
    'returns_lag_5',
    'sentiment_mean',
    'sentiment_median',
    'sentiment_std',
    'sentiment_min',
    'sentiment_max',
    'article_count',
    'positive_ratio',
]


class UnifiedPredictor:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sequence_length = 30

        # Load model with 49 features (not 50 - one is dropped during training)
        model_path = Path('Code/models/lstm_model_sentiment_QQQ.pt')
        self.model = ImprovedLSTMModel(49, 64, 2, 0.3).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        # Load historical data
        data_path = Path('Code/data/processed/qqq_features_with_sentiment.csv')
        self.historical_df = pd.read_csv(data_path)
        self.historical_df['Date'] = pd.to_datetime(self.historical_df['Date'])

        self.last_historical_date = self.historical_df['Date'].max()

        print(f"✓ Model loaded ({self.device})")
        print(f"✓ Historical data: {len(self.historical_df)} days")
        print(f"  Last date: {self.last_historical_date.strftime('%Y-%m-%d')}")
        print(f"✓ Using {len(FEATURE_COLUMNS)} feature columns")

    def fetch_and_process_live_data(self):
        """Fetch live data and compute indicators"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=150)

        ticker = yf.Ticker('QQQ')
        df = ticker.history(start=start_date, end=end_date)

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

        # Rename columns to lowercase to match training data
        df.columns = [col.lower() if col != 'Date' else col for col in df.columns]

        # Compute indicators (lowercase column names)
        df = self.compute_indicators(df)

        return df

    def compute_indicators(self, df):
        """Compute all technical indicators with lowercase names"""
        # Note: 'close' column already exists from yfinance (lowercase)

        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14 + 1e-10)

        df['bb_mid'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_high'] = df['bb_mid'] + (bb_std * 2)
        df['bb_low'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = df['bb_high'] - df['bb_low']

        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()

        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma_20'] + 1e-10)

        # OBV
        df['obv'] = (df['volume'] * (~df['close'].diff().le(0) * 2 - 1)).cumsum()

        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14).mean()

        # ADX (simplified)
        df['adx'] = df['rsi'].rolling(14).mean()  # Placeholder - proper ADX is complex

        # CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (typical_price - typical_price.rolling(20).mean()) / (
                    0.015 * typical_price.rolling(20).std() + 1e-10)

        # Price position
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Lagged features
        df['close_lag_1'] = df['close'].shift(1)
        df['close_lag_2'] = df['close'].shift(2)
        df['close_lag_3'] = df['close'].shift(3)
        df['close_lag_5'] = df['close'].shift(5)

        df['returns_lag_1'] = df['returns'].shift(1)
        df['returns_lag_2'] = df['returns'].shift(2)
        df['returns_lag_3'] = df['returns'].shift(3)
        df['returns_lag_5'] = df['returns'].shift(5)

        return df.dropna()

    def get_sentiment(self, target_date):
        """Get sentiment from historical data"""
        target_date = pd.to_datetime(target_date)
        before = self.historical_df[self.historical_df['Date'] <= target_date]

        if before.empty:
            last = self.historical_df.iloc[0]
        else:
            last = before.iloc[-1]

        return {
            'sentiment_mean': last['sentiment_mean'],
            'sentiment_median': last['sentiment_median'],
            'sentiment_std': last['sentiment_std'],
            'sentiment_min': last['sentiment_min'],
            'sentiment_max': last['sentiment_max'],
            'article_count': last['article_count'],
            'positive_ratio': last['positive_ratio'],
        }

    def make_prediction(self, df, current_date):
        """Make prediction given prepared data"""
        # Get last 30 days
        if len(df) < self.sequence_length:
            return None

        df_seq = df.tail(self.sequence_length).copy()

        # Add sentiment
        sentiment = self.get_sentiment(current_date)
        for key, value in sentiment.items():
            df_seq[key] = value

        # Check which features are available
        available_features = [col for col in FEATURE_COLUMNS if col in df_seq.columns]
        missing_features = [col for col in FEATURE_COLUMNS if col not in df_seq.columns]

        if missing_features:
            print(f"\n⚠️  Warning: Missing features: {missing_features[:5]}...")
            # Use only available features
            use_features = available_features
        else:
            use_features = FEATURE_COLUMNS

        # Prepare features
        features = df_seq[use_features].values

        # If we have 50 features but model expects 49, drop the first one ('close')
        if features.shape[1] == 50:
            features = features[:, 1:]  # Drop first column (close)
            print(f"✓ Adjusted features from 50 to 49 (dropped 'close' column)")

        # Normalize
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0)
        feature_std[feature_std == 0] = 1
        features_norm = (features - feature_mean) / feature_std

        # Predict
        with torch.no_grad():
            X = torch.FloatTensor(features_norm).unsqueeze(0).to(self.device)
            pred_norm = self.model(X).cpu().item()

        # Denormalize using 'close' column
        if 'close' in df_seq.columns:
            target_mean = df_seq['close'].mean()
            target_std = df_seq['close'].std()
            current_price = df_seq['close'].iloc[-1]
        else:
            target_mean = df_seq['Close'].mean()
            target_std = df_seq['Close'].std()
            current_price = df_seq['Close'].iloc[-1]

        pred_price = pred_norm * target_std + target_mean

        change = pred_price - current_price
        change_pct = (change / current_price) * 100

        # Next trading day
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)

        return {
            'current_date': current_date.strftime('%Y-%m-%d'),
            'prediction_date': next_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predicted_price': float(pred_price),
            'predicted_change': float(change),
            'predicted_change_pct': float(change_pct),
            'sentiment_mean': float(sentiment['sentiment_mean']),
            'article_count': int(sentiment['article_count']),
            'timestamp': datetime.now().isoformat()
        }

    def predict_live(self):
        """Predict tomorrow using live data"""
        print(f"\n{'=' * 80}")
        print(f"LIVE QQQ PREDICTION")
        print(f"{'=' * 80}\n")

        df = self.fetch_and_process_live_data()
        current_date = df['Date'].iloc[-1]

        print(f"Latest data: {current_date.strftime('%Y-%m-%d')}")

        result = self.make_prediction(df, current_date)

        if result:
            print(f"\nCurrent:   ${result['current_price']:.2f}")
            print(f"Predicted: ${result['predicted_price']:.2f} (for {result['prediction_date']})")
            print(f"Change:    {result['predicted_change_pct']:+.2f}%")

        return result

    def predict_range(self, start_date, end_date):
        """Predict for date range - intelligently uses historical or live data"""
        print(f"\n{'=' * 80}")
        print(f"BATCH PREDICTIONS: {start_date} to {end_date}")
        print(f"{'=' * 80}\n")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Check if range is beyond historical data
        if start_date > self.last_historical_date:
            print(f"⚠️  Requested range is beyond historical data")
            print(f"   Historical ends: {self.last_historical_date.strftime('%Y-%m-%d')}")
            print(f"   Using LIVE data mode...\n")

            # Fetch live data
            df_live = self.fetch_and_process_live_data()

            # Generate predictions for each day we have data for
            predictions = []
            available_dates = df_live['Date'].tolist()

            for target_date in available_dates:
                if start_date <= target_date <= end_date:
                    # Get data up to this date
                    df_subset = df_live[df_live['Date'] <= target_date]

                    pred = self.make_prediction(df_subset, target_date)
                    if pred:
                        predictions.append(pred)
                        print(
                            f"✓ {target_date.strftime('%Y-%m-%d')}: ${pred['predicted_price']:.2f} ({pred['predicted_change_pct']:+.2f}%)")

            return predictions

        else:
            # Use historical data
            print(f"Using HISTORICAL data mode...\n")

            date_range = self.historical_df[
                (self.historical_df['Date'] >= start_date) &
                (self.historical_df['Date'] <= end_date)
                ]['Date'].tolist()

            if not date_range:
                print(f"❌ No data in range")
                return []

            predictions = []

            for target_date in date_range:
                df_subset = self.historical_df[self.historical_df['Date'] < target_date]

                pred = self.make_prediction(df_subset, target_date)
                if pred:
                    predictions.append(pred)
                    print(
                        f"✓ {target_date.strftime('%Y-%m-%d')}: ${pred['predicted_price']:.2f} ({pred['predicted_change_pct']:+.2f}%)")

            return predictions

    def save(self, predictions):
        """Save predictions"""
        if not predictions:
            print("\n⚠️  No predictions to save")
            return

        if isinstance(predictions, dict):
            predictions = [predictions]

        output = Path('exhibition/realtime_predictions_QQQ.jsonl')
        output.parent.mkdir(exist_ok=True)

        with open(output, 'a') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')

        csv_file = output.with_suffix('.csv')
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            new_rows = pd.DataFrame(predictions)
            df = pd.concat([df, new_rows], ignore_index=True)
            df = df.drop_duplicates(subset=['prediction_date'], keep='last')
        else:
            df = pd.DataFrame(predictions)

        df.to_csv(csv_file, index=False)

        print(f"\n{'=' * 80}")
        print(f"✓ SAVED {len(predictions)} PREDICTION(S)")
        print(f"{'=' * 80}")
        print(f"  JSONL: {output}")
        print(f"  CSV:   {csv_file}")

        if len(predictions) > 1:
            changes = [p['predicted_change_pct'] for p in predictions]
            print(f"\n  Summary:")
            print(f"    Avg change: {np.mean(changes):+.2f}%")
            print(f"    Range: {np.min(changes):+.2f}% to {np.max(changes):+.2f}%")

        print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    predictor = UnifiedPredictor()

    # Batch mode
    if args.start and args.end:
        predictions = predictor.predict_range(args.start, args.end)
        predictor.save(predictions)

    # Live mode (tomorrow)
    else:
        prediction = predictor.predict_live()
        predictor.save(prediction)


if __name__ == '__main__':
    main()