"""
Multi-Stock Real-Time Prediction System
Predicts SPY + 7 individual tech stocks
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

class MultiStockPredictor:
    def __init__(self, model_path, results_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Initializing Multi-Stock Prediction System")
        print(f"Device: {self.device}")
        
        with open(results_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"\n📦 Loading trained model...")
        self.model = StockLSTM(
            input_size=46,
            hidden_size=self.config.get('hidden_size', 128),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        self.sequence_length = self.config.get('sequence_length', 30)
        
        print(f"🤖 Loading FinBERT for sentiment analysis...")
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.sentiment_model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
        self.sentiment_model.eval()
        
        self.av_api_key = "GR9K9CDZ4SK596YU"
        
        print(f"✅ System initialized!")
    
    def fetch_stock_data(self, symbol, days=60):
        """Fetch historical data for any stock"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
        
        df = df.reset_index()
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={'date': 'Date'})
        
        return df
    
    def fetch_all_news(self, tickers):
        """Fetch news for all companies"""
        print(f"\n📰 Fetching news for {len(tickers)} companies...")
        
        all_articles = {}
        
        for ticker in tickers:
            try:
                url = f"https://www.alphavantage.co/query"
                params = {
                    'function': 'NEWS_SENTIMENT',
                    'tickers': ticker,
                    'apikey': self.av_api_key,
                    'limit': 50
                }
                
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                articles = []
                if 'feed' in data:
                    for item in data['feed']:
                        articles.append({
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'time_published': item.get('time_published', ''),
                            'ticker': ticker
                        })
                
                all_articles[ticker] = articles
                print(f"  ✓ {ticker}: {len(articles)} articles")
                
            except Exception as e:
                print(f"  ✗ {ticker}: {str(e)}")
                all_articles[ticker] = []
        
        return all_articles
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using FinBERT"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        positive = predictions[0][0].item()
        negative = predictions[0][1].item()
        
        return positive - negative
    
    def process_sentiment(self, articles, show_progress=False):
        """Process sentiment for a list of articles"""
        if not articles:
            return {
                'sentiment_mean': 0.0,
                'sentiment_std': 0.0,
                'article_count': 0,
                'positive_ratio': 0.5
            }
        
        sentiments = []
        
        for i, article in enumerate(articles):
            text = f"{article['title']}. {article['summary']}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment)
            
            if show_progress and (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(articles)} articles...")
        
        sentiments = np.array(sentiments)
        
        return {
            'sentiment_mean': sentiments.mean(),
            'sentiment_std': sentiments.std(),
            'article_count': len(sentiments),
            'positive_ratio': (sentiments > 0).sum() / len(sentiments)
        }
    
    def compute_technical_indicators(self, df):
        """Compute all 42 technical indicators"""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_high'] = df['bb_mid'] + (bb_std * 2)
        df['bb_low'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = df['bb_high'] - df['bb_low']
        
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
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
        
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        
        df['price_position'] = (df['close'] - df['low'].rolling(14).min()) / \
                              (df['high'].rolling(14).max() - df['low'].rolling(14).min())
        
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        return df
    
    def prepare_features(self, df, sentiment_features):
        """Prepare features for prediction"""
        for key, value in sentiment_features.items():
            df[key] = value
        
        features_df = df[EXPECTED_FEATURES].iloc[-self.sequence_length:]
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        features_normalized = (features_df - features_df.mean()) / (features_df.std() + 1e-8)
        
        target_mean = df['close'].mean()
        target_std = df['close'].std()
        
        return features_normalized.values, target_mean, target_std
    
    def predict_price(self, features, target_mean, target_std, current_price):
        """Make price prediction"""
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            pred_normalized = self.model(x).cpu().item()
        
        pred_price = pred_normalized * target_std + target_mean
        change = pred_price - current_price
        change_pct = (change / current_price) * 100
        
        return pred_price, change, change_pct
    
    def predict_stock(self, symbol, sentiment_features, show_progress=False):
        """Predict a single stock"""
        if show_progress:
            print(f"\n📊 Processing {symbol}...")
        
        # Fetch data
        df = self.fetch_stock_data(symbol, days=60)
        if df is None:
            print(f"  ✗ Could not fetch data for {symbol}")
            return None
        
        current_price = df['close'].iloc[-1]
        current_date = df['Date'].iloc[-1]
        
        # Compute indicators
        df = self.compute_technical_indicators(df)
        
        # Prepare features
        features, target_mean, target_std = self.prepare_features(df, sentiment_features)
        
        # Predict
        pred_price, change, change_pct = self.predict_price(features, target_mean, target_std, current_price)
        
        # Calculate next trading day
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += timedelta(days=1)
        
        return {
            'symbol': symbol,
            'current_date': current_date.strftime('%Y-%m-%d'),
            'prediction_date': next_date.strftime('%Y-%m-%d'),
            'current_price': float(current_price),
            'predicted_price': float(pred_price),
            'predicted_change': float(change),
            'predicted_change_pct': float(change_pct),
            'sentiment_mean': float(sentiment_features['sentiment_mean']),
            'article_count': int(sentiment_features['article_count'])
        }
    
    def run_multi_stock_prediction(self):
        """Predict SPY + 7 tech stocks"""
        print("\n" + "="*80)
        print("MULTI-STOCK REAL-TIME PREDICTION SYSTEM")
        print("="*80)
        
        # Define stocks to predict
        stocks = ['SPY', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'GOOGL']
        tech_stocks = ['AAPL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'GOOGL']
        
        # Fetch all news
        all_news = self.fetch_all_news(tech_stocks)
        
        # For SPY: Use combined sentiment from all tech stocks
        print(f"\n🧠 Analyzing combined sentiment for SPY...")
        all_articles = []
        for ticker, articles in all_news.items():
            all_articles.extend(articles)
        
        combined_sentiment = self.process_sentiment(all_articles, show_progress=True)
        print(f"  ✓ Combined sentiment: {combined_sentiment['sentiment_mean']:.3f}")
        print(f"  ✓ Total articles: {combined_sentiment['article_count']}")
        
        # Store all predictions
        predictions = []
        
        # Predict SPY with combined sentiment
        print(f"\n{'='*80}")
        print(f"PREDICTING SPY (S&P 500 ETF) - Market Benchmark")
        print(f"{'='*80}")
        spy_pred = self.predict_stock('SPY', combined_sentiment, show_progress=True)
        if spy_pred:
            predictions.append(spy_pred)
            direction = "↑" if spy_pred['predicted_change'] > 0 else "↓"
            print(f"  Current: ${spy_pred['current_price']:.2f}")
            print(f"  Predicted: ${spy_pred['predicted_price']:.2f}")
            print(f"  Change: {direction} ${abs(spy_pred['predicted_change']):.2f} ({spy_pred['predicted_change_pct']:+.2f}%)")
        
        # Predict individual tech stocks
        print(f"\n{'='*80}")
        print(f"PREDICTING INDIVIDUAL TECH STOCKS")
        print(f"{'='*80}")
        
        for ticker in tech_stocks:
            print(f"\n📱 {ticker}:")
            
            # Use individual sentiment for each stock
            individual_sentiment = self.process_sentiment(all_news[ticker])
            
            pred = self.predict_stock(ticker, individual_sentiment)
            if pred:
                predictions.append(pred)
                direction = "↑" if pred['predicted_change'] > 0 else "↓"
                print(f"  Current: ${pred['current_price']:.2f}")
                print(f"  Predicted: ${pred['predicted_price']:.2f}")
                print(f"  Change: {direction} ${abs(pred['predicted_change']):.2f} ({pred['predicted_change_pct']:+.2f}%)")
                print(f"  Sentiment: {pred['sentiment_mean']:.3f} ({pred['article_count']} articles)")
        
        # Summary table
        print(f"\n{'='*80}")
        print(f"PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'Symbol':<8} {'Current':<12} {'Predicted':<12} {'Change':<12} {'% Change':<10} {'Direction':<10}")
        print("-" * 80)
        
        for pred in predictions:
            direction = "↑ UP" if pred['predicted_change'] > 0 else "↓ DOWN"
            print(f"{pred['symbol']:<8} ${pred['current_price']:<11.2f} ${pred['predicted_price']:<11.2f} "
                  f"${abs(pred['predicted_change']):<11.2f} {pred['predicted_change_pct']:>+7.2f}%   {direction:<10}")
        
        # Market consensus
        up_count = sum(1 for p in predictions if p['predicted_change'] > 0)
        down_count = len(predictions) - up_count
        
        print(f"\n📊 Market Consensus:")
        print(f"  Bullish (↑): {up_count}/{len(predictions)} stocks")
        print(f"  Bearish (↓): {down_count}/{len(predictions)} stocks")
        
        if up_count > down_count:
            print(f"  Overall sentiment: 🟢 BULLISH")
        elif down_count > up_count:
            print(f"  Overall sentiment: 🔴 BEARISH")
        else:
            print(f"  Overall sentiment: ⚪ NEUTRAL")
        
        # Save all predictions
        predictions_file = Path('Exhibition/multi_stock_predictions.jsonl')
        predictions_file.parent.mkdir(exist_ok=True)
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'predictions': predictions,
            'market_consensus': {
                'bullish': up_count,
                'bearish': down_count,
                'total': len(predictions)
            }
        }
        
        with open(predictions_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
        
        print(f"\n💾 Predictions saved to: {predictions_file}")
        print(f"⏰ Check back tomorrow to verify accuracy!")
        print(f"{'='*80}\n")
        
        return predictions

def main():
    model_path = Path('Code/models/checkpoints/lstm_with_sentiment_best.pth')
    results_path = Path('Code/results/lstm_with_sentiment_results.json')
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        return
    
    predictor = MultiStockPredictor(model_path, results_path)
    predictions = predictor.run_multi_stock_prediction()

if __name__ == "__main__":
    main()
