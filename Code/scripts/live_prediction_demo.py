"""
Live Prediction Demo for Final Presentation
Loads trained model and makes predictions on recent data
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Import your model architecture
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.lstm.model import StockLSTM
from models.lstm.dataset import StockDataset

def safe_normalize(data, epsilon=1e-8):
    """Normalize with protection against zero variance"""
    mean = data.mean()
    std = data.std()
    
    if std < epsilon:
        return data - mean, mean, std
    
    return (data - mean) / std, mean, std

class LivePredictor:
    def __init__(self, model_path, data_path, results_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load configuration from results
        with open(results_path, 'r') as f:
            self.config = json.load(f)
        
        # Load data
        print(f"\nLoading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Prepare features
        feature_cols = [col for col in self.df.columns if col not in ['Date', 'close']]
        features = self.df[feature_cols]
        
        # Normalize features
        self.features_normalized = (features - features.mean()) / (features.std() + 1e-8)
        self.features = self.features_normalized.values
        
        # Normalize targets and save normalization params
        targets = self.df['close']
        self.target_mean = targets.mean()
        self.target_std = targets.std()
        self.targets_normalized = (targets - self.target_mean) / self.target_std
        self.targets = self.targets_normalized.values
        self.targets_raw = self.df['close'].values  # Keep raw for comparison
        self.dates = self.df['Date'].values
        
        print(f"  ✓ Loaded {len(self.df)} data points")
        print(f"  ✓ Features: {len(feature_cols)}")
        print(f"  ✓ Target normalization: mean=${self.target_mean:.2f}, std=${self.target_std:.2f}")
        
        # Load model
        print(f"\nLoading model from {model_path}...")
        self.model = StockLSTM(
            input_size=self.features.shape[1],
            hidden_size=self.config.get('hidden_size', 128),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        
        self.sequence_length = self.config.get('sequence_length', 30)
        print(f"✓ Model loaded successfully!")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - Sequence length: {self.sequence_length}")
        print(f"  - Features: {self.features.shape[1]}")
    
    def denormalize_prediction(self, pred_normalized):
        """Convert normalized prediction back to actual price"""
        return pred_normalized * self.target_std + self.target_mean
    
    def predict_next_days(self, n_days=5, start_idx=-30):
        """
        Predict next N days starting from start_idx
        """
        print(f"\n{'='*60}")
        print(f"MAKING PREDICTIONS FOR NEXT {n_days} DAYS")
        print(f"{'='*60}")
        
        # Get the sequence to start from
        if start_idx < 0:
            start_idx = len(self.features) + start_idx
        
        # Ensure we have enough data
        if start_idx < self.sequence_length:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length} points")
        
        # Get initial sequence
        current_sequence = self.features[start_idx - self.sequence_length:start_idx]
        last_actual_price = self.targets_raw[start_idx - 1]
        last_date = self.dates[start_idx - 1]
        
        print(f"\nStarting from:")
        print(f"  Date: {pd.Timestamp(last_date).strftime('%Y-%m-%d')}")
        print(f"  Last known price: ${last_actual_price:.2f}")
        
        predictions = []
        prediction_dates = []
        
        with torch.no_grad():
            for day in range(n_days):
                # Prepare input
                x = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Predict (normalized)
                pred_normalized = self.model(x).cpu().item()
                
                # Denormalize to actual price
                pred_actual = self.denormalize_prediction(pred_normalized)
                predictions.append(pred_actual)
                
                # Update date
                next_date = pd.Timestamp(last_date) + timedelta(days=day+1)
                # Skip weekends (simple approach)
                while next_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    next_date += timedelta(days=1)
                prediction_dates.append(next_date)
                
                # For demo purposes, shift the sequence
                if day < n_days - 1:
                    current_sequence = np.roll(current_sequence, -1, axis=0)
                    current_sequence[-1] = current_sequence[-2]
        
        return predictions, prediction_dates, last_actual_price, last_date
    
    def visualize_recent_performance(self, days_back=30):
        """
        Show model performance on recent data
        """
        print(f"\n{'='*60}")
        print(f"RECENT PERFORMANCE (Last {days_back} days)")
        print(f"{'='*60}")
        
        # Get test set indices (last 15% of data)
        test_start = int(len(self.features) * 0.85)
        
        # Use last N days from test set
        start_idx = max(test_start, len(self.features) - days_back - self.sequence_length)
        end_idx = len(self.features)
        
        predictions = []
        actuals = []
        dates = []
        
        with torch.no_grad():
            for i in range(start_idx + self.sequence_length, end_idx):
                x = torch.FloatTensor(
                    self.features[i - self.sequence_length:i]
                ).unsqueeze(0).to(self.device)
                
                # Get normalized prediction and denormalize
                pred_normalized = self.model(x).cpu().item()
                pred = self.denormalize_prediction(pred_normalized)
                actual = self.targets_raw[i]
                
                predictions.append(pred)
                actuals.append(actual)
                dates.append(self.dates[i])
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        print(f"\nPerformance Metrics:")
        print(f"  MAE:  ${mae:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: ${rmse:.2f}")
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Predictions vs Actuals
        dates_plot = [pd.Timestamp(d) for d in dates]
        ax1.plot(dates_plot, actuals, 'b-', linewidth=2, label='Actual Price', marker='o', markersize=4)
        ax1.plot(dates_plot, predictions, 'r--', linewidth=2, label='Predicted Price', marker='s', markersize=4)
        ax1.fill_between(dates_plot, actuals, predictions, alpha=0.3, color='gray')
        ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax1.set_ylabel('SPY Price ($)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Recent Predictions vs Actual (Last {len(dates)} Trading Days)\nMAPE: {mape:.2f}%', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Prediction Error
        errors = predictions - actuals
        colors = ['red' if e > 0 else 'green' for e in errors]
        ax2.bar(dates_plot, errors, color=colors, alpha=0.6, edgecolor='black')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Prediction Error ($)', fontsize=12, fontweight='bold')
        ax2.set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_dir = Path('Exhibition/figures')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'recent_performance.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved: {output_dir / 'recent_performance.png'}")
        
        return predictions, actuals, dates, mape
    
    def demo_prediction(self):
        """
        Full demo: show recent performance + future predictions
        """
        print("\n" + "="*60)
        print("LIVE PREDICTION DEMO - SPY PRICE PREDICTION")
        print("Group 5: Adarsh, Venkatesh, Mayur")
        print("="*60)
        
        # Show recent performance
        pred, actual, dates, mape = self.visualize_recent_performance(days_back=30)
        
        # Make future predictions
        future_preds, future_dates, last_price, last_date = self.predict_next_days(n_days=5)
        
        print(f"\n{'='*60}")
        print("FUTURE PRICE PREDICTIONS")
        print(f"{'='*60}")
        print(f"\nStarting from: {pd.Timestamp(last_date).strftime('%Y-%m-%d')} @ ${last_price:.2f}")
        print(f"\nNext 5 Trading Days:")
        print(f"{'Date':<15} {'Predicted Price':<20} {'Change':<15}")
        print("-" * 50)
        
        for date, pred in zip(future_dates, future_preds):
            change = pred - last_price
            change_pct = (change / last_price) * 100
            direction = "↑" if change > 0 else "↓"
            print(f"{date.strftime('%Y-%m-%d'):<15} ${pred:>8.2f}           {direction} ${abs(change):>6.2f} ({change_pct:>+5.2f}%)")
        
        print(f"\n{'='*60}")
        print(f"Model Accuracy (Recent 30 days): {mape:.2f}% MAPE")
        print(f"{'='*60}\n")
        
        return future_preds, future_dates

def main():
    """
    Run the live demo
    """
    # Paths
    model_path = Path('Code/models/checkpoints/lstm_with_sentiment_best.pth')
    data_path = Path('Code/data/processed/spy_features_with_full_sentiment.csv')
    results_path = Path('Code/results/lstm_with_sentiment_results.json')
    
    # Check if files exist
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Available models:")
        for f in Path('Code/models/checkpoints').glob('*.pth'):
            print(f"  - {f}")
        return
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        return
    
    if not results_path.exists():
        print(f"❌ Results not found: {results_path}")
        return
    
    # Run demo
    predictor = LivePredictor(model_path, data_path, results_path)
    future_preds, future_dates = predictor.demo_prediction()
    
    print("\n✓ Demo complete! Check Exhibition/figures/ for visualization.")
    print("\nPress Enter to close...")
    input()

if __name__ == "__main__":
    main()
