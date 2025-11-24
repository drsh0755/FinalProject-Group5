import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm.model import StockLSTM

class StockDatasetWithSentiment(Dataset):
    """Dataset for stock data with sentiment features"""
    def __init__(self, features, targets, sequence_length=30):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return X, y

def safe_normalize(data, epsilon=1e-8):
    """Normalize with protection against zero variance"""
    mean = data.mean()
    std = data.std()
    
    # If std is too small, don't normalize (just center)
    if std < epsilon:
        return data - mean
    
    return (data - mean) / std

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    batch_count = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions.squeeze(), y_batch)
        
        # Check for NaN
        if torch.isnan(loss):
            print("    WARNING: NaN loss detected, skipping batch")
            continue
            
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        batch_count += 1
    
    return total_loss / max(batch_count, 1)

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            loss = criterion(predictions.squeeze(), y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def calculate_metrics(predictions, actuals):
    """Calculate MAPE, MAE, RMSE"""
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # MAPE
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # MAE
    mae = np.mean(np.abs(actuals - predictions))
    
    # RMSE
    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    
    return mape, mae, rmse

def main():
    # Configuration
    config = {
        'sequence_length': 30,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 30,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("LSTM TRAINING - WITH SENTIMENT FEATURES (FIXED)")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load data - try both possible filenames
    data_files = [
        'data/processed/spy_features_with_full_sentiment.csv',
        'data/processed/spy_features_with_sentiment.csv'
    ]
    
    df = None
    for data_file in data_files:
        try:
            df = pd.read_csv(data_file)
            print(f"\n✓ Loaded data from: {data_file}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        print("\n✗ Error: Could not find sentiment data file!")
        print("Expected files:")
        for f in data_files:
            print(f"  - {f}")
        return
    
    print(f"  Total days: {len(df)}")
    
    # Separate target from features
    target_col = 'close'
    exclude_cols = ['Date', 'date', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"  Total features: {len(feature_cols)}")
    
    # Check for sentiment features
    sentiment_cols = [col for col in feature_cols if 'sentiment' in col.lower() or 'article' in col.lower() or 'positive' in col.lower() or 'negative' in col.lower()]
    if sentiment_cols:
        print(f"  Sentiment features: {len(sentiment_cols)}")
        print(f"    {sentiment_cols}")
    else:
        print("  ⚠ No sentiment features found!")
    
    # Remove zero-variance features
    feature_data = df[feature_cols]
    stds = feature_data.std()
    non_zero_var_cols = stds[stds > 1e-8].index.tolist()
    
    removed_cols = set(feature_cols) - set(non_zero_var_cols)
    if removed_cols:
        print(f"\n⚠ Removed {len(removed_cols)} zero-variance features:")
        for col in sorted(removed_cols):
            print(f"    {col}")
    
    feature_cols = non_zero_var_cols
    print(f"  Active features: {len(feature_cols)}")
    
    # Check for NaN or Inf in raw data
    if df[feature_cols].isnull().any().any():
        print("\n⚠ NaN values found in features, filling with 0...")
        df[feature_cols] = df[feature_cols].fillna(0)
    
    if np.isinf(df[feature_cols].values).any():
        print("\n⚠ Infinite values found, clipping...")
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0)
    
    # Normalize with safe method
    print("\nNormalizing data...")
    features_normalized = pd.DataFrame()
    for col in feature_cols:
        features_normalized[col] = safe_normalize(df[col])
    
    targets_normalized = safe_normalize(df[target_col])
    
    # Final check for NaN/Inf after normalization
    if features_normalized.isnull().any().any():
        print("\n✗ NaN values detected after normalization!")
        nan_cols = features_normalized.columns[features_normalized.isnull().any()].tolist()
        print(f"  Columns with NaN: {nan_cols}")
        return
    
    if np.isinf(features_normalized.values).any():
        print("\n✗ Infinite values detected after normalization!")
        return
    
    print("✓ Data normalized successfully")
    
    # Create sequences
    features = features_normalized.values
    targets = targets_normalized.values
    
    # Split data
    n = len(features) - config['sequence_length']
    train_size = int(n * config['train_split'])
    val_size = int(n * config['val_split'])
    
    print(f"\nCreating sequences...")
    print(f"  Total data points: {len(features)}")
    print(f"  Usable sequences: {n}")
    
    # Create datasets
    train_dataset = StockDatasetWithSentiment(
        features[:train_size + config['sequence_length']],
        targets[:train_size + config['sequence_length']],
        config['sequence_length']
    )
    
    val_dataset = StockDatasetWithSentiment(
        features[train_size:train_size + val_size + config['sequence_length']],
        targets[train_size:train_size + val_size + config['sequence_length']],
        config['sequence_length']
    )
    
    test_dataset = StockDatasetWithSentiment(
        features[train_size + val_size:],
        targets[train_size + val_size:],
        config['sequence_length']
    )
    
    print(f"✓ Sequences created:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Create model
    model = StockLSTM(
        input_size=len(feature_cols),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Model created:")
    print(f"  Input size: {len(feature_cols)}")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/checkpoints/lstm_with_sentiment_best.pth')
            print(f"  ✓ Best model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model for testing
    model.load_state_dict(torch.load('models/checkpoints/lstm_with_sentiment_best.pth'))
    
    # Testing
    print("\n" + "=" * 60)
    print("TESTING")
    print("=" * 60)
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            predictions.extend(pred.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # Denormalize
    target_mean = df[target_col].mean()
    target_std = df[target_col].std()
    
    predictions = np.array(predictions).flatten() * target_std + target_mean
    actuals = np.array(actuals) * target_std + target_mean
    
    # Calculate metrics
    mape, mae, rmse = calculate_metrics(predictions, actuals)
    
    print(f"\n✓ Test Results:")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    
    # Save results
    results = {
        'config': config,
        'test_metrics': {
            'mape': float(mape),
            'mae': float(mae),
            'rmse': float(rmse)
        },
        'feature_count': len(feature_cols),
        'sentiment_features': sentiment_cols,
        'removed_features': list(removed_cols),
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'test_samples': len(test_dataset)
    }
    
    with open('results/lstm_with_sentiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Results saved to: results/lstm_with_sentiment_results.json")
    print(f"Model saved to: models/checkpoints/lstm_with_sentiment_best.pth")

if __name__ == '__main__':
    main()
