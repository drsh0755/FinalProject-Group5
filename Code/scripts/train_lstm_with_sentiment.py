"""
Train LSTM with technical indicators + sentiment features
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from models.lstm.model import StockLSTM
from models.lstm.dataset import StockDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, actuals = [], []

    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    return total_loss / len(dataloader), mae, rmse


def main():
    print("\n" + "=" * 60)
    print("LSTM TRAINING - WITH SENTIMENT ANALYSIS")
    print("=" * 60 + "\n")

    # Config
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

    print("Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")

    # Load data with sentiment
    print("=" * 60)
    print("Loading data with sentiment features...")
    print("=" * 60)

    data_file = Path(__file__).parent.parent / 'data' / 'processed' / 'spy_features_with_sentiment.csv'
    dataset = StockDataset(
        data_file=data_file,
        sequence_length=config['sequence_length']
    )

    print(f"✓ Dataset loaded:")
    print(f"  Total days: {len(dataset.data)}")
    print(f"  Features: {dataset.feature_dim} (includes sentiment)")
    print(f"  Usable sequences: {len(dataset)}")

    # Split
    train_size = int(config['train_split'] * len(dataset))
    val_size = int(config['val_split'] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"  Test: {len(test_dataset)} sequences")

    # Model
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60 + "\n")

    model = StockLSTM(
        input_size=dataset.feature_dim,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Training
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    best_val_loss = float('inf')
    train_losses, val_losses, val_maes, val_rmses = [], [], [], []

    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{config['epochs']}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_mae:.4f}")
            print(f"  Val RMSE: {val_rmse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint_dir = Path(__file__).parent.parent / 'models' / 'checkpoints'
            torch.save(model.state_dict(), checkpoint_dir / 'lstm_sentiment_best.pth')

    # Test
    print("\n" + "=" * 60)
    print("Testing...")
    print("=" * 60 + "\n")

    test_loss, test_mae, test_rmse = validate(model, test_loader, criterion, device)

    # MAPE
    predictions, actuals = [], []
    model.eval()
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print("Test Results:")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # Save results
    results = {
        'config': config,
        'training_history': {
            'train_loss': [float(x) for x in train_losses],
            'val_loss': [float(x) for x in val_losses],
            'val_mae': [float(x) for x in val_maes],
            'val_rmse': [float(x) for x in val_rmses]
        },
        'test_metrics': {
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'mape': float(mape)
        },
        'best_epoch': int(best_epoch),
        'timestamp': datetime.now().isoformat()
    }

    results_dir = Path(__file__).parent.parent / 'results'
    results_file = results_dir / 'lstm_sentiment_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved: {results_file}")
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()