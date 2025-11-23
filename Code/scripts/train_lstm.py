#!/usr/bin/env python3
"""
Train LSTM model for stock price prediction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm.model import StockLSTM
from models.lstm.dataset import StockDataset

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate metrics
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    return avg_loss, mae, rmse, mape

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def main():
    print("\n" + "="*60)
    print("LSTM MODEL TRAINING")
    print("="*60)
    
    # ============================================================
    # CONFIGURATION
    # ============================================================
    config = {
        'sequence_length': 60,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 50,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # ============================================================
    # SETUP
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Paths
    script_dir = Path(__file__).parent
    code_dir = script_dir.parent
    data_path = code_dir / 'data' / 'processed' / 'SPY_with_indicators.csv'
    model_dir = code_dir / 'models' / 'checkpoints'
    results_dir = code_dir / 'results'
    
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # LOAD DATASET
    # ============================================================
    print("\n" + "="*60)
    print("Loading data...")
    print("="*60)
    
    dataset = StockDataset(
        data_path,
        sequence_length=config['sequence_length']
    )
    
    # Split dataset
    total_size = len(dataset)
    train_size = int(config['train_split'] * total_size)
    val_size = int(config['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} sequences")
    print(f"  Validation: {len(val_dataset)} sequences")
    print(f"  Test: {len(test_dataset)} sequences")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # ============================================================
    # CREATE MODEL
    # ============================================================
    print("\n" + "="*60)
    print("Creating model...")
    print("="*60)
    
    input_size = len(dataset.feature_cols)
    model = StockLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    
    summary = model.get_model_summary()
    print(f"\nModel summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # ============================================================
    # TRAINING LOOP
    # ============================================================
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': []
    }
    
    for epoch in range(config['epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_mae, val_rmse, val_mape = validate(model, val_loader, criterion, device)
        
        # Save history
        training_history['train_loss'].append(float(train_loss))
        training_history['val_loss'].append(float(val_loss))
        training_history['val_mae'].append(float(val_mae))
        training_history['val_rmse'].append(float(val_rmse))
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config['epochs']}]")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val MAE: {val_mae:.4f}")
            print(f"  Val RMSE: {val_rmse:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': float(val_loss),
                'config': config
            }, model_dir / 'lstm_best.pth')
    
    # ============================================================
    # EVALUATION
    # ============================================================
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(model_dir / 'lstm_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_mae, test_rmse, test_mape = validate(model, test_loader, criterion, device)
    
    print(f"\nTest Results:")
    print(f"  MSE Loss: {test_loss:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    
    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results = {
        'config': config,
        'training_history': training_history,
        'test_metrics': {
            'mse': float(test_loss),
            'mae': float(test_mae),
            'rmse': float(test_rmse),
            'mape': float(test_mape)
        },
        'best_epoch': int(checkpoint['epoch']),
        'timestamp': datetime.now().isoformat()
    }
    
    # Convert all numpy types to Python types
    results = convert_to_serializable(results)
    
    with open(results_dir / 'lstm_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_dir / 'lstm_training_results.json'}")
    print(f"✓ Model saved to: {model_dir / 'lstm_best.pth'}")
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
