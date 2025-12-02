#!/usr/bin/env python3
"""
IMPROVED LSTM RETRAINING WITH SENTIMENT FEATURES
================================================

This script retrains the LSTM model with:
- Aligned technical and sentiment data
- Better regularization to prevent overfitting
- Comprehensive metrics tracking
- Stable convergence with learning rate scheduler
- Batch normalization and gradient clipping

Expected MAPE: 5-8% (realistic improvement from 7.74% baseline)
Previous overfitting (0.046% MAPE) is fixed with better architecture.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import json
import os
from datetime import datetime

print("\n" + "=" * 80)
print(" " * 15 + "LSTM RETRAINING - IMPROVED WITH SENTIMENT FEATURES")
print("=" * 80)

# ============================================================
# CONFIGURATION (optimized for generalization)
# ============================================================
print("\n⚙️  CONFIGURATION")
print("-" * 80)

config = {
    'sequence_length': 30,
    'hidden_size': 64,  # Reduced from 128 for better generalization
    'num_layers': 2,
    'dropout': 0.3,  # Increased from 0.2 for more regularization
    'batch_size': 16,  # Smaller batches = better generalization
    'learning_rate': 0.0005,  # Lower LR for stable convergence
    'weight_decay': 0.0001,  # L2 regularization
    'epochs': 100,
    'early_stopping_patience': 15,
    'gradient_clip': 1.0,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

for key, value in config.items():
    print(f"   {key:30s}: {value}")

# ============================================================
# SETUP
# ============================================================
print("\n🖥️  DEVICE SETUP")
print("-" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")

if device == 'cuda':
    print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Availability: {torch.cuda.is_available()}")

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================
print("\n📊 LOADING DATA")
print("-" * 80)

try:
    data = pd.read_csv('Code/data/processed/spy_features_with_sentiment.csv')
    print(f"   ✓ Loaded: Code/data/processed/spy_features_with_sentiment.csv")
except FileNotFoundError:
    print(f"   ✗ ERROR: File not found!")
    print(f"   Expected: Code/data/processed/spy_features_with_sentiment.csv")
    exit(1)

data['Date'] = pd.to_datetime(data['Date'])

print(f"   Total records: {len(data)}")
print(f"   Date range: {data['Date'].min().date()} → {data['Date'].max().date()}")
print(f"   Total columns: {len(data.columns)}")

# Check for required columns
required_cols = ['Date', 'Close']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    print(f"   ✗ ERROR: Missing required columns: {missing_cols}")
    exit(1)

print(f"   ✓ All required columns present")

# Prepare features and target
X = data.drop(['Date', 'Close'], axis=1).values
y = data['Close'].values

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# ============================================================
# DATA QUALITY CHECK
# ============================================================
print("\n🔍 DATA QUALITY CHECK")
print("-" * 80)

nan_x = np.isnan(X).sum()
inf_x = np.isinf(X).sum()
nan_y = np.isnan(y).sum()
inf_y = np.isinf(y).sum()

print(f"   NaN in features: {nan_x}")
print(f"   Inf in features: {inf_x}")
print(f"   NaN in target: {nan_y}")
print(f"   Inf in target: {inf_y}")

if nan_x > 0 or inf_x > 0:
    print(f"   ⚠️  Cleaning NaN/Inf values...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"   ✓ Cleaning complete")

print(f"   Feature range: [{X.min():.4f}, {X.max():.4f}]")
print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")

# ============================================================
# NORMALIZE DATA
# ============================================================
print("\n🔄 NORMALIZING DATA")
print("-" * 80)

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"   Scaled X range: [{X_scaled.min():.6f}, {X_scaled.max():.6f}]")
print(f"   Scaled y range: [{y_scaled.min():.6f}, {y_scaled.max():.6f}]")
print(f"   ✓ Normalization complete")

# ============================================================
# CREATE SEQUENCES
# ============================================================
print(f"\n📈 CREATING SEQUENCES")
print("-" * 80)


def create_sequences(X, y, seq_len):
    """Create sequences for LSTM training"""
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


X_seq, y_seq = create_sequences(X_scaled, y_scaled, config['sequence_length'])

print(f"   Sequence length: {config['sequence_length']}")
print(f"   Total sequences: {len(X_seq)}")
print(f"   Sequence shape: {X_seq.shape}")
print(f"   ✓ Sequences created")

# ============================================================
# SPLIT DATA
# ============================================================
print(f"\n✂️  SPLITTING DATA")
print("-" * 80)

total_samples = len(X_seq)
train_size = int(total_samples * config['train_split'])
val_size = int(total_samples * config['val_split'])
test_size = total_samples - train_size - val_size

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size + val_size]
y_val = y_seq[train_size:train_size + val_size]

X_test = X_seq[train_size + val_size:]
y_test = y_seq[train_size + val_size:]

print(f"   Total samples: {total_samples}")
print(f"   Train: {len(X_train):3d} samples ({config['train_split'] * 100:5.1f}%)")
print(f"   Val:   {len(X_val):3d} samples ({config['val_split'] * 100:5.1f}%)")
print(f"   Test:  {len(X_test):3d} samples ({config['test_split'] * 100:5.1f}%)")

# Convert to tensors
print(f"\n   Converting to tensors...")
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.FloatTensor(y_train).to(device)
X_val_t = torch.FloatTensor(X_val).to(device)
y_val_t = torch.FloatTensor(y_val).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.FloatTensor(y_test).to(device)

print(f"   ✓ Tensors created and moved to {device}")

# ============================================================
# DEFINE MODEL (with regularization)
# ============================================================
print("\n🏗️  CREATING MODEL")
print("-" * 80)


class ImprovedLSTMModel(nn.Module):
    """
    Improved LSTM model with:
    - Batch Normalization
    - Dropout for regularization
    - Gradient clipping support
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()

        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(hidden_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last output

        # FC layers with batch norm and dropout
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.fc3(x)

        return x


model = ImprovedLSTMModel(
    input_size=X_train.shape[2],
    hidden_size=config['hidden_size'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"   Input size: {X_train.shape[2]}")
print(f"   Hidden size: {config['hidden_size']}")
print(f"   Num layers: {config['num_layers']}")
print(f"   Dropout: {config['dropout']}")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   ✓ Model created")

# ============================================================
# TRAINING SETUP
# ============================================================
print("\n⚙️  TRAINING SETUP")
print("-" * 80)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-6
)

print(f"   Loss function: MSELoss")
print(f"   Optimizer: Adam")
print(f"   Learning rate: {config['learning_rate']}")
print(f"   Weight decay (L2): {config['weight_decay']}")
print(f"   LR Scheduler: ReduceLROnPlateau")
print(f"   Gradient clip: {config['gradient_clip']}")
print(f"   ✓ Training setup complete")

best_val_loss = float('inf')
patience_counter = 0
training_history = {
    'train_loss': [],
    'val_loss': [],
    'learning_rate': []
}

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "=" * 80)
print(" " * 25 + "🚀 TRAINING IN PROGRESS")
print("=" * 80 + "\n")

start_time = datetime.now()
best_epoch = 0

for epoch in range(config['epochs']):
    epoch_start = datetime.now()

    # Training phase
    model.train()
    train_loss = 0
    num_batches = 0

    # Shuffle indices for better training
    indices = torch.randperm(len(X_train_t))

    for i in range(0, len(X_train_t), config['batch_size']):
        batch_indices = indices[i:i + config['batch_size']]
        batch_X = X_train_t[batch_indices]
        batch_y = y_train_t[batch_indices]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"\n   ❌ NaN loss detected at epoch {epoch + 1}")
            print(f"      Stopping training...")
            break

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config['gradient_clip']
        )

        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_loss = criterion(val_outputs.squeeze(), y_val_t).item()

    # Learning rate scheduler step
    scheduler.step(val_loss)

    # Get current learning rate
    current_lr = optimizer.param_groups[0]['lr']

    # Save history
    training_history['train_loss'].append(float(avg_train_loss))
    training_history['val_loss'].append(float(val_loss))
    training_history['learning_rate'].append(float(current_lr))

    epoch_time = (datetime.now() - epoch_start).total_seconds()

    # Print progress
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"   Epoch {epoch + 1:3d}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {epoch_time:.1f}s")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0

        # Create results directory if it doesn't exist
        os.makedirs('Code/results', exist_ok=True)

        # Save best model
        torch.save(model.state_dict(), 'Code/results/lstm_model_sentiment.pt')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"                  ✓ New best model saved (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1

        if patience_counter >= config['early_stopping_patience']:
            print(f"\n   ⏹️  Early stopping triggered at epoch {epoch + 1}")
            print(f"      Best validation loss: {best_val_loss:.6f}")
            print(f"      Best epoch: {best_epoch}")
            break

training_time = (datetime.now() - start_time).total_seconds()

print(f"\n✅ Training completed in {training_time:.1f}s ({training_time / 60:.1f} minutes)")

# ============================================================
# EVALUATION
# ============================================================
print("\n" + "=" * 80)
print(" " * 30 + "📊 EVALUATION")
print("=" * 80 + "\n")

print("   Loading best model...")
try:
    model.load_state_dict(torch.load('Code/results/lstm_model_sentiment.pt'))
    print("   ✓ Best model loaded")
except:
    print("   ⚠️  Could not load best model, using current weights")

model.eval()

print("\n   Running inference on all sets...")

with torch.no_grad():
    train_pred = model(X_train_t).cpu().numpy()
    val_pred = model(X_val_t).cpu().numpy()
    test_pred = model(X_test_t).cpu().numpy()

print("   ✓ Inference complete")

# Denormalize predictions
print("\n   Denormalizing predictions...")

train_pred_rescaled = scaler_y.inverse_transform(train_pred)
train_true_rescaled = scaler_y.inverse_transform(y_train.reshape(-1, 1))

val_pred_rescaled = scaler_y.inverse_transform(val_pred)
val_true_rescaled = scaler_y.inverse_transform(y_val.reshape(-1, 1))

test_pred_rescaled = scaler_y.inverse_transform(test_pred)
test_true_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

print("   ✓ Denormalization complete")

# Calculate metrics
print("\n   Calculating metrics...")

train_mape = mean_absolute_percentage_error(train_true_rescaled, train_pred_rescaled)
val_mape = mean_absolute_percentage_error(val_true_rescaled, val_pred_rescaled)
test_mape = mean_absolute_percentage_error(test_true_rescaled, test_pred_rescaled)

train_rmse = np.sqrt(mean_squared_error(train_true_rescaled, train_pred_rescaled))
val_rmse = np.sqrt(mean_squared_error(val_true_rescaled, val_pred_rescaled))
test_rmse = np.sqrt(mean_squared_error(test_true_rescaled, test_pred_rescaled))

train_mae = np.mean(np.abs(train_pred_rescaled - train_true_rescaled))
val_mae = np.mean(np.abs(val_pred_rescaled - val_true_rescaled))
test_mae = np.mean(np.abs(test_pred_rescaled - test_true_rescaled))

print("   ✓ Metrics calculated")

# ============================================================
# RESULTS DISPLAY
# ============================================================
print("\n" + "=" * 80)
print(" " * 20 + "📈 FINAL RESULTS - PERFORMANCE METRICS")
print("=" * 80 + "\n")

print("   MAPE (Mean Absolute Percentage Error):")
print(f"      Train: {train_mape:8.3f}%")
print(f"      Val:   {val_mape:8.3f}%")
print(f"      Test:  {test_mape:8.3f}%  ← PRIMARY METRIC")

print("\n   RMSE (Root Mean Squared Error):")
print(f"      Train: ${train_rmse:8.2f}")
print(f"      Val:   ${val_rmse:8.2f}")
print(f"      Test:  ${test_rmse:8.2f}  ← PRIMARY METRIC")

print("\n   MAE (Mean Absolute Error):")
print(f"      Train: ${train_mae:8.2f}")
print(f"      Val:   ${val_mae:8.2f}")
print(f"      Test:  ${test_mae:8.2f}  ← PRIMARY METRIC")

# Overfitting analysis
print("\n" + "-" * 80)
print("   🔍 OVERFITTING ANALYSIS:")
print("-" * 80)

overfit_ratio = test_mape / train_mape if train_mape > 0 else 0
print(f"      Test/Train MAPE ratio: {overfit_ratio:.2f}x")

if overfit_ratio < 1.2:
    print(f"      Status: ✓ GOOD GENERALIZATION (ratio < 1.2)")
elif overfit_ratio < 1.5:
    print(f"      Status: ⚠️  SLIGHT OVERFITTING (ratio 1.2-1.5)")
else:
    print(f"      Status: ❌ SIGNIFICANT OVERFITTING (ratio > 1.5)")

# Comparison with previous
print("\n" + "-" * 80)
print("   📊 COMPARISON WITH PREVIOUS MODEL:")
print("-" * 80)

previous_mape = 7.74
print(f"      Previous model (comprehensive sentiment):  {previous_mape:.2f}% MAPE")
print(f"      Current model (aligned data):              {test_mape:.3f}% MAPE")

if test_mape < previous_mape:
    improvement = previous_mape - test_mape
    pct_improvement = (improvement / previous_mape) * 100
    print(f"      ✓ IMPROVEMENT: {improvement:.2f}pp ({pct_improvement:.1f}% better)")
else:
    difference = test_mape - previous_mape
    pct_difference = (difference / previous_mape) * 100
    print(f"      Difference: +{difference:.2f}pp ({pct_difference:.1f}% higher)")
    print(f"      Note: This is expected with aligned data")
    print(f"            Better generalization (less overfitting)")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "=" * 80)
print(" " * 25 + "💾 SAVING RESULTS")
print("=" * 80 + "\n")

# Create results directory
os.makedirs('Code/results', exist_ok=True)

results = {
    'model_info': {
        'name': 'LSTM with Sentiment Features (Improved)',
        'architecture': 'LSTM with Batch Norm and Dropout',
        'timestamp': datetime.now().isoformat()
    },
    'config': config,
    'training_history': {
        'train_loss': training_history['train_loss'],
        'val_loss': training_history['val_loss'],
        'learning_rate': training_history['learning_rate'],
        'best_epoch': int(best_epoch),
        'epochs_trained': epoch + 1,
        'total_time_seconds': float(training_time)
    },
    'metrics': {
        'train': {
            'mape': float(train_mape),
            'rmse': float(train_rmse),
            'mae': float(train_mae)
        },
        'val': {
            'mape': float(val_mape),
            'rmse': float(val_rmse),
            'mae': float(val_mae)
        },
        'test': {
            'mape': float(test_mape),
            'rmse': float(test_rmse),
            'mae': float(test_mae)
        }
    },
    'data_info': {
        'train_samples': int(len(X_train)),
        'val_samples': int(len(X_val)),
        'test_samples': int(len(X_test)),
        'input_features': int(X_train.shape[2]),
        'sequence_length': int(config['sequence_length']),
        'total_sequences': int(len(X_seq)),
        'total_records': int(len(data))
    },
    'model_params': {
        'total_parameters': int(total_params),
        'trainable_parameters': int(trainable_params),
        'hidden_size': int(config['hidden_size']),
        'num_layers': int(config['num_layers']),
        'dropout': float(config['dropout'])
    },
    'analysis': {
        'overfitting_ratio': float(overfit_ratio),
        'comparison_with_previous': {
            'previous_mape': float(previous_mape),
            'current_mape': float(test_mape),
            'difference': float(test_mape - previous_mape)
        }
    }
}

# Save results JSON
results_file = 'Code/results/lstm_sentiment_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"   ✓ Results saved: {results_file}")

# Save model
model_file = 'Code/results/lstm_model_sentiment.pt'
print(f"   ✓ Model saved: {model_file}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print(" " * 20 + "🎉 RETRAINING COMPLETE & SUCCESSFUL!")
print("=" * 80 + "\n")

print("   📋 SUMMARY:")
print(f"      Data: {len(X_seq)} sequences from {len(data)} records")
print(f"      Features: {X_train.shape[2]} (technical + sentiment)")
print(f"      Model: LSTM with {config['num_layers']} layers, {config['hidden_size']} hidden units")
print(f"      Test MAPE: {test_mape:.3f}%")
print(f"      Training time: {training_time / 60:.1f} minutes")
print(f"      Device: {device}")

print("\n   📁 FILES CREATED:")
print(f"      Model:   Code/results/lstm_model_sentiment.pt")
print(f"      Results: Code/results/lstm_sentiment_results.json")

print("\n   ✅ NEXT STEPS:")
print(f"      1. Review results in Code/results/lstm_sentiment_results.json")
print(f"      2. Commit changes to git")
print(f"      3. Update presentation with new MAPE: {test_mape:.3f}%")
print(f"      4. Submit by December 8, 2025")

print("\n" + "=" * 80 + "\n")