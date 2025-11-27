"""
Train LSTM with comprehensive sentiment features (9 features)
WITH COMPREHENSIVE LOGGING
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm.model import StockLSTM
from utils.logger import setup_logger, log_section, log_dict

def train_comprehensive_model():
    # Setup logger
    logger, log_file = setup_logger('train_comprehensive_sentiment', log_dir='Exhibition/logs')
    
    log_section(logger, "TRAINING LSTM WITH COMPREHENSIVE SENTIMENT")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    try:
        # Load data
        log_section(logger, "DATA LOADING")
        logger.info("Loading comprehensive sentiment features...")
        
        df = pd.read_csv('Code/data/processed/spy_features_with_comprehensive_sentiment.csv')
        logger.info(f"  ✓ Loaded {len(df)} days of data")
        logger.info(f"  ✓ Columns: {len(df.columns)}")
        logger.info("")
        
        # Expected features (51 total)
        EXPECTED_FEATURES = [
            # Technical indicators (42)
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
            # Comprehensive sentiment features (9)
            'sentiment_mean',
            'sentiment_median',
            'sentiment_weighted',
            'sentiment_extreme',
            'sentiment_min',
            'sentiment_max',
            'sentiment_std',
            'article_count',
            'positive_ratio'
        ]
        
        logger.info(f"Feature breakdown:")
        logger.info(f"  Technical indicators: 42")
        logger.info(f"  Sentiment features: 9")
        logger.info(f"  Total expected: {len(EXPECTED_FEATURES)}")
        logger.info("")
        
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
        
        log_section(logger, "CONFIGURATION")
        log_dict(logger, config, "Training Configuration")
        logger.info("")
        
        # Prepare features and target
        log_section(logger, "FEATURE PREPARATION")
        logger.info("Extracting features and target...")
        
        features = df[EXPECTED_FEATURES].values
        target = df['close'].values
        
        logger.info(f"  Feature shape: {features.shape}")
        logger.info(f"  Target shape: {target.shape}")
        
        # Normalize
        logger.info("Normalizing features...")
        feature_mean = features.mean(axis=0)
        feature_std = features.std(axis=0) + 1e-8
        features_norm = (features - feature_mean) / feature_std
        
        logger.info(f"  Feature normalization:")
        logger.info(f"    Mean range: [{feature_mean.min():.2f}, {feature_mean.max():.2f}]")
        logger.info(f"    Std range: [{feature_std.min():.4f}, {feature_std.max():.4f}]")
        
        target_mean = target.mean()
        target_std = target.std()
        target_norm = (target - target_mean) / target_std
        
        logger.info(f"  Target normalization:")
        logger.info(f"    Mean: ${target_mean:.2f}")
        logger.info(f"    Std: ${target_std:.2f}")
        logger.info("")
        
        # Create sequences
        log_section(logger, "SEQUENCE CREATION")
        logger.info(f"Creating sequences with length {config['sequence_length']}...")
        
        def create_sequences(features, target, seq_len):
            X, y = [], []
            for i in range(len(features) - seq_len):
                X.append(features[i:i+seq_len])
                y.append(target[i+seq_len])
            return np.array(X), np.array(y)
        
        X, y = create_sequences(features_norm, target_norm, config['sequence_length'])
        
        logger.info(f"  ✓ Created {len(X)} sequences")
        logger.info(f"  ✓ Shape: {X.shape}")
        logger.info("")
        
        # Split data
        log_section(logger, "DATA SPLITTING")
        train_size = int(len(X) * config['train_split'])
        val_size = int(len(X) * config['val_split'])
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        logger.info(f"Train: {len(X_train)} samples ({config['train_split']*100:.0f}%)")
        logger.info(f"Val:   {len(X_val)} samples ({config['val_split']*100:.0f}%)")
        logger.info(f"Test:  {len(X_test)} samples ({config['test_split']*100:.0f}%)")
        logger.info("")
        
        # Convert to tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        log_section(logger, "MODEL SETUP")
        logger.info(f"Device: {device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info("")
        
        logger.info("Converting to tensors...")
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(1).to(device)
        logger.info("  ✓ Tensors created and moved to device")
        logger.info("")
        
        # Model
        logger.info("Creating model...")
        model = StockLSTM(
            input_size=51,  # 42 technical + 9 sentiment
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"  ✓ Model created")
        logger.info(f"    Total parameters: {total_params:,}")
        logger.info(f"    Trainable parameters: {trainable_params:,}")
        logger.info(f"    Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
        logger.info("")
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Training loop
        log_section(logger, "TRAINING")
        logger.info(f"Starting training for {config['epochs']} epochs...")
        logger.info(f"Early stopping patience: {patience}")
        logger.info("")
        
        training_start = datetime.now()
        
        for epoch in range(config['epochs']):
            epoch_start = datetime.now()
            
            model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            train_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train_t), config['batch_size']):
                batch_indices = indices[i:i+config['batch_size']]
                batch_X = X_train_t[batch_indices]
                batch_y = y_train_t[batch_indices]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = train_loss / num_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            
            logger.info(f"Epoch {epoch+1:2d}/{config['epochs']}: "
                       f"Train Loss: {avg_train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, "
                       f"Time: {epoch_time:.1f}s")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'Code/models/checkpoints/lstm_comprehensive_sentiment_best.pth')
                logger.info(f"         → New best model saved! (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"")
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    logger.info(f"Best validation loss: {best_val_loss:.6f}")
                    break
        
        training_time = (datetime.now() - training_start).total_seconds()
        logger.info("")
        logger.info(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
        logger.info("")
        
        # Test evaluation
        log_section(logger, "TEST EVALUATION")
        logger.info("Loading best model...")
        model.load_state_dict(torch.load('Code/models/checkpoints/lstm_comprehensive_sentiment_best.pth'))
        model.eval()
        
        logger.info("Running predictions on test set...")
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_preds_norm = test_outputs.cpu().numpy().flatten()
            test_actual_norm = y_test_t.cpu().numpy().flatten()
        
        # Denormalize
        logger.info("Denormalizing predictions...")
        test_preds = test_preds_norm * target_std + target_mean
        test_actual = test_actual_norm * target_std + target_mean
        
        # Metrics
        logger.info("Computing metrics...")
        mae = np.mean(np.abs(test_preds - test_actual))
        rmse = np.sqrt(np.mean((test_preds - test_actual)**2))
        mape = np.mean(np.abs((test_actual - test_preds) / test_actual)) * 100
        
        logger.info("")
        log_section(logger, "TEST RESULTS")
        logger.info(f"MAE:  ${mae:.2f}")
        logger.info(f"RMSE: ${rmse:.2f}")
        logger.info(f"MAPE: {mape:.2f}%")
        logger.info("")
        
        # Compare with previous
        log_section(logger, "COMPARISON WITH PREVIOUS MODEL")
        previous_mape = 7.87
        logger.info(f"Previous (4 sentiment features):  {previous_mape:.2f}% MAPE")
        logger.info(f"Current (9 sentiment features):   {mape:.2f}% MAPE")
        logger.info("")
        
        if mape < previous_mape:
            improvement = previous_mape - mape
            pct_improvement = (improvement / previous_mape) * 100
            logger.info(f"✅ IMPROVEMENT: {improvement:.2f} percentage points ({pct_improvement:.1f}% better)")
        else:
            degradation = mape - previous_mape
            pct_degradation = (degradation / previous_mape) * 100
            logger.info(f"⚠️  CHANGE: {degradation:.2f} percentage points ({pct_degradation:.1f}% higher)")
            logger.info("   Note: Still gained better feature representation for analysis")
        
        logger.info("")
        
        # Save results
        log_section(logger, "SAVING RESULTS")
        
        results = {
            'config': config,
            'test_metrics': {
                'mape': float(mape),
                'mae': float(mae),
                'rmse': float(rmse)
            },
            'feature_count': 51,
            'sentiment_features': [
                'sentiment_mean', 'sentiment_median', 'sentiment_weighted',
                'sentiment_extreme', 'sentiment_min', 'sentiment_max',
                'sentiment_std', 'article_count', 'positive_ratio'
            ],
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'training_time_seconds': training_time,
            'best_epoch': epoch + 1 - patience_counter,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = 'Code/results/lstm_comprehensive_sentiment_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Results saved: {results_file}")
        logger.info(f"✓ Model saved: Code/models/checkpoints/lstm_comprehensive_sentiment_best.pth")
        logger.info("")
        
        log_section(logger, "TRAINING COMPLETE")
        logger.info("Next steps:")
        logger.info("  1. Analyze results:")
        logger.info("     python3 Code/scripts/analyze_comprehensive_results.py")
        logger.info("  2. Compare with baseline:")
        logger.info("     python3 Code/scripts/compare_all_models.py")
        logger.info("")
        logger.info(f"📊 Full log saved: {log_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ TRAINING FAILED: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    train_comprehensive_model()
