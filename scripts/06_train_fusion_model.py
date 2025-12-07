# scripts/06_train_fusion_model.py
"""
Train the fusion model (price LSTM + sentiment).
"""

import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from config import DataConfig, PROCESSED_DATA_DIR, MODELS_DIR, ModelConfig, SequenceConfig
from models import PriceLSTMModel, FusionMLP
from utils import build_sequences, train_val_test_split

data_config = DataConfig()
model_config = ModelConfig()
seq_config = SequenceConfig()

def main():
    print("=" * 60)
    print("Training Fusion Model (Price LSTM + Sentiment)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load sequences
    symbol = data_config.symbol
    sequences_path = PROCESSED_DATA_DIR / f"{symbol}_sequences.pkl"
    print(f"Loading sequences from {sequences_path}...")
    
    with open(sequences_path, "rb") as f:
        sequences = pickle.load(f)
    
    X_train = torch.tensor(sequences["X_train"], dtype=torch.float32).to(device)
    y_train = torch.tensor(sequences["y_train"], dtype=torch.float32).to(device)
    X_val = torch.tensor(sequences["X_val"], dtype=torch.float32).to(device)
    y_val = torch.tensor(sequences["y_val"], dtype=torch.float32).to(device)
    
    # Load sentiment features
    features_path = PROCESSED_DATA_DIR / f"{symbol}_features_with_sentiment.csv"
    print(f"Loading sentiment features from {features_path}...")
    sentiment_df = pd.read_csv(features_path)
    
    sentiment_cols = ["sentiment_mean", "sentiment_std", "sentiment_count"]
    sentiment_data = sentiment_df[sentiment_cols].values.astype("float32")
    
    # Align sentiment with sequences (skip first `window` rows)
    window = seq_config.input_window
    sentiment_data = sentiment_data[window:, :]
    
    S_train = torch.tensor(sentiment_data[:len(X_train)], dtype=torch.float32).to(device)
    S_val = torch.tensor(sentiment_data[len(X_train):len(X_train)+len(X_val)], dtype=torch.float32).to(device)
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  S_train: {S_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  S_val: {S_val.shape}")
    
    # Load pre-trained price LSTM
    price_ckpt_path = MODELS_DIR / "price_lstm_best.pt"
    print(f"Loading pre-trained price LSTM from {price_ckpt_path}...")
    price_ckpt = torch.load(price_ckpt_path, map_location=device)
    
    feature_dim = price_ckpt["feature_dim"]
    price_model = PriceLSTMModel(
        input_dim=feature_dim,
        hidden_dim=model_config.price_hidden_dim,
        num_layers=model_config.price_num_layers,
        dropout=model_config.dropout,
    ).to(device)
    price_model.load_state_dict(price_ckpt["model_state_dict"])
    price_model.eval()
    
    # Extract price representations
    print("Extracting price representations...")
    with torch.no_grad():
        _, (h_n_train, _) = price_model.lstm(X_train)
        price_repr_train = h_n_train[-1]  # (batch, hidden_dim)
        
        _, (h_n_val, _) = price_model.lstm(X_val)
        price_repr_val = h_n_val[-1]  # (batch, hidden_dim)
    
    print(f"  Price repr train: {price_repr_train.shape}")
    print(f"  Price repr val: {price_repr_val.shape}")
    
    # Fusion model
    fusion_model = FusionMLP(
        price_repr_dim=model_config.price_hidden_dim,
        sentiment_dim=len(sentiment_cols),
        hidden_dim=model_config.fusion_hidden_dim,
        dropout=model_config.dropout,
    ).to(device)
    
    print(f"Fusion model parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    batch_size = 32
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    
    print(f"Training for {num_epochs} epochs, batch_size={batch_size}...")
    
    for epoch in range(num_epochs):
        # Train
        fusion_model.train()
        train_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(price_repr_train), batch_size):
            pr_batch = price_repr_train[i : i + batch_size]
            s_batch = S_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size].unsqueeze(1)
            
            optimizer.zero_grad()
            pred = fusion_model(pr_batch, s_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation
        fusion_model.eval()
        with torch.no_grad():
            val_pred = fusion_model(price_repr_val, S_val)
            val_loss = criterion(val_pred, y_val.unsqueeze(1)).item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            ckpt_path = MODELS_DIR / "fusion_mlp_best.pt"
            torch.save(
                {
                    "model_state_dict": fusion_model.state_dict(),
                    "price_repr_dim": model_config.price_hidden_dim,
                    "sentiment_dim": len(sentiment_cols),
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best model to {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    print(f"\n✓ Training complete! Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
