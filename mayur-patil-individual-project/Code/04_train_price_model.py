import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from config import DataConfig, PROCESSED_DATA_DIR, MODELS_DIR, ModelConfig, SequenceConfig
from models import PriceLSTMModel

data_config = DataConfig()
model_config = ModelConfig()
seq_config = SequenceConfig()

def main():
    print("=" * 60)
    print("Training Price LSTM Model")
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
    
    feature_dim = X_train.shape[-1]
    print(f"Feature dimension: {feature_dim}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Model
    model = PriceLSTMModel(
        input_dim=feature_dim,
        hidden_dim=model_config.price_hidden_dim,
        num_layers=model_config.price_num_layers,
        dropout=model_config.dropout,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    batch_size = 32
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    
    print(f"Training for {num_epochs} epochs, batch_size={batch_size}...")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size].unsqueeze(1)
            
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss /= num_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val.unsqueeze(1)).item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            ckpt_path = MODELS_DIR / "price_lstm_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_dim": feature_dim,
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
