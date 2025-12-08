import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from config import DataConfig, PROCESSED_DATA_DIR, SequenceConfig
from utils import build_sequences, train_val_test_split

data_config = DataConfig()
seq_config = SequenceConfig()

def main():
    print("=" * 60)
    print("Sequence Preparation Pipeline")
    print("=" * 60)
    
    symbol = data_config.symbol
    merged_path = PROCESSED_DATA_DIR / f"{symbol}_features_merged.csv"
    
    print(f"Loading features from {merged_path}...")
    df = pd.read_csv(merged_path, parse_dates=["date"])
    
    # Feature columns (all numeric except date and target)
    feature_cols = [col for col in df.columns if col not in ["date", seq_config.target_column]]
    
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:5]}...")
    print(f"Target column: {seq_config.target_column}")
    
    # Build sequences
    print(f"Building sequences with window={seq_config.input_window}...")
    X, y = build_sequences(
        df,
        feature_cols=feature_cols,
        target_col=seq_config.target_column,
        window=seq_config.input_window,
    )
    
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    
    # Train-Val-Test split
    print(f"Splitting data: train={seq_config.train_val_test_split[0]}, "
          f"val={seq_config.train_val_test_split[1]}, "
          f"test={seq_config.train_val_test_split[2]}")
    
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y,
        train_ratio=seq_config.train_val_test_split[0],
        val_ratio=seq_config.train_val_test_split[1],
    )
    
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    
    # Save sequences
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    sequences = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_cols": feature_cols,
    }
    
    output_path = PROCESSED_DATA_DIR / f"{symbol}_sequences.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)
    
    print(f"\n✓ Saved sequences to {output_path}")


if __name__ == "__main__":
    main()
