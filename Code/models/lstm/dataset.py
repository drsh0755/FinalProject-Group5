"""
Dataset class for LSTM training
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class StockDataset(Dataset):
    """
    PyTorch Dataset for stock price prediction
    
    Creates sequences of historical data for LSTM input
    """
    
    def __init__(self, data_path, sequence_length=60, target_col='Close', 
                 feature_cols=None, scale=True):
        """
        Args:
            data_path: Path to CSV with technical indicators
            sequence_length: Number of days to look back
            target_col: Column to predict
            feature_cols: List of feature columns (None = all except target)
            scale: Whether to standardize features
        """
        self.sequence_length = sequence_length
        self.target_col = target_col
        self.scale = scale
        
        # Load data
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        # Select features
        if feature_cols is None:
            # Use all columns except target
            feature_cols = [col for col in df.columns if col != target_col]
        
        self.feature_cols = feature_cols
        
        # Extract features and target
        self.features = df[feature_cols].values
        self.targets = df[target_col].values
        
        # Scale features
        if self.scale:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = None
        
        print(f"✓ Dataset loaded:")
        print(f"  Total samples: {len(self.features)}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Usable sequences: {len(self)}")
    
    def __len__(self):
        """Number of sequences available"""
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        """
        Get one sequence
        
        Returns:
            features: Tensor of shape (sequence_length, num_features)
            target: Scalar tensor (next day's price)
        """
        # Get sequence of features
        sequence = self.features[idx:idx + self.sequence_length]
        
        # Get target (next day's price)
        target = self.targets[idx + self.sequence_length]
        
        return (
            torch.FloatTensor(sequence),
            torch.FloatTensor([target])
        )
    
    def get_scaler(self):
        """Return the fitted scaler"""
        return self.scaler


if __name__ == "__main__":
    # Test the dataset
    from pathlib import Path
    
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'SPY_with_indicators.csv'
    
    print("Testing StockDataset...")
    dataset = StockDataset(data_path, sequence_length=60)
    
    # Get one sample
    features, target = dataset[0]
    print(f"\nSample data:")
    print(f"  Features shape: {features.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Target value: {target.item():.2f}")
