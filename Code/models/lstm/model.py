"""
LSTM Model for Stock Price Prediction
"""

import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """
    LSTM model for stock price prediction
    
    Architecture:
    - Input: Sequence of technical indicators
    - LSTM layers with dropout
    - Fully connected layers
    - Output: Next day's closing price
    """
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(StockLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out

    def get_model_summary(self):
        """Get model architecture summary"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }


if __name__ == "__main__":
    # Test the model
    print("Testing StockLSTM model...")
    
    # Model parameters
    input_size = 46  # Number of features
    sequence_length = 60  # Days of history
    batch_size = 32
    
    # Create model
    model = StockLSTM(input_size=input_size)
    
    # Create dummy input
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # Forward pass
    output = model(x)
    
    print(f"✓ Model created successfully")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"\nModel summary:")
    summary = model.get_model_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
