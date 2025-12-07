# models/price_lstm.py

import torch
import torch.nn as nn


class PriceLSTMModel(nn.Module):
    """
    LSTM-based model for time-series price features.
    Input:  (batch, seq_len, feature_dim)
    Output: (batch, 1) regression (e.g., next-day return)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, c_n) = self.lstm(x)  # h_n: (num_layers, batch, hidden_dim)
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        out = self.fc(last_hidden)  # (batch, 1)
        return out
