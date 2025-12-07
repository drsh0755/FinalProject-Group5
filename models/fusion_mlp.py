# models/fusion_mlp.py

import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """
    Fuse price representation (from LSTM) and daily sentiment features.
    price_repr_dim: dimension of representation from price model
    sentiment_dim: dimension of aggregated sentiment features
    """

    def __init__(self, price_repr_dim: int, sentiment_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(price_repr_dim + sentiment_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, price_repr, sentiment_features):
        # price_repr: (batch, d1)
        # sentiment_features: (batch, d2)
        x = torch.cat([price_repr, sentiment_features], dim=-1)
        return self.net(x)
