"""
Test prediction with extreme sentiment values
"""

import torch
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm.model import StockLSTM

print("Testing with EXTREME sentiment values...")
print("")

# Get actual recent data
df = yf.Ticker('SPY').history(start=datetime.now()-timedelta(60), end=datetime.now())
df = df.reset_index()
current_price = df['Close'].iloc[-1]

# [Calculate technical indicators - abbreviated]
# ... (same as your script)

# Load model
model = StockLSTM(input_size=46, hidden_size=128, num_layers=2, dropout=0.3)
checkpoint = torch.load('Code/models/checkpoints/lstm_with_sentiment_best.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Test scenarios
scenarios = [
    ("VERY NEGATIVE", -0.9, 0.1, 300, 0.05),
    ("ACTUAL TODAY", -0.115, 0.87, 300, 0.46),
    ("NEUTRAL", 0.0, 0.0, 0, 0.5),
    ("VERY POSITIVE", 0.9, 0.1, 300, 0.95),
]

target_mean = 671.05
target_std = 8.66

print("=" * 70)
print("SCENARIO TESTING: Impact of Extreme Sentiment")
print("=" * 70)
print("")

for name, sent_mean, sent_std, articles, pos_ratio in scenarios:
    # Create features with this sentiment
    # [Include actual technical data + test sentiment]
    
    # Make prediction
    # pred_price = ...
    
    print(f"{name:20} Sentiment: {sent_mean:+.2f} → Price: $XXX.XX")

print("")
print("If predictions vary significantly, sentiment IS working!")
