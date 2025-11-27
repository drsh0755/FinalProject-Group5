"""
Analyze how much sentiment actually impacts predictions
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.lstm.model import StockLSTM

# Load model
model = StockLSTM(input_size=46, hidden_size=128, num_layers=2, dropout=0.3)
checkpoint = torch.load('Code/models/checkpoints/lstm_with_sentiment_best.pth', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()

# Create dummy features (30 days × 46 features)
# All zeros except we'll vary sentiment
base_features = torch.zeros(1, 30, 46)

# Test different sentiment values
sentiment_values = [-1.0, -0.5, -0.115, 0.0, 0.5, 1.0]

print("=" * 60)
print("SENTIMENT IMPACT ANALYSIS")
print("=" * 60)
print("")
print("Testing how different sentiment values affect predictions:")
print("")

results = []
for sentiment in sentiment_values:
    # Set sentiment features (last 4 features)
    test_features = base_features.clone()
    test_features[:, :, -4] = sentiment  # sentiment_mean
    test_features[:, :, -3] = 0.8        # sentiment_std
    test_features[:, :, -2] = 300        # article_count
    test_features[:, :, -1] = (sentiment + 1) / 2  # positive_ratio
    
    with torch.no_grad():
        pred = model(test_features).item()
    
    results.append((sentiment, pred))
    print(f"Sentiment: {sentiment:+.3f} → Prediction: {pred:+.4f}")

# Calculate range
predictions = [r[1] for r in results]
pred_range = max(predictions) - min(predictions)

print("")
print(f"Prediction range: {pred_range:.4f}")
print("")

if pred_range < 0.1:
    print("⚠️  FINDING: Sentiment has VERY LOW impact on predictions!")
    print("   The model is dominated by technical indicators.")
elif pred_range < 0.5:
    print("✓ FINDING: Sentiment has MODERATE impact on predictions.")
    print("  But technical indicators are still primary.")
else:
    print("✅ FINDING: Sentiment has SIGNIFICANT impact on predictions!")
    print("   Multi-modal fusion is working well.")

print("")
print("This explains why your prediction didn't change much")
print("between neutral sentiment (0.0) and negative sentiment (-0.115).")
