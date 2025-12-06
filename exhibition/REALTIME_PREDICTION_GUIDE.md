# Real-Time Prediction System Guide

## 🚀 How to Use

### Step 1: Make Today's Prediction
```bash
cd ~/DL/"Final Project"
python3 Code/scripts/live_realtime_prediction.py
```

**What it does:**
1. Fetches latest SPY price data (last 60 days)
2. Fetches today's news for 7 major tech companies
3. Analyzes sentiment using FinBERT
4. Computes 46 technical indicators
5. Makes prediction for tomorrow
6. Saves prediction to `Exhibition/realtime_predictions.jsonl`

**Note:** You need an Alpha Vantage API key. Get a free one at: https://www.alphavantage.co/support/#api-key

Update the script with your key:
```python
self.av_api_key = "YOUR_KEY_HERE"  # Line 37 in live_realtime_prediction.py
```

### Step 2: Verify Tomorrow

**Next day, run:**
```bash
python3 Code/scripts/verify_predictions.py
```

**What it does:**
1. Loads yesterday's prediction
2. Fetches actual price for today
3. Calculates prediction error (MAPE)
4. Checks if direction was correct
5. Saves verification to `Exhibition/verified_predictions.jsonl`

## 📊 Example Output

**Making Prediction (Today):**
```
🔮 PREDICTION FOR 2025-11-26 Tuesday:
   Predicted Price: $625.34
   Expected Change: ↑ $5.67 (+0.91%)

📰 News Sentiment:
   Articles analyzed: 127
   Mean sentiment: 0.245
   Positive ratio: 62.3%
```

**Verification (Tomorrow):**
```
✅ ACTUAL RESULTS:
💵 Actual price: $627.89
📊 Actual change: +8.22 (+1.33%)

📏 PREDICTION ACCURACY:
   Absolute error: $2.55
   MAPE: 0.41%
   Direction: ✅ CORRECT (predicted up, actual up)

🎯 ASSESSMENT: Excellent prediction!
```

## 📈 Track Performance Over Time
```bash
# View all predictions
cat Exhibition/realtime_predictions.jsonl | jq '.'

# View all verifications
cat Exhibition/verified_predictions.jsonl | jq '.'

# Calculate average MAPE
cat Exhibition/verified_predictions.jsonl | jq '.mape' | awk '{sum+=$1; n++} END {print "Average MAPE:", sum/n "%"}'
```

## ⚠️ Important Notes

1. **Market Hours:** Stock market is open Mon-Fri 9:30 AM - 4:00 PM EST
2. **Weekend Predictions:** If you run on Friday, it predicts for Monday
3. **API Limits:** Free Alpha Vantage key allows 25 requests/day
4. **News Freshness:** Alpha Vantage news updates every few minutes

## 🎯 For Your Presentation

You can demonstrate:
1. **Live Prediction:** Run the script during presentation
2. **Historical Verification:** Show past predictions and their accuracy
3. **Real-World Viability:** Prove the model works on unseen future data

**Talking Points:**
- "We've been running live predictions since [date]"
- "Our real-time MAPE is [X]%, confirming our test set results"
- "The model correctly predicted direction in [X]% of cases"

## 🔧 Troubleshooting

**Error: No data returned**
- Check internet connection
- Verify API key is valid
- Market might be closed (try during trading hours)

**Error: Module not found**
- Install missing packages: `pip install yfinance transformers --break-system-packages`

**Error: CUDA out of memory**
- FinBERT runs on CPU by default, should work fine

