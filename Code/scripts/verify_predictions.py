"""
Verify Real-Time Predictions
Check how accurate yesterday's predictions were
"""

import json
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

def verify_latest_prediction():
    """
    Load latest prediction and check against actual price
    """
    predictions_file = Path('Exhibition/realtime_predictions.jsonl')
    
    if not predictions_file.exists():
        print("❌ No predictions found. Run live_realtime_prediction.py first!")
        return
    
    # Load all predictions
    predictions = []
    with open(predictions_file, 'r') as f:
        for line in f:
            predictions.append(json.loads(line))
    
    if not predictions:
        print("❌ No predictions in file!")
        return
    
    # Get the latest prediction
    latest = predictions[-1]
    
    print("\n" + "="*60)
    print("PREDICTION VERIFICATION")
    print("="*60)
    
    print(f"\n📅 Prediction made on: {latest['current_date']}")
    print(f"🎯 Prediction for: {latest['prediction_date']}")
    print(f"💰 Price at prediction time: ${latest['current_price']:.2f}")
    print(f"🔮 Predicted price: ${latest['predicted_price']:.2f}")
    print(f"📈 Predicted change: {latest['predicted_change']:+.2f} ({latest['predicted_change_pct']:+.2f}%)")
    
    # Fetch actual data
    print(f"\n📡 Fetching actual data...")
    
    pred_date = datetime.fromisoformat(latest['prediction_date'])
    end_date = pred_date + timedelta(days=3)
    
    ticker = yf.Ticker('SPY')
    df = ticker.history(start=pred_date, end=end_date)
    
    if df.empty:
        print("⚠️  No actual data available yet (market might be closed)")
        return
    
    actual_price = df['Close'].iloc[0]
    actual_change = actual_price - latest['current_price']
    actual_change_pct = (actual_change / latest['current_price']) * 100
    
    # Calculate error
    prediction_error = abs(actual_price - latest['predicted_price'])
    mape = (prediction_error / actual_price) * 100
    
    print(f"\n✅ ACTUAL RESULTS:")
    print(f"💵 Actual price: ${actual_price:.2f}")
    print(f"📊 Actual change: {actual_change:+.2f} ({actual_change_pct:+.2f}%)")
    
    print(f"\n📏 PREDICTION ACCURACY:")
    print(f"   Absolute error: ${prediction_error:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Direction accuracy
    pred_direction = "up" if latest['predicted_change'] > 0 else "down"
    actual_direction = "up" if actual_change > 0 else "down"
    direction_correct = pred_direction == actual_direction
    
    print(f"   Direction: {'✅ CORRECT' if direction_correct else '❌ INCORRECT'}")
    print(f"      Predicted: {pred_direction}")
    print(f"      Actual: {actual_direction}")
    
    # Overall assessment
    print(f"\n🎯 ASSESSMENT:")
    if mape < 2:
        assessment = "Excellent"
    elif mape < 5:
        assessment = "Very Good"
    elif mape < 10:
        assessment = "Good"
    else:
        assessment = "Needs Improvement"
    
    print(f"   {assessment} prediction!")
    print(f"   (Model average: 7.87% MAPE)")
    
    print("="*60)
    
    # Update prediction record with actual
    latest['actual_price'] = float(actual_price)
    latest['actual_change'] = float(actual_change)
    latest['error'] = float(prediction_error)
    latest['mape'] = float(mape)
    latest['direction_correct'] = direction_correct
    latest['verified_at'] = datetime.now().isoformat()
    
    # Save updated record
    verified_file = Path('Exhibition/verified_predictions.jsonl')
    with open(verified_file, 'a') as f:
        f.write(json.dumps(latest) + '\n')
    
    print(f"\n💾 Verification saved to: {verified_file}")

if __name__ == "__main__":
    verify_latest_prediction()
