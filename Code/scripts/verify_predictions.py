# !/usr/bin/env python3
"""
Verification Script - Verify daily predictions (Fixed)
"""

import json
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np


class PredictionVerifier:
    def __init__(self):
        print("\n" + "=" * 80)
        print("🔍 INITIALIZING PREDICTION VERIFIER")
        print("=" * 80)

        self.predictions_file = Path('Exhibition/realtime_predictions.jsonl')
        self.verified_file = Path('Exhibition/verified_predictions.jsonl')
        self.predictions_file.parent.mkdir(exist_ok=True)

        print(f"✓ Predictions file: {self.predictions_file}")
        print(f"✓ Verified file: {self.verified_file}")

    def load_latest_prediction(self):
        """Load the latest prediction from JSONL"""
        print("\n📖 LOADING LATEST PREDICTION")
        print("-" * 80)

        if not self.predictions_file.exists():
            print(f"❌ No predictions file found")
            return None

        with open(self.predictions_file, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("❌ Predictions file is empty")
                return None
            latest = json.loads(lines[-1])

        print(f"✓ Loaded prediction:")
        print(f"  Current date: {latest['current_date']}")
        print(f"  Prediction date: {latest['prediction_date']}")
        print(f"  Predicted price: ${latest['predicted_price']:.2f}")
        print(f"  Predicted change: {latest['predicted_change_pct']:+.2f}%")

        return latest

    def get_actual_price(self, date_str):
        """Fetch actual closing price"""
        print(f"\n📊 FETCHING ACTUAL PRICE FOR {date_str}")
        print("-" * 80)

        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')

            # Today's date
            today = datetime.now().date()
            pred_date = date_obj.date()

            # Check if prediction date is in the future
            if pred_date > today:
                print(f"⚠️  Prediction date {date_str} hasn't occurred yet")
                print(f"   Today is {today}")
                return None

            # Get data for broader range
            start = date_obj - timedelta(days=5)
            end = min(date_obj + timedelta(days=5), datetime.now())

            ticker = yf.Ticker('SPY')
            df = ticker.history(start=start, end=end)

            if df.empty:
                print(f"⚠️  No data available")
                return None

            # Convert index to dates for comparison
            df.index = pd.to_datetime(df.index).date

            # Try exact date first
            if pred_date in df.index:
                price = df.loc[pred_date, 'Close']
                print(f"✓ Found exact price for {date_str}: ${price:.2f}")
                return float(price)

            # Find next trading day
            future_dates = [d for d in df.index if d >= pred_date]
            if future_dates:
                actual_date = min(future_dates)
                price = df.loc[actual_date, 'Close']
                print(f"⚠️  {date_str} was not a trading day")
                print(f"✓ Using next trading day {actual_date}: ${price:.2f}")
                return float(price)

            # No future dates available
            print(f"⚠️  No trading data available for or after {date_str}")
            return None

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def verify_prediction(self, prediction, actual_price):
        """Verify prediction"""
        print("\n🎯 VERIFYING PREDICTION")
        print("-" * 80)

        predicted = prediction['predicted_price']
        current = prediction['current_price']

        # Calculate errors
        absolute_error = abs(predicted - actual_price)
        mape = (absolute_error / actual_price) * 100

        # Calculate actual change
        actual_change = ((actual_price - current) / current) * 100
        pred_change = prediction['predicted_change_pct']

        # Check direction
        direction_correct = (pred_change > 0 and actual_change > 0) or (pred_change <= 0 and actual_change <= 0)

        print(f"Current price:    ${current:.2f}")
        print(f"Predicted price:  ${predicted:.2f}")
        print(f"Actual price:     ${actual_price:.2f}")
        print(f"Absolute error:   ${absolute_error:.2f}")
        print(f"MAPE:             {mape:.3f}%")
        print(f"\nPredicted change: {pred_change:+.2f}%")
        print(f"Actual change:    {actual_change:+.2f}%")
        print(f"Direction:        {'✓ CORRECT' if direction_correct else '✗ WRONG'}")

        return {
            'actual_price': float(actual_price),
            'actual_change_pct': float(actual_change),
            'absolute_error': float(absolute_error),  # KEY FIX: Use consistent name
            'mape': float(mape),
            'direction_correct': bool(direction_correct),
            'verified_at': datetime.now().isoformat()
        }

    def save_verified_prediction(self, prediction, verification):
        """Save verified prediction"""
        print(f"\n💾 SAVING VERIFIED PREDICTION")
        print("-" * 80)

        verified_record = {**prediction, **verification}

        with open(self.verified_file, 'a') as f:
            f.write(json.dumps(verified_record) + '\n')

        print(f"✓ Saved to {self.verified_file}")

    def show_statistics(self):
        """Show statistics"""
        print("\n📈 VERIFICATION STATISTICS")
        print("-" * 80)

        if not self.verified_file.exists():
            print("No verified predictions yet")
            return

        verified = []
        with open(self.verified_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        verified.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not verified:
            print("No verified predictions yet")
            return

        # Extract metrics (handle both old and new key names)
        mapes = []
        errors = []
        directions = []

        for p in verified:
            mapes.append(p.get('mape', 0))
            # Handle both 'absolute_error' and 'prediction_error'
            errors.append(p.get('absolute_error', p.get('prediction_error', 0)))
            directions.append(p.get('direction_correct', False))

        print(f"Total verified: {len(verified)}")
        print(f"\nMAPE:")
        print(f"  Average: {np.mean(mapes):.3f}%")
        print(f"  Best:    {np.min(mapes):.3f}%")
        print(f"  Worst:   {np.max(mapes):.3f}%")
        print(f"\nAverage error: ${np.mean(errors):.2f}")
        print(f"\nDirection accuracy: {np.mean(directions) * 100:.1f}%")
        print(f"  Correct: {int(np.sum(directions))}/{len(directions)}")

        # Comparison with training
        avg_mape = np.mean(mapes)
        print(f"\n📊 Comparison with training MAPE (0.028%):")
        if avg_mape <= 0.028:
            print(f"  ✓ Better than or equal to training!")
        elif avg_mape < 0.5:
            print(f"  🌟 EXCEPTIONAL performance (< 0.5%)")
        elif avg_mape < 1.0:
            print(f"  ⭐ EXCELLENT performance (< 1%)")
        elif avg_mape < 2.0:
            print(f"  👍 VERY GOOD performance (< 2%)")
        elif avg_mape < 5.0:
            print(f"  ✓ GOOD performance (< 5%)")
        else:
            print(f"  ⚠️  Needs improvement (> 5%)")

    def show_history(self, n=10):
        """Show last n predictions"""
        print("\n📋 VERIFICATION HISTORY")
        print("-" * 80)

        if not self.verified_file.exists():
            print("No verified predictions yet")
            return

        verified = []
        with open(self.verified_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        verified.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not verified:
            print("No verified predictions yet")
            return

        recent = verified[-n:]

        print(f"\n{'Date':<12} {'Predicted':<11} {'Actual':<11} {'Error':<9} {'MAPE':<8} {'Dir'}")
        print("-" * 62)

        for p in recent:
            date = p['prediction_date']
            predicted = f"${p['predicted_price']:.2f}"
            actual = f"${p['actual_price']:.2f}"
            error = p.get('absolute_error', p.get('prediction_error', 0))
            error_str = f"${error:.2f}"
            mape = f"{p['mape']:.2f}%"
            direction = "✓" if p['direction_correct'] else "✗"

            print(f"{date:<12} {predicted:<11} {actual:<11} {error_str:<9} {mape:<8} {direction}")

    def verify_latest(self, force=False):
        """Verify latest prediction"""
        prediction = self.load_latest_prediction()
        if not prediction:
            return False

        # Check if already verified
        if not force and self.verified_file.exists():
            with open(self.verified_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    try:
                        latest_verified = json.loads(lines[-1])
                        if latest_verified.get('prediction_date') == prediction['prediction_date']:
                            print("\n⚠️  This prediction has already been verified!")
                            print(f"   Prediction date: {prediction['prediction_date']}")
                            print(f"   MAPE: {latest_verified.get('mape', 'N/A'):.3f}%")
                            print(f"\n   Use --force to re-verify")
                            return False
                    except (json.JSONDecodeError, KeyError):
                        pass

        # Get actual price
        actual_price = self.get_actual_price(prediction['prediction_date'])

        if actual_price is None:
            print("\n⚠️  Cannot verify yet - prediction date hasn't occurred or no data available")
            return False

        # Verify
        verification = self.verify_prediction(prediction, actual_price)

        # Save
        self.save_verified_prediction(prediction, verification)

        # Show stats
        self.show_statistics()

        return True


def main():
    """Main execution"""
    import sys

    verifier = PredictionVerifier()

    # Check for command line arguments
    force = '--force' in sys.argv or '-f' in sys.argv
    show_history = '--history' in sys.argv or '-h' in sys.argv

    print(f"\n✓ System ready for verification")

    if show_history:
        verifier.show_history(n=20)
        return

    success = verifier.verify_latest(force=force)

    if success:
        print("\n✅ Verification complete!")
    else:
        print("\n⚠️  Verification skipped or not possible yet")
        print("\nTips:")
        print("  • Wait until after the prediction date (currently: Dec 3, 2025)")
        print("  • Use --force to re-verify an already verified prediction")
        print("  • Use --history to see all verifications")


if __name__ == "__main__":
    main()