#!/usr/bin/env python3
"""
Unified Verification Script - Single OR batch mode
Usage:
  python3 07_verify_predictions.py                    # Verify latest
  python3 07_verify_predictions.py --all              # Verify all unverified
  python3 07_verify_predictions.py --start 2025-11-01 --end 2025-11-30  # Range
  python3 07_verify_predictions.py --history          # Show history
  python3 07_verify_predictions.py --stats            # Show stats only
"""

import json
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import argparse
import warnings

warnings.filterwarnings('ignore')


class UnifiedVerifier:
    def __init__(self):
        self.predictions_file = Path('exhibition/realtime_predictions.jsonl')
        self.verified_file = Path('exhibition/verified_predictions.jsonl')
        self.predictions_file.parent.mkdir(exist_ok=True)
        
        print(f"✓ Predictions file: {self.predictions_file}")
        print(f"✓ Verified file: {self.verified_file}")
    
    def load_predictions(self):
        """Load all predictions"""
        if not self.predictions_file.exists():
            return []
        
        predictions = []
        with open(self.predictions_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        predictions.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return predictions
    
    def load_verified(self):
        """Load all verified predictions"""
        if not self.verified_file.exists():
            return []
        
        verified = []
        with open(self.verified_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        verified.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        return verified
    
    def get_actual_price(self, date_str):
        """Fetch actual closing price for a date"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Check if date is in the future
            if date_obj.date() > datetime.now().date():
                return None
            
            # Get data with buffer
            start = date_obj - timedelta(days=7)
            end = min(date_obj + timedelta(days=7), datetime.now())
            
            ticker = yf.Ticker('SPY')
            df = ticker.history(start=start, end=end)
            
            if df.empty:
                return None
            
            # Convert index to dates
            df.index = pd.to_datetime(df.index).date
            pred_date = date_obj.date()
            
            # Try exact date
            if pred_date in df.index:
                return float(df.loc[pred_date, 'Close'])
            
            # Find next trading day
            future_dates = [d for d in df.index if d >= pred_date]
            if future_dates:
                actual_date = min(future_dates)
                return float(df.loc[actual_date, 'Close'])
            
            return None
            
        except Exception as e:
            return None
    
    def verify_prediction(self, prediction):
        """Verify a single prediction"""
        pred_date = prediction['prediction_date']
        
        # Get actual price
        actual_price = self.get_actual_price(pred_date)
        
        if actual_price is None:
            return None
        
        # Calculate metrics
        predicted = prediction['predicted_price']
        current = prediction['current_price']
        
        absolute_error = abs(predicted - actual_price)
        mape = (absolute_error / actual_price) * 100
        
        actual_change = ((actual_price - current) / current) * 100
        pred_change = prediction['predicted_change_pct']
        
        direction_correct = (pred_change > 0 and actual_change > 0) or \
                           (pred_change <= 0 and actual_change <= 0)
        
        return {
            **prediction,
            'actual_price': float(actual_price),
            'actual_change_pct': float(actual_change),
            'absolute_error': float(absolute_error),
            'mape': float(mape),
            'direction_correct': bool(direction_correct),
            'verified_at': datetime.now().isoformat()
        }
    
    def save_verified(self, verified_predictions):
        """Save verified predictions"""
        if not verified_predictions:
            return
        
        if isinstance(verified_predictions, dict):
            verified_predictions = [verified_predictions]
        
        with open(self.verified_file, 'a') as f:
            for v in verified_predictions:
                f.write(json.dumps(v) + '\n')
        
        # Update CSV
        all_verified = self.load_verified()
        if all_verified:
            df = pd.DataFrame(all_verified)
            df = df.drop_duplicates(subset=['prediction_date'], keep='last')
            csv_file = self.verified_file.with_suffix('.csv')
            df.to_csv(csv_file, index=False)
    
    def verify_latest(self, force=False):
        """Verify the latest prediction"""
        print(f"\n{'='*80}")
        print(f"VERIFYING LATEST PREDICTION")
        print(f"{'='*80}\n")
        
        predictions = self.load_predictions()
        if not predictions:
            print("❌ No predictions found")
            return False
        
        latest = predictions[-1]
        pred_date = latest['prediction_date']
        
        # Check if already verified
        if not force:
            verified = self.load_verified()
            if any(v.get('prediction_date') == pred_date for v in verified):
                print(f"⚠️  Prediction for {pred_date} already verified")
                print(f"   Use --force to re-verify\n")
                return False
        
        print(f"Prediction date: {pred_date}")
        print(f"Predicted price: ${latest['predicted_price']:.2f}")
        print(f"Predicted change: {latest['predicted_change_pct']:+.2f}%\n")
        
        # Verify
        verified = self.verify_prediction(latest)
        
        if verified is None:
            pred_dt = datetime.strptime(pred_date, '%Y-%m-%d')
            if pred_dt.date() > datetime.now().date():
                print(f"⏳ Cannot verify - prediction date hasn't occurred yet")
                print(f"   Today: {datetime.now().strftime('%Y-%m-%d')}")
                print(f"   Prediction: {pred_date}\n")
            else:
                print(f"⚠️  No data available for {pred_date}\n")
            return False
        
        # Save
        self.save_verified(verified)
        
        # Show results
        print(f"{'='*80}")
        print(f"VERIFICATION RESULTS")
        print(f"{'='*80}")
        print(f"Actual price:     ${verified['actual_price']:.2f}")
        print(f"Actual change:    {verified['actual_change_pct']:+.2f}%")
        print(f"Absolute error:   ${verified['absolute_error']:.2f}")
        print(f"MAPE:             {verified['mape']:.3f}%")
        print(f"Direction:        {'✓ CORRECT' if verified['direction_correct'] else '✗ WRONG'}")
        print(f"{'='*80}\n")
        
        return True
    
    def verify_all_unverified(self):
        """Verify all predictions that haven't been verified"""
        print(f"\n{'='*80}")
        print(f"VERIFYING ALL UNVERIFIED PREDICTIONS")
        print(f"{'='*80}\n")
        
        predictions = self.load_predictions()
        verified = self.load_verified()
        
        verified_dates = {v.get('prediction_date') for v in verified}
        unverified = [p for p in predictions if p['prediction_date'] not in verified_dates]
        
        print(f"Total predictions: {len(predictions)}")
        print(f"Already verified: {len(verified)}")
        print(f"To verify: {len(unverified)}\n")
        
        if not unverified:
            print("✓ All predictions already verified!\n")
            return
        
        newly_verified = []
        skipped = 0
        
        for i, pred in enumerate(unverified, 1):
            pred_date = pred['prediction_date']
            print(f"[{i}/{len(unverified)}] {pred_date}...", end=' ')
            
            verified_pred = self.verify_prediction(pred)
            
            if verified_pred:
                newly_verified.append(verified_pred)
                print(f"✓ MAPE: {verified_pred['mape']:.3f}% | Dir: {'✓' if verified_pred['direction_correct'] else '✗'}")
            else:
                skipped += 1
                print(f"⏳ Not yet available")
        
        if newly_verified:
            self.save_verified(newly_verified)
            print(f"\n✓ Verified {len(newly_verified)} new prediction(s)")
            print(f"⏳ Skipped {skipped} (not yet available)\n")
        else:
            print(f"\n⏳ No predictions ready for verification yet\n")
    
    def verify_date_range(self, start_date, end_date):
        """Verify predictions in a date range"""
        print(f"\n{'='*80}")
        print(f"VERIFYING DATE RANGE: {start_date} to {end_date}")
        print(f"{'='*80}\n")
        
        predictions = self.load_predictions()
        verified = self.load_verified()
        
        verified_dates = {v.get('prediction_date') for v in verified}
        
        # Filter predictions in range
        in_range = [
            p for p in predictions 
            if start_date <= p['prediction_date'] <= end_date 
            and p['prediction_date'] not in verified_dates
        ]
        
        print(f"Unverified predictions in range: {len(in_range)}\n")
        
        if not in_range:
            print("No unverified predictions in this range\n")
            return
        
        newly_verified = []
        skipped = 0
        
        for i, pred in enumerate(in_range, 1):
            pred_date = pred['prediction_date']
            print(f"[{i}/{len(in_range)}] {pred_date}...", end=' ')
            
            verified_pred = self.verify_prediction(pred)
            
            if verified_pred:
                newly_verified.append(verified_pred)
                print(f"✓ MAPE: {verified_pred['mape']:.3f}%")
            else:
                skipped += 1
                print(f"⏳ Not available")
        
        if newly_verified:
            self.save_verified(newly_verified)
            print(f"\n✓ Verified {len(newly_verified)} prediction(s)")
            print(f"⏳ Skipped {skipped}\n")
    
    def show_statistics(self):
        """Show verification statistics"""
        verified = self.load_verified()
        
        if not verified:
            print("\n⚠️  No verified predictions yet\n")
            return
        
        mapes = [v['mape'] for v in verified]
        errors = [v['absolute_error'] for v in verified]
        directions = [v['direction_correct'] for v in verified]
        
        print(f"\n{'='*80}")
        print(f"STATISTICS")
        print(f"{'='*80}")
        print(f"Total verified: {len(verified)}")
        print(f"\nMAPE:")
        print(f"  Average: {np.mean(mapes):.3f}%")
        print(f"  Median:  {np.median(mapes):.3f}%")
        print(f"  Best:    {np.min(mapes):.3f}%")
        print(f"  Worst:   {np.max(mapes):.3f}%")
        print(f"\nAbsolute Error:")
        print(f"  Average: ${np.mean(errors):.2f}")
        print(f"  Median:  ${np.median(errors):.2f}")
        print(f"\nDirection Accuracy: {np.mean(directions)*100:.1f}%")
        print(f"  Correct: {sum(directions)}/{len(directions)}")
        
        # Performance rating
        avg_mape = np.mean(mapes)
        print(f"\nPerformance Rating:")
        if avg_mape < 0.5:
            print(f"  🌟 EXCEPTIONAL (MAPE < 0.5%)")
        elif avg_mape < 1.0:
            print(f"  ⭐ EXCELLENT (MAPE < 1%)")
        elif avg_mape < 2.0:
            print(f"  👍 VERY GOOD (MAPE < 2%)")
        elif avg_mape < 5.0:
            print(f"  ✓ GOOD (MAPE < 5%)")
        else:
            print(f"  ⚠️  NEEDS IMPROVEMENT (MAPE > 5%)")
        
        print(f"{'='*80}\n")
    
    def show_history(self, n=20):
        """Show verification history"""
        verified = self.load_verified()
        
        if not verified:
            print("\n⚠️  No verified predictions yet\n")
            return
        
        print(f"\n{'='*80}")
        print(f"VERIFICATION HISTORY (Last {min(n, len(verified))})")
        print(f"{'='*80}\n")
        
        recent = verified[-n:]
        
        print(f"{'Date':<12} {'Predicted':<11} {'Actual':<11} {'Error':<9} {'MAPE':<8} {'Dir'}")
        print("-" * 62)
        
        for v in recent:
            date = v['prediction_date']
            predicted = f"${v['predicted_price']:.2f}"
            actual = f"${v['actual_price']:.2f}"
            error = f"${v['absolute_error']:.2f}"
            mape = f"{v['mape']:.2f}%"
            direction = "✓" if v['direction_correct'] else "✗"
            
            print(f"{date:<12} {predicted:<11} {actual:<11} {error:<9} {mape:<8} {direction}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='Verify stock predictions')
    parser.add_argument('--all', action='store_true', help='Verify all unverified')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--force', '-f', action='store_true', help='Force re-verification')
    parser.add_argument('--history', action='store_true', help='Show history')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    args = parser.parse_args()
    
    verifier = UnifiedVerifier()
    
    # Show history
    if args.history:
        verifier.show_history(n=30)
        verifier.show_statistics()
        return
    
    # Show stats only
    if args.stats:
        verifier.show_statistics()
        return
    
    # Verify all unverified
    if args.all:
        verifier.verify_all_unverified()
        verifier.show_statistics()
        return
    
    # Verify date range
    if args.start and args.end:
        verifier.verify_date_range(args.start, args.end)
        verifier.show_statistics()
        return
    
    # Verify latest (default)
    success = verifier.verify_latest(force=args.force)
    if success:
        verifier.show_statistics()


if __name__ == '__main__':
    main()
