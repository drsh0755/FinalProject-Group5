#!/usr/bin/env python3
"""
Identify the EXACT 43 features used during training
Run this on your EC2 instance in the project directory
"""

import pandas as pd
import sys

print("=" * 80)
print("IDENTIFYING THE 43 FEATURES FROM TRAINING DATA")
print("=" * 80)

try:
    # Load the exact same file used in training
    df = pd.read_csv('Code/data/processed/spy_features_with_sentiment.csv')
    print(f"\n✓ Loaded: Code/data/processed/spy_features_with_sentiment.csv")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Total rows: {len(df)}")

    # Show all columns
    print(f"\n{'=' * 80}")
    print("ALL COLUMNS IN THE CSV:")
    print(f"{'=' * 80}")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

    # Get feature columns (excluding Date and Close - as done in training)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]

    print(f"\n{'=' * 80}")
    print(f"FEATURE COLUMNS (Total: {len(feature_cols)}):")
    print(f"{'=' * 80}")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")

    # Generate Python list for easy copy-paste
    print(f"\n{'=' * 80}")
    print("PYTHON LIST FORMAT - COPY THIS TO YOUR SCRIPT:")
    print(f"{'=' * 80}\n")

    print("FEATURE_COLUMNS = [")
    for col in feature_cols:
        print(f"    '{col}',")
    print("]")

    print(f"\n✓ Total features: {len(feature_cols)}")

    if len(feature_cols) == 43:
        print("✅ PERFECT! This matches the expected 43 features.")
    else:
        print(f"⚠️  WARNING: Expected 43 features, got {len(feature_cols)}")
        print(f"   Difference: {len(feature_cols) - 43}")

        if len(feature_cols) > 43:
            print(f"   You have {len(feature_cols) - 43} EXTRA features")
            print(f"   These need to be removed or the model needs retraining")
        else:
            print(f"   You are MISSING {43 - len(feature_cols)} features")
            print(f"   Check your feature engineering")

    # Show sample data
    print(f"\n{'=' * 80}")
    print("SAMPLE DATA (first 3 rows, first 10 features):")
    print(f"{'=' * 80}")
    print(df[feature_cols[:10]].head(3))

    # Check for NaN
    nan_counts = df[feature_cols].isnull().sum()
    if nan_counts.sum() > 0:
        print(f"\n⚠️  NaN VALUES DETECTED:")
        print(nan_counts[nan_counts > 0])

except FileNotFoundError:
    print(f"\n❌ ERROR: File not found!")
    print(f"   Expected: Code/data/processed/spy_features_with_sentiment.csv")
    print(f"\n   Make sure you run this script from your project root directory:")
    print(f"   cd ~/DL/Final\\ Project")
    print(f"   python3 ~/identify_43_features.py")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print(f"\n{'=' * 80}")
print("✅ ANALYSIS COMPLETE")
print(f"{'=' * 80}\n")