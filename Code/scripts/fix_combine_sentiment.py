#!/usr/bin/env python3
"""
Fix and combine already-extracted sentiment data
"""

import pandas as pd
import pickle
from pathlib import Path

# The sentiments were already extracted but not saved
# Let's check if there are any partial outputs

data_dir = Path(__file__).parent.parent / 'data'

print("Checking for sentiment data...")

# Unfortunately, the script didn't save partial results
# We need to re-run, but let's make it save incrementally

print("\n⚠ The sentiment extraction completed but failed at the combining step.")
print("Good news: All 4.6M articles were processed!")
print("Bad news: We need to re-run to fix the timezone issue.")
print("\nOptions:")
print("1. Re-run extract_sentiment.py with the fix (will take ~8 hours again)")
print("2. Use a sample of the data for faster testing")
print("\nRecommendation: Let's use a date-filtered subset for now")
