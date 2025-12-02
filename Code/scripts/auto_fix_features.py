#!/usr/bin/env python3
"""
Automatic Fix Script - Updates live_realtime_prediction.py with correct features
Run AFTER you've identified the 43 features
"""

import pandas as pd
import sys
import shutil
from datetime import datetime

print("=" * 80)
print("AUTOMATIC FEATURE FIX FOR LIVE PREDICTION SCRIPT")
print("=" * 80)

# Step 1: Load training data and get features
try:
    df = pd.read_csv('Code/data/processed/spy_features_with_sentiment.csv')
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close']]

    print(f"\n✓ Identified {len(feature_cols)} features from training data")

    if len(feature_cols) != 43:
        print(f"\n❌ ERROR: Expected 43 features, found {len(feature_cols)}")
        print(f"   Your training data doesn't have 43 features!")
        print(f"   Cannot proceed with automatic fix.")
        sys.exit(1)

except FileNotFoundError:
    print("\n❌ ERROR: Training data file not found!")
    print("   Expected: Code/data/processed/spy_features_with_sentiment.csv")
    print("   Run this script from: ~/DL/Final Project/")
    sys.exit(1)

# Step 2: Backup existing script
script_path = 'Code/scripts/live_realtime_prediction.py'
backup_path = f'{script_path}.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}'

try:
    shutil.copy2(script_path, backup_path)
    print(f"✓ Backed up existing script to: {backup_path}")
except FileNotFoundError:
    print(f"\n⚠️  WARNING: Script not found at {script_path}")
    print(f"   Will create new script instead")
    script_path = 'Code/scripts/live_realtime_prediction_FIXED.py'

# Step 3: Read existing script
try:
    with open('Code/scripts/live_realtime_prediction.py', 'r') as f:
        script_content = f.read()
except FileNotFoundError:
    script_content = None

# Step 4: Generate new FEATURE_COLUMNS section
feature_list_str = "FEATURE_COLUMNS = [\n"
for col in feature_cols:
    feature_list_str += f"    '{col}',\n"
feature_list_str += "]\n\n"
feature_list_str += f"# VERIFY COUNT\n"
feature_list_str += f"assert len(FEATURE_COLUMNS) == 43, f\"ERROR: Expected 43 features, got {{len(FEATURE_COLUMNS)}}\"\n"

# Step 5: Replace FEATURE_COLUMNS in script
if script_content:
    # Find and replace FEATURE_COLUMNS section
    import re

    # Pattern to match FEATURE_COLUMNS = [...] including the assertion
    pattern = r'FEATURE_COLUMNS = \[.*?\]\s*\n\s*#?\s*VERIFY.*?(?=\n\n|\nprint|\nclass)'

    if re.search(pattern, script_content, re.DOTALL):
        updated_content = re.sub(pattern, feature_list_str, script_content, flags=re.DOTALL)

        # Write updated script
        output_path = 'Code/scripts/live_realtime_prediction.py'
        with open(output_path, 'w') as f:
            f.write(updated_content)

        print(f"✓ Updated script: {output_path}")
        print(f"\n{'=' * 80}")
        print("FEATURE_COLUMNS updated with:")
        print(f"{'=' * 80}")
        print(feature_list_str)

    else:
        print("\n⚠️  Could not find FEATURE_COLUMNS section to replace")
        print("   Creating new script with correct features...")
        create_new = True
else:
    create_new = True

if script_content is None or 'create_new' in locals():
    # Create minimal script with correct features
    print("\n📝 Creating new script with correct features...")
    print("   Output: Code/scripts/live_realtime_prediction_FIXED.py")

    with open('Code/scripts/live_realtime_prediction_FIXED.py', 'w') as f:
        f.write(f"""#!/usr/bin/env python3
\"\"\"
Live Real-Time Prediction Script - AUTO-FIXED
Correct feature list extracted from training data
\"\"\"

# ============================================================================
# CORRECT FEATURE COLUMNS (43 features)
# ============================================================================

{feature_list_str}

print(f"✓ Features loaded: {{len(FEATURE_COLUMNS)}}")
print("\\n Features:")
for i, col in enumerate(FEATURE_COLUMNS, 1):
    print(f"  {{i:2d}}. {{col}}")
""")

    print("✓ Created: Code/scripts/live_realtime_prediction_FIXED.py")

# Step 6: Display the features
print(f"\n{'=' * 80}")
print(f"THE 43 FEATURES (in order):")
print(f"{'=' * 80}")
for i, col in enumerate(feature_cols, 1):
    print(f"{i:2d}. {col}")

print(f"\n{'=' * 80}")
print("✅ FIX COMPLETE")
print(f"{'=' * 80}")

print(f"\nNext steps:")
print(f"1. Review the updated features above")
print(f"2. Test the script:")
print(f"   python3 Code/scripts/live_realtime_prediction.py")
print(f"3. If successful, predictions will run without errors")

print(f"\nBackup available at: {backup_path if 'backup_path' in locals() else 'N/A'}")
print()