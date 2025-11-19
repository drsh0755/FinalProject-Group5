import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import RAW_DATA_DIR

print("\nData Quality Check")
print("="*60)

for csv_file in RAW_DATA_DIR.glob("*.csv"):
    print(f"\n{csv_file.name}:")
    df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Missing values: {df.isnull().sum().sum()}")
    
print("\n" + "="*60)
print("✓ Data quality check complete\n")
