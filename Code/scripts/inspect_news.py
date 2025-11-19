import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.config import RAW_DATA_DIR

print("\nNews Dataset Inspection")
print("="*60)

# Find the news CSV file
news_dir = RAW_DATA_DIR / 'news'
news_files = list(news_dir.glob("*.csv"))

if not news_files:
    print("⚠ No news CSV files found")
    sys.exit(1)

for news_file in news_files:
    print(f"\nFile: {news_file.name}")
    print("-"*60)
    
    # Load first 1000 rows to inspect
    df = pd.read_csv(news_file, nrows=1000)
    
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nSample rows:")
    print(df.head(3))
    
    # Check date column
    if 'date' in df.columns or 'Date' in df.columns:
        date_col = 'date' if 'date' in df.columns else 'Date'
        print(f"\nDate range:")
        print(f"  From: {df[date_col].min()}")
        print(f"  To: {df[date_col].max()}")

print("\n" + "="*60)
print("✓ News dataset inspection complete\n")
