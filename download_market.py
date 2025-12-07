import yfinance as yf
from pathlib import Path

INDICES = ["^GSPC", "^VIX", "XLK", "XLF", "DX-Y.NYB"]
START = "2020-01-01"
END = "2025-11-30"

data_dir = Path("data/raw/market")
data_dir.mkdir(parents=True, exist_ok=True)

for symbol in INDICES:
    df = yf.download(symbol, start=START, end=END)
    # replace ^ in filenames to keep them filesystem-friendly
    safe_name = symbol.replace("^", "")
    df.to_csv(data_dir / f"{safe_name}.csv")
    print(f"Downloaded {symbol} -> {safe_name}.csv")
