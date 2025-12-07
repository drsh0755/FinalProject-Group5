import yfinance as yf
from pathlib import Path

TICKERS = ["AAPL", "TSLA", "JPM", "MSFT", "GOOGL"]
START = "2020-01-01"
END = "2025-11-30"

data_dir = Path("data/raw/prices")
data_dir.mkdir(parents=True, exist_ok=True)

for t in TICKERS:
    df = yf.download(t, start=START, end=END)
    df.to_csv(data_dir / f"{t}.csv")
    print(f"Downloaded {t}")

