
import yfinance as yf
from pathlib import Path
from config import DataConfig, RAW_DATA_DIR

data_config = DataConfig()

def download_price_data():
    symbol = data_config.symbol
    print(f"Downloading {symbol} data from {data_config.start_date} to {data_config.end_date}...")
    
    df = yf.download(
        symbol,
        start=data_config.start_date,
        end=data_config.end_date,
        progress=False
    )
    
    price_dir = RAW_DATA_DIR / "prices"
    price_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = price_dir / f"{symbol}.csv"
    df.to_csv(output_path)
    print(f"✓ Saved {symbol} data to {output_path}")
    return output_path


def download_market_indices():
    """Download market indices for feature engineering."""
    indices = data_config.market_indices
    print(f"Downloading market indices: {indices}...")
    
    indices_dir = RAW_DATA_DIR / "market_indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    
    for idx in indices:
        print(f"  Downloading {idx}...")
        df = yf.download(
            idx,
            start=data_config.start_date,
            end=data_config.end_date,
            progress=False
        )
        
        safe_name = idx.replace("^", "")
        output_path = indices_dir / f"{safe_name}.csv"
        df.to_csv(output_path)
        print(f"    ✓ Saved to {output_path}")


if __name__ == "__main__":
    download_price_data()
    download_market_indices()
    print("\n✓ Data download complete!")
