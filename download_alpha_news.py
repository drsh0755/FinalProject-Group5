import requests
import json
from pathlib import Path

API_KEY = "YOUR_ALPHA_VANTAGE_KEY_HERE"  # ⚠️ Replace this with your key

params = {
    "function": "NEWS_SENTIMENT",
    "apikey": API_KEY,
    "topics": "financial_markets",
    "sort": "latest"
}

url = "https://www.alphavantage.co/query"

data_dir = Path("data/raw/news")
data_dir.mkdir(parents=True, exist_ok=True)

response = requests.get(url, params=params)
response.raise_for_status()

out_path = data_dir / "alpha_news.json"
with open(out_path, "w") as f:
    json.dump(response.json(), f, indent=4)

print(f"Saved:", out_path)
