import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from models.model_wrapper import TFTModelWrapper
from data_sources.market_data_provider import MarketDataProvider
from features.technical_indicators import TechnicalIndicators

def load_latest_wrapper() -> TFTModelWrapper:
    import glob
    matches = glob.glob("checkpoints/*/model_wrapper")
    if not matches:
        raise FileNotFoundError("No model_wrapper directory found under checkpoints/")
    model_dir = sorted(matches)[-1]
    print(f"Using model: {model_dir}")
    return TFTModelWrapper.load(model_dir)

def build_features_for_ticker(ticker: str, days: int = 365, seq_len: int = 60):
    provider = MarketDataProvider(tickers=[ticker])
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = provider.get_historical_data(start_date=start, end_date=end)
    if df.empty:
        raise ValueError(f"No data for {ticker}")

    df = TechnicalIndicators.add_all_indicators(df)

    return df[df["Ticker"] == ticker].sort_values("Date").tail(seq_len)

def run_single_ticker_inference(ticker: str):
    wrapper = load_latest_wrapper()
    feature_cols = wrapper.feature_cols
    ticker_to_id = wrapper.ticker_to_id

    df_ticker = build_features_for_ticker(ticker)
    if len(df_ticker) < 60:
        raise ValueError(f"Need at least 60 rows, got {len(df_ticker)}")

    # Ensure every feature expected by the model exists
    missing = [c for c in feature_cols if c not in df_ticker.columns]
    for c in missing:
        df_ticker[c] = 0.0

    x = df_ticker[feature_cols].values
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    encoder_cont = torch.from_numpy(x).unsqueeze(0)          # [1, seq_len, F]
    ticker_id = torch.tensor([[ticker_to_id.get(ticker, 0)]])  # [1, 1]

    features = {
        "encoder_cont": encoder_cont,
        "ticker_id": ticker_id,
    }

    out = wrapper.predict(features)
    pred_class = int(out["predictions"][0])
    probs = out["probabilities"][0]

    direction = "UP" if pred_class == 1 else "DOWN"
    confidence = float(probs[pred_class])

    print(f"{ticker} prediction: {direction}, confidence={confidence:.3f}")
    print(f"Prob UP={float(probs[1]):.3f}, Prob DOWN={float(probs[0]):.3f}")

if __name__ == "__main__":
    run_single_ticker_inference("AAPL")
