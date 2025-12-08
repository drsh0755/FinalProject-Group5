from datetime import datetime

import torch
import pandas as pd

from config import (
    PROCESSED_DATA_DIR,
    LIVE_DATA_DIR,
    MODELS_DIR,
    data_config,
    seq_config,
    model_config,
)
from models.price_lstm import PriceLSTMModel
from models.fusion_mlp import FusionMLP
from scripts._utils_live_sentiment import get_today_sentiment_vector


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load last N days of features
    symbol = data_config.symbol
    merged_path = PROCESSED_DATA_DIR / f"{symbol}_features_merged.csv"
    df = pd.read_csv(merged_path, parse_dates=["date"]).sort_values("date")

    window = seq_config.input_window
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "return", "log_return",
        "ma_5", "ma_10", "ma_20",
        "std_10", "rsi_14",
    ]
    for idx_symbol in data_config.market_indices:
        feature_cols.append(f"{idx_symbol}_return")

    last_window = df.tail(window)[feature_cols].values.astype("float32")
    last_window = torch.tensor(last_window).unsqueeze(0).to(device)  # (1, seq_len, feat_dim)

    # Load price LSTM
    price_ckpt = torch.load(MODELS_DIR / "price_lstm_best.pt", map_location=device)
    price_model = PriceLSTMModel(
        input_dim=price_ckpt["feature_dim"],
        hidden_dim=model_config.price_hidden_dim,
        num_layers=model_config.price_num_layers,
        dropout=model_config.dropout,
    ).to(device)
    price_model.load_state_dict(price_ckpt["model_state_dict"])
    price_model.eval()

    # Extract representation from LSTM
    with torch.no_grad():
        _, (h_n, c_n) = price_model.lstm(last_window)
        price_repr = h_n[-1]  # (1, hidden_dim)

    # Load sentiment for "today"
    sentiment_vec = get_today_sentiment_vector()  # shape (3,)
    sentiment_vec = torch.tensor(sentiment_vec, dtype=torch.float32).unsqueeze(0).to(device)

    # Load fusion model
    fusion_ckpt = torch.load(MODELS_DIR / "fusion_mlp_best.pt", map_location=device)
    fusion_model = FusionMLP(
        price_repr_dim=fusion_ckpt["price_repr_dim"],
        sentiment_dim=fusion_ckpt["sentiment_dim"],
        hidden_dim=model_config.fusion_hidden_dim,
        dropout=model_config.dropout,
    ).to(device)
    fusion_model.load_state_dict(fusion_ckpt["model_state_dict"])
    fusion_model.eval()

    with torch.no_grad():
        pred_return = fusion_model(price_repr, sentiment_vec).item()

    print(f"Predicted next-day return for {symbol}: {pred_return:.4%}")

    LIVE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LIVE_DATA_DIR / f"live_prediction_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(
        {
            "symbol": [symbol],
            "prediction_time": [datetime.now()],
            "predicted_return_1d": [pred_return],
        }
    ).to_csv(out_path, index=False)
    print("Saved live prediction to:", out_path)


if __name__ == "__main__":
    main()
