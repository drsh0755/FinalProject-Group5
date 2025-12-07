# utils/data_utils.py

import pandas as pd
import numpy as np
from pathlib import Path


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: DataFrame with columns: ['open','high','low','close','volume']
    Adds simple technical indicators.
    """
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"]).diff()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["std_10"] = df["close"].rolling(window=10).std()
    df["rsi_14"] = _rsi(df["close"], window=14)
    df["target_close_shift1"] = df["close"].shift(-1)
    df["close_return_1d"] = df["target_close_shift1"] / df["close"] - 1.0
    df = df.dropna().reset_index(drop=True)
    return df


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(window=window).mean()
    roll_down = pd.Series(down).rolling(window=window).mean()
    rs = roll_up / (roll_down + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    window: int,
):
    """
    Convert a feature dataframe into overlapping sequences for supervised learning.
    Returns X (N, window, F), y (N, 1)
    """
    features = df[feature_cols].values
    target = df[target_col].values

    X, y = [], []
    for i in range(len(df) - window):
        X.append(features[i : i + window])
        y.append(target[i + window])
    
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """
    Split data into train, val, test sets.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    n = len(X)
    train_idx = int(n * train_ratio)
    val_idx = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    return X_train, X_val, X_test, y_train, y_val, y_test
