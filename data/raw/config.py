# config.py

from dataclasses import dataclass
from pathlib import Path

# Base paths (relative to this file)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LIVE_DATA_DIR = DATA_DIR / "live"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
MODELS_DIR = BASE_DIR / "models" / "checkpoints"

for p in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, LIVE_DATA_DIR,
    RESULTS_DIR, FIGURES_DIR, LOGS_DIR, PREDICTIONS_DIR, MODELS_DIR
]:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    symbol: str = "AAPL"
    market_indices: tuple = ("SPY", "QQQ", "DIA", "^VIX")
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-01"
    price_interval: str = "daily"
    news_csv_path: Path = RAW_DATA_DIR / "news.csv"  # Kaggle news dataset
    alpha_vantage_api_key_env: str = "ALPHA_VANTAGE_KEY"


@dataclass
class SequenceConfig:
    input_window: int = 30
    forecast_horizon: int = 1
    train_val_test_split: tuple = (0.7, 0.15, 0.15)
    target_column: str = "close_return_1d"  # next-day return


@dataclass
class ModelConfig:
    price_input_dim: int = 16  # will be set after feature engineering
    price_hidden_dim: int = 64
    price_num_layers: int = 2
    sentiment_dim: int = 3   # mean, std, count
    fusion_hidden_dim: int = 64
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda"  # or "cpu"


data_config = DataConfig()
seq_config = SequenceConfig()
model_config = ModelConfig()
training_config = TrainingConfig()
