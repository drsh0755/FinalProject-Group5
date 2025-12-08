"""
Configuration for Streamlit app.
"""

import os
from pathlib import Path

# Paths
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODELS_DIR = PROJECT_ROOT / 'checkpoints'

# Default settings
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
SEQUENCE_LENGTH = 60
REFRESH_INTERVAL = 300  # 5 minutes in seconds

# Model paths
MODEL_PATHS = {
    "baseline": str(MODELS_DIR / "tft_baseline" / "model_wrapper"),
    "sentiment": str(MODELS_DIR / "tft_sentiment" / "model_wrapper"),
    "full": str(MODELS_DIR / "tft_full" / "model_wrapper")
}

# Chart settings
CHART_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17becf'
}

# Display settings
MAX_TICKERS_DISPLAY = 10
DAYS_OF_HISTORY = 365

# API settings
ALPHAVANTAGE_RATE_LIMIT = 5  # calls per minute
CACHE_TTL = 3600  # 1 hour
