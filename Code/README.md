# Code Directory Documentation

## Directory Structure
```
Code/
├── data/                   # Data storage (not tracked in git)
│   ├── raw/               # Raw downloaded data
│   ├── processed/         # Preprocessed, model-ready data
│   └── live/              # Live prediction data and logs
├── models/                # Model implementations
│   ├── lstm/              # LSTM architecture and training
│   ├── sentiment/         # FinBERT sentiment analysis
│   ├── fusion/            # Multi-modal fusion network
│   └── checkpoints/       # Saved model weights
├── notebooks/             # Jupyter notebooks
├── scripts/               # Executable Python scripts
├── utils/                 # Helper functions and utilities
└── results/               # Outputs and visualizations
```

## Scripts Overview

### Data Scripts (`scripts/`)
- `download_data.py` - Download historical stock and news data
- `preprocess_data.py` - Clean and prepare data for models
- `create_features.py` - Generate technical indicators

### Model Scripts
- `train_lstm.py` - Train LSTM on technical indicators
- `train_sentiment.py` - Run FinBERT sentiment analysis
- `train_fusion.py` - Train multi-modal fusion model
- `evaluate_models.py` - Generate performance metrics

### Deployment Scripts
- `live_predict.py` - Generate daily predictions
- `fetch_live_data.py` - Collect real-time data

## Usage Examples

### 1. Initial Setup
```bash
# Download historical data
python scripts/download_data.py --start 2025-05-01 --end 2025-11-17

# Preprocess data
python scripts/preprocess_data.py --input data/raw --output data/processed
```

### 2. Training Models
```bash
# Train LSTM
python scripts/train_lstm.py --epochs 50 --batch-size 32

# Run sentiment analysis
python scripts/train_sentiment.py --dataset data/processed/news.csv

# Train fusion model
python scripts/train_fusion.py --lstm-checkpoint models/checkpoints/lstm_best.pth
```

### 3. Live Predictions
```bash
# Run daily prediction (schedule for 4 PM EST)
python scripts/live_predict.py --save-results data/live/predictions.csv
```

## Configuration

Edit hyperparameters and settings in `utils/config.py`:
- Model architectures
- Training parameters
- API endpoints
- File paths

## Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

**CUDA not available:**
- Models will run on CPU (slower but functional)
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

**API rate limits:**
- Alpha Vantage: 25 calls/day (free tier)
- Use cached data in `data/raw/` when possible

## Testing

Run unit tests:
```bash
python -m pytest tests/
```

## Code Style

This project follows PEP 8 style guidelines:
```bash
# Format code
black scripts/ models/ utils/

# Check style
flake8 scripts/ models/ utils/
```
EOF