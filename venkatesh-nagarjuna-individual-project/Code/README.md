# Stock Direction Forecasting System

Production-ready deep learning system for 1-day-ahead stock direction prediction using **Temporal Fusion Transformer (TFT)** with **DDG-DA meta-learning** for concept drift adaptation.

## Overview

This system predicts next-day direction (UP/DOWN) for top tech stocks (AAPL, MSFT, GOOGL, NVDA, AMZN) using:

- **Temporal Fusion Transformer (TFT)**: State-of-the-art attention-based forecasting
- **DDG-DA Meta-Learning**: Adapts to predictable market regime shifts
- **Multi-Modal Features**: Price/volume, technical indicators, and news sentiment
- **GPU Optimization**: Optimized for NVIDIA A10/A10G on AWS EC2 G5 instances
- **Real-Time Inference**: Streamlit dashboard with live predictions


### Key Features

✅ **End-to-End Pipeline**: Data ingestion → Feature engineering → Training → Deployment
✅ **Mixed Precision Training**: Leverages A10 Tensor Cores for 2-3x speedup
✅ **Interpretability**: Attention weights and feature importance visualization
✅ **Production-Ready**: Docker support, logging, monitoring, checkpointing
✅ **Adaptive Learning**: DDG-DA handles market regime changes

***

## 📁 Repository Structure

```
stock-direction-forecasting/
├── app/                          # Streamlit frontend
│   ├── main.py                   # Main application
│   ├── utils.py                  # App utilities
│   ├── config.py                 # App configuration
│   ├── pages/                    # Multi-page components
│   ├── Dockerfile                # Container for deployment
│   └── run.sh                    # Launch script
│
├── data_sources/                 # Data acquisition
│   ├── yfinance_loader.py        # Historical data (yfinance)
│   ├── alphavantage_loader.py    # Real-time data (Alpha Vantage)
│   └── market_data_provider.py   # Unified data interface
│
├── features/                     # Feature engineering
│   ├── technical_indicators.py   # Price/volume features
│   └── sentiment_processing.py   # News sentiment (FinBERT)
│
├── data/                         # PyTorch datasets
│   ├── dataset.py                # StockDataset implementation
│   └── dataloaders.py            # GPU-optimized DataLoaders
│
├── ddg_da/                       # Meta-learning
│   ├── distribution_predictor.py # Regime prediction
│   └── sampler.py                # Adaptive sampling
│
├── models/                       # Neural networks
│   ├── tft.py                    # Temporal Fusion Transformer
│   └── model_wrapper.py          # Inference wrapper
│
├── training/                     # Training pipeline
│   ├── train_tft.py              # Training script
│   ├── train_config.yaml         # Training configuration
│   └── __init__.py
│
├── configs/                      # Hardware-specific configs
│   └── aws_g5_a10.yaml           # AWS A10G optimizations
│
├── scripts/                      # Utility scripts
│   ├── download_data.py          # Data acquisition
│   ├── preprocess_data.py        # Feature engineering
│   ├── run_backtest.py           # Historical evaluation
│   ├── train_model.sh            # Training launcher
│   ├── monitor_gpu.py            # GPU monitoring
│   └── optimize_batch_size.py    # Batch size tuning
│
├── tests/                        # Unit tests
│   ├── test_data_pipeline.py
│   ├── test_ddg_da.py
│   ├── test_tft.py
│   └── test_integration.py
│
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_evaluation.ipynb
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation
├── .env.example                  # Environment variables template
├── docker-compose.yml            # Multi-container setup
└── README.md                     # This file
```


***

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10+
- **CUDA**: 11.8+ (for GPU training)
- **Hardware**: NVIDIA GPU with 16GB+ VRAM (recommended: A10/A10G)
- **OS**: Linux (Ubuntu 20.04+) or macOS


### Installation

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/stock-direction-forecasting.git
cd stock-direction-forecasting
```


#### 2. Create Virtual Environment

```bash
# Using conda
conda create -n stock-forecast python=3.10
conda activate stock-forecast

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


#### 4. Set Up API Keys

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Alpha Vantage (for real-time data)
ALPHAVANTAGE_API_KEY=your_key_here

# NewsAPI (optional, for sentiment)
NEWSAPI_KEY=your_key_here
```

> **Get API Keys:**
> - Alpha Vantage: https://www.alphavantage.co/support/\#api-key
> - NewsAPI: https://newsapi.org/register

***

## Usage

### Step 1: Download Data

```bash
python3 scripts/download_data.py \
    --tickers AAPL MSFT GOOGL NVDA AMZN \
    --start-date 2014-11-01 \
    --end-date 2025-11-01 \
    --output data/raw/historical_data.csv
```


### Step 2: Preprocess \& Engineer Features

```bash
python3 scripts/preprocess_data.py \
    --input data/raw/historical_data.csv \
    --output data/processed/features_with_sentiment.csv \
    --add-sentiment
```


### Step 3: Train DDG-DA Predictor (Optional)

```bash
python3 -c "
from ddg_da import DDGDADistributionPredictor
import pandas as pd

data = pd.read_csv('data/processed/features_with_sentiment.csv')
predictor = DDGDADistributionPredictor()
predictor.train(data, epochs=500, batch_size=32)
predictor.save('checkpoints/ddg_da_predictor.pkl')
"
```


### Step 4: Train TFT Model

```bash
python3 training/train_tft.py \
    --config training/train_config.yaml \
    --data data/processed/features_with_sentiment.csv \
    --use-ddg-da \
    --ddg-da-model checkpoints/ddg_da_predictor.pkl
```

Or use the convenience script:

```bash
bash scripts/train_model.sh \
    --config configs/aws_g5_a10.yaml \
    --data data/processed/features_with_sentiment.csv \
    --use-ddg-da
```


### Step 5: Run Streamlit App

```bash
streamlit run app/main.py
```

Or use the launch script:

```bash
bash app/run.sh
```

The app will be available at `http://localhost:8501`

***

## 🔧 Configuration

### Training Configuration (`training/train_config.yaml`)

```yaml
# Model architecture
hidden_dim: 128
lstm_layers: 2
num_heads: 4
dropout: 0.1

# Training hyperparameters
epochs: 100
batch_size: 64
learning_rate: 0.001
optimizer: adamw

# GPU optimization
use_amp: true  # Mixed precision for A10 Tensor Cores

# Early stopping
patience: 15
metric_to_monitor: f1
```


### AWS A10 Optimization (`configs/aws_g5_a10.yaml`)

```yaml
# Optimized for NVIDIA A10G (24GB VRAM)
batch_size: 64
hidden_dim: 160
lstm_layers: 3
num_heads: 8

# Data loading (tuned for vCPU count)
num_workers: 4
pin_memory: true
prefetch_factor: 2

# Performance targets
# Expected: 200-300 samples/sec, >80% GPU utilization
```


***

## AWS EC2 Setup (G5 Instance)

### Launch G5 Instance

```bash
# Recommended: g5.xlarge or g5.2xlarge
# OS: Ubuntu 22.04 LTS with Deep Learning AMI
# Storage: 100GB+ EBS
```


### Install NVIDIA Drivers

```bash
# Check GPU
nvidia-smi

# Should show: NVIDIA A10G, CUDA 11.8+
```


### Clone and Setup

```bash
cd /home/ubuntu
git clone https://github.com/yourusername/stock-direction-forecasting.git
cd stock-direction-forecasting

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```


### Monitor GPU During Training

In a separate terminal:

```bash
python scripts/monitor_gpu.py --interval 1 --output gpu_stats.csv
```

Or use `nvidia-smi`:

```bash
watch -n 1 nvidia-smi
```


### Optimize Batch Size

```bash
python scripts/optimize_batch_size.py \
    --num-features 20 \
    --hidden-dim 128 \
    --sequence-length 60
```


***

## Model Evaluation

### Run Backtest

```bash
python scripts/run_backtest.py \
    --model checkpoints/tft_full/model_wrapper \
    --data data/processed/features_with_sentiment.csv \
    --output results/backtest_results.json
```


### Evaluation Metrics

The system reports:

- **Accuracy**: Overall prediction correctness
- **Precision/Recall/F1**: Class-specific performance
- **AUC**: Area under ROC curve
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst peak-to-trough decline


### Interpret Results

Example output:

```json
{
  "accuracy": 0.623,
  "precision": 0.641,
  "recall": 0.598,
  "f1": 0.619,
  "auc": 0.667,
  "sharpe_ratio": 1.43,
  "max_drawdown": -0.087,
  "total_return": 0.234
}
```


***

## Docker Deployment

### Build Image

```bash
docker build -t stock-forecast-app -f app/Dockerfile .
```


### Run Container

```bash
docker run -p 8501:8501 \
    -v $(pwd)/data:/app/data:ro \
    -v $(pwd)/checkpoints:/app/checkpoints:ro \
    --gpus all \
    stock-forecast-app
```


### Docker Compose

```bash
docker-compose up -d
```

Access app at `http://localhost:8501`

***

## Testing

### Run All Tests

```bash
pytest tests/ -v --cov=. --cov-report=html
```


### Individual Test Suites

```bash
# Data pipeline
pytest tests/test_data_pipeline.py -v

# DDG-DA
pytest tests/test_ddg_da.py -v

# TFT model
pytest tests/test_tft.py -v

# Integration
pytest tests/test_integration.py -v
```


***

## Documentation

### Architecture

The system uses a multi-stage pipeline:

1. **Data Ingestion**: yfinance (historical) + Alpha Vantage (real-time)
2. **Feature Engineering**: Technical indicators + FinBERT sentiment
3. **Regime Detection**: DDG-DA identifies market conditions
4. **Adaptive Sampling**: Reweight training data based on predicted regime
5. **TFT Training**: Temporal Fusion Transformer with attention
6. **Inference**: Real-time predictions via Streamlit

### DDG-DA Meta-Learning

DDG-DA (Data Distribution Generation for Predictable Concept Drift Adaptation) handles non-stationary markets:

1. Extract regime features (volatility, returns, sentiment)
2. Predict future regime using neural network
3. Reweight historical data to match predicted regime
4. Train model on adapted distribution

### TFT Architecture

Components:

- **Variable Selection Networks**: Learn feature importance
- **LSTM Encoder/Decoder**: Temporal sequence modeling
- **Multi-Head Attention**: Capture long-range dependencies
- **Gated Residual Networks**: Feature transformation with skip connections

***

## Performance Optimization

### Mixed Precision Training

Enable AMP for 2-3x speedup on A10:

```python
# Automatically enabled in train_config.yaml
use_amp: true
```


### Batch Size Tuning

Larger batches = better GPU utilization:

```bash
# Find optimal batch size
python scripts/optimize_batch_size.py
```

Expected on A10G (24GB):

- Batch size: 64-128
- Throughput: 200-300 samples/sec
- GPU utilization: >80%


### Data Loading

Optimize `num_workers` based on vCPU count:

```yaml
# For g5.2xlarge (8 vCPUs)
num_workers: 4
pin_memory: true
persistent_workers: true
```


***

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size
batch_size: 32  # Down from 64

# Or reduce model size
hidden_dim: 96  # Down from 128
```


### Poor Predictions

- **Check data quality**: Missing values, outliers
- **Add more features**: Sentiment, fundamentals
- **Tune hyperparameters**: Learning rate, dropout
- **Enable DDG-DA**: Better adaptation to regime shifts


### Slow Training

- **Enable mixed precision**: `use_amp: true`
- **Increase batch size**: Use `optimize_batch_size.py`
- **More workers**: Match vCPU count
- **Use faster storage**: EBS with provisioned IOPS

***

## Example Results

Evaluated on test set (Jan 2024 - Dec 2024):


| Model Variant | Accuracy | F1 Score | Sharpe Ratio |
| :-- | :-- | :-- | :-- |
| TFT (Price only) | 58.3% | 0.561 | 0.87 |
| TFT + Sentiment | 61.7% | 0.602 | 1.21 |
| **TFT + Sentiment + DDG-DA** | **62.3%** | **0.619** | **1.43** |


***

## Contributing

Contributions welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

***

## Citation

If you use this code in your research, please cite:

```bibtex
@software{stock_direction_forecasting_2025,
  title={Stock Direction Forecasting with Temporal Fusion Transformer and DDG-DA},
  author={Venkatesh Nagarjuna},
  year={2025}
}
```


***

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

***

## Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Trading involves risk of loss
- Consult a financial advisor before making investment decisions

***

## Acknowledgments

- **Temporal Fusion Transformer**: [Lim et al. (2021)](https://arxiv.org/abs/1912.09363)
- **DDG-DA**: Concept drift adaptation methodology
- **PyTorch**: Deep learning framework
- **Streamlit**: Interactive web applications
- **yfinance**: Historical market data
- **Alpha Vantage**: Real-time market data

***

## Contact

- **Author**: Venkatesh
- **Email**: venkateshbn98@gmail.com
- **GitHub**: [@asbetos](https://github.com/Asbetos)

***

**Built with ❤️ using PyTorch**
