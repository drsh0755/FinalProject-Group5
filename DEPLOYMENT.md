# Deployment Guide - FinalProject-Group5

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/mayur212626/FinalProject-Group5.git
cd FinalProject-Group5
git checkout mayur
```

### 2. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Dependencies
```bash
# Install from requirements.txt
pip install -r requirements.txt

# OR install with development tools
pip install -e ".[dev,jupyter]"
```

### 4. Verify Installation
```bash
# Test imports
python3 << 'EOF'
import torch
import pandas as pd
import numpy as np
from models import PriceLSTMModel, FusionMLP
from utils import compute_technical_indicators
print("✓ All imports successful!")
EOF
```

---

## Project Structure

```
FinalProject-Group5/
├── models/                          # PyTorch model implementations
│   ├── __init__.py
│   ├── price_lstm.py               # LSTM model for prices
│   └── fusion_mlp.py               # Fusion model (price + sentiment)
│
├── utils/                           # Utility functions
│   ├── __init__.py
│   └── data_utils.py               # Feature engineering & data prep
│
├── scripts/                         # Pipeline scripts (01-07)
│   ├── 01_download_data.py         # Download prices & indices
│   ├── 02_feature_engineering.py   # Technical indicators
│   ├── 03_prepare_sequences.py     # Build sequences
│   ├── 04_train_price_model.py     # Train LSTM
│   ├── 05_build_sentiment_features.py # NLP sentiment
│   ├── 06_train_fusion_model.py    # Train fusion model
│   ├── 07_live_predict.py          # Live predictions
│   └── _utils_live_sentiment.py    # Sentiment utilities
│
├── data/                            # Data directory
│   ├── raw/                        # Raw data (config, prices, news, etc)
│   ├── processed/                  # Generated features & sequences
│   └── live/                       # Live predictions output
│
├── requirements.txt                 # Python dependencies
├── setup.py                        # Package setup
└── .gitignore                      # Git ignore rules
```

---

## Running the Pipeline

### Sequential Execution (Recommended)

```bash
# Step 1: Download data
python scripts/01_download_data.py
# Output: data/raw/prices/*.csv, data/raw/market_indices/*.csv

# Step 2: Feature engineering
python scripts/02_feature_engineering.py
# Output: data/processed/{SYMBOL}_features_merged.csv

# Step 3: Prepare sequences
python scripts/03_prepare_sequences.py
# Output: data/processed/{SYMBOL}_sequences.pkl

# Step 4: Train price model
python scripts/04_train_price_model.py
# Output: models/checkpoints/price_lstm_best.pt

# Step 5: Build sentiment features
python scripts/05_build_sentiment_features.py
# Output: data/processed/{SYMBOL}_features_with_sentiment.csv

# Step 6: Train fusion model
python scripts/06_train_fusion_model.py
# Output: models/checkpoints/fusion_mlp_best.pt

# Step 7: Make live predictions
python scripts/07_live_predict.py
# Output: data/live/live_prediction_{SYMBOL}_{timestamp}.csv
```

### Configuration
Edit `data/raw/config.py` to customize:
- Target symbol (default: "AAPL")
- Market indices (default: "SPY", "QQQ", "DIA", "^VIX")
- Date ranges
- Model hyperparameters

---

## Docker Deployment (Optional)

### Build Docker Image
```bash
docker build -t finalproject-group5 .
```

### Run Container
```bash
docker run -it --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  finalproject-group5 \
  bash
```

---

## Cloud Deployment Options

### AWS EC2
```bash
# Launch GPU instance
aws ec2 run-instances \
  --image-id ami-XXXXXXX \
  --instance-type g4dn.xlarge \
  --key-name your-key \
  --security-group-ids sg-XXXXXXX

# SSH into instance
ssh -i your-key.pem ec2-user@instance-ip

# Follow quick start steps above
```

### Google Colab (Free GPU)
```python
# In Colab notebook
!git clone https://github.com/mayur212626/FinalProject-Group5.git
%cd FinalProject-Group5
!pip install -r requirements.txt
!python scripts/01_download_data.py
# ... continue with other steps
```

### Azure ML
```bash
az ml environment create \
  --name finalproject \
  --conda-file environment.yml \
  --image mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-runtime:latest
```

---

## Troubleshooting

### Issue: PyTorch CUDA not found
```bash
# Install CPU-only version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or install with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: yfinance data download fails
```bash
# Try updating yfinance
pip install --upgrade yfinance

# Check internet connection and API rate limits
```

### Issue: ImportError in scripts
```bash
# Ensure you're in the project root directory
cd /home/ubuntu/FinalProject-Group5

# Verify PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: CUDA out of memory
```bash
# Reduce batch size in config.py
# Or use CPU for testing
export CUDA_VISIBLE_DEVICES=''
python scripts/04_train_price_model.py
```

---

## Performance Notes

### GPU Acceleration
- CUDA-compatible GPU strongly recommended
- Models use PyTorch (supports CUDA/CPU fallback)
- Training time: ~15-20 minutes on GPU, ~2-3 hours on CPU

### Hardware Requirements
- **Minimum:** 4GB RAM, 2-core CPU
- **Recommended:** 8GB RAM, GPU with 4GB VRAM
- **Optimal:** 16GB+ RAM, GPU with 8GB+ VRAM

### Data Size
- Raw data: ~500MB (includes historical price data)
- Processed data: ~100MB (features + sequences)
- Model checkpoints: ~50MB

---

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Test
```bash
# Run all pipeline steps on small sample
python scripts/01_download_data.py --sample
python scripts/02_feature_engineering.py --sample
```

---

## Monitoring & Logging

### Enable Logging
```bash
export LOG_LEVEL=DEBUG
python scripts/04_train_price_model.py --verbose
```

### Model Evaluation
```bash
# After training, check validation metrics
cat models/checkpoints/training_log.txt
```

---

## Production Deployment Checklist

- [ ] All tests pass
- [ ] Requirements.txt updated and tested
- [ ] Environment variables configured (.env file)
- [ ] Data paths verified
- [ ] Model checkpoints saved
- [ ] Logging configured
- [ ] Error handling in place
- [ ] Documentation complete
- [ ] Git commits pushed to main branch
- [ ] CI/CD pipeline configured (GitHub Actions)

---

## Support & Contributions

- **Issues:** Report on GitHub Issues
- **Pull Requests:** Create PR to develop branch
- **Documentation:** See README.md for more info

---

## Version History

- **v0.1.0** (2025-12-07) - Initial release with complete restructuring
  - Fixed directory structure
  - Restored all Python files
  - Created deployment guide

---

**Last Updated:** December 7, 2025
