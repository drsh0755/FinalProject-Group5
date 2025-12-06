# Code Directory
**Multi-Modal LSTM Stock Market Prediction - Production Code**

## 🚀 Quick Start

### Setup Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure API Keys
```bash
# Set Alpha Vantage API key
export ALPHA_VANTAGE_API_KEY="your_key_here"

# Or create .env file
echo "ALPHA_VANTAGE_API_KEY=your_key_here" > .env
```

## 📊 Production Pipeline

Run these scripts in order:

### 1️⃣ Download Data
```bash
cd scripts
python 01_download_data.py
```
**Purpose:** Fetches latest SPY stock data and financial news from Alpha Vantage  
**Output:** `data/raw/spy_data.csv`, `data/raw/news_data.json`  
**Time:** ~2-3 minutes

### 2️⃣ Create Features
```bash
python 02_create_features.py
```
**Purpose:** Generates 43 features (36 technical + 7 sentiment)  
**Input:** Raw stock and news data  
**Output:** `data/processed/features.csv`  
**Time:** ~5 minutes

### 3️⃣ Train Model
```bash
python 03_train_model.py
```
**Purpose:** Trains multi-modal LSTM model  
**Input:** Processed features  
**Output:** `models/lstm_model_sentiment.pt`, training logs  
**Time:** ~15-30 minutes (with GPU)

### 4️⃣ Evaluate Model
```bash
python 04_evaluate.py
```
**Purpose:** Evaluates model performance  
**Input:** Trained model, test data  
**Output:** `results/lstm_comprehensive_sentiment_results.json`  
**Metrics:** MAPE, MAE, direction accuracy

### 4️⃣b Create Plots
```bash
python 04b_create_plots.py
```
**Purpose:** Creates visualization plots for presentation  
**Input:** Results and predictions  
**Output:** `results/figures/*.png`

### 5️⃣ Live Prediction
```bash
python 05_live_prediction.py
```
**Purpose:** Generates real-time SPY price predictions  
**Input:** Latest data, trained model  
**Output:** Live prediction with confidence intervals

## 📁 Directory Structure
```
Code/
├── scripts/                    # Production pipeline
│   ├── 01_download_data.py    # Data fetching
│   ├── 02_create_features.py  # Feature engineering
│   ├── 03_train_model.py      # Model training
│   ├── 04_evaluate.py         # Performance evaluation
│   ├── 04b_create_plots.py    # Visualization
│   ├── 05_live_prediction.py  # Live predictions
│   └── [supporting scripts]   # Utilities
│
├── models/                     # Model architecture
│   └── lstm/
│       ├── model.py           # LSTM model definition
│       └── dataset.py         # Dataset loader
│
├── data/                       # Data storage
│   ├── raw/                   # Raw downloaded data
│   ├── processed/             # Processed features
│   └── live/                  # Live prediction data
│
├── results/                    # Outputs
│   ├── figures/               # Visualization plots
│   ├── logs/                  # Training logs
│   └── *.json                 # Results files
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_data_exploration.ipynb
│
├── utils/                      # Utility functions
│   ├── config.py              # Configuration
│   └── logger.py              # Logging utilities
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 Supporting Scripts

### Alternative Data Sources
- `download_data.py` - Alternative data fetcher
- `create_features.py` - Alternative feature generator

### Utilities
- `merge_sentiment_with_features.py` - Merge sentiment with technical features
- `retrain_lstm_model.py` - Retrain existing model
- `live_prediction_demo.py` - Demo version of live predictions

## 📦 Dependencies

Main packages (see `requirements.txt` for complete list):
- `torch` - PyTorch for deep learning
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `transformers` - FinBERT sentiment analysis
- `yfinance` - Stock data (backup source)
- `ta` - Technical analysis indicators
- `matplotlib` - Plotting
- `scikit-learn` - Preprocessing and metrics

## 🧪 Testing

### Test Imports
```bash
python -c "from models.lstm.model import *; print('✓ Model OK')"
python -c "from models.lstm.dataset import *; print('✓ Dataset OK')"
python -c "from utils.config import *; print('✓ Config OK')"
```

### Test Pipeline Components
```bash
cd scripts
python 01_download_data.py --test  # Test data fetching
python 02_create_features.py --test  # Test feature creation
```

## ⚙️ Configuration

### Environment Variables
- `ALPHA_VANTAGE_API_KEY` - API key for Alpha Vantage (required)
- `MODEL_PATH` - Custom model path (optional)
- `DATA_PATH` - Custom data path (optional)

### Config File
Edit `utils/config.py` to customize:
- Data sources
- Model hyperparameters
- Feature selection
- Training parameters

## 📊 Model Details

### Architecture
- **Input:** 60-day sequence of 43 features
- **LSTM Layers:** 2 layers × 128 units
- **Dropout:** 0.2
- **Output:** Next day SPY price prediction

### Training
- **Optimizer:** Adam (lr=0.001)
- **Loss:** MSE
- **Batch Size:** 32
- **Epochs:** 100 (with early stopping)
- **Device:** CUDA if available, else CPU

### Performance
- **MAPE:** 7.87%
- **MAE:** $32.54
- **Direction Accuracy:** 68.2%

## 🚨 Troubleshooting

### Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt
```

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, model will run on CPU (slower but functional)
```

### API Rate Limits
- Alpha Vantage: 5 requests/minute, 500 requests/day
- Solution: Use cached data or implement rate limiting

### Data Alignment Issues
- See `../docs/DATA_ALIGNMENT_FIX.md` for details
- Ensure news data matches stock data time period

## 📝 Development Notes

### Adding New Features
1. Edit `02_create_features.py`
2. Add feature calculation
3. Update feature count in `models/lstm/model.py`
4. Retrain model with `03_train_model.py`

### Modifying Model
1. Edit `models/lstm/model.py`
2. Adjust architecture as needed
3. Update configuration in `utils/config.py`
4. Retrain and evaluate

## 📞 Support

- **Documentation:** `../docs/` folder
- **Issues:** See `CURRENT_STATUS.md` for known issues
- **Team:** Contact via GWU email or Git branches

---

**Last Updated:** December 6, 2025  
**Status:** Production Ready ✅
