# Multi-Modal Stock Market Prediction System
## DATS 6303 Deep Learning - Final Project - Group 5

**Team Members:** 
- Adarsh Singh
- Venkatesh Nagarjuna
- Mayur Patil

**Instructor:** Dr. Amir Jafari  
**Semester:** Fall 2025

DATS6303– Deep Learning | Final Project | Fall 2025


---

## Project Overview

This project implements a multi-modal deep learning system for stock market prediction that integrates:
- **LSTM Networks** for technical analysis of historical price data
- **FinBERT Transformer** for financial news sentiment analysis
- **Dense Networks** for market context integration

The system performs both historical backtesting and real-time prediction capabilities.

---

## Repository Structure
```
FinalProject-Group5/
├── Group-Proposal/
│   └── proposal.pdf                    # Project proposal
├── Final-Group-Project-Report/
│   └── (Final report - Due Dec 8)
├── Final-Group-Presentation/
│   └── (Presentation slides - Due Dec 8)
├── Code/                               # Main codebase
│   ├── data/                          # Data storage
│   │   ├── raw/                       # Raw downloaded data
│   │   ├── processed/                 # Preprocessed data
│   │   └── live/                      # Live prediction data
│   ├── models/                        # Model implementations
│   │   ├── lstm/                      # LSTM model code
│   │   ├── sentiment/                 # FinBERT sentiment code
│   │   ├── fusion/                    # Fusion network code
│   │   └── checkpoints/               # Saved model weights
│   ├── notebooks/                     # Jupyter notebooks for EDA
│   ├── scripts/                       # Executable scripts
│   ├── utils/                         # Helper functions
│   ├── results/                       # Outputs
│   │   ├── figures/                   # Plots and visualizations
│   │   ├── predictions/               # Prediction results
│   │   └── logs/                      # Training logs
│   ├── requirements.txt               # Python dependencies
│   └── README.md                      # Code documentation
├── adarsh-singh-individual-project/
│   ├── Individual-Final-Project-Report/
│   └── Code/                          # Individual contributions
├── venkatesh-nagarjuna-individual-project/
│   ├── Individual-Final-Project-Report/
│   └── Code/
└── mayur-patil-individual-project/
    ├── Individual-Final-Project-Report/
    └── Code/
```

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/drsh0755/FinalProject-Group5.git
cd FinalProject-Group5/Code
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup API Keys
Create a `.env` file in the `Code/` directory:
```bash
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here  # Optional
```

### 5. Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

---

## Data Sources

1. **Historical Stock Prices:** Alpha Vantage API
2. **Financial News:** Kaggle Daily Financial News Dataset
3. **Market Indices:** Yahoo Finance (SPY, QQQ, DIA, VIX)

---

## Project Timeline

| Phase | Dates | Tasks |
|-------|-------|-------|
| **Phase 1: Development** | Nov 17-24 | Historical data processing, model development |
| **Phase 2: Live Deployment** | Nov 25-Dec 1 | Real-time predictions, daily monitoring |
| **Phase 3: Documentation** | Dec 2-8 | Final report, presentation, submission |

---

## Running the Code

### Data Collection
```bash
cd Code
python scripts/download_data.py
```

### Model Training
```bash
python scripts/train_lstm.py
python scripts/train_sentiment.py
python scripts/train_fusion.py
```

### Live Predictions
```bash
python scripts/live_predict.py
```

---

## Results

Results will be available in `Code/results/`:
- Training curves and metrics in `figures/`
- Daily predictions in `predictions/`
- Model performance logs in `logs/`

---

## Contact

For questions or issues, contact:
- Adarsh Singh: adarsh.singh@gwu.edu
- Venkatesh Nagarjuna: venkatesh.nagarjuna@gwu.edu
- Mayur Patil: mayur.patil@gwu.edu

---

## License

This project is submitted as coursework for DATS 6303 at George Washington University.
