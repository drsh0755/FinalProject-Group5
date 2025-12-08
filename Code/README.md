# Multi-Modal LSTM Stock Market Prediction System
**DATS 6303 Deep Learning - Final Project**  
**Group 5:** Adarsh, Venkatesh Nagarjuna, Mayur Patil  
**Instructor:** Dr. Amir Jafari  
**George Washington University - Fall 2025**

## Project Overview

This project implements a sophisticated multi-modal LSTM-based stock market prediction system that combines technical analysis with sentiment analysis from financial news. The system targets SPY and QQQ ETF price forecasting, achieving exceptional accuracy with 7.87% MAPE on SPY predictions - an 85% improvement over baseline models.

### Key Features
- **Multi-Modal Architecture:** Integrates 36+ technical indicators with sentiment features from financial news
- **Real-Time Predictions:** Live stock price forecasting with GPU-accelerated inference
- **Comprehensive Dashboard:** Interactive Streamlit application for visualization and analysis
- **Production-Ready Deployment:** AWS EC2 deployment with automated data pipelines
- **Model Generalizability:** Demonstrated across multiple market indices (SPY, QQQ)

### Performance Metrics
- **SPY Model MAPE:** 7.87% (Target: <15%)
- **Baseline Improvement:** 85% reduction in prediction error
- **Architecture:** 2-layer LSTM (128 hidden units, dropout 0.4)
- **Training Framework:** PyTorch with GPU acceleration

## Repository Structure

```
Final-Project-Group5/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── Group-Proposal/                    # Initial project proposal
│   └── Group5_Proposal.pdf
│
├── Final-Group-Project-Report/        # Comprehensive project report
│   └── Group5_Final_Report.pdf
│
├── Final-Group-Presentation/          # Presentation materials
│   └── Group5_Presentation.pdf
│
├── Code/                              # All project code
│   ├── scripts/                       # Production pipeline scripts
│   │   ├── 01_download_stock_data.py      # Yahoo Finance data collection
│   │   ├── 02_download_news.py            # Alpha Vantage news fetching
│   │   ├── 03_process_news.py             # FinBERT sentiment analysis
│   │   ├── 04_create_technical_features.py # Technical indicators (36+)
│   │   ├── 05_merge_features.py           # Feature integration pipeline
│   │   ├── 06_train_model.py              # LSTM model training
│   │   ├── 07_live_predictions.py         # Real-time inference
│   │   └── 08_verify_predictions.py       # Model validation
│   │
│   ├── app.py                         # Streamlit dashboard application
│   │
│   ├── models/                        # Trained model artifacts
│   │   ├── spy_lstm_model.pth
│   │   ├── qqq_lstm_model.pth
│   │   └── scalers/
│   │
│   ├── data/                          # Data storage
│   │   ├── raw/                       # Original downloaded data
│   │   ├── processed/                 # Processed features
│   │   └── predictions/               # Model outputs
│   │
│   └── utils/                         # Helper functions
│       ├── feature_engineering.py
│       ├── data_loader.py
│       └── visualization.py
│
└── Individual-Final-Project-Report/   # Individual contribution reports
    ├── adarsh-individual-project/
    ├── venkatesh-individual-project/
    └── mayur-individual-project/
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- AWS account (for deployment)
- API Keys:
  - Alpha Vantage API key (for news sentiment data)
  - Yahoo Finance access (free, via yfinance library)

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Final-Project-Group5.git
cd Final-Project-Group5
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up API credentials:**
Create a `.env` file in the project root:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

5. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage Guide

### Running the Complete Pipeline

Execute scripts in numerical order:

```bash
cd Code/scripts

# Step 1: Download stock data
python 01_download_stock_data.py

# Step 2: Fetch financial news
python 02_download_news.py

# Step 3: Process news with sentiment analysis
python 03_process_news.py

# Step 4: Create technical indicators
python 04_create_technical_features.py

# Step 5: Merge all features
python 05_merge_features.py

# Step 6: Train the model
python 06_train_model.py

# Step 7: Generate live predictions
python 07_live_predictions.py

# Step 8: Verify predictions
python 08_verify_predictions.py
```

### Running the Streamlit Dashboard

Launch the interactive web application:

```bash
cd Code
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## Quick Start (Minimal Setup)

If you just want to see predictions with pre-trained models:

```bash
cd Code/scripts
python 07_live_predictions.py
```

## AWS Deployment

To deploy on AWS EC2:

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone and setup
git clone https://github.com/yourusername/Final-Project-Group5.git
cd Final-Project-Group5
pip install -r requirements.txt

# Run the Streamlit app
streamlit run Code/app.py --server.port 8501 --server.address 0.0.0.0
```

Access via: `http://your-ec2-ip:8501`

## Project Results

- **SPY MAPE:** 7.87% (85% improvement over baseline)
- **Target Achieved:** <15% MAPE threshold exceeded
- **Features Used:** 36+ technical indicators + news sentiment
- **Model Architecture:** 2-layer LSTM, 128 hidden units, dropout 0.4

## Team Contributions

### Adarsh
- Multi-modal LSTM architecture design
- AWS EC2 deployment and infrastructure
- Streamlit dashboard development
- Feature engineering pipeline

### Venkatesh Nagarjuna
- Data collection and preprocessing
- Technical indicators implementation
- Model training and optimization

### Mayur Patil
- News sentiment analysis integration
- FinBERT implementation
- Model validation and testing

## References

- **PyTorch Documentation:** https://pytorch.org/docs/
- **Alpha Vantage API:** https://www.alphavantage.co/
- **Yahoo Finance:** https://finance.yahoo.com/
- **FinBERT:** ProsusAI/finbert (HuggingFace)
- **Streamlit:** https://streamlit.io/

## Contact

For questions about this project, please contact:
- **Adarsh:** [your-email]@gwu.edu
- **Course:** DATS 6303 Deep Learning
- **Instructor:** Dr. Amir Jafari (ajafari@gwu.edu)

## License

This project is submitted as coursework for DATS 6303 at George Washington University.
