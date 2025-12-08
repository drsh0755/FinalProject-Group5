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


## Repository Structure

```
Final-Project-Group5/
│
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
├── Code/                                      # All project code
│   ├── scripts/                               # Production pipeline scripts
│   │   ├── 01_download_data.py                # Yahoo Finance data collection
│   │   ├── 02_download_news.py                # Alpha Vantage news fetching
│   │   ├── 03_create_technical_features.py    # Technical indicators (36+)
│   │   ├── 04_merge_features.py               # Feature integration pipeline
│   │   ├── 05_train_model.py                  # LSTM model training
│   │   ├── 06_live_prediction.py             # Real-time inference
│   │   └── 07_verify_predictions.py           # Model validation
│   │
│   ├── app.py                         # Streamlit dashboard application
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
git clone https://github.com/drsh0755/FinalProject-Group5.git
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
python 01_download_data.py

# Step 2: Fetch financial news
python 02_download_news.py

# Step 3: Create technical indicators
python 03_create_technical_features.py

# Step 4: Merge all features
python 04_merge_features.py

# Step 5: Train the model
python 05_train_model.py

# Step 6: Generate live predictions
python 06_live_prediction.py

# Step 7: Verify predictions
python 07_verify_predictions.py
```

```bash
Note: Script execution mentioned above is for SPY (S&P 500).
For similar execution for QQQ (NASDAQ) run its respective scripts suffixed with "_QQQ".
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
git clone https://github.com/drsh0755/FinalProject-Group5.git
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
