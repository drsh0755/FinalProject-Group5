# Deep Learning Final Project - Group 5
**Multi-Modal LSTM Stock Market Prediction**

## Team
- Adarsh, Venkatesh Nagarjuna, Mayur Patil
- Instructor: Dr. Amir Jafari
- Course: DATS 6303 - Deep Learning
- Due: December 8, 2025

## Overview
Multi-modal LSTM for SPY prediction using 43 features (36 technical + 7 sentiment)

**Performance:** 7.87% MAPE (target: <15%)

## Quick Start
```bash
cd Code
source venv/bin/activate
python scripts/01_download_data.py
python scripts/02_download_news.py
python scripts/03_create_technical_features.py
python scripts/04_merge_features.py
python scripts/05_train_model.py
python scripts/06_live_prediction.py
python scripts/07_verify_predictions.py
```

## Structure
- Code/scripts/ - Production pipeline (7 scripts)
- Code/data/ - Stock & news data (4 files)
- Code/models/ - Trained LSTM
- Code/results/ - 7.87% MAPE results
- Documentation/ - Reports & slides
- archive/ - Experimental code

## Data Pipeline
1. Stock data (Yahoo Finance, 407 days)
2. News data (Alpha Vantage, 25K+ articles)
3. Technical features (36 indicators)
4. Sentiment features (7 aggregations)
5. Training data (43 features, 407 samples)

## Model
- 2-layer LSTM, 128 hidden units
- Dropout: 0.4, Sequence: 60 days
- Parameters: 232,577

See Documentation/ for complete reports.
