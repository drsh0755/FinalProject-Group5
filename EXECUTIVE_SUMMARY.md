# Multi-Modal Stock Prediction: Executive Summary
**Group 5 | DATS 6303 Deep Learning | George Washington University**

---

## The Challenge
Predict S&P 500 ETF (SPY) price movements with <15% Mean Absolute Percentage Error using deep learning.

## Our Approach
**Multi-modal LSTM combining:**
1. **Technical Analysis:** 46 indicators from 5 market indices
2. **Sentiment Analysis:** FinBERT processing 25,639 news articles on 7 tech giants

## Key Results
| Metric | Value | Status |
|--------|-------|--------|
| **Final MAPE** | **7.87%** | ✅ **Target Exceeded** (15% target) |
| **Total Improvement** | **85.0%** | From 52.31% baseline |
| **Training Data** | 452 days | Feb 2024 - Nov 2025 |
| **Model Parameters** | 228,993 | PyTorch LSTM |

## The Journey
```
Baseline → 2-Year Data → + Sentiment
52.31%      18.03%         7.87%
  └─65.5% improvement─┘
            └─56.3% improvement─┘
```

## Why It Works
1. **Scale matters:** 3.8× more data = 65.5% improvement
2. **Sentiment adds value:** News analysis = +56.3% boost
3. **Feature engineering:** 46 technical + 4 sentiment features capture market dynamics
4. **Sequence optimization:** 30-day window balances context and recency

## Business Impact
- **Accuracy:** 7.87% MAPE enables confident trading decisions
- **Real-time ready:** Trained through Nov 19, 2025
- **Scalable:** Architecture applicable to any stock/index
- **Transparent:** Interpretable features and predictions

## Technologies
PyTorch | FinBERT | Alpha Vantage API | AWS EC2 (GPU) | Python | Git

---
*"By combining technical indicators with news sentiment, we achieved 85% improvement in prediction accuracy, demonstrating the power of multi-modal deep learning for financial forecasting."*
