# Multi-Modal LSTM Stock Market Prediction System
## Group 5: Adarsh, Venkatesh Nagarjuna, Mayur Patil

### 🎯 Project Overview
**Objective:** Predict SPY (S&P 500 ETF) price movements using multi-modal deep learning

**Target:** S&P 500 ETF (SPY)
- Represents 500 largest U.S. companies
- Most liquid ETF (~$500B daily volume)
- Training Period: Feb 2, 2024 → Nov 19, 2025 (452 trading days)

### 📊 Data Sources

#### 1. Technical Data (SPY)
- **Source:** Alpha Vantage API
- **Features:** 46 technical indicators
  - Price metrics: OHLC, volume
  - Moving averages: SMA/EMA (5, 10, 20, 50 days)
  - Momentum: RSI, Stochastic, Williams %R
  - Trend: MACD, ADX, CCI
  - Volatility: Bollinger Bands, ATR
  - Lagged features: 1, 2, 3, 5 days

#### 2. Market Context
- QQQ (NASDAQ-100), DIA (Dow Jones), ^VIX (Volatility), ^TNX (10Y Treasury)

#### 3. Sentiment Data
- **Source:** 25,639 financial news articles from Alpha Vantage
- **Companies Tracked:** AAPL, MSFT, AMZN, TSLA, NVDA, META, GOOGL
- **Analysis:** FinBERT sentiment model
- **Coverage:** 63.7% of trading days (288/452 days)
- **Features:** sentiment_mean, sentiment_std, article_count, positive_ratio

### 🏗️ Architecture

**Model:** PyTorch LSTM
- **Input:** 48 features × 30-day sequences
- **Parameters:** 228,993 trainable parameters
- **Layers:**
  - LSTM: 2 layers, 128 hidden units, 0.3 dropout
  - Dense: 64 → 32 → 1 (output)
- **Training:** 70% train, 15% val, 15% test

### 📈 Results Evolution

| Stage | MAPE | Improvement | Key Enhancement |
|-------|------|-------------|-----------------|
| **Baseline LSTM** | 52.31% | - | 133 days, basic features |
| **2-Year Dataset** | 18.03% | 65.5% ↓ | 501 days, optimized sequence |
| **+ Sentiment** | **7.87%** | **85.0% ↓** | Multi-modal fusion |

### 🎯 Key Achievements
- ✅ **Sub-15% MAPE target exceeded** (7.87% achieved)
- ✅ **Multi-modal integration** (technical + sentiment)
- ✅ **Real-time capable** (trained through Nov 19, 2025)
- ✅ **Comprehensive feature engineering** (46 technical + 4 sentiment features)

### 💡 Key Insights
1. **Dataset size matters:** 3.8× more data → 65.5% improvement
2. **Sentiment adds value:** News sentiment → additional 56.3% improvement
3. **Feature engineering crucial:** 46 technical indicators capture market dynamics
4. **Sequence optimization:** 30-day window balances context and recency

### 🚀 Future Work
- Live deployment with real-time predictions
- Additional sentiment sources (Twitter, Reddit, earnings calls)
- Multi-stock portfolio optimization
- Transfer learning to other market indices
