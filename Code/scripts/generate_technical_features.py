#!/usr/bin/env python3
import pandas as pd
import numpy as np

print("📈 Generating technical indicators...")

# Load the updated stock data
df = pd.read_csv('Code/data/processed/spy_raw_updated.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Convert numeric columns to float
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# Remove any rows with NaN values
df = df.dropna()

df = df.set_index('Date').sort_index()

print(f"   Loaded {len(df)} trading days")
print(f"   Date range: {df.index[0].date()} → {df.index[-1].date()}")

# ============================================================
# 1. MOVING AVERAGES
# ============================================================
print("  - Moving averages...")
df['SMA_5'] = df['Close'].rolling(window=5).mean()
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# ============================================================
# 2. MACD (Moving Average Convergence Divergence)
# ============================================================
print("  - MACD...")
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# ============================================================
# 3. RSI (Relative Strength Index)
# ============================================================
print("  - RSI...")
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# ============================================================
# 4. BOLLINGER BANDS
# ============================================================
print("  - Bollinger Bands...")
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

# ============================================================
# 5. MOMENTUM INDICATORS
# ============================================================
print("  - Momentum indicators...")
df['Momentum'] = df['Close'] - df['Close'].shift(10)
df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

# ============================================================
# 6. VOLATILITY
# ============================================================
print("  - Volatility...")
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility_10'] = df['Daily_Return'].rolling(window=10).std()
df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std()
df['Volatility_50'] = df['Daily_Return'].rolling(window=50).std()

# ============================================================
# 7. VOLUME INDICATORS
# ============================================================
print("  - Volume indicators...")
df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']

# ============================================================
# 8. PRICE CHANGES
# ============================================================
print("  - Price changes...")
df['Price_Change'] = df['Close'].diff()
df['Price_Change_Pct'] = df['Close'].pct_change() * 100
df['High_Low_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
df['Open_Close_Pct'] = ((df['Close'] - df['Open']) / df['Open']) * 100

# ============================================================
# 9. LAG FEATURES
# ============================================================
print("  - Lag features...")
df['Close_Lag1'] = df['Close'].shift(1)
df['Close_Lag2'] = df['Close'].shift(2)
df['Close_Lag3'] = df['Close'].shift(3)
df['Close_Lag5'] = df['Close'].shift(5)

# Remove rows with NaN
df_clean = df.dropna()

# Save
df_clean.reset_index().to_csv('Code/data/processed/spy_features_2year_updated.csv', index=False)

print(f"\n✅ Features generated!")
print(f"   Rows: {len(df_clean)}")
print(f"   Columns: {len(df_clean.columns)}")
print(f"   Date range: {df_clean.index[0].date()} → {df_clean.index[-1].date()}")
print(f"   Saved to: Code/data/processed/spy_features_2year_updated.csv")
