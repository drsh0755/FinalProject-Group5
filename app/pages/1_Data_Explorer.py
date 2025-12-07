"""
Data exploration page for Streamlit app.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_sources.market_data_provider import MarketDataProvider
from features.technical_indicators import TechnicalIndicators

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Data Explorer")

st.markdown("""
Explore historical market data, technical indicators, and data quality metrics.
""")

# Initialize data provider
tickers = st.multiselect(
    "Select tickers to explore",
    options=["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"],
    default=["AAPL"]
)

if tickers:
    provider = MarketDataProvider(tickers)

    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

    # Load data
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            data = provider.get_historical_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )

            # Add indicators
            data = TechnicalIndicators.add_all_indicators(data)

            st.success(f"Loaded {len(data)} rows of data")

            # Summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(data.describe())

            # Missing data
            st.subheader("Data Quality")
            missing = data.isnull().sum()
            missing = missing[missing > 0]

            if len(missing) > 0:
                st.warning("Missing values detected:")
                st.dataframe(missing)
            else:
                st.success("No missing values!")

            # Raw data
            st.subheader("Raw Data")
            st.dataframe(data.tail(100))

            # Download
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="market_data.csv",
                mime="text/csv"
            )
