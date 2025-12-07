#!/bin/bash
# Script to run Streamlit app

set -e

echo "Starting Stock Direction Forecasting App..."

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true

# Run streamlit
streamlit run app/main.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=$STREAMLIT_SERVER_HEADLESS \
    --theme.base="light" \
    --theme.primaryColor="#FF4B4B" \
    --theme.backgroundColor="#FFFFFF" \
    --theme.secondaryBackgroundColor="#F0F2F6" \
    --theme.textColor="#262730"
