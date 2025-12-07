"""
Streamlit application for real-time stock direction forecasting.
Integrates TFT model with live market data and sentiment analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sources.market_data_provider import MarketDataProvider
from features.technical_indicators import TechnicalIndicators
from features.sentiment_processing import SentimentProcessor
from models.model_wrapper import TFTModelWrapper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Stock Direction Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path: str = None):
    """Load trained TFT model (cached)."""
    try:
        # If no path provided, search for latest checkpoint
        if model_path is None:
            checkpoint_dirs = [
                "checkpoints/tft_*/model_wrapper",
                "checkpoints/*/model_wrapper"
            ]

            import glob
            for pattern in checkpoint_dirs:
                matches = glob.glob(pattern)
                if matches:
                    model_path = sorted(matches)[-1]  # Get latest
                    break

        if model_path is None or not os.path.exists(model_path):
            st.warning("⚠️ No trained model found. Please train a model first.")
            st.info(
                "Run: `python training/train_tft.py --config training/train_config.yaml --data data/processed/features_with_sentiment.csv`")
            return None

        model_wrapper = TFTModelWrapper.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model_wrapper

    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure you have trained a model first.")
        return None


@st.cache_resource
def initialize_data_provider(tickers: list, alphavantage_key: str = None):
    """Initialize market data provider (cached)."""
    return MarketDataProvider(
        tickers=tickers,
        alphavantage_key=alphavantage_key if alphavantage_key else None
    )


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_historical_data(provider, start_date: str, end_date: str):
    """Load historical data with caching."""
    return provider.get_historical_data(start_date=start_date, end_date=end_date)


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_latest_quotes(provider):
    """Get latest quotes with short cache."""
    return provider.get_latest_quotes(use_cache=True, cache_timeout=60)


def create_candlestick_chart(df: pd.DataFrame, ticker: str):
    """Create candlestick chart with volume."""
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('Date')

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{ticker} Price', 'Volume')
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=ticker_data['Date'],
            open=ticker_data['Open'],
            high=ticker_data['High'],
            low=ticker_data['Low'],
            close=ticker_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ['red' if close < open else 'green'
              for close, open in zip(ticker_data['Close'], ticker_data['Open'])]

    fig.add_trace(
        go.Bar(
            x=ticker_data['Date'],
            y=ticker_data['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{ticker} - Last 90 Days',
        yaxis_title='Price ($)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=600,
        hovermode='x unified'
    )

    return fig


def create_technical_indicators_chart(df: pd.DataFrame, ticker: str):
    """Create technical indicators chart."""
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('Date').tail(90)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price with Moving Averages', 'RSI', 'MACD')
    )

    # Price and MAs
    fig.add_trace(
        go.Scatter(x=ticker_data['Date'], y=ticker_data['Close'],
                   name='Close', line=dict(color='blue', width=2)),
        row=1, col=1
    )

    if 'sma_20' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['Date'], y=ticker_data['sma_20'],
                       name='SMA 20', line=dict(color='orange', dash='dash')),
            row=1, col=1
        )

    if 'sma_50' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['Date'], y=ticker_data['sma_50'],
                       name='SMA 50', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

    # RSI
    if 'rsi' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['Date'], y=ticker_data['rsi'],
                       name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    if 'macd' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['Date'], y=ticker_data['macd'],
                       name='MACD', line=dict(color='blue')),
            row=3, col=1
        )

    if 'macd_signal' in ticker_data.columns:
        fig.add_trace(
            go.Scatter(x=ticker_data['Date'], y=ticker_data['macd_signal'],
                       name='Signal', line=dict(color='orange')),
            row=3, col=1
        )

    if 'macd_hist' in ticker_data.columns:
        colors = ['green' if val >= 0 else 'red' for val in ticker_data['macd_hist']]
        fig.add_trace(
            go.Bar(x=ticker_data['Date'], y=ticker_data['macd_hist'],
                   name='Histogram', marker_color=colors),
            row=3, col=1
        )

    fig.update_layout(height=800, hovermode='x unified')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig


def create_prediction_card(prediction: dict):
    """Create prediction display card."""
    direction = prediction['prediction']
    confidence = prediction['confidence']
    prob_up = prediction['prob_up']
    prob_down = prediction['prob_down']

    # Color based on prediction
    color = "green" if direction == "UP" else "red"
    icon = "📈" if direction == "UP" else "📉"

    # Create card
    st.markdown(f"""
    <div style="
        padding: 20px;
        border-radius: 10px;
        border: 2px solid {color};
        background-color: rgba({'0,255,0' if direction == 'UP' else '255,0,0'}, 0.1);
        margin: 10px 0;
    ">
        <h2 style="margin: 0; color: {color};">{icon} {direction}</h2>
        <h3 style="margin: 10px 0;">Confidence: {confidence:.1%}</h3>
        <div style="margin-top: 15px;">
            <div style="margin: 5px 0;">
                <span>Prob UP:</span>
                <div style="
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    height: 25px;
                    position: relative;
                    margin-top: 5px;
                ">
                    <div style="
                        background-color: green;
                        width: {prob_up * 100}%;
                        height: 100%;
                        border-radius: 5px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                    ">{prob_up:.1%}</div>
                </div>
            </div>
            <div style="margin: 5px 0;">
                <span>Prob DOWN:</span>
                <div style="
                    background-color: #e0e0e0;
                    border-radius: 5px;
                    height: 25px;
                    position: relative;
                    margin-top: 5px;
                ">
                    <div style="
                        background-color: red;
                        width: {prob_down * 100}%;
                        height: 100%;
                        border-radius: 5px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                    ">{prob_down:.1%}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def create_attention_heatmap(attention_weights: np.ndarray, dates: list = None):
    """Create attention weights heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        colorscale='Viridis',
        x=dates if dates else list(range(attention_weights.shape[1])),
        y=dates if dates else list(range(attention_weights.shape[0]))
    ))

    fig.update_layout(
        title='Temporal Attention Weights',
        xaxis_title='Time Steps',
        yaxis_title='Time Steps',
        height=400
    )

    return fig


def run_backtest(predictions_df: pd.DataFrame, actual_prices: pd.DataFrame):
    """Run simple backtest based on predictions."""
    results = []

    for ticker in predictions_df['ticker'].unique():
        ticker_preds = predictions_df[predictions_df['ticker'] == ticker].copy()
        ticker_prices = actual_prices[actual_prices['Ticker'] == ticker].copy()

        # Merge predictions with next-day prices
        ticker_preds = ticker_preds.sort_values('date')
        ticker_prices = ticker_prices.sort_values('Date')

        # Simple strategy: buy on UP prediction, sell/short on DOWN
        equity = 1000  # Start with $1000
        equity_curve = [equity]

        for _, pred in ticker_preds.iterrows():
            pred_date = pd.to_datetime(pred['date'])
            next_day = pred_date + timedelta(days=1)

            # Find actual return
            try:
                current_price = ticker_prices[ticker_prices['Date'] == pred_date]['Close'].iloc[0]
                next_price = ticker_prices[ticker_prices['Date'] >= next_day]['Close'].iloc[0]
                actual_return = (next_price - current_price) / current_price

                # Apply strategy
                if pred['prediction'] == 'UP':
                    equity *= (1 + actual_return)
                else:
                    equity *= (1 - actual_return)  # Short position

                equity_curve.append(equity)
            except:
                continue

        total_return = (equity - 1000) / 1000

        results.append({
            'ticker': ticker,
            'total_return': total_return,
            'final_equity': equity,
            'num_trades': len(ticker_preds)
        })

    return pd.DataFrame(results)


def main():
    """Main Streamlit application."""

    st.title("📈 Stock Direction Forecasting System")
    st.markdown("**Powered by Temporal Fusion Transformer + DDG-DA**")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")

    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]

    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=DEFAULT_TICKERS,
        default=[]
    )

    # Find and load model
    import glob
    available_models = glob.glob("checkpoints/*/model_wrapper")

    if not available_models:
        st.error("❌ No trained models found!")
        st.stop()

    model_options = {os.path.basename(os.path.dirname(m)): m for m in available_models}
    selected_model_name = st.sidebar.selectbox("Model", options=list(model_options.keys()))
    model_path = model_options[selected_model_name]

    with st.spinner("Loading model..."):
        model = load_model(model_path)

    if model is None:
        st.error("Failed to load model")
        st.stop()

    st.success(f"✅ Model loaded: {selected_model_name}")

    if not selected_tickers:
        st.info("Select tickers from the sidebar to generate predictions.")
        st.stop()

    st.header("🎯 Stock Direction Predictions")

    # Initialize data provider
    with st.spinner("Loading market data..."):
        provider = MarketDataProvider(tickers=selected_tickers)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

        try:
            data = provider.get_historical_data(start_date=start_date, end_date=end_date)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.stop()

    if data.empty:
        st.warning("No data available")
        st.stop()

    # Add technical indicators
    with st.spinner("Computing features..."):
        data = TechnicalIndicators.add_all_indicators(data)

    # Add missing sentiment features with zeros
    sentiment_features = [col for col in model.feature_cols if 'sentiment' in col.lower()]
    for feat in sentiment_features:
        if feat not in data.columns:
            data[feat] = 0.0

    # Generate predictions
    predictions = []

    for ticker in selected_tickers:
        ticker_data = data[data['Ticker'] == ticker].sort_values('Date')

        if len(ticker_data) < 60:
            st.warning(f"Insufficient data for {ticker} ({len(ticker_data)} < 60 days)")
            continue

        ticker_data = ticker_data.tail(60)

        try:
            # Ensure all required features exist
            missing_features = [f for f in model.feature_cols if f not in ticker_data.columns]
            if missing_features:
                st.warning(f"Missing features for {ticker}: {missing_features[:5]}")
                # Add missing features as zeros
                for feat in missing_features:
                    ticker_data[feat] = 0.0

            # Prepare features in correct order
            features = ticker_data[model.feature_cols].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features_tensor = torch.FloatTensor(features).unsqueeze(0)

            # Get ticker ID
            ticker_id = model.ticker_to_id.get(ticker, 0)
            ticker_id_tensor = torch.LongTensor([[ticker_id]])

            input_features = {
                'encoder_cont': features_tensor,
                'ticker_id': ticker_id_tensor
            }

            # Predict
            output = model.predict(input_features)

            pred_class = output['predictions'][0]
            probs = output['probabilities'][0]

            prediction = {
                'ticker': ticker,
                'prediction': 'UP' if pred_class == 1 else 'DOWN',
                'confidence': float(probs[pred_class]),
                'prob_up': float(probs[1]),
                'prob_down': float(probs[0]),
                'date': ticker_data['Date'].iloc[-1],
                'current_price': float(ticker_data['Close'].iloc[-1])
            }

            predictions.append(prediction)

        except Exception as e:
            st.error(f"Error predicting {ticker}: {str(e)[:200]}")
            continue

    # Display predictions
    if predictions:
        cols = st.columns(min(len(predictions), 3))

        for idx, pred in enumerate(predictions):
            with cols[idx % 3]:
                st.markdown(f"### {pred['ticker']}")

                direction_color = "green" if pred['prediction'] == "UP" else "red"
                direction_icon = "📈" if pred['prediction'] == "UP" else "📉"

                st.markdown(f"""
                <div style="
                    padding: 20px;
                    border-radius: 10px;
                    border: 2px solid {direction_color};
                    background-color: rgba({'0,255,0' if pred['prediction'] == 'UP' else '255,0,0'}, 0.1);
                    text-align: center;
                ">
                    <h2 style="margin: 0; color: {direction_color};">{direction_icon} {pred['prediction']}</h2>
                    <h3 style="margin: 10px 0;">Confidence: {pred['confidence']:.1%}</h3>
                    <p style="margin: 5px 0;">Price: ${pred['current_price']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

                st.metric("Prob UP", f"{pred['prob_up']:.1%}")
                st.metric("Prob DOWN", f"{pred['prob_down']:.1%}")

        # Summary
        st.subheader("📊 Summary")
        summary_df = pd.DataFrame(predictions)[['ticker', 'prediction', 'confidence', 'current_price']]
        summary_df.columns = ['Ticker', 'Direction', 'Confidence', 'Price']
        st.dataframe(
            summary_df.style.format({'Confidence': '{:.1%}', 'Price': '${:.2f}'}),
            hide_index=True,
            use_container_width=True
        )
    else:
        st.warning("No predictions generated. Try different tickers or check data availability.")


if __name__ == "__main__":
    main()
