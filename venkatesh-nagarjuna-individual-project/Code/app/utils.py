"""
Utility functions for Streamlit app.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.graph_objects as go


def calculate_performance_metrics(predictions: pd.DataFrame, actuals: pd.DataFrame) -> Dict:
    """
    Calculate performance metrics for predictions.

    Args:
        predictions: DataFrame with predictions
        actuals: DataFrame with actual prices

    Returns:
        Dictionary of performance metrics
    """
    # Merge predictions with actuals
    merged = predictions.merge(
        actuals,
        left_on=['ticker', 'date'],
        right_on=['Ticker', 'Date'],
        how='inner'
    )

    if len(merged) == 0:
        return {}

    # Calculate actual direction
    merged['actual_direction'] = (merged['Close'].shift(-1) > merged['Close']).astype(int)
    merged['predicted_direction'] = (merged['prediction'] == 'UP').astype(int)

    # Accuracy
    accuracy = (merged['actual_direction'] == merged['predicted_direction']).mean()

    # Precision, Recall, F1
    tp = ((merged['predicted_direction'] == 1) & (merged['actual_direction'] == 1)).sum()
    fp = ((merged['predicted_direction'] == 1) & (merged['actual_direction'] == 0)).sum()
    fn = ((merged['predicted_direction'] == 0) & (merged['actual_direction'] == 1)).sum()
    tn = ((merged['predicted_direction'] == 0) & (merged['actual_direction'] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_predictions': len(merged)
    }


def create_equity_curve(returns: List[float], initial_capital: float = 10000) -> go.Figure:
    """Create equity curve plot from returns."""
    equity = [initial_capital]

    for ret in returns:
        equity.append(equity[-1] * (1 + ret))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equity,
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Trade Number',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified'
    )

    return fig


def format_large_number(num: float) -> str:
    """Format large numbers with K, M, B suffixes."""
    if abs(num) >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.2f}"


def create_comparison_table(predictions: List[Dict]) -> pd.DataFrame:
    """Create formatted comparison table for predictions."""
    df = pd.DataFrame(predictions)

    # Add color coding column
    df['signal_strength'] = df.apply(
        lambda row: '🟢 Strong' if row['confidence'] > 0.75
        else '🟡 Moderate' if row['confidence'] > 0.6
        else '🔴 Weak',
        axis=1
    )

    return df


def get_market_summary(quotes_df: pd.DataFrame) -> Dict:
    """Get overall market summary statistics."""
    if quotes_df.empty:
        return {}

    avg_change = quotes_df['change_percent'].astype(float).mean()

    bullish = (quotes_df['change_percent'].astype(float) > 0).sum()
    bearish = (quotes_df['change_percent'].astype(float) < 0).sum()

    total_volume = quotes_df['volume'].sum()

    return {
        'average_change': avg_change,
        'bullish_stocks': bullish,
        'bearish_stocks': bearish,
        'total_volume': total_volume,
        'market_sentiment': 'Bullish' if avg_change > 0 else 'Bearish'
    }
