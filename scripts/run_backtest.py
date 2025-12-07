"""
Run backtest on historical data to evaluate model performance.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import TFTModelWrapper
from data import StockDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simple backtesting engine for stock predictions."""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.equity_curve = [initial_capital]
        self.trades = []

    def run(
            self,
            predictions: pd.DataFrame,
            actual_prices: pd.DataFrame,
            transaction_cost: float = 0.001
    ) -> dict:
        """
        Run backtest.

        Args:
            predictions: DataFrame with predictions
            actual_prices: DataFrame with actual prices
            transaction_cost: Transaction cost as fraction

        Returns:
            Dictionary with backtest results
        """
        equity = self.initial_capital

        for _, pred in predictions.iterrows():
            ticker = pred['ticker']
            pred_date = pd.to_datetime(pred['date'])
            direction = pred['prediction']
            confidence = pred['confidence']

            # Find actual prices
            ticker_prices = actual_prices[actual_prices['Ticker'] == ticker]
            current_price_row = ticker_prices[ticker_prices['Date'] == pred_date]

            if current_price_row.empty:
                continue

            current_price = current_price_row['Close'].iloc[0]

            # Find next day price
            next_day_prices = ticker_prices[ticker_prices['Date'] > pred_date]
            if next_day_prices.empty:
                continue

            next_price = next_day_prices['Close'].iloc[0]

            # Calculate return
            actual_return = (next_price - current_price) / current_price

            # Apply strategy (with transaction costs)
            if direction == 'UP':
                strategy_return = actual_return - transaction_cost
            else:
                strategy_return = -actual_return - transaction_cost

            # Update equity
            equity *= (1 + strategy_return)
            self.equity_curve.append(equity)

            # Record trade
            self.trades.append({
                'date': pred_date,
                'ticker': ticker,
                'direction': direction,
                'confidence': confidence,
                'entry_price': current_price,
                'exit_price': next_price,
                'actual_return': actual_return,
                'strategy_return': strategy_return,
                'equity': equity
            })

        # Calculate metrics
        total_return = (equity - self.initial_capital) / self.initial_capital

        trades_df = pd.DataFrame(self.trades)

        if len(trades_df) > 0:
            win_rate = (trades_df['strategy_return'] > 0).mean()
            avg_win = trades_df[trades_df['strategy_return'] > 0]['strategy_return'].mean()
            avg_loss = trades_df[trades_df['strategy_return'] < 0]['strategy_return'].mean()

            # Sharpe ratio (assuming 252 trading days)
            returns = trades_df['strategy_return'].values
            sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)

            # Max drawdown
            equity_curve = np.array(self.equity_curve)
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            win_rate = avg_win = avg_loss = sharpe = max_drawdown = 0

        results = {
            'total_return': total_return,
            'final_equity': equity,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'trades': self.trades
        }

        return results


def main():
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model wrapper directory'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to preprocessed data CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/backtest_results.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=10000,
        help='Initial capital for backtest'
    )
    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=0.001,
        help='Transaction cost (0.001 = 0.1%)'
    )

    args = parser.parse_args()

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = TFTModelWrapper.load(args.model)

    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    df['Date'] = pd.to_datetime(df['Date'])

    # Use last 20% as test set
    test_start_idx = int(len(df) * 0.8)
    test_df = df.iloc[test_start_idx:].copy()

    logger.info(f"Backtest period: {test_df['Date'].min()} to {test_df['Date'].max()}")

    # Generate predictions
    logger.info("Generating predictions...")
    predictions = model.batch_predict_directions(test_df)

    logger.info(f"Generated {len(predictions)} predictions")

    # Run backtest
    logger.info("Running backtest...")
    engine = BacktestEngine(initial_capital=args.initial_capital)
    results = engine.run(
        predictions,
        test_df,
        transaction_cost=args.transaction_cost
    )

    # Remove detailed trades for summary
    summary = {k: v for k, v in results.items() if k != 'trades'}

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("Backtest Results")
    logger.info("=" * 60)
    logger.info(f"Total Return: {summary['total_return']:.2%}")
    logger.info(f"Final Equity: ${summary['final_equity']:.2f}")
    logger.info(f"Number of Trades: {summary['num_trades']}")
    logger.info(f"Win Rate: {summary['win_rate']:.2%}")
    logger.info(f"Avg Win: {summary['avg_win']:.2%}")
    logger.info(f"Avg Loss: {summary['avg_loss']:.2%}")
    logger.info(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {summary['max_drawdown']:.2%}")
    logger.info("=" * 60)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
