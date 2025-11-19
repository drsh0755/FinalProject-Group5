"""
Exploratory Data Analysis for Stock Market Data
Generates all visualizations and saves them to results/figures/

Usage:
    cd ~/DL/Final Project/Code
    python3 notebooks/01_data_exploration_script.py
"""

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for script
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    """Main function to run all EDA"""

    print("=" * 60)
    print("STOCK MARKET DATA - EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Setup paths - always relative to Code directory
    # Get the script's directory (notebooks/)
    script_dir = Path(__file__).resolve().parent
    # Go up one level to Code/
    code_dir = script_dir.parent

    data_dir = code_dir / 'data' / 'raw'
    results_dir = code_dir / 'results' / 'figures'

    print(f"\nWorking directory: {os.getcwd()}")
    print(f"Script directory: {script_dir}")
    print(f"Code directory: {code_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")

    # Create results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    print("\n" + "=" * 60)
    print("1. Loading Data...")
    print("=" * 60)

    spy_file = data_dir / 'SPY_historical.csv'
    print(f"Looking for: {spy_file}")
    print(f"File exists: {spy_file.exists()}")

    if not spy_file.exists():
        print(f"\n✗ Error: Could not find {spy_file}")
        print("\nAvailable files in data/raw/:")
        for f in data_dir.glob('*.csv'):
            print(f"  - {f.name}")
        sys.exit(1)

    try:
        spy_df = pd.read_csv(spy_file, index_col=0, parse_dates=True)
        print(f"✓ SPY data loaded: {spy_df.shape}")
        print(f"  Date range: {spy_df.index[0].date()} to {spy_df.index[-1].date()}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    # ============================================================
    # 2. BASIC STATISTICS
    # ============================================================
    print("\n" + "=" * 60)
    print("2. Summary Statistics")
    print("=" * 60)
    stats = spy_df[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
    print(stats)

    # ============================================================
    # 3. PRICE HISTORY PLOT
    # ============================================================
    print("\n" + "=" * 60)
    print("3. Generating price history plot...")
    print("=" * 60)
    plt.figure(figsize=(14, 6))
    plt.plot(spy_df.index, spy_df['Close'], label='Close Price',
             linewidth=2, color='steelblue')
    plt.title('SPY Closing Price History', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = results_dir / 'spy_price_history.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # ============================================================
    # 4. VOLUME ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("4. Generating volume analysis...")
    print("=" * 60)
    plt.figure(figsize=(14, 6))
    plt.bar(spy_df.index, spy_df['Volume'], alpha=0.7, color='steelblue')
    plt.title('SPY Trading Volume', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volume', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_file = results_dir / 'spy_volume.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # ============================================================
    # 5. DAILY RETURNS
    # ============================================================
    print("\n" + "=" * 60)
    print("5. Calculating daily returns...")
    print("=" * 60)
    spy_df['Daily_Return'] = spy_df['Close'].pct_change() * 100

    plt.figure(figsize=(14, 6))
    plt.plot(spy_df.index, spy_df['Daily_Return'], alpha=0.7,
             linewidth=1, color='steelblue')
    plt.title('SPY Daily Returns (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Return (%)', fontsize=12)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = results_dir / 'spy_daily_returns.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    print(f"\nDaily Return Statistics:")
    print(f"  Mean: {spy_df['Daily_Return'].mean():.3f}%")
    print(f"  Std:  {spy_df['Daily_Return'].std():.3f}%")
    print(f"  Min:  {spy_df['Daily_Return'].min():.3f}%")
    print(f"  Max:  {spy_df['Daily_Return'].max():.3f}%")

    # ============================================================
    # 6. RETURNS DISTRIBUTION
    # ============================================================
    print("\n" + "=" * 60)
    print("6. Generating returns distribution...")
    print("=" * 60)
    plt.figure(figsize=(10, 6))
    spy_df['Daily_Return'].hist(bins=50, alpha=0.7, color='steelblue',
                                edgecolor='black')
    plt.title('Distribution of Daily Returns', fontsize=16, fontweight='bold')
    plt.xlabel('Return (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_file = results_dir / 'spy_returns_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # ============================================================
    # 7. MULTIPLE INDICES COMPARISON
    # ============================================================
    print("\n" + "=" * 60)
    print("7. Comparing multiple indices...")
    print("=" * 60)
    tickers = ['SPY', 'QQQ', 'DIA']
    dfs = {}

    for ticker in tickers:
        ticker_file = data_dir / f'{ticker}_historical.csv'
        if ticker_file.exists():
            df = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
            dfs[ticker] = df['Close']
            print(f"✓ Loaded {ticker}")
        else:
            print(f"⚠ Warning: {ticker}_historical.csv not found, skipping...")

    if len(dfs) > 1:
        combined_df = pd.DataFrame(dfs)
        normalized = (combined_df / combined_df.iloc[0]) * 100

        plt.figure(figsize=(14, 6))
        for ticker in dfs.keys():
            plt.plot(normalized.index, normalized[ticker], label=ticker, linewidth=2)

        plt.title('Normalized Index Comparison (Base = 100)',
                  fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        output_file = results_dir / 'indices_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()

        # ========================================================
        # 8. CORRELATION ANALYSIS
        # ========================================================
        print("\n" + "=" * 60)
        print("8. Generating correlation matrix...")
        print("=" * 60)
        returns_df = combined_df.pct_change().dropna()

        plt.figure(figsize=(8, 6))
        sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, fmt='.3f')
        plt.title('Correlation Matrix of Daily Returns',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        output_file = results_dir / 'correlation_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    else:
        print("⚠ Not enough data for comparison plots")

    # ============================================================
    # 9. VOLATILITY ANALYSIS
    # ============================================================
    print("\n" + "=" * 60)
    print("9. Analyzing volatility...")
    print("=" * 60)
    spy_df['Volatility_20'] = spy_df['Daily_Return'].rolling(window=20).std()

    plt.figure(figsize=(14, 6))
    plt.plot(spy_df.index, spy_df['Volatility_20'], linewidth=2, color='orange')
    plt.title('SPY 20-Day Rolling Volatility', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volatility (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = results_dir / 'spy_volatility.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()

    # ============================================================
    # 10. SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    summary = {
        'Total Trading Days': len(spy_df),
        'Date Range': f"{spy_df.index[0].date()} to {spy_df.index[-1].date()}",
        'Current Price': f"${spy_df['Close'].iloc[-1]:.2f}",
        'Price Change': f"${spy_df['Close'].iloc[-1] - spy_df['Close'].iloc[0]:.2f}",
        'Percent Change': f"{((spy_df['Close'].iloc[-1] / spy_df['Close'].iloc[0]) - 1) * 100:.2f}%",
        'Mean Daily Return': f"{spy_df['Daily_Return'].mean():.3f}%",
        'Volatility (Std)': f"{spy_df['Daily_Return'].std():.3f}%",
        'Max Daily Gain': f"{spy_df['Daily_Return'].max():.3f}%",
        'Max Daily Loss': f"{spy_df['Daily_Return'].min():.3f}%",
        'Average Volume': f"{spy_df['Volume'].mean():,.0f}",
    }

    for key, value in summary.items():
        print(f"{key:.<30} {value}")

    print("=" * 60)
    print("\n✓ EDA Complete!")
    print(f"✓ All figures saved to: {results_dir.relative_to(code_dir)}/")
    print("\nGenerated files:")
    for i, file in enumerate(sorted(results_dir.glob("*.png")), 1):
        print(f"  {i}. {file.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
