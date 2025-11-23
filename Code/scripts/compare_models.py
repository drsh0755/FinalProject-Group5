"""
Compare baseline vs 2-year vs sentiment models
"""

import json
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60 + "\n")

    results_dir = Path(__file__).parent.parent / 'results'

    # Load all results
    models = {
        'Baseline (133 days)': results_dir / 'lstm_training_results.json',
        '2-Year Data': results_dir / 'lstm_2year_results.json',
        'With Sentiment': results_dir / 'lstm_sentiment_results.json'
    }

    comparison = []

    for name, filepath in models.items():
        if filepath.exists():
            with open(filepath) as f:
                results = json.load(f)
                comparison.append({
                    'Model': name,
                    'MAE': results['test_metrics']['mae'],
                    'RMSE': results['test_metrics']['rmse'],
                    'MAPE (%)': results['test_metrics']['mape'],
                    'Sequences': len(results['training_history']['train_loss'])
                })
        else:
            print(f"⚠ Not found: {filepath.name}")

    df = pd.DataFrame(comparison)

    print("\nTest Set Performance:")
    print(df.to_string(index=False))

    # Calculate improvements
    if len(df) >= 2:
        print("\n" + "=" * 60)
        print("IMPROVEMENTS")
        print("=" * 60)

        baseline_mape = df.iloc[0]['MAPE (%)']

        for idx in range(1, len(df)):
            model_name = df.iloc[idx]['Model']
            model_mape = df.iloc[idx]['MAPE (%)']
            improvement = baseline_mape - model_mape
            improvement_pct = (improvement / baseline_mape) * 100

            print(f"\n{model_name}:")
            print(f"  MAPE improvement: {improvement:.2f} percentage points")
            print(f"  Relative improvement: {improvement_pct:.1f}%")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['MAE', 'RMSE', 'MAPE (%)']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(df['Model'], df[metric], color=['#ff7f0e', '#2ca02c', '#1f77b4'])
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_file = results_dir / 'figures' / 'model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_file}")

    print("\n" + "=" * 60)
    print("✓ COMPARISON COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()