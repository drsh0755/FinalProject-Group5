"""
Visualize improvement from 2-year data
"""

import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def main():
    results_dir = Path(__file__).parent.parent / 'results'

    # Load results
    with open(results_dir / 'lstm_training_results.json') as f:
        baseline = json.load(f)

    with open(results_dir / 'lstm_2year_results.json') as f:
        improved = json.load(f)

    # Create comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. MAPE Comparison
    ax = axes[0, 0]
    models = ['Baseline\n(133 days)', '2-Year Data\n(501 days)']
    mapes = [baseline['test_metrics']['mape'], improved['test_metrics']['mape']]
    colors = ['#ff7f0e', '#2ca02c']
    bars = ax.bar(models, mapes, color=colors, alpha=0.8)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title('Test MAPE Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, mape in zip(bars, mapes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{mape:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement arrow
    improvement = mapes[0] - mapes[1]
    ax.annotate(f'-{improvement:.2f} pts\n(-65.5%)',
                xy=(0.5, (mapes[0] + mapes[1]) / 2),
                xytext=(1.5, (mapes[0] + mapes[1]) / 2),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'),
                fontsize=11, fontweight='bold', color='red',
                ha='left', va='center')

    # 2. Training Loss Curves
    ax = axes[0, 1]
    epochs_baseline = range(1, len(baseline['training_history']['train_loss']) + 1)
    epochs_improved = range(1, len(improved['training_history']['train_loss']) + 1)

    ax.plot(epochs_baseline, baseline['training_history']['val_loss'],
            label='Baseline', color='#ff7f0e', linewidth=2)
    ax.plot(epochs_improved, improved['training_history']['val_loss'],
            label='2-Year', color='#2ca02c', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title('Validation Loss During Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 3. MAE Comparison
    ax = axes[1, 0]
    maes = [baseline['test_metrics']['mae'], improved['test_metrics']['mae']]
    bars = ax.bar(models, maes, color=colors, alpha=0.8)
    ax.set_ylabel('MAE ($)', fontsize=12)
    ax.set_title('Mean Absolute Error Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'${mae:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 4. Key Statistics Table
    ax = axes[1, 1]
    ax.axis('off')

    table_data = [
        ['Metric', 'Baseline', '2-Year', 'Improvement'],
        ['MAPE', f"{baseline['test_metrics']['mape']:.2f}%",
         f"{improved['test_metrics']['mape']:.2f}%",
         f"-{improvement:.2f} pts"],
        ['MAE', f"${baseline['test_metrics']['mae']:.2f}",
         f"${improved['test_metrics']['mae']:.2f}",
         f"-${baseline['test_metrics']['mae'] - improved['test_metrics']['mae']:.2f}"],
        ['RMSE', f"${baseline['test_metrics']['rmse']:.2f}",
         f"${improved['test_metrics']['rmse']:.2f}",
         f"-${baseline['test_metrics']['rmse'] - improved['test_metrics']['rmse']:.2f}"],
        ['Sequences', '28', '422', '+394 (15x)'],
        ['Data Days', '133', '501', '+368'],
        ['Seq Length', '60', '30', 'Optimized']
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style improvement column
    for i in range(1, 7):
        table[(i, 3)].set_facecolor('#E8F5E9')

    ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)

    plt.suptitle('2-Year Data Enhancement: 65.5% MAPE Improvement',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    output_file = results_dir / 'figures' / '2year_improvement_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")


if __name__ == '__main__':
    main()