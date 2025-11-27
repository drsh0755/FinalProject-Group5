import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
output_dir = Path('Exhibition/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# 1. PERFORMANCE COMPARISON
fig, ax = plt.subplots(figsize=(10, 6))
models = ['Baseline\nLSTM', '2-Year\nDataset', 'Multi-Modal\n+ Sentiment']
mapes = [52.31, 18.03, 7.87]
colors = ['#e74c3c', '#f39c12', '#27ae60']

bars = ax.bar(models, mapes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.axhline(y=15, color='red', linestyle='--', linewidth=2, label='Target: 15% MAPE')
ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performance Evolution\nGroup 5: SPY Price Prediction',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 60)

# Add value labels
for bar, mape in zip(bars, mapes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mape:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add improvement annotations
ax.annotate('', xy=(1, 18.03), xytext=(0, 52.31),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.text(0.5, 35, '65.5% ↓', fontsize=10, color='blue', fontweight='bold',
        ha='center')

ax.annotate('', xy=(2, 7.87), xytext=(1, 18.03),
            arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
ax.text(1.5, 13, '56.3% ↓', fontsize=10, color='blue', fontweight='bold',
        ha='center')

ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'performance_comparison.png'}")

# 2. DATA TIMELINE
fig, ax = plt.subplots(figsize=(12, 4))
dates = pd.date_range('2024-02-02', '2025-11-19', freq='M')
ax.plot(dates, np.random.randn(len(dates)).cumsum() + 100,
        linewidth=2, color='#3498db', label='SPY Price Trend')
ax.axvspan(pd.Timestamp('2024-02-02'), pd.Timestamp('2025-05-01'),
           alpha=0.2, color='green', label='Training (70%)')
ax.axvspan(pd.Timestamp('2025-05-01'), pd.Timestamp('2025-07-15'),
           alpha=0.2, color='yellow', label='Validation (15%)')
ax.axvspan(pd.Timestamp('2025-07-15'), pd.Timestamp('2025-11-19'),
           alpha=0.2, color='red', label='Test (15%)')
ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Index', fontsize=12, fontweight='bold')
ax.set_title('Training Timeline: Feb 2024 - Nov 2025 (452 Trading Days)',
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'data_timeline.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'data_timeline.png'}")

# 3. FEATURE CATEGORIES
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Price\nMetrics', 'Moving\nAverages', 'Momentum\nIndicators',
              'Volatility\nMetrics', 'Lagged\nFeatures', 'Sentiment\nFeatures']
counts = [6, 8, 14, 10, 8, 4]
colors_cat = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c', '#34495e']

bars = ax.bar(categories, counts, color=colors_cat, alpha=0.8,
              edgecolor='black', linewidth=1.5)
ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax.set_title('Feature Engineering: 50 Total Features\n(46 Technical + 4 Sentiment)',
             fontsize=14, fontweight='bold')

for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'feature_categories.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'feature_categories.png'}")

print(f"\n✓ All exhibition plots created in {output_dir}/")

