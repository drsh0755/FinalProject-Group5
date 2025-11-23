#!/usr/bin/env python3
"""
Visualize LSTM training results
"""

import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    # Load results
    results_dir = Path(__file__).parent.parent / 'results'
    results_file = results_dir / 'lstm_training_results.json'
    
    if not results_file.exists():
        print(f"✗ Results file not found: {results_file}")
        sys.exit(1)
    
    with open(results_file) as f:
        results = json.load(f)
    
    history = results['training_history']
    test_metrics = results['test_metrics']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Training Results - Baseline Model', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0, 0]
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation MAE
    ax2 = axes[0, 1]
    ax2.plot(history['val_mae'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.set_title('Validation MAE Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Validation RMSE
    ax3 = axes[1, 0]
    ax3.plot(history['val_rmse'], linewidth=2, color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Validation RMSE Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Test Metrics Summary
    ax4 = axes[1, 1]
    metrics = ['MAE', 'RMSE', 'MAPE (%)']
    values = [test_metrics['mae'], test_metrics['rmse'], test_metrics['mape']]
    colors = ['green', 'orange', 'red']
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax4.set_ylabel('Value')
    ax4.set_title('Test Set Metrics')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    output_file = results_dir / 'figures' / 'lstm_training_results.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to: {output_file}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY - BASELINE MODEL")
    print("="*60)
    print(f"Best Epoch: {results['best_epoch'] + 1}/{results['config']['epochs']}")
    print(f"\nDataset:")
    print(f"  Sequence length: {results['config']['sequence_length']} days")
    print(f"  Training samples: ~19 sequences")
    print(f"\nTest Metrics:")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"\n⚠ Note: High MAPE indicates model needs improvement")
    print(f"  Possible causes:")
    print(f"  - Small dataset (only 28 sequences)")
    print(f"  - Need more historical data")
    print(f"  - May need different architecture")
    print("="*60)

if __name__ == "__main__":
    main()
