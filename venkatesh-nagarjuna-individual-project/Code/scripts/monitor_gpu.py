"""
Monitor GPU utilization during training.
Useful for optimizing batch size and hyperparameters for A10G GPU.
"""

import subprocess
import time
import pandas as pd
from datetime import datetime
import argparse


def get_gpu_stats():
    """Get current GPU statistics using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        stats = result.stdout.strip().split(', ')

        return {
            'timestamp': stats[0],
            'gpu_name': stats[1],
            'gpu_util': float(stats[2]),
            'mem_util': float(stats[3]),
            'mem_used': float(stats[4]),
            'mem_total': float(stats[5]),
            'temperature': float(stats[6]),
            'power_draw': float(stats[7])
        }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return None


def monitor_gpu(interval=1, duration=None, output_file=None):
    """
    Monitor GPU statistics.

    Args:
        interval: Sampling interval in seconds
        duration: Total duration in seconds (None for infinite)
        output_file: CSV file to save statistics
    """
    stats_list = []
    start_time = time.time()

    print(f"Monitoring GPU (interval={interval}s)...")
    print("Press Ctrl+C to stop\n")

    try:
        while True:
            stats = get_gpu_stats()

            if stats:
                stats['elapsed_time'] = time.time() - start_time
                stats_list.append(stats)

                # Print current stats
                print(f"[{stats['elapsed_time']:.1f}s] "
                      f"GPU: {stats['gpu_util']:.1f}% | "
                      f"MEM: {stats['mem_used']:.0f}/{stats['mem_total']:.0f}MB ({stats['mem_util']:.1f}%) | "
                      f"TEMP: {stats['temperature']:.1f}°C | "
                      f"POWER: {stats['power_draw']:.1f}W")

            time.sleep(interval)

            if duration and (time.time() - start_time) >= duration:
                break

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    # Save to CSV if requested
    if output_file and stats_list:
        df = pd.DataFrame(stats_list)
        df.to_csv(output_file, index=False)
        print(f"\nStatistics saved to {output_file}")

        # Print summary
        print("\nSummary:")
        print(f"  Average GPU utilization: {df['gpu_util'].mean():.1f}%")
        print(f"  Average memory usage: {df['mem_used'].mean():.0f}MB ({df['mem_util'].mean():.1f}%)")
        print(f"  Peak memory usage: {df['mem_used'].max():.0f}MB")
        print(f"  Average temperature: {df['temperature'].mean():.1f}°C")
        print(f"  Average power draw: {df['power_draw'].mean():.1f}W")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor GPU statistics')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds')
    parser.add_argument('--duration', type=float, default=None,
                        help='Monitoring duration in seconds (default: infinite)')
    parser.add_argument('--output', type=str, default='gpu_stats.csv',
                        help='Output CSV file')

    args = parser.parse_args()

    monitor_gpu(
        interval=args.interval,
        duration=args.duration,
        output_file=args.output
    )
