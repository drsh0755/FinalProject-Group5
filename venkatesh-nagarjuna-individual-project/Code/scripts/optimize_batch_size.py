"""
Automatically find optimal batch size for A10G GPU.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TFTClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_optimal_batch_size(
        num_features=20,
        num_tickers=5,
        sequence_length=60,
        hidden_dim=128,
        max_batch_size=512,
        memory_fraction=0.85
):
    """
    Binary search for optimal batch size.

    Args:
        num_features: Number of input features
        num_tickers: Number of tickers
        sequence_length: Sequence length
        hidden_dim: Model hidden dimension
        max_batch_size: Maximum batch size to try
        memory_fraction: Target memory utilization fraction

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return 32

    device = torch.device('cuda')

    # Create model
    model = TFTClassifier(
        num_features=num_features,
        num_tickers=num_tickers,
        hidden_dim=hidden_dim,
        sequence_length=sequence_length
    ).to(device)

    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    logger.info("Searching for optimal batch size...")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    low, high = 1, max_batch_size
    optimal_batch_size = low

    while low <= high:
        batch_size = (low + high) // 2

        logger.info(f"\nTrying batch_size={batch_size}...")

        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Create sample batch
            features = {
                'encoder_cont': torch.randn(batch_size, sequence_length, num_features, device=device),
                'ticker_id': torch.randint(0, num_tickers, (batch_size, 1), device=device)
            }
            targets = torch.randint(0, 2, (batch_size,), device=device)

            # Forward pass
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(features)
                loss = criterion(outputs['logits'], targets)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Check memory usage
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9

            utilization = mem_allocated / mem_total

            logger.info(f"  Memory: {mem_allocated:.2f}GB allocated, "
                        f"{mem_reserved:.2f}GB reserved, "
                        f"{utilization * 100:.1f}% utilization")

            # Success - try larger batch
            optimal_batch_size = batch_size

            if utilization < memory_fraction:
                low = batch_size + 1
            else:
                # Close to target, stop here
                break

            # Cleanup
            del features, targets, outputs, loss
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.warning(f"  OOM at batch_size={batch_size}")
                high = batch_size - 1
                torch.cuda.empty_cache()
            else:
                raise e

    # Apply safety margin
    final_batch_size = int(optimal_batch_size * 0.9)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Optimal batch size: {final_batch_size}")
    logger.info(f"{'=' * 60}")

    return final_batch_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-features', type=int, default=20)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--sequence-length', type=int, default=60)

    args = parser.parse_args()

    optimal_bs = find_optimal_batch_size(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        sequence_length=args.sequence_length
    )

    print(f"\nRecommended batch_size for your configuration: {optimal_bs}")
