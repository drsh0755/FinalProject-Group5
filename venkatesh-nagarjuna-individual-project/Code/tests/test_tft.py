"""
Test script for Temporal Fusion Transformer.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from models import TFTClassifier, TFTModelWrapper, count_parameters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_tft_architecture():
    """Test TFT architecture and components."""

    logger.info("=" * 60)
    logger.info("Testing Temporal Fusion Transformer")
    logger.info("=" * 60)

    # Test configuration
    batch_size = 16
    seq_len = 60
    num_features = 15
    num_tickers = 5

    logger.info(f"\nTest configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Sequence length: {seq_len}")
    logger.info(f"  Number of features: {num_features}")
    logger.info(f"  Number of tickers: {num_tickers}")

    # Create model
    logger.info("\n1. Initializing TFT model...")
    model = TFTClassifier(
        num_features=num_features,
        num_tickers=num_tickers,
        hidden_dim=128,
        lstm_layers=2,
        num_heads=4,
        dropout=0.1,
        num_classes=2,
        sequence_length=seq_len
    )

    num_params = count_parameters(model)
    logger.info(f"Model initialized with {num_params:,} parameters")

    # Create sample data
    logger.info("\n2. Creating sample input...")
    features_dict = {
        'encoder_cont': torch.randn(batch_size, seq_len, num_features),
        'ticker_id': torch.randint(0, num_tickers, (batch_size, 1)),
        'static_features': torch.FloatTensor([])
    }

    # Forward pass
    logger.info("\n3. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(features_dict)

    logger.info(f"Output shapes:")
    logger.info(f"  Logits: {outputs['logits'].shape}")
    logger.info(f"  Attention weights: {outputs['attention_weights'].shape}")
    logger.info(f"  VSN weights: {outputs['variable_selection_weights'].shape}")

    # Test predictions
    logger.info("\n4. Testing predictions...")
    probs = torch.softmax(outputs['logits'], dim=-1)
    preds = torch.argmax(probs, dim=-1)

    logger.info(f"Sample predictions:")
    logger.info(f"  Probabilities: {probs[:3]}")
    logger.info(f"  Predicted classes: {preds[:3]}")

    # Test GPU if available
    if torch.cuda.is_available():
        logger.info("\n5. Testing GPU execution...")
        device = torch.device('cuda')
        model_gpu = model.to(device)

        features_dict_gpu = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in features_dict.items()
        }

        # Test mixed precision
        with torch.cuda.amp.autocast():
            outputs_gpu = model_gpu(features_dict_gpu)

        logger.info(f"GPU forward pass successful!")
        logger.info(f"  Output device: {outputs_gpu['logits'].device}")
        logger.info(f"  GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"  GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Benchmark speed
        logger.info("\n6. Benchmarking inference speed...")
        import time

        torch.cuda.synchronize()
        start = time.time()

        n_iterations = 100
        for _ in range(n_iterations):
            with torch.cuda.amp.autocast():
                _ = model_gpu(features_dict_gpu)

        torch.cuda.synchronize()
        elapsed = time.time() - start

        logger.info(f"  {n_iterations} iterations in {elapsed:.2f}s")
        logger.info(f"  Throughput: {batch_size * n_iterations / elapsed:.1f} samples/sec")
        logger.info(f"  Latency: {elapsed / n_iterations * 1000:.2f}ms per batch")

    # Test model wrapper
    logger.info("\n7. Testing model wrapper...")
    feature_cols = [f'feature_{i}' for i in range(num_features)]

    wrapper = TFTModelWrapper(
        model=model,
        feature_cols=feature_cols,
        config={
            'num_features': num_features,
            'num_tickers': num_tickers,
            'hidden_dim': 128,
            'lstm_layers': 2,
            'num_heads': 4,
            'num_classes': 2,
            'sequence_length': seq_len
        }
    )

    probs_wrapper = wrapper.predict_proba(features_dict)
    logger.info(f"Wrapper predictions shape: {probs_wrapper.shape}")

    # Test save/load
    logger.info("\n8. Testing model save/load...")
    save_dir = 'test_tft_model'
    wrapper.save(save_dir)

    loaded_wrapper = TFTModelWrapper.load(save_dir)
    probs_loaded = loaded_wrapper.predict_proba(features_dict)

    # Verify predictions match
    assert np.allclose(probs_wrapper, probs_loaded, atol=1e-5), "Loaded model predictions don't match!"
    logger.info("Model save/load successful and predictions match!")

    # Cleanup
    import shutil
    shutil.rmtree(save_dir)

    logger.info("\n" + "=" * 60)
    logger.info("✓ All TFT tests passed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    test_tft_architecture()
