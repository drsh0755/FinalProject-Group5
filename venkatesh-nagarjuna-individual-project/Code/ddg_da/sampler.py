"""
DDG-DA Sampler: Reweights and resamples training data based on predicted regimes.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from typing import List, Dict, Optional
import logging
from scipy.spatial.distance import euclidean, cosine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RegimeSimilarityCalculator:
    """Calculate similarity between regimes for sample weighting."""

    @staticmethod
    def compute_similarity(regime1: np.ndarray, regime2: np.ndarray,
                           metric: str = 'euclidean') -> float:
        """
        Compute similarity between two regime feature vectors.

        Args:
            regime1: First regime feature vector
            regime2: Second regime feature vector
            metric: Similarity metric ('euclidean', 'cosine', 'rbf')

        Returns:
            Similarity score (higher = more similar)
        """
        if metric == 'euclidean':
            dist = euclidean(regime1, regime2)
            similarity = 1.0 / (1.0 + dist)
        elif metric == 'cosine':
            similarity = 1.0 - cosine(regime1, regime2)
        elif metric == 'rbf':
            dist = euclidean(regime1, regime2)
            gamma = 1.0 / len(regime1)
            similarity = np.exp(-gamma * dist ** 2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return similarity

    @staticmethod
    def compute_regime_weights(
            historical_regimes: List[np.ndarray],
            target_regime: np.ndarray,
            metric: str = 'rbf',
            temperature: float = 1.0
    ) -> np.ndarray:
        """
        Compute weights for historical regimes based on similarity to target.

        Args:
            historical_regimes: List of historical regime feature vectors
            target_regime: Target regime to match
            metric: Similarity metric
            temperature: Temperature for softmax normalization

        Returns:
            Array of weights (sums to 1)
        """
        similarities = np.array([
            RegimeSimilarityCalculator.compute_similarity(hist, target_regime, metric)
            for hist in historical_regimes
        ])

        # Apply temperature and softmax
        similarities = similarities / temperature
        weights = np.exp(similarities - np.max(similarities))
        weights = weights / weights.sum()

        return weights


class DDGDASampler(Sampler):
    """
    PyTorch Sampler that reweights samples based on regime similarity.
    Compatible with DataLoader.
    """

    def __init__(
            self,
            dataset,
            regime_weights: np.ndarray,
            sample_indices_per_regime: List[List[int]],
            num_samples: Optional[int] = None,
            replacement: bool = True
    ):
        """
        Initialize DDG-DA sampler.

        Args:
            dataset: PyTorch dataset
            regime_weights: Weight for each regime
            sample_indices_per_regime: List of sample indices for each regime
            num_samples: Number of samples to draw per epoch
            replacement: Whether to sample with replacement
        """
        super().__init__(dataset)

        self.dataset = dataset
        self.regime_weights = regime_weights
        self.sample_indices_per_regime = sample_indices_per_regime
        self.replacement = replacement

        # Compute per-sample weights
        self.sample_weights = self._compute_sample_weights()

        # Number of samples per epoch
        self.num_samples = num_samples or len(dataset)

        logger.info(f"DDG-DA Sampler initialized with {len(regime_weights)} regimes")
        logger.info(f"Regime weights: {regime_weights[:5]}... (showing first 5)")

    def _compute_sample_weights(self) -> np.ndarray:
        """Compute weight for each sample based on regime assignment."""
        sample_weights = np.zeros(len(self.dataset))

        for regime_idx, indices in enumerate(self.sample_indices_per_regime):
            if regime_idx < len(self.regime_weights):
                regime_weight = self.regime_weights[regime_idx]
                for idx in indices:
                    if idx < len(sample_weights):
                        sample_weights[idx] = regime_weight

        # Normalize
        if sample_weights.sum() > 0:
            sample_weights = sample_weights / sample_weights.sum()
        else:
            sample_weights = np.ones(len(self.dataset)) / len(self.dataset)

        return sample_weights

    def __iter__(self):
        """Generate sample indices."""
        if self.replacement:
            # Sample with replacement according to weights
            indices = np.random.choice(
                len(self.dataset),
                size=self.num_samples,
                replace=True,
                p=self.sample_weights
            )
        else:
            # Weighted sampling without replacement (approximate)
            indices = np.random.choice(
                len(self.dataset),
                size=min(self.num_samples, len(self.dataset)),
                replace=False,
                p=self.sample_weights
            )

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


class DDGDADataAdapter:
    """High-level interface for DDG-DA data adaptation."""

    def __init__(
            self,
            predictor,
            similarity_metric: str = 'rbf',
            temperature: float = 2.0,
            top_k_regimes: Optional[int] = None
    ):
        """
        Initialize data adapter.

        Args:
            predictor: DDGDADistributionPredictor instance
            similarity_metric: Metric for regime similarity
            temperature: Temperature for weight normalization
            top_k_regimes: Only use top K most similar regimes (None = use all)
        """
        self.predictor = predictor
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        self.top_k_regimes = top_k_regimes

        self.regime_weights = None
        self.sample_indices_per_regime = None

    def adapt_dataset(
            self,
            dataset,
            predicted_regime: Optional[np.ndarray] = None
    ):
        """
        Adapt dataset to predicted future regime.

        Args:
            dataset: PyTorch dataset with sequences
            predicted_regime: Predicted regime features (auto-predict if None)

        Returns:
            DDGDASampler for adapted sampling
        """
        # Predict future regime if not provided
        if predicted_regime is None:
            logger.info("Predicting future regime...")
            predicted_regime = self.predictor.predict_future_regime()

        # Get historical regimes
        if not self.predictor.regimes:
            raise ValueError("No historical regimes available")

        historical_regime_features = [
            self.predictor.regime_extractor.transform_regime(r)
            for r in self.predictor.regimes
        ]

        # Compute regime weights based on similarity to predicted regime
        logger.info("Computing regime similarities...")
        self.regime_weights = RegimeSimilarityCalculator.compute_regime_weights(
            historical_regime_features,
            predicted_regime,
            metric=self.similarity_metric,
            temperature=self.temperature
        )

        # Filter to top-K regimes if specified
        if self.top_k_regimes:
            top_k_indices = np.argsort(self.regime_weights)[-self.top_k_regimes:]
            mask = np.zeros_like(self.regime_weights)
            mask[top_k_indices] = self.regime_weights[top_k_indices]
            self.regime_weights = mask / mask.sum()
            logger.info(f"Using top {self.top_k_regimes} regimes")

        # Map dataset samples to regimes
        logger.info("Mapping samples to regimes...")
        self.sample_indices_per_regime = self._map_samples_to_regimes(dataset)

        # Create sampler
        sampler = DDGDASampler(
            dataset,
            self.regime_weights,
            self.sample_indices_per_regime,
            num_samples=len(dataset),
            replacement=True
        )

        # Log statistics
        effective_regimes = (self.regime_weights > 0.01).sum()
        max_weight_regime = np.argmax(self.regime_weights)
        logger.info(f"Effective regimes: {effective_regimes}/{len(self.regime_weights)}")
        logger.info(f"Max weight regime: {max_weight_regime} (weight={self.regime_weights[max_weight_regime]:.4f})")

        return sampler

    def _map_samples_to_regimes(self, dataset) -> List[List[int]]:
        """
        Map dataset samples to regime windows based on timestamps.

        Args:
            dataset: Dataset with sequences

        Returns:
            List of sample indices per regime
        """
        sample_indices_per_regime = [[] for _ in self.predictor.regimes]

        # Get date information from dataset sequences
        for sample_idx in range(len(dataset)):
            seq_info = dataset.sequences[sample_idx]

            # Get date range for this sample
            sample_data = seq_info['data']
            sample_start_date = sample_data.iloc[seq_info['start_idx']]['Date']
            sample_end_date = sample_data.iloc[seq_info['end_idx'] - 1]['Date']

            # Find which regime this sample belongs to
            for regime_idx, regime in enumerate(self.predictor.regimes):
                regime_start = regime['start_date']
                regime_end = regime['end_date']

                # Check if sample overlaps with regime
                if sample_start_date <= regime_end and sample_end_date >= regime_start:
                    sample_indices_per_regime[regime_idx].append(sample_idx)
                    break  # Assign to first matching regime

        # Log regime sizes
        regime_sizes = [len(indices) for indices in sample_indices_per_regime]
        logger.info(f"Samples per regime: min={min(regime_sizes)}, max={max(regime_sizes)}, "
                    f"mean={np.mean(regime_sizes):.1f}")

        return sample_indices_per_regime

    def get_adapted_dataloader(
            self,
            dataset,
            batch_size: int = 64,
            num_workers: int = 4,
            **kwargs
    ):
        """
        Create adapted DataLoader with DDG-DA sampling.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            num_workers: Number of workers
            **kwargs: Additional DataLoader arguments

        Returns:
            DataLoader with DDG-DA sampler
        """
        from torch.utils.data import DataLoader
        from data.dataset import collate_fn

        # Get adapted sampler
        sampler = self.adapt_dataset(dataset)

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_fn,
            **kwargs
        )

        logger.info(f"Created DDG-DA adapted DataLoader with {len(loader)} batches")
        return loader


def evaluate_regime_adaptation(
        original_loader,
        adapted_loader,
        num_batches: int = 100
) -> Dict[str, float]:
    """
    Evaluate the effect of regime adaptation on data distribution.

    Args:
        original_loader: Original DataLoader
        adapted_loader: DDG-DA adapted DataLoader
        num_batches: Number of batches to analyze

    Returns:
        Dictionary of statistics
    """
    logger.info("Evaluating regime adaptation effects...")

    def collect_statistics(loader, n_batches):
        feature_means = []
        feature_stds = []
        target_dist = []

        for i, (features, targets) in enumerate(loader):
            if i >= n_batches:
                break

            encoder_features = features['encoder_cont']  # [B, L, F]
            feature_means.append(encoder_features.mean(dim=[0, 1]).numpy())
            feature_stds.append(encoder_features.std(dim=[0, 1]).numpy())
            target_dist.append(targets.float().mean().item())

        return {
            'mean_features': np.mean(feature_means, axis=0),
            'std_features': np.mean(feature_stds, axis=0),
            'target_positive_ratio': np.mean(target_dist)
        }

    original_stats = collect_statistics(original_loader, num_batches)
    adapted_stats = collect_statistics(adapted_loader, num_batches)

    # Compute differences
    mean_shift = np.linalg.norm(adapted_stats['mean_features'] - original_stats['mean_features'])
    std_shift = np.linalg.norm(adapted_stats['std_features'] - original_stats['std_features'])
    target_shift = abs(adapted_stats['target_positive_ratio'] - original_stats['target_positive_ratio'])

    results = {
        'mean_feature_shift': mean_shift,
        'std_feature_shift': std_shift,
        'target_distribution_shift': target_shift,
        'original_target_ratio': original_stats['target_positive_ratio'],
        'adapted_target_ratio': adapted_stats['target_positive_ratio']
    }

    logger.info(f"Adaptation results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.6f}")

    return results


if __name__ == "__main__":
    # Example usage
    from ddg_da.distribution_predictor import DDGDADistributionPredictor
    import pandas as pd

    # Create synthetic data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    data_list = []

    for ticker in ['AAPL', 'MSFT']:
        n = len(dates)
        df = pd.DataFrame({
            'Date': dates,
            'Ticker': ticker,
            'Close': 100 + np.cumsum(np.random.randn(n)),
            'Volume': np.random.randint(1e6, 1e8, n),
            'sentiment_score': np.random.uniform(-0.5, 0.5, n),
            'rsi': np.random.uniform(30, 70, n),
            'volatility_10d': np.random.uniform(0.01, 0.03, n)
        })
        data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    # Train predictor
    predictor = DDGDADistributionPredictor()
    predictor.train(data, epochs=20, batch_size=16)

    # Create adapter
    adapter = DDGDADataAdapter(predictor, temperature=2.0)

    print("DDG-DA Sampler initialized successfully")
