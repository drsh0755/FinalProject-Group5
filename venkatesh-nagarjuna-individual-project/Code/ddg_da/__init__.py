"""
DDG-DA (Data Distribution Generation for Predictable Concept Drift Adaptation) module.
Handles regime prediction and adaptive sampling for financial time series.
"""

from .distribution_predictor import (
    RegimeExtractor,
    RegimePredictor,
    DDGDADistributionPredictor
)

from .sampler import (
    RegimeSimilarityCalculator,
    DDGDASampler,
    DDGDADataAdapter,
    evaluate_regime_adaptation
)

__all__ = [
    'RegimeExtractor',
    'RegimePredictor',
    'DDGDADistributionPredictor',
    'RegimeSimilarityCalculator',
    'DDGDASampler',
    'DDGDADataAdapter',
    'evaluate_regime_adaptation'
]
