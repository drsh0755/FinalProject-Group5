"""
Analyze comprehensive sentiment model results
Compare with previous models and assess sentiment impact
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, log_section

def analyze_comprehensive_results():
    """
    Comprehensive analysis of sentiment feature improvements
    """
    
    # Setup logger
    logger, log_file = setup_logger('analyze_comprehensive_results', log_dir='Exhibition/logs')
    
    log_section(logger, "COMPREHENSIVE SENTIMENT ANALYSIS")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    try:
        # Load all model results
        log_section(logger, "LOADING MODEL RESULTS")
        
        logger.info("Loading baseline model results...")
        with open('Code/results/lstm_training_results.json', 'r') as f:
            baseline = json.load(f)
        logger.info(f"  ✓ Baseline: {baseline['test_metrics']['mape']:.2f}% MAPE")
        
        logger.info("Loading 4-sentiment-feature model results...")
        with open('Code/results/lstm_with_sentiment_results.json', 'r') as f:
            sentiment_4 = json.load(f)
        logger.info(f"  ✓ 4 sentiment features: {sentiment_4['test_metrics']['mape']:.2f}% MAPE")
        
        logger.info("Loading 9-sentiment-feature model results...")
        with open('Code/results/lstm_comprehensive_sentiment_results.json', 'r') as f:
            sentiment_9 = json.load(f)
        logger.info(f"  ✓ 9 sentiment features: {sentiment_9['test_metrics']['mape']:.2f}% MAPE")
        logger.info("")
        
        # Comparison table
        log_section(logger, "MODEL COMPARISON")
        
        models = {
            'Baseline (No Sentiment)': {
                'mape': baseline['test_metrics']['mape'],
                'features': 42,
                'sentiment': 0
            },
            'Simple Sentiment': {
                'mape': sentiment_4['test_metrics']['mape'],
                'features': 46,
                'sentiment': 4
            },
            'Comprehensive Sentiment': {
                'mape': sentiment_9['test_metrics']['mape'],
                'features': 51,
                'sentiment': 9
            }
        }
        
        logger.info("Performance Summary:")
        logger.info("-" * 70)
        logger.info(f"{'Model':<30} {'MAPE':<10} {'Features':<12} {'Sentiment'}")
        logger.info("-" * 70)
        
        for name, data in models.items():
            logger.info(f"{name:<30} {data['mape']:>6.2f}%   "
                       f"{data['features']:>3} total    {data['sentiment']:>2} sentiment")
        
        logger.info("-" * 70)
        logger.info("")
        
        # Improvement analysis
        log_section(logger, "IMPROVEMENT ANALYSIS")
        
        baseline_mape = baseline['test_metrics']['mape']
        sentiment4_mape = sentiment_4['test_metrics']['mape']
        sentiment9_mape = sentiment_9['test_metrics']['mape']
        
        improvement_4 = baseline_mape - sentiment4_mape
        improvement_9_from_baseline = baseline_mape - sentiment9_mape
        improvement_9_from_4 = sentiment4_mape - sentiment9_mape
        
        logger.info("Improvements:")
        logger.info(f"  Baseline → 4 sentiment features:")
        logger.info(f"    Absolute: {improvement_4:.2f} percentage points")
        logger.info(f"    Relative: {(improvement_4/baseline_mape)*100:.1f}% better")
        logger.info("")
        
        logger.info(f"  Baseline → 9 sentiment features:")
        logger.info(f"    Absolute: {improvement_9_from_baseline:.2f} percentage points")
        logger.info(f"    Relative: {(improvement_9_from_baseline/baseline_mape)*100:.1f}% better")
        logger.info("")
        
        logger.info(f"  4 features → 9 features:")
        logger.info(f"    Absolute: {improvement_9_from_4:.2f} percentage points")
        
        if improvement_9_from_4 > 0:
            logger.info(f"    Relative: {(improvement_9_from_4/sentiment4_mape)*100:.1f}% better ✅")
            logger.info(f"    Status: Comprehensive features provide additional value")
        elif improvement_9_from_4 < -0.1:
            logger.info(f"    Relative: {abs(improvement_9_from_4/sentiment4_mape)*100:.1f}% worse ⚠️")
            logger.info(f"    Status: More features added complexity without benefit")
        else:
            logger.info(f"    Relative: Negligible change (~0%)")
            logger.info(f"    Status: Similar performance, but richer feature set")
        
        logger.info("")
        
        # Feature contribution analysis
        log_section(logger, "FEATURE CONTRIBUTION ANALYSIS")
        
        # Calculate marginal contribution
        total_improvement = baseline_mape - sentiment9_mape
        
        logger.info("Approximate feature contributions:")
        logger.info(f"  Total improvement: {total_improvement:.2f} percentage points")
        logger.info("")
        
        # Assume improvement came mainly from hyperparameters and 4 sentiment features
        hyperparameter_improvement = baseline_mape - 18.03  # From your analysis
        sentiment_4_contribution = 18.03 - sentiment4_mape
        sentiment_9_additional = sentiment4_mape - sentiment9_mape if sentiment9_mape < sentiment4_mape else 0
        
        logger.info(f"  Hyperparameter tuning:  ~{hyperparameter_improvement:.2f} pp ({(hyperparameter_improvement/total_improvement)*100:.1f}%)")
        logger.info(f"  4 sentiment features:   ~{sentiment_4_contribution:.2f} pp ({(sentiment_4_contribution/total_improvement)*100:.1f}%)")
        
        if sentiment_9_additional > 0:
            logger.info(f"  Additional 5 features:  ~{sentiment_9_additional:.2f} pp ({(sentiment_9_additional/total_improvement)*100:.1f}%)")
        
        logger.info("")
        
        # Sentiment feature effectiveness
        log_section(logger, "SENTIMENT FEATURE EFFECTIVENESS")
        
        # Load feature data to check variance
        df = pd.read_csv('Code/data/processed/spy_features_with_comprehensive_sentiment.csv')
        
        sentiment_features = [
            'sentiment_mean',
            'sentiment_median',
            'sentiment_weighted',
            'sentiment_extreme',
            'sentiment_min',
            'sentiment_max',
            'sentiment_std'
        ]
        
        logger.info("Sentiment feature statistics:")
        logger.info("-" * 70)
        logger.info(f"{'Feature':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max'}")
        logger.info("-" * 70)
        
        for feat in sentiment_features:
            if feat in df.columns:
                mean = df[feat].mean()
                std = df[feat].std()
                min_val = df[feat].min()
                max_val = df[feat].max()
                logger.info(f"{feat:<25} {mean:>8.4f}  {std:>8.4f}  {min_val:>8.4f}  {max_val:>8.4f}")
        
        logger.info("-" * 70)
        logger.info("")
        
        # Key findings
        log_section(logger, "KEY FINDINGS")
        
        if sentiment9_mape < sentiment4_mape:
            logger.info("✅ SUCCESS: Comprehensive sentiment features improved performance")
            logger.info(f"   The additional 5 features provided {improvement_9_from_4:.2f} pp improvement")
            logger.info("")
            logger.info("Likely reasons:")
            logger.info("  - Extreme sentiment (min/max) captures market-moving events")
            logger.info("  - Weighted sentiment emphasizes high-confidence articles")
            logger.info("  - Multiple aggregation methods provide robust signal")
        else:
            logger.info("⚠️  LIMITED IMPACT: Comprehensive features didn't improve over simple")
            logger.info(f"   Performance similar or slightly worse ({improvement_9_from_4:.2f} pp change)")
            logger.info("")
            logger.info("Likely reasons:")
            logger.info("  - Technical indicators still dominate (42 vs 9 features)")
            logger.info("  - SPY is less sensitive to news than individual stocks")
            logger.info("  - Additional features may have added noise")
            logger.info("")
            logger.info("Value of comprehensive features:")
            logger.info("  - Better understanding of sentiment aggregation methods")
            logger.info("  - Richer feature set for future analysis")
            logger.info("  - Framework for individual stock prediction")
        
        logger.info("")
        
        # Recommendations
        log_section(logger, "RECOMMENDATIONS")
        
        logger.info("Based on this analysis:")
        logger.info("")
        
        if sentiment9_mape < sentiment4_mape - 0.5:
            logger.info("1. ✅ USE comprehensive sentiment model (9 features)")
            logger.info("   Benefit: Best performance with richer feature representation")
        elif sentiment9_mape < sentiment4_mape:
            logger.info("1. ⚖️  EITHER model works well")
            logger.info("   Simple (4 features): Easier to explain, similar performance")
            logger.info("   Comprehensive (9 features): Better for future analysis")
        else:
            logger.info("1. ✅ USE simple sentiment model (4 features)")
            logger.info("   Benefit: Simpler, equal or better performance")
        
        logger.info("")
        logger.info("2. 📊 For presentation:")
        logger.info("   - Show the journey: Baseline → Simple → Comprehensive")
        logger.info("   - Explain sentiment aggregation challenge (low variance)")
        logger.info("   - Demonstrate three aggregation methods (median, weighted, extreme)")
        logger.info("   - Discuss lessons learned about feature engineering")
        
        logger.info("")
        logger.info("3. 🔮 Future work:")
        logger.info("   - Apply comprehensive features to individual stocks")
        logger.info("   - Test during high-volatility periods (2020 COVID, 2022 inflation)")
        logger.info("   - Add event-driven features (earnings, Fed announcements)")
        logger.info("   - Incorporate intraday sentiment shifts")
        
        logger.info("")
        
        # Save analysis
        log_section(logger, "SAVING ANALYSIS")
        
        analysis_results = {
            'models': models,
            'improvements': {
                'baseline_to_4_features': improvement_4,
                'baseline_to_9_features': improvement_9_from_baseline,
                '4_to_9_features': improvement_9_from_4
            },
            'best_model': min(models.items(), key=lambda x: x[1]['mape'])[0],
            'best_mape': min(m['mape'] for m in models.values()),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        output_file = 'Exhibition/comprehensive_sentiment_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"✓ Analysis saved: {output_file}")
        logger.info(f"✓ Log saved: {log_file}")
        logger.info("")
        
        log_section(logger, "ANALYSIS COMPLETE")
        logger.info("Review the log file for detailed findings")
        logger.info("Use this analysis for your presentation and report")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"❌ ANALYSIS FAILED: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    analyze_comprehensive_results()
