"""
Create comprehensive sentiment features using all three methods:
1. Median (robust to outliers)
2. Weighted by confidence (emphasize extreme articles)
3. Capture extremes (min, max, extreme)

WITH COMPREHENSIVE LOGGING
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import setup_logger, log_section, log_dict

def create_comprehensive_sentiment_features():
    """
    Generate 9 sentiment features from raw news data
    """
    
    # Setup logger
    logger, log_file = setup_logger('comprehensive_sentiment_features', log_dir='Exhibition/logs')
    
    log_section(logger, "COMPREHENSIVE SENTIMENT FEATURE CREATION")
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    try:
        # Load data
        log_section(logger, "LOADING DATA")
        logger.info("Loading news sentiment data...")
        news = pd.read_csv('Code/data/processed/news_sentiment.csv', parse_dates=['date'])
        logger.info(f"  ✓ Loaded {len(news):,} news articles")
        logger.info(f"  Date range: {news['date'].min()} to {news['date'].max()}")
        
        logger.info("Loading stock features...")
        stock = pd.read_csv('Code/data/processed/spy_features_2year.csv', parse_dates=['Date'])
        logger.info(f"  ✓ Loaded {len(stock)} days of stock data")
        logger.info(f"  Date range: {stock['Date'].min()} to {stock['Date'].max()}")
        logger.info("")
        
        # Prepare
        log_section(logger, "DATA PREPARATION")
        news['date_only'] = news['date'].dt.date
        news['abs_sentiment'] = news['sentiment_score'].abs()
        
        logger.info(f"Prepared news data:")
        logger.info(f"  Articles per day (avg): {news.groupby('date_only').size().mean():.1f}")
        logger.info(f"  Raw sentiment std: {news['sentiment_score'].std():.4f}")
        logger.info("")
        
        # Aggregate with ALL methods
        log_section(logger, "COMPUTING SENTIMENT FEATURES")
        logger.info("Computing basic aggregations (mean, median, std, min, max, count)...")
        
        daily_agg = news.groupby('date_only').agg({
            'sentiment_score': [
                'mean',    # Method 0: Original mean
                'median',  # Method 1: Median (robust)
                'std',     # Volatility
                'min',     # Method 3a: Most negative
                'max',     # Method 3b: Most positive
                'count'    # Article count
            ]
        })
        
        # Flatten column names
        daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns.values]
        daily_agg = daily_agg.reset_index()
        
        # Rename for clarity
        daily_agg.columns = [
            'date',
            'sentiment_mean',
            'sentiment_median',
            'sentiment_std',
            'sentiment_min',
            'sentiment_max',
            'article_count'
        ]
        
        logger.info(f"  ✓ Computed basic features for {len(daily_agg)} days")
        
        # Method 2: Weighted by confidence
        logger.info("Computing weighted sentiment (Method 2: Confidence weighting)...")
        
        weighted_sentiment = []
        for date_val in daily_agg['date']:
            day_articles = news[news['date_only'] == date_val]
            
            if len(day_articles) > 0:
                sentiments = day_articles['sentiment_score'].values
                weights = day_articles['abs_sentiment'].values
                
                if weights.sum() > 0:
                    weighted = (sentiments * weights).sum() / weights.sum()
                else:
                    weighted = sentiments.mean()
            else:
                weighted = 0.0
            
            weighted_sentiment.append(weighted)
        
        daily_agg['sentiment_weighted'] = weighted_sentiment
        logger.info(f"  ✓ Computed weighted sentiment")
        logger.info(f"    Formula: Σ(sentiment_i × |sentiment_i|) / Σ|sentiment_i|")
        
        # Method 3: Extreme sentiment (max absolute)
        logger.info("Computing extreme sentiment (Method 3: Capture extremes)...")
        
        extreme_sentiment = []
        for date_val in daily_agg['date']:
            row = daily_agg[daily_agg['date'] == date_val].iloc[0]
            
            # Choose the sentiment with larger absolute value
            if abs(row['sentiment_max']) > abs(row['sentiment_min']):
                extreme = row['sentiment_max']
            else:
                extreme = row['sentiment_min']
            
            extreme_sentiment.append(extreme)
        
        daily_agg['sentiment_extreme'] = extreme_sentiment
        logger.info(f"  ✓ Computed extreme sentiment")
        logger.info(f"    Formula: argmax(|sentiment_min|, |sentiment_max|)")
        
        # Positive ratio
        logger.info("Computing positive ratio...")
        
        positive_counts = news[news['sentiment_score'] > 0.1].groupby('date_only').size()
        daily_agg['positive_ratio'] = positive_counts / daily_agg['article_count']
        daily_agg['positive_ratio'] = daily_agg['positive_ratio'].fillna(0)
        
        logger.info(f"  ✓ Computed positive ratio")
        logger.info("")
        
        # Show statistics
        log_section(logger, "FEATURE STATISTICS")
        
        features = [
            'sentiment_mean',
            'sentiment_median',
            'sentiment_weighted',
            'sentiment_extreme',
            'sentiment_min',
            'sentiment_max',
            'sentiment_std'
        ]
        
        logger.info("Standard deviation of each feature:")
        for feat in features:
            std = daily_agg[feat].std()
            logger.info(f"  {feat:<25} std = {std:.4f}")
        
        logger.info("")
        
        # Signal preservation
        raw_std = news['sentiment_score'].std()
        
        log_section(logger, "SIGNAL PRESERVATION ANALYSIS")
        logger.info(f"Raw article sentiment std: {raw_std:.4f}")
        logger.info("")
        logger.info("Signal preservation by method:")
        
        for feat in features:
            feat_std = daily_agg[feat].std()
            preservation = (feat_std / raw_std) * 100
            logger.info(f"  {feat:<25} {preservation:>5.1f}% of raw signal")
        
        logger.info("")
        logger.info("Key findings:")
        best_feat = max(features, key=lambda f: daily_agg[f].std())
        worst_feat = min(features, key=lambda f: daily_agg[f].std())
        logger.info(f"  Best preserved:  {best_feat} ({daily_agg[best_feat].std():.4f})")
        logger.info(f"  Worst preserved: {worst_feat} ({daily_agg[worst_feat].std():.4f})")
        logger.info(f"  Improvement: {daily_agg[best_feat].std() / daily_agg[worst_feat].std():.1f}x better")
        logger.info("")
        
        # Merge with stock data
        log_section(logger, "MERGING WITH STOCK DATA")
        logger.info("Merging sentiment features with stock technical indicators...")
        
        daily_agg['date'] = pd.to_datetime(daily_agg['date'])
        
        merged = stock.merge(
            daily_agg,
            left_on=stock['Date'].dt.date,
            right_on=daily_agg['date'].dt.date,
            how='left'
        )
        
        logger.info(f"  ✓ Initial merge: {len(merged)} days")
        
        # Forward fill missing days
        sentiment_cols = [
            'sentiment_mean',
            'sentiment_median',
            'sentiment_weighted',
            'sentiment_extreme',
            'sentiment_min',
            'sentiment_max',
            'sentiment_std',
            'article_count',
            'positive_ratio'
        ]
        
        missing_before = merged[sentiment_cols].isnull().sum().sum()
        logger.info(f"  Missing values before forward fill: {missing_before}")
        
        for col in sentiment_cols:
            merged[col] = merged[col].ffill().fillna(0)
        
        missing_after = merged[sentiment_cols].isnull().sum().sum()
        logger.info(f"  Missing values after forward fill: {missing_after}")
        
        merged = merged.drop(['key_0', 'date'], axis=1, errors='ignore')
        
        logger.info(f"  ✓ Final merged data: {len(merged)} days")
        logger.info("")
        
        # Show sample
        log_section(logger, "SAMPLE DATA")
        logger.info("First 3 days of merged data:")
        sample_cols = ['Date', 'close'] + sentiment_cols[:5]  # Show first 5 sentiment features
        for idx, row in merged[sample_cols].head(3).iterrows():
            logger.info(f"  {row['Date'].date()}: close=${row['close']:.2f}, "
                       f"mean={row['sentiment_mean']:.3f}, "
                       f"extreme={row['sentiment_extreme']:.3f}")
        logger.info("")
        
        # Save
        log_section(logger, "SAVING RESULTS")
        output = 'Code/data/processed/spy_features_with_comprehensive_sentiment.csv'
        merged.to_csv(output, index=False)
        
        logger.info(f"✅ COMPLETE!")
        logger.info(f"   Output file: {output}")
        logger.info(f"   File size: {Path(output).stat().st_size / 1024 / 1024:.1f} MB")
        logger.info(f"   Total columns: {len(merged.columns)}")
        logger.info(f"   Sentiment features: 9 (was 4)")
        logger.info(f"   Technical features: 42")
        logger.info(f"   Total: 51 features")
        logger.info("")
        
        # Summary
        log_section(logger, "SUMMARY")
        logger.info("Features created:")
        for i, col in enumerate(sentiment_cols, 1):
            logger.info(f"  {i}. {col}")
        logger.info("")
        logger.info("Next step: Train model with comprehensive features")
        logger.info("  python3 Code/scripts/train_lstm_with_comprehensive_sentiment.py")
        logger.info("")
        
        return merged
        
    except Exception as e:
        logger.error(f"❌ FAILED: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    create_comprehensive_sentiment_features()
