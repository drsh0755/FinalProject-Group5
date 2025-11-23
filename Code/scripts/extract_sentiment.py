#!/usr/bin/env python3
"""
Extract sentiment from financial news using FinBERT
WITH DETAILED LOGGING AND PROGRESS TRACKING
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings
import logging
from datetime import datetime
import sys
warnings.filterwarnings('ignore')

# Setup logging
def setup_logging():
    """Setup detailed logging to file and console"""
    log_dir = Path(__file__).parent.parent / 'results' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'sentiment_extraction_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('sentiment_extraction')
    logger.setLevel(logging.INFO)
    
    # File handler - detailed
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    
    # Console handler - less verbose
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, log_file

def load_finbert(logger):
    """Load pre-trained FinBERT model"""
    logger.info("Loading FinBERT model...")
    model_name = "ProsusAI/finbert"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    logger.info(f"✓ FinBERT loaded on {device}")
    return tokenizer, model, device

def get_sentiment(text, tokenizer, model, device):
    """Get sentiment score for text"""
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        pos_score = probs[0][0].item()
        neg_score = probs[0][1].item()
        neu_score = probs[0][2].item()
        
        sentiment_score = pos_score - neg_score
        
        return {
            'sentiment_score': sentiment_score,
            'positive': pos_score,
            'negative': neg_score,
            'neutral': neu_score
        }
        
    except Exception as e:
        return {
            'sentiment_score': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0
        }

def main():
    # Setup logging
    logger, log_file = setup_logging()
    
    logger.info("="*60)
    logger.info("FINBERT SENTIMENT EXTRACTION - WITH LOGGING")
    logger.info("="*60)
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    # Load model
    tokenizer, model, device = load_finbert(logger)
    
    news_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'news'
    
    logger.info("\nLooking for news files...")
    news_files = list(news_dir.glob('*.csv'))
    
    if not news_files:
        logger.error("✗ No news CSV files found!")
        logger.error(f"  Expected location: {news_dir}")
        return
    
    logger.info(f"✓ Found {len(news_files)} news file(s)")
    
    all_sentiments = []
    total_processed = 0
    file_stats = {}
    
    for file_idx, news_file in enumerate(news_files, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file {file_idx}/{len(news_files)}: {news_file.name}")
        logger.info(f"{'='*60}")
        
        file_start_time = datetime.now()
        
        try:
            # Load file
            logger.info("  Loading CSV...")
            df = pd.read_csv(news_file)
            logger.info(f"  ✓ Loaded {len(df):,} articles")
            
            # Find text column
            text_cols = ['text', 'content', 'article', 'body', 'headline', 'title']
            text_col = None
            for col in text_cols:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                logger.warning(f"  ⚠ No text column found in {df.columns.tolist()}")
                logger.warning("  Skipping this file")
                continue
            
            logger.info(f"  ✓ Using text column: '{text_col}'")
            
            # Find date column
            date_cols = ['date', 'published', 'publish_date', 'timestamp']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                logger.info(f"  ✓ Using date column: '{date_col}'")
                
                # Parse dates with robust handling
                logger.info("  Parsing dates...")
                try:
                    # First try: parse with UTC
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
                    df[date_col] = df[date_col].dt.tz_localize(None)
                    logger.info("    ✓ Parsed dates with UTC conversion")
                except Exception as e1:
                    logger.warning(f"    ⚠ UTC parsing failed: {e1}")
                    try:
                        # Second try: parse without timezone
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                        logger.info("    ✓ Parsed dates without timezone")
                    except Exception as e2:
                        logger.error(f"    ✗ Date parsing failed: {e2}")
                        date_col = None
                
                if date_col:
                    valid_dates = df[date_col].notna().sum()
                    logger.info(f"    Valid dates: {valid_dates:,} / {len(df):,}")
            else:
                logger.warning("  ⚠ No date column found")
            
            # Extract sentiment
            logger.info(f"\n  Extracting sentiment from {len(df):,} articles...")
            logger.info(f"  Estimated time: ~{len(df) / 150 / 60:.1f} minutes")
            
            sentiments = []
            batch_size = 10000
            error_count = 0
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Progress", 
                                file=sys.stdout, dynamic_ncols=True):
                try:
                    text = str(row[text_col])[:512]
                    sentiment = get_sentiment(text, tokenizer, model, device)
                    
                    result = {
                        'date': row[date_col] if date_col else None,
                        'text': text[:200],
                        **sentiment
                    }
                    sentiments.append(result)
                    
                    # Save in batches
                    if len(sentiments) >= batch_size:
                        batch_df = pd.DataFrame(sentiments)
                        all_sentiments.append(batch_df)
                        total_processed += len(sentiments)
                        logger.info(f"\n    ✓ Saved batch: {len(sentiments):,} articles (Total: {total_processed:,})")
                        sentiments = []
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # Log first 5 errors
                        logger.error(f"    Error processing row {idx}: {e}")
            
            # Save remaining
            if sentiments:
                batch_df = pd.DataFrame(sentiments)
                all_sentiments.append(batch_df)
                total_processed += len(sentiments)
                logger.info(f"\n    ✓ Saved final batch: {len(sentiments):,} articles")
            
            file_end_time = datetime.now()
            file_duration = (file_end_time - file_start_time).total_seconds() / 60
            
            file_stats[news_file.name] = {
                'articles': len(df),
                'processed': len(df) - error_count,
                'errors': error_count,
                'duration_minutes': file_duration
            }
            
            logger.info(f"\n  ✓ File complete:")
            logger.info(f"    Articles: {len(df):,}")
            logger.info(f"    Processed: {len(df) - error_count:,}")
            logger.info(f"    Errors: {error_count}")
            logger.info(f"    Time: {file_duration:.1f} minutes")
            
        except Exception as e:
            logger.error(f"  ✗ File processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    if not all_sentiments:
        logger.error("\n✗ No sentiments extracted!")
        return
    
    # Combine all sentiments
    logger.info(f"\n{'='*60}")
    logger.info("Combining all sentiments...")
    logger.info(f"{'='*60}")
    
    combined_df = pd.concat(all_sentiments, ignore_index=True)
    logger.info(f"✓ Combined {len(combined_df):,} total articles")
    
    # Filter to valid dates and sort
    if 'date' in combined_df.columns:
        initial_count = len(combined_df)
        combined_df = combined_df.dropna(subset=['date'])
        dropped = initial_count - len(combined_df)
        logger.info(f"  Dropped {dropped:,} articles with invalid dates")
        
        # Ensure timezone-naive before sorting
        try:
            combined_df['date'] = pd.to_datetime(combined_df['date']).dt.tz_localize(None)
            combined_df = combined_df.sort_values('date')
            logger.info(f"✓ Sorted by date")
            logger.info(f"  Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        except Exception as e:
            logger.error(f"⚠ Could not sort by date: {e}")
    
    # Save
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'news_sentiment.csv'
    
    logger.info(f"\nSaving to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"✓ Saved: {output_file}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    
    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY STATISTICS")
    logger.info(f"{'='*60}")
    
    logger.info("\nPer-file statistics:")
    for filename, stats in file_stats.items():
        logger.info(f"\n  {filename}:")
        logger.info(f"    Articles: {stats['articles']:,}")
        logger.info(f"    Processed: {stats['processed']:,}")
        logger.info(f"    Errors: {stats['errors']}")
        logger.info(f"    Time: {stats['duration_minutes']:.1f} min")
    
    logger.info(f"\nSentiment distribution:")
    logger.info(f"  Total articles: {len(combined_df):,}")
    logger.info(f"  Mean sentiment: {combined_df['sentiment_score'].mean():.4f}")
    logger.info(f"  Std sentiment: {combined_df['sentiment_score'].std():.4f}")
    logger.info(f"  Min sentiment: {combined_df['sentiment_score'].min():.4f}")
    logger.info(f"  Max sentiment: {combined_df['sentiment_score'].max():.4f}")
    
    # Sentiment categories
    positive = (combined_df['sentiment_score'] > 0.1).sum()
    negative = (combined_df['sentiment_score'] < -0.1).sum()
    neutral = len(combined_df) - positive - negative
    
    logger.info(f"\nSentiment categories:")
    logger.info(f"  Positive (>0.1): {positive:,} ({positive/len(combined_df)*100:.1f}%)")
    logger.info(f"  Negative (<-0.1): {negative:,} ({negative/len(combined_df)*100:.1f}%)")
    logger.info(f"  Neutral: {neutral:,} ({neutral/len(combined_df)*100:.1f}%)")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ SENTIMENT EXTRACTION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"\nLog saved to: {log_file}")

if __name__ == '__main__':
    main()
