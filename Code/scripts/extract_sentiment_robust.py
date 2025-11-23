#!/usr/bin/env python3
"""
Extract sentiment - ROBUST timezone handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def load_finbert():
    """Load pre-trained FinBERT model"""
    print("Loading FinBERT model...")
    model_name = "ProsusAI/finbert"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"✓ FinBERT loaded on {device}")
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
        
    except:
        return {
            'sentiment_score': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0
        }

def normalize_date(date_series):
    """Normalize dates to timezone-naive"""
    # Convert to datetime
    dates = pd.to_datetime(date_series, errors='coerce')
    
    # Remove timezone if present
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    
    return dates

def main():
    print("\n" + "="*60)
    print("FINBERT SENTIMENT EXTRACTION - ROBUST VERSION")
    print("="*60 + "\n")
    
    # Date filter (as strings for comparison)
    start_year = 2023
    end_year = 2025
    print(f"Filtering news to: {start_year}-{end_year}\n")
    
    tokenizer, model, device = load_finbert()
    
    news_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'news'
    output_dir = Path(__file__).parent.parent / 'data' / 'processed'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    news_files = list(news_dir.glob('*.csv'))
    
    if not news_files:
        print("✗ No news CSV files found!")
        return
    
    print(f"✓ Found {len(news_files)} news file(s)\n")
    
    all_sentiments = []
    total_processed = 0
    
    for news_file in news_files:
        print(f"Processing: {news_file.name}")
        
        try:
            df = pd.read_csv(news_file)
            print(f"  Loaded {len(df):,} articles")
            
            # Find text column
            text_cols = ['headline', 'title', 'text', 'content', 'body']
            text_col = None
            for col in text_cols:
                if col in df.columns:
                    text_col = col
                    break
            
            if not text_col:
                print(f"  ⚠ No text column found, skipping")
                continue
            
            print(f"  Using text column: '{text_col}'")
            
            # Find date column
            date_cols = ['date', 'published', 'publish_date', 'timestamp']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                print(f"  ⚠ No date column found, skipping")
                continue
            
            print(f"  Using date column: '{date_col}'")
            
            # Normalize dates
            df['normalized_date'] = normalize_date(df[date_col])
            df = df.dropna(subset=['normalized_date'])
            
            # Filter by year
            df['year'] = df['normalized_date'].dt.year
            df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
            
            print(f"  After date filter: {len(df):,} articles")
            
            if len(df) == 0:
                print(f"  ⚠ No articles in date range, skipping\n")
                continue
            
            # Process sentiment
            print(f"  Extracting sentiment...")
            sentiments = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Progress"):
                text = str(row[text_col])[:512]
                sentiment = get_sentiment(text, tokenizer, model, device)
                
                result = {
                    'date': row['normalized_date'],
                    'text': text[:200],
                    **sentiment
                }
                sentiments.append(result)
                
                # Save every 10k to avoid memory issues
                if len(sentiments) >= 10000:
                    sentiment_df = pd.DataFrame(sentiments)
                    all_sentiments.append(sentiment_df)
                    total_processed += len(sentiments)
                    sentiments = []
                    print(f"    → Saved batch, total: {total_processed:,}")
            
            # Save remaining
            if sentiments:
                sentiment_df = pd.DataFrame(sentiments)
                all_sentiments.append(sentiment_df)
                total_processed += len(sentiments)
            
            print(f"  ✓ Processed {len(df):,} articles\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_sentiments:
        print("\n✗ No sentiments extracted!")
        return
    
    # Combine all
    print("Combining all sentiments...")
    combined_df = pd.concat(all_sentiments, ignore_index=True)
    combined_df = combined_df.sort_values('date')
    
    print(f"\n✓ Total articles processed: {len(combined_df):,}")
    print(f"  Date range: {combined_df['date'].min().date()} to {combined_df['date'].max().date()}")
    
    # Save
    output_file = output_dir / 'news_sentiment.csv'
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Saved: {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024**2):.1f} MB")
    
    # Summary
    print("\nSentiment Summary:")
    print(f"  Mean: {combined_df['sentiment_score'].mean():.4f}")
    print(f"  Std: {combined_df['sentiment_score'].std():.4f}")
    print(f"  Min: {combined_df['sentiment_score'].min():.4f}")
    print(f"  Max: {combined_df['sentiment_score'].max():.4f}")
    
    # Daily stats
    print("\nDaily article counts:")
    daily_counts = combined_df.groupby(combined_df['date'].dt.date).size()
    print(f"  Mean per day: {daily_counts.mean():.0f}")
    print(f"  Median per day: {daily_counts.median():.0f}")
    print(f"  Max per day: {daily_counts.max():.0f}")
    
    print("\n" + "="*60)
    print("✓ SENTIMENT EXTRACTION COMPLETE!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
