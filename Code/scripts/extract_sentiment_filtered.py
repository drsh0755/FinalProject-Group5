#!/usr/bin/env python3
"""
Extract sentiment from financial news - FILTERED TO 2023-2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm
from datetime import datetime
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

def main():
    print("\n" + "="*60)
    print("FINBERT SENTIMENT EXTRACTION - FILTERED 2023-2025")
    print("="*60 + "\n")
    
    # Date filter
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2025-12-31')
    print(f"Filtering news to: {start_date.date()} to {end_date.date()}\n")
    
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
    total_articles = 0
    filtered_articles = 0
    
    for news_file in news_files:
        print(f"Processing: {news_file.name}")
        
        try:
            # Load with date parsing
            df = pd.read_csv(news_file)
            total_articles += len(df)
            print(f"  Loaded {len(df):,} articles")
            
            # Find text column
            text_cols = ['headline', 'title', 'text', 'content']
            text_col = None
            for col in text_cols:
                if col in df.columns:
                    text_col = col
                    break
            
            # Find date column
            date_cols = ['date', 'published', 'publish_date']
            date_col = None
            for col in date_cols:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                print(f"  ⚠ No date column found, skipping")
                continue
            
            # Parse dates and filter
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
            
            # Remove timezone
            df[date_col] = df[date_col].dt.tz_localize(None)
            
            filtered_articles += len(df)
            print(f"  After filtering: {len(df):,} articles")
            
            if len(df) == 0:
                continue
            
            # Extract sentiment
            print(f"  Extracting sentiment...")
            sentiments = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Progress"):
                text = str(row[text_col])[:512]
                sentiment = get_sentiment(text, tokenizer, model, device)
                
                result = {
                    'date': row[date_col],
                    'text': text[:200],
                    **sentiment
                }
                sentiments.append(result)
            
            sentiment_df = pd.DataFrame(sentiments)
            all_sentiments.append(sentiment_df)
            
            print(f"  ✓ Processed {len(sentiment_df):,} articles\n")
            
        except Exception as e:
            print(f"  ✗ Error: {e}\n")
            continue
    
    if not all_sentiments:
        print("\n✗ No sentiments extracted!")
        return
    
    # Combine all
    print("Combining all sentiments...")
    combined_df = pd.concat(all_sentiments, ignore_index=True)
    combined_df = combined_df.sort_values('date')
    
    print(f"\n✓ Total articles in dataset: {total_articles:,}")
    print(f"✓ Articles after date filter: {filtered_articles:,}")
    print(f"✓ Articles processed: {len(combined_df):,}")
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
    
    print("\n" + "="*60)
    print("✓ SENTIMENT EXTRACTION COMPLETE!")
    print("="*60 + "\n")

if __name__ == '__main__':
    main()
