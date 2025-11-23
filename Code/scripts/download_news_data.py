#!/usr/bin/env python3
"""
Download financial news dataset from Kaggle
"""

import subprocess
from pathlib import Path

def main():
    print("\n" + "="*60)
    print("DOWNLOADING FINANCIAL NEWS DATA")
    print("="*60 + "\n")
    
    data_dir = Path(__file__).parent.parent / 'data' / 'raw' / 'news'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Using Kaggle API to download...")
    print("Dataset: jeet2016/us-financial-news-articles\n")
    
    try:
        # Download using kaggle CLI
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', 'jeet2016/us-financial-news-articles',
            '-p', str(data_dir),
            '--unzip'
        ], check=True)
        
        print("\n✓ Download complete!")
        print(f"✓ Saved to: {data_dir}")
        
        # List what was downloaded
        print("\nDownloaded files:")
        for file in data_dir.iterdir():
            print(f"  - {file.name}")
            
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error downloading: {e}")
        print("\nAlternative: Manual download from:")
        print("https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles")

if __name__ == '__main__':
    main()
