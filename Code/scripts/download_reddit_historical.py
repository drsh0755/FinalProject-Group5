import praw
import pandas as pd
from datetime import datetime, timedelta

# Register free app at: https://www.reddit.com/prefs/apps
reddit = praw.Reddit(
    client_id='YOUR_CLIENT_ID',
    client_secret='YOUR_CLIENT_SECRET',
    user_agent='stock_sentiment_v1.0'
)

print("=" * 60)
print("DOWNLOADING REDDIT FINANCIAL DISCUSSIONS")
print("=" * 60)

subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
all_posts = []

for sub_name in subreddits:
    print(f"\nFetching r/{sub_name}...")
    subreddit = reddit.subreddit(sub_name)
    
    # Get posts from different time periods
    for time_filter in ['year', 'month']:
        print(f"  {time_filter}...")
        for post in subreddit.top(time_filter=time_filter, limit=1000):
            post_date = datetime.fromtimestamp(post.created_utc)
            
            # Only posts from 2024-2025
            if post_date >= datetime(2024, 2, 1):
                all_posts.append({
                    'date': post_date,
                    'headline': post.title,
                    'text': post.selftext[:500],  # First 500 chars
                    'score': post.score,
                    'subreddit': sub_name,
                    'num_comments': post.num_comments
                })
        
        print(f"    Found: {len(all_posts)} posts so far")

# Save
df = pd.DataFrame(all_posts)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

output_file = 'data/raw/news/reddit_finance_2024_2025.csv'
df.to_csv(output_file, index=False)

print(f"\n✓ Total: {len(df)} posts")
print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
print(f"✓ Saved to: {output_file}")
