import os
import subprocess
import json
from pathlib import Path

print("=" * 80)
print("GITHUB vs AWS COMPARISON")
print("=" * 80)

# Get what's on GitHub
print("\n📦 Fetching GitHub branch info...")
result = subprocess.run(['git', 'ls-tree', '-r', '--name-only', 'adarsh'], 
                       capture_output=True, text=True)
github_files = set(result.stdout.strip().split('\n'))

print(f"✓ GitHub has {len(github_files)} files")

# Get what's in AWS working directory
print("\n💻 Scanning AWS directory...")
aws_files = set()
for root, dirs, files in os.walk('Code'):
    # Skip certain directories
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', '.ipynb_checkpoints']]
    
    for file in files:
        # Skip certain file types
        if file.endswith(('.pyc', '.pyo', '.DS_Store')):
            continue
        
        rel_path = os.path.join(root, file)
        aws_files.add(rel_path)

print(f"✓ AWS has {len(aws_files)} files (excluding venv, __pycache__, etc.)")

# Compare
on_github_not_aws = github_files - aws_files
on_aws_not_github = aws_files - github_files
on_both = github_files & aws_files

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

print(f"\n✅ Files on BOTH GitHub and AWS: {len(on_both)}")

print(f"\n📤 Files on AWS but NOT on GitHub: {len(on_aws_not_github)}")
if on_aws_not_github:
    print("\nIMPORTANT FILES NOT ON GITHUB:")
    important_extensions = ['.py', '.json', '.md', '.txt', '.sh']
    important_files = []
    
    for f in sorted(on_aws_not_github):
        if any(f.endswith(ext) for ext in important_extensions):
            size = os.path.getsize(f) if os.path.exists(f) else 0
            important_files.append((f, size))
    
    if important_files:
        print(f"\n⚠️  CRITICAL - Code/Config files missing from GitHub:")
        for f, size in important_files[:20]:  # Top 20
            print(f"  - {f} ({size:,} bytes)")
        
        if len(important_files) > 20:
            print(f"\n  ... and {len(important_files) - 20} more files")
    
    # Show large data files (expected to be missing)
    print(f"\n📊 Large data files (expected to be missing from GitHub):")
    data_files = [f for f in on_aws_not_github if 'data/' in f]
    for f in sorted(data_files)[:10]:
        if os.path.exists(f):
            size = os.path.getsize(f)
            if size > 1_000_000:  # > 1MB
                print(f"  - {f} ({size/1_000_000:.1f} MB)")

print(f"\n📥 Files on GitHub but NOT on AWS: {len(on_github_not_aws)}")
if on_github_not_aws:
    for f in sorted(on_github_not_aws)[:10]:
        print(f"  - {f}")
    if len(on_github_not_aws) > 10:
        print(f"  ... and {len(on_github_not_aws) - 10} more")

# Check critical files
print("\n" + "=" * 80)
print("CRITICAL FILES CHECK")
print("=" * 80)

critical_files = {
    'Scripts': [
        'Code/scripts/download_alphavantage_historical_news.py',
        'Code/scripts/align_all_news.py',
        'Code/scripts/train_lstm_with_sentiment_fixed.py',
        'Code/scripts/create_features_2year.py',
        'Code/scripts/download_data.py',
    ],
    'Models': [
        'Code/models/lstm/model.py',
        'Code/models/lstm/dataset.py',
    ],
    'Results': [
        'Code/results/lstm_training_results.json',
        'Code/results/lstm_2year_results.json',
        'Code/results/lstm_with_sentiment_results.json',
    ],
    'Config': [
        'Code/.gitignore',
    ]
}

for category, files in critical_files.items():
    print(f"\n{category}:")
    for f in files:
        aws_exists = os.path.exists(f)
        github_exists = f in github_files
        
        status = "✅" if (aws_exists and github_exists) else "⚠️"
        aws_status = "✓" if aws_exists else "✗"
        gh_status = "✓" if github_exists else "✗"
        
        print(f"  {status} {f}")
        print(f"      AWS: {aws_status}  GitHub: {gh_status}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total files on GitHub: {len(github_files)}")
print(f"Total files on AWS: {len(aws_files)}")
print(f"Files in sync: {len(on_both)}")
print(f"Files only on AWS: {len(on_aws_not_github)}")
print(f"Files only on GitHub: {len(on_github_not_aws)}")

# Check if any important files are missing
important_missing = [f for f in on_aws_not_github 
                     if f.endswith(('.py', '.json', '.md', '.txt', '.sh'))]

if important_missing:
    print(f"\n⚠️  WARNING: {len(important_missing)} important files not on GitHub!")
    print("   Run: git add <files> && git commit && git push")
else:
    print("\n✅ All important code files are on GitHub!")

