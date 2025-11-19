import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

api_key = os.getenv('ALPHA_VANTAGE_KEY')
print(f"API Key loaded: {api_key[:10]}..." if api_key else "API Key not found!")

# Test API call
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=SPY&apikey={api_key}'
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    if 'Time Series (Daily)' in data:
        print("✓ API working correctly!")
        latest_date = list(data['Time Series (Daily)'].keys())[0]
        print(f"Latest data available: {latest_date}")
    else:
        print("⚠ API call succeeded but unexpected format")
        print(data)
else:
    print(f"✗ API call failed with status {response.status_code}")
