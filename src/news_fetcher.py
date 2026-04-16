import os
import requests
from datetime import datetime, timedelta

def fetch_news(ticker: str, days_back: int = 30) -> list:
    """
    Fetch related news headlines for a stock ticker using NewsAPI.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        days_back (int): Number of days back to search for news.
        
    Returns:
        list: A list of dictionaries containing news headlines and dates.
    """
    # Retrieve the API key from environment variables
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise ValueError("NEWS_API_KEY environment variable not set. Please check your .env file or environment variables.")
    
    # Calculate the date range for fetching news
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Construct the NewsAPI URL for everything endpoint
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={ticker}&"
        f"from={start_date.strftime('%Y-%m-%d')}&"
        f"to={end_date.strftime('%Y-%m-%d')}&"
        f"sortBy=publishedAt&"
        f"apiKey={api_key}&"
        f"language=en"
    )
    
    print(f"Fetching news for {ticker} over the last {days_back} days...")
    try:
        # Make the GET request to NewsAPI
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        
        # Parse the JSON response
        data = response.json()
        articles = data.get("articles", [])
        
        # Extract relevant information from each article
        news_data = []
        for article in articles:
            news_data.append({
                "date": article.get("publishedAt")[:10], # Extract only the date part YYYY-MM-DD
                "title": article.get("title"),
                "source": article.get("source", {}).get("name")
            })
            
        print(f"Fetched {len(news_data)} news articles.")
        return news_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []
