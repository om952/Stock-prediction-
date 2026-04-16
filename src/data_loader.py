import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        
    Returns:
        pd.DataFrame: DataFrame containing the historical stock data.
    """
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    try:
        # Download historical data from Yahoo Finance
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # Check if the returned dataframe is empty
        if stock_data.empty:
            print(f"Warning: No data found for {ticker}.")
            
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()
