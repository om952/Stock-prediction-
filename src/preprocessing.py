import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import re

def preprocess_stock_data(df: pd.DataFrame):
    """
    Handle missing values, select 'Close' price, and normalize using MinMaxScaler.
    
    Args:
        df: Raw dataframe from yfinance.
        
    Returns:
        scaled_data: Numpy array of scaled 'Close' prices.
        scaler: The fitted MinMaxScaler object.
    """
    # 0. Flatten multi-level column headers produced by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # 1. Handle missing values
    df = df.dropna(subset=['Close'])
    
    # 2. Select 'Close' price (converting to 2D array for scaler)
    close_prices = df[['Close']].values
    
    # 3. Normalize using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    return scaled_data, scaler

def create_sequences(data: np.ndarray, window_size: int = 60):
    """
    Convert time series into sequences of length `window_size`.
    
    Args:
        data: Numpy array of scaled data (e.g., shape [N, 1]).
        window_size: Number of past days to use.
        
    Returns:
        X: Numpy array of input sequences (shape: [samples, window_size, 1]).
        y: Numpy array of target values (shape: [samples]).
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        # Past `window_size` days
        X.append(data[i-window_size:i, 0])
        # Next day's price
        y.append(data[i, 0])
        
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to be [samples, time steps, features] required for CNN-LSTM
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y

def preprocess_text_data(text_list: list, tokenizer, max_length: int = 128):
    """
    Clean text and tokenize using a BERT tokenizer.
    
    Args:
        text_list: List of strings (news headlines).
        tokenizer: HuggingFace BERT tokenizer.
        max_length: Maximum sequence length for padding/truncation.
        
    Returns:
        input_ids: Tensor of token ids.
        attention_mask: Tensor of attention masks.
    """
    # Clean text
    cleaned_texts = []
    for text in text_list:
        if not isinstance(text, str):
            text = ""
        # Lowercase
        text = text.lower()
        # Remove special characters or non-ascii
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Normalize whitespace 
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_texts.append(text)
        
    # Tokenize
    encoded = tokenizer(
        cleaned_texts,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'  # Use TensorFlow since CNN-LSTM is in Keras
    )
    
    return encoded['input_ids'], encoded['attention_mask']
