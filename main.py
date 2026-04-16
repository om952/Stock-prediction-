"""
main.py
=======
Complete Multi-Modal Stock Price Prediction and Trading Signal System.
Integrates CNN-LSTM price prediction with BERT sentiment analysis to
generate actionable trading recommendations.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# --- Module imports ---
from src.data_loader import fetch_stock_data
from src.news_fetcher import fetch_news
from src.preprocessing import preprocess_stock_data, create_sequences
from src.cnn_lstm_model import build_cnn_lstm_model, train_model
from src.sentiment_model import load_bert_model, predict_sentiment
from src.fusion import generate_signal


def main():
    # Load environment variables (.env contains NEWS_API_KEY)
    load_dotenv()

    # ──────────────────────────────────────────
    # 1. USER INPUT
    # ──────────────────────────────────────────
    try:
        ticker = input("Enter stock ticker (e.g. AAPL): ").strip().upper()
    except EOFError:
        print("Non-interactive mode detected. Defaulting to 'AAPL'.")
        ticker = "AAPL"

    if not ticker:
        print("Error: No ticker entered. Exiting.")
        return

    print(f"\n{'='*55}")
    print(f"  Stock Analysis Pipeline — {ticker}")
    print(f"{'='*55}")

    # ──────────────────────────────────────────
    # 2. FETCH STOCK DATA
    # ──────────────────────────────────────────
    # Fetch 5 years of data for robust training
    print("\n[Step 1/7] Fetching stock data...")
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.now() - datetime.timedelta(days=1825)).strftime('%Y-%m-%d')

    df = fetch_stock_data(ticker, start_date, end_date)

    if df is None or df.empty:
        print(f"Error: No stock data found for '{ticker}'. Exiting.")
        return

    # Flatten multi-level column headers from newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    try:
        current_price = float(df['Close'].iloc[-1])
    except Exception as e:
        print(f"Error reading current price: {e}")
        return

    print(f"  → {len(df)} trading days loaded.")
    print(f"  → Latest close: ${current_price:.2f}")

    # ──────────────────────────────────────────
    # 3. PREPROCESS STOCK DATA
    # ──────────────────────────────────────────
    print("\n[Step 2/7] Preprocessing stock data...")
    scaled_data, scaler = preprocess_stock_data(df)

    window_size = 60
    if len(scaled_data) < window_size + 1:
        print(f"Error: Not enough data ({len(scaled_data)} days). Need at least {window_size + 1}. Exiting.")
        return

    X, y = create_sequences(scaled_data, window_size=window_size)
    print(f"  → Created {X.shape[0]} training sequences (window={window_size}).")

    # ──────────────────────────────────────────
    # 4. TRAIN CNN-LSTM MODEL
    # ──────────────────────────────────────────
    print("\n[Step 3/7] Training CNN-LSTM model...")
    model = build_cnn_lstm_model((X.shape[1], X.shape[2]))
    model = train_model(model, X, y, epochs=25, batch_size=32)

    # ──────────────────────────────────────────
    # 5. PREDICT NEXT PRICE
    # ──────────────────────────────────────────
    print("\n[Step 4/7] Predicting next-day price...")
    last_sequence = scaled_data[-window_size:]
    last_sequence = np.reshape(last_sequence, (1, window_size, 1))

    predicted_scaled = model.predict(last_sequence, verbose=0)
    predicted_price = float(scaler.inverse_transform(predicted_scaled)[0][0])

    price_change_pct = ((predicted_price - current_price) / current_price) * 100
    print(f"  → Predicted price: ${predicted_price:.2f} ({price_change_pct:+.2f}%)")

    # ──────────────────────────────────────────
    # 6. FETCH NEWS & SENTIMENT
    # ──────────────────────────────────────────
    print("\n[Step 5/7] Fetching news headlines...")
    news_items = fetch_news(ticker, days_back=30)
    headlines = [item['title'] for item in news_items if item.get('title')]

    print("\n[Step 6/7] Analyzing sentiment...")
    if headlines:
        # Load BERT once
        tokenizer, bert_model = load_bert_model()

        # Predict sentiment for all headlines
        sentiments = predict_sentiment(headlines, tokenizer, bert_model)

        # Convert to numeric: 1 → +1 (positive), 0 → -1 (negative)
        mapped = [1 if s == 1 else -1 for s in sentiments]
        sentiment_score = float(np.mean(mapped))

        pos_count = mapped.count(1)
        neg_count = mapped.count(-1)
        print(f"  → Analyzed {len(headlines)} headlines.")
        print(f"  → Positive: {pos_count}  |  Negative: {neg_count}")
        print(f"  → Aggregate sentiment: {sentiment_score:+.2f}")
    else:
        print("  → No news found. Using neutral sentiment (0.0).")
        sentiment_score = 0.0

    # ──────────────────────────────────────────
    # 7. GENERATE TRADING SIGNAL
    # ──────────────────────────────────────────
    print("\n[Step 7/7] Generating trading signal...")
    signal = generate_signal(predicted_price, current_price, sentiment_score)

    # ──────────────────────────────────────────
    # OUTPUT
    # ──────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"              ANALYSIS RESULTS")
    print(f"{'='*55}")
    print(f"  Ticker:              {ticker}")
    print(f"  Current Price:       ${current_price:.2f}")
    print(f"  Predicted Price:     ${predicted_price:.2f}")
    print(f"  Change:              {price_change_pct:+.2f}%")
    print(f"  Sentiment Score:     {sentiment_score:+.2f}")
    print(f"  Recommendation:      {signal}")
    print(f"{'='*55}\n")

    # ──────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────
    print("Generating price chart...")
    try:
        # Use last 100 actual prices (or all if fewer)
        close_prices = df['Close'].values.flatten()
        display_count = min(100, len(close_prices))
        recent_prices = close_prices[-display_count:]

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot actual prices
        ax.plot(range(display_count), recent_prices, color='#2196F3', linewidth=1.8, label='Actual Price')

        # Mark predicted next-day point
        ax.scatter(
            display_count,               # one step beyond the last actual
            predicted_price,
            color='#FF5722',
            s=120,
            zorder=5,
            label=f'Predicted: ${predicted_price:.2f}'
        )

        # Dashed line connecting last actual to prediction
        ax.plot(
            [display_count - 1, display_count],
            [recent_prices[-1], predicted_price],
            color='#FF5722',
            linestyle='--',
            linewidth=1.5
        )

        ax.set_title(f'{ticker} — Price History & Next-Day Prediction', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = f"{ticker}_prediction_plot.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  → Chart saved to '{plot_path}'")
    except Exception as e:
        print(f"  → Chart generation failed: {e}")


if __name__ == "__main__":
    main()
