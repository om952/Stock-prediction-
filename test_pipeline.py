"""
test_pipeline.py
================
End-to-end validation of every module in the stock prediction pipeline.
Runs with 1 training epoch so it stays fast — the goal is to verify that
all pieces connect, not to produce a good model.
"""

import sys
import numpy as np
import yfinance as yf

# ──────────────────────────────────────────────
# 1. IMPORTS — verify all modules are loadable
# ──────────────────────────────────────────────
print("=" * 50)
print("  PIPELINE VALIDATION TEST")
print("=" * 50)

print("\n[0] Importing modules...")
try:
    from src.preprocessing import preprocess_stock_data, create_sequences
    from src.cnn_lstm_model import build_cnn_lstm_model, train_model
    from src.sentiment_model import load_bert_model, predict_sentiment
    print("    ✅ All modules imported successfully.")
except ImportError as e:
    print(f"    ❌ Import failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. FETCH SAMPLE STOCK DATA
# ──────────────────────────────────────────────
print("\n[1] Fetching sample stock data (AAPL, 6 months)...")
ticker = "AAPL"
df = yf.download(ticker, period="6mo")

if df is None or df.empty:
    print("    ❌ No data returned from yfinance. Cannot continue.")
    sys.exit(1)

print(f"    ✅ Downloaded {len(df)} rows for {ticker}.")

# Flatten multi-level column headers from newer yfinance versions
import pandas as pd
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

print(f"    Columns: {list(df.columns)}")

# ──────────────────────────────────────────────
# 3. PREPROCESS STOCK DATA
# ──────────────────────────────────────────────
print("\n[2] Running preprocess_stock_data...")
try:
    scaled_data, scaler = preprocess_stock_data(df)
    print(f"    ✅ Scaled data shape: {scaled_data.shape}")
    print(f"    Value range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
except Exception as e:
    print(f"    ❌ preprocess_stock_data failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 4. CREATE SEQUENCES
# ──────────────────────────────────────────────
print("\n[3] Running create_sequences (window=60)...")
try:
    X, y = create_sequences(scaled_data, window_size=60)
    print(f"    ✅ X shape: {X.shape}  (expected: (samples, 60, 1))")
    print(f"    ✅ y shape: {y.shape}  (expected: (samples,))")

    # Safety check: validate dimensions
    assert X.shape[1] == 60, f"Expected time steps = 60, got {X.shape[1]}"
    assert X.shape[2] == 1,  f"Expected features = 1, got {X.shape[2]}"
    assert len(y) == len(X),  "X and y sample counts do not match"
    print("    ✅ Shape assertions passed.")
except Exception as e:
    print(f"    ❌ create_sequences failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 5. BUILD CNN-LSTM MODEL
# ──────────────────────────────────────────────
print("\n[4] Building CNN-LSTM model...")
try:
    input_shape = (X.shape[1], X.shape[2])  # (60, 1)
    model = build_cnn_lstm_model(input_shape)
    print("    ✅ Model built. Summary:")
    model.summary()
except Exception as e:
    print(f"    ❌ build_cnn_lstm_model failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 6. TRAIN FOR 1 EPOCH (quick sanity check)
# ──────────────────────────────────────────────
print("\n[5] Training CNN-LSTM for 1 epoch (validation only)...")
try:
    # Use a small subset to keep it fast
    subset_size = min(200, len(X))
    X_subset = X[:subset_size]
    y_subset = y[:subset_size]

    model = train_model(model, X_subset, y_subset, epochs=1, batch_size=32)
    print("    ✅ Training completed without errors.")
except Exception as e:
    print(f"    ❌ train_model failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 7. MAKE A TEST PREDICTION
# ──────────────────────────────────────────────
print("\n[6] Making a test prediction with the last sequence...")
try:
    last_sequence = X[-1:]  # shape (1, 60, 1)
    pred_scaled = model.predict(last_sequence, verbose=0)
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    actual_price = float(df['Close'].iloc[-1])

    print(f"    ✅ Predicted (scaled): {pred_scaled[0][0]:.4f}")
    print(f"    ✅ Predicted price:    ${pred_price:.2f}")
    print(f"    ✅ Actual last price:  ${actual_price:.2f}")
except Exception as e:
    print(f"    ❌ Prediction failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 8. LOAD BERT SENTIMENT MODEL
# ──────────────────────────────────────────────
print("\n[7] Loading BERT model from HuggingFace...")
try:
    tokenizer, bert_model = load_bert_model()
    print("    ✅ BERT tokenizer and model loaded.")
except Exception as e:
    print(f"    ❌ load_bert_model failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 9. RUN SENTIMENT PREDICTION
# ──────────────────────────────────────────────
print("\n[8] Running sentiment prediction on sample headlines...")
sample_texts = [
    "Apple stock surges to all-time high amid strong earnings",
    "Markets crash as recession fears grow",
    "Tech sector remains stable with mixed signals"
]

try:
    sentiments = predict_sentiment(sample_texts, tokenizer, bert_model)
    labels = {0: "NEGATIVE", 1: "POSITIVE"}

    for text, sent in zip(sample_texts, sentiments):
        print(f'    → "{text}"')
        print(f'      Sentiment: {labels.get(sent, "UNKNOWN")} ({sent})')

    # Also test empty list edge case
    empty_result = predict_sentiment([], tokenizer, bert_model)
    assert empty_result == [], "Empty input should return empty list"
    print("    ✅ Empty-list edge case passed.")
    print("    ✅ Sentiment predictions completed.")
except Exception as e:
    print(f"    ❌ predict_sentiment failed: {e}")
    sys.exit(1)

# ──────────────────────────────────────────────
# 10. FINAL SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  ✅ ALL VALIDATION TESTS PASSED")
print("=" * 50)
print(f"""
Summary:
  Stock data rows:    {len(df)}
  Sequence samples:   {X.shape[0]}
  X shape:            {X.shape}
  y shape:            {y.shape}
  Model params:       {model.count_params():,}
  Test prediction:    ${pred_price:.2f}
  Sentiment outputs:  {sentiments}
""")
