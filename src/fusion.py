def generate_signal(predicted_price: float, current_price: float, sentiment_score: float) -> str:
    """
    Generate trading signal based on predicted price change and sentiment score.

    Args:
        predicted_price (float): The predicted next-day price.
        current_price (float): The actual current price.
        sentiment_score (float): Aggregated sentiment score (-1.0 to 1.0).

    Returns:
        str: Trading recommendation string.
    """
    price_diff = predicted_price - current_price
    price_change_pct = (price_diff / current_price) * 100

    if price_change_pct > 1 and sentiment_score > 0:
        return "STRONG BUY"
    elif price_change_pct > 0:
        return "BUY"
    elif price_change_pct < -1 and sentiment_score < 0:
        return "STRONG SELL"
    elif price_change_pct < 0:
        return "SELL"
    else:
        return "HOLD"
