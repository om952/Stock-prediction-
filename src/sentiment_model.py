import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_bert_model():
    """
    Load a pretrained sentiment analysis model from HuggingFace.
    Uses 'distilbert-base-uncased-finetuned-sst-2-english' which is already
    fine-tuned on the SST-2 sentiment dataset — no additional training needed.

    Returns:
        tokenizer: Initialized tokenizer.
        model: Fine-tuned sentiment classification model.
    """
    # This model is already fine-tuned for binary sentiment (NEGATIVE / POSITIVE)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading tokenizer and model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode (no dropout)

    return tokenizer, model

def predict_sentiment(text_list: list, tokenizer, model) -> list:
    """
    Predict sentiment labels for a list of text inputs.

    The model outputs two classes:
        0 = NEGATIVE
        1 = POSITIVE

    Args:
        text_list: List of text strings (news headlines).
        tokenizer: Pre-initialized tokenizer.
        model: Pre-initialized sentiment model.

    Returns:
        predictions: List of sentiment labels (1 for positive, 0 for negative).
    """
    if not text_list:
        return []

    # Tokenize input
    encoded = tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Run inference without computing gradients
    with torch.no_grad():
        outputs = model(**encoded)

    # Obtain logits from model output
    logits = outputs.logits

    # Apply softmax & take argmax to get predicted class
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=1).tolist()

    return predictions
