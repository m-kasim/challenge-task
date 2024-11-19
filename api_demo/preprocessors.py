import re
import numpy as np
from transformers import AutoTokenizer

CONFIG_MODEL_SCIBERT = "allenai/scibert_scivocab_uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(CONFIG_MODEL_SCIBERT)

def preprocess_text(text: str):
    """
    Cleans and tokenizes input text for model prediction.

    Args:
        text (str): The input text to preprocess.

    Returns:
        tuple: A tuple containing input_ids and attention_mask as numpy arrays.
    """
    if not text:
        raise ValueError("Input text cannot be empty or None.")

    # Remove newlines, URLs, and HTML tags
    text = text.replace("\n", " ").strip()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)          # Remove HTML tags

    # Tokenize and encode
    encoded = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )

    return np.array(encoded["input_ids"]), np.array(encoded["attention_mask"])

