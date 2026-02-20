"""
Text Preprocessing Pipeline for Railway Complaints.
Handles cleaning, normalization, and vectorization-ready text.
"""
import re
import string
from typing import List, Optional

# Try NLTK; fallback to simple tokenization
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

# Fallback stopwords if NLTK unavailable
FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "now",
}


def clean_text(text: str) -> str:
    """Normalize and clean raw complaint text."""
    if not text or not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower().strip()
    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Remove email-like patterns (keep context)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Replace numbers with placeholder to retain "train number" etc. sense
    text = re.sub(r"\d+", " ", text)
    # Remove extra whitespace and punctuation runs
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize and optionally remove stopwords."""
    text = clean_text(text)
    if not text:
        return []
    if _HAS_NLTK:
        tokens = word_tokenize(text)
        stops = set(stopwords.words("english"))
    else:
        tokens = text.split()
        stops = FALLBACK_STOPWORDS
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stops and len(t) > 1]
    return tokens


def preprocess_for_model(text: str) -> str:
    """
    Return a single string suitable for TF-IDF (space-joined tokens).
    Use this for inference pipeline.
    """
    tokens = tokenize(text, remove_stopwords=True)
    return " ".join(tokens)


def preprocess_batch(texts: List[str]) -> List[str]:
    """Preprocess a list of complaint texts."""
    return [preprocess_for_model(t) for t in texts]
