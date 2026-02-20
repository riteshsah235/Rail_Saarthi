"""Configuration for Rail Saarthi - Complaint Intelligence System."""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Complaint categories (problem statement)
CATEGORIES = [
    "cleanliness",
    "delay_issues",
    "staff_behavior",
    "food_quality",
    "safety_concerns",
    "ticketing_problems",
]

# Severity levels
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# High-priority keywords for severity tagging
CRITICAL_KEYWORDS = [
    "accident", "safety", "emergency", "fire", "theft", "assault",
    "derailment", "collapse", "critical", "urgent", "life", "death",
]
HIGH_KEYWORDS = [
    "delay", "cancelled", "stranded", "stuck", "hours", "no water",
    "no food", "harassment", "abuse", "discrimination", "refund",
]

# Model paths
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
CLASSIFIER_PATH = MODELS_DIR / "category_classifier.joblib"
CLUSTER_MODEL_PATH = MODELS_DIR / "cluster_model.joblib"
SEVERITY_MODEL_PATH = MODELS_DIR / "severity_model.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
