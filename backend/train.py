"""
Training script: Preprocessing pipeline, Classification, Clustering, Severity, Evaluation.
Run once to generate models and metrics for the hackathon demo.
"""
import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import (
    MODELS_DIR,
    CATEGORIES,
    SEVERITY_LEVELS,
    VECTORIZER_PATH,
    CLASSIFIER_PATH,
    CLUSTER_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    METRICS_PATH,
)
from preprocessing import preprocess_batch
from data.sample_complaints import get_training_data


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    data = get_training_data()
    texts = [d["text"] for d in data]
    categories = [d["category"] for d in data]
    severities = [d["severity"] for d in data]

    # 1) Preprocessing pipeline
    X_processed = preprocess_batch(texts)
    print("Preprocessing pipeline: done.")

    # 2) TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(X_processed)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("TF-IDF vectorizer: saved.")

    # 3) Category classifier (Multinomial NB for text)
    clf = MultinomialNB(alpha=0.1)
    y_cat = np.array(categories)
    clf.fit(X_tfidf, y_cat)
    joblib.dump(clf, CLASSIFIER_PATH)
    # Evaluation: accuracy & F1
    pred_cat = cross_val_predict(MultinomialNB(alpha=0.1), X_tfidf, y_cat, cv=min(5, len(data) // 2))
    acc = accuracy_score(y_cat, pred_cat)
    f1 = f1_score(y_cat, pred_cat, average="weighted", zero_division=0)
    report = classification_report(y_cat, pred_cat, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_cat, pred_cat)
    print("Category classifier: saved. Accuracy =", round(acc, 4), "F1 =", round(f1, 4))

    # 4) Severity model (same TF-IDF, different target)
    sev_map = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    y_sev = np.array([sev_map[s] for s in severities])
    sev_clf = SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)
    sev_clf.fit(X_tfidf, y_sev)
    joblib.dump(sev_clf, SEVERITY_MODEL_PATH)
    pred_sev = cross_val_predict(SGDClassifier(loss="log_loss", max_iter=1000, random_state=42), X_tfidf, y_sev, cv=min(5, len(data) // 2))
    sev_acc = accuracy_score(y_sev, pred_sev)
    sev_f1 = f1_score(y_sev, pred_sev, average="weighted", zero_division=0)
    print("Severity model: saved. Accuracy =", round(sev_acc, 4), "F1 =", round(sev_f1, 4))

    # 5) Clustering (recurring issue clusters)
    n_clusters = min(6, max(2, len(data) // 10))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X_tfidf)
    joblib.dump(km, CLUSTER_MODEL_PATH)
    print("Clustering model: saved. n_clusters =", n_clusters)

    # 6) Persist evaluation metrics
    metrics = {
        "category": {
            "accuracy": round(float(acc), 4),
            "f1_weighted": round(float(f1), 4),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        },
        "severity": {
            "accuracy": round(float(sev_acc), 4),
            "f1_weighted": round(float(sev_f1), 4),
        },
        "clustering": {"n_clusters": int(n_clusters)},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Evaluation metrics: saved to", METRICS_PATH)
    print("Training complete. Run the API and frontend to use the system.")


if __name__ == "__main__":
    main()
