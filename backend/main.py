"""
Rail Saarthi - Indian Railways Complaint Intelligence API.
Endpoints: classify, severity, cluster, trends, metrics, clustering visualization.
"""
import json
import joblib
import numpy as np
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.decomposition import PCA

from .config import (
    CATEGORIES,
    SEVERITY_LEVELS,
    VECTORIZER_PATH,
    CLASSIFIER_PATH,
    CLUSTER_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    METRICS_PATH,
)
from .preprocessing import preprocess_for_model, preprocess_batch
from .severity import resolve_severity

app = FastAPI(
    title="Rail Saarthi API",
    description="AI-driven complaint categorization & pattern detection for Indian Railways",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load models
_vectorizer = _classifier = _cluster_model = _severity_model = _metrics = None


def _load_models():
    global _vectorizer, _classifier, _cluster_model, _severity_model, _metrics
    if _vectorizer is None:
        if not VECTORIZER_PATH.exists():
            raise HTTPException(status_code=503, detail="Models not trained. Run: python backend/train.py")
        _vectorizer = joblib.load(VECTORIZER_PATH)
        _classifier = joblib.load(CLASSIFIER_PATH)
        _cluster_model = joblib.load(CLUSTER_MODEL_PATH)
        _severity_model = joblib.load(SEVERITY_MODEL_PATH)
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                _metrics = json.load(f)
        else:
            _metrics = {}
    return _vectorizer, _classifier, _cluster_model, _severity_model, _metrics


class ComplaintInput(BaseModel):
    text: str


class BatchComplaintInput(BaseModel):
    texts: List[str]


@app.post("/api/classify", response_model=dict)
def classify_complaint(body: ComplaintInput):
    """Automatic complaint categorization."""
    vec, clf, _, _, _ = _load_models()
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty complaint text")
    processed = preprocess_for_model(text)
    X = vec.transform([processed])
    pred = clf.predict(X)[0]
    proba = getattr(clf, "predict_proba", None)
    conf = float(np.max(proba(X)[0])) if proba else 1.0
    return {"category": pred, "confidence": round(conf, 4), "all_categories": list(CATEGORIES)}


@app.post("/api/severity", response_model=dict)
def get_severity(body: ComplaintInput):
    """Severity tagging (low/medium/high/critical)."""
    vec, _, _, sev_clf, _ = _load_models()
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty complaint text")
    processed = preprocess_for_model(text)
    X = vec.transform([processed])
    pred_idx = sev_clf.predict(X)[0]
    pred_sev = SEVERITY_LEVELS[int(pred_idx)]
    final = resolve_severity(pred_sev, text)
    return {"severity": final, "levels": SEVERITY_LEVELS}


@app.post("/api/analyze", response_model=dict)
def analyze_complaint(body: ComplaintInput):
    """Single endpoint: category + severity for one complaint."""
    vec, clf, km, sev_clf, _ = _load_models()
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty complaint text")
    processed = preprocess_for_model(text)
    X = vec.transform([processed])
    cat = clf.predict(X)[0]
    sev_idx = sev_clf.predict(X)[0]
    sev = resolve_severity(SEVERITY_LEVELS[int(sev_idx)], text)
    cluster = int(km.predict(X)[0])
    return {"category": cat, "severity": sev, "cluster_id": cluster}


@app.post("/api/cluster-batch", response_model=dict)
def cluster_batch(body: BatchComplaintInput):
    """Return cluster IDs for a list of complaints (for visualization)."""
    vec, _, km, _, _ = _load_models()
    if not body.texts:
        return {"labels": [], "n_clusters": km.n_clusters}
    processed = preprocess_batch(body.texts)
    X = vec.transform(processed)
    labels = km.predict(X).tolist()
    return {"labels": labels, "n_clusters": int(km.n_clusters)}


@app.get("/api/clustering-viz", response_model=dict)
def clustering_visualization():
    """
    2D projection of complaint embeddings + cluster labels for dashboard.
    Uses stored training data and current vectorizer/cluster model.
    """
    from .data.sample_complaints import get_training_data
    vec, _, km, _, _ = _load_models()
    data = get_training_data()
    texts = [d["text"] for d in data]
    processed = preprocess_batch(texts)
    X = vec.transform(processed)
    labels = km.predict(X)
    # PCA to 2D for scatter plot
    n_components = min(2, X.shape[1], X.shape[0] - 1)
    if n_components < 2:
        return {"points": [], "n_clusters": int(km.n_clusters)}
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X.toarray())
    points = [
        {"x": float(coords[i, 0]), "y": float(coords[i, 1]), "cluster": int(labels[i]), "text": texts[i][:80]}
        for i in range(len(texts))
    ]
    return {"points": points, "n_clusters": int(km.n_clusters)}


@app.get("/api/trends", response_model=dict)
def complaint_trends():
    """Aggregate counts by category and severity for dashboard trends."""
    from .data.sample_complaints import get_training_data
    data = get_training_data()
    by_cat = {}
    by_sev = {}
    for d in data:
        by_cat[d["category"]] = by_cat.get(d["category"], 0) + 1
        by_sev[d["severity"]] = by_sev.get(d["severity"], 0) + 1
    return {
        "by_category": [{"category": k, "count": v} for k, v in sorted(by_cat.items())],
        "by_severity": [{"severity": k, "count": v} for k, v in sorted(by_sev.items())],
    }


@app.get("/api/metrics", response_model=dict)
def evaluation_metrics():
    """Accuracy and F1 for category and severity models."""
    _, _, _, _, metrics = _load_models()
    return metrics if metrics else {"category": {}, "severity": {}, "clustering": {}}


@app.get("/api/categories")
def list_categories():
    return {"categories": CATEGORIES}


@app.get("/api/severity-levels")
def list_severity_levels():
    return {"severity_levels": SEVERITY_LEVELS}


# Serve frontend (dashboard) at /
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
    @app.get("/")
    def serve_dashboard():
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/")
    def root():
        return {"message": "Rail Saarthi API", "docs": "/docs"}
