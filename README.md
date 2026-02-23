# Rail Saarthi — Intelligent Complaint Categorization & Pattern Detection

An AI-driven system that processes textual passenger complaints to **automatically categorize** them, **detect recurring issue clusters**, **tag severity**, and **visualize trends** for faster prioritization and response by railway authorities.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAIL SAARTHI SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  FRONTEND (Responsive Web Dashboard)                                         │
│  • Submit complaint → Analyze (category + severity + cluster)                │
│  • Model metrics (Accuracy, F1)                                               │
│  • Trends: by category (bar), by severity (doughnut)                          │
│  • Clustering: 2D PCA scatter of complaint clusters                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  BACKEND (FastAPI)                                                            │
│  • POST /api/analyze      → category, severity, cluster_id                    │
│  • POST /api/classify    → category only                                     │
│  • POST /api/severity    → severity only                                     │
│  • GET  /api/trends      → by_category, by_severity                           │
│  • GET  /api/clustering-viz → 2D points + cluster labels                     │
│  • GET  /api/metrics     → accuracy, F1 (category & severity)                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ML PIPELINE                                                                  │
│  1. Preprocessing: clean_text → tokenize (NLTK/fallback) → space-join         │
│  2. Vectorization: TF-IDF (1–2 grams, max 5000 features)                      │
│  3. Category classifier: Multinomial Naive Bayes → 6 classes                  │
│  4. Severity model: SGDClassifier (log loss) + keyword upgrade (critical/high)│
│  5. Clustering: KMeans on TF-IDF → recurring issue clusters                  │
│  6. Evaluation: cross_val_predict → Accuracy, F1, confusion matrix           │
├─────────────────────────────────────────────────────────────────────────────┤
│  DATA                                                                         │
│  • Sample complaints: backend/data/sample_complaints.py (6 categories,        │
│    severity labels) — used for training and demo trends/clustering            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complaint Categories (Problem Statement)

- **cleanliness** — Coach, platform, toilet, hygiene  
- **delay_issues** — Late trains, cancellation, no information  
- **staff_behavior** — TTE, station staff, rudeness, bribe  
- **food_quality** — Pantry, water, stale food, pricing  
- **safety_concerns** — Overcrowding, footboard, emergency, eve teasing  
- **ticketing_problems** — IRCTC, refund, PNR, wrong deduction  

### Severity Levels

- **low**, **medium**, **high**, **critical**  
- ML model + rule-based upgrade using keywords (e.g. accident, safety, delay, stranded → critical/high).

---

## Project Structure

```
rail_saarthi/
├── README.md                 # This file — architecture & hackathon guide
├── requirements.txt          # Python dependencies
├── backend/
│   ├── config.py             # Paths, categories, severity levels, keywords
│   ├── preprocessing.py     # Text preprocessing pipeline
│   ├── severity.py           # Severity tagging (ML + keyword rules)
│   ├── train.py              # Train classifier, severity model, clustering; save metrics
│   ├── main.py               # FastAPI app + serve frontend
│   ├── data/
│   │   ├── __init__.py
│   │   └── sample_complaints.py   # Training/demo data
│   └── models/               # Created by train.py
│       ├── tfidf_vectorizer.joblib
│       ├── category_classifier.joblib
│       ├── cluster_model.joblib
│       ├── severity_model.joblib
│       └── evaluation_metrics.json
└── frontend/
    ├── index.html            # Dashboard UI
    └── js/
        └── app.js            # API calls, charts, analyze flow
```

---

## Quick Start (Run for Hackathon Demo)

### 1. Create virtual environment and install dependencies

```bash
cd d:\rail_saarthi
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train models (required once)

```bash
cd backend
python train.py
cd ..
```

This creates `backend/models/` with vectorizer, category classifier, severity model, cluster model, and `evaluation_metrics.json`.

### 3. Start the application

From project root:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- **Dashboard:** open **http://localhost:8000** in the browser.  
- **API docs:** **http://localhost:8000/docs** (Swagger UI).

### 4. Use the dashboard

- Type a complaint in the text area and click **Analyze complaint** to get category, severity, and cluster.
- View **Model performance** (Accuracy / F1).
- See **Complaints by category** and **by severity** (from sample data).
- See **Recurring issue clusters** (2D PCA scatter).

---

## API Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves dashboard (index.html) |
| GET | `/docs` | Swagger UI |
| POST | `/api/analyze` | Body: `{"text": "..."}` → category, severity, cluster_id |
| POST | `/api/classify` | Body: `{"text": "..."}` → category, confidence |
| POST | `/api/severity` | Body: `{"text": "..."}` → severity |
| GET | `/api/trends` | by_category, by_severity (counts) |
| GET | `/api/clustering-viz` | points (x, y, cluster, text), n_clusters |
| GET | `/api/metrics` | accuracy, F1 for category and severity |
| GET | `/api/categories` | list of category names |
| GET | `/api/severity-levels` | list of severity levels |

---

## Evaluation Metrics (Stored and Exposed)

- **Category model:** Accuracy, weighted F1, classification report, confusion matrix (saved in `evaluation_metrics.json`, shown on dashboard).  
- **Severity model:** Accuracy, weighted F1.  
- **Clustering:** number of clusters (configurable in `train.py`).

---

## Optional: Add More Training Data

Edit `backend/data/sample_complaints.py`: add more entries to `SAMPLE_COMPLAINTS` or `EXTRA_SAMPLES` with keys `text`, `category`, `severity`. Then run:

```bash
cd backend
python train.py
```

---

## License & Credits

Built for **AIML-03 Indian Railways Hackathon**.  
Rail Saarthi — Intelligent Complaint Categorization & Pattern Detection System.
