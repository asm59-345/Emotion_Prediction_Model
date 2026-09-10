# 🎭 Moodline Pro — Emotion & Sentiment Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Live_Demo-moodline--pro.onrender.com-brightgreen?style=for-the-badge&logo=render&logoColor=white)](https://moodline-pro.onrender.com/)
[![API Docs](https://img.shields.io/badge/Swagger_UI-Docs-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://moodline-pro.onrender.com/docs)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Deep Learning](https://img.shields.io/badge/Model-Bidirectional_GRU-orange.svg)](https://keras.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-14%20Passed%20(100%25)-brightgreen.svg)]()

> An enterprise-grade, high-performance Deep Learning Emotion & Sentiment Intelligence platform powered by a stacked **Bidirectional Gated Recurrent Unit (BiGRU)** neural network, featuring real-time token attribution, narrative paragraph arcs, voice dictation, CSV batch processing, and persistent SQLite analytics.

---

## 🌐 Live Deployment Links

| Resource | URL | Status |
| :--- | :--- | :--- |
| **🚀 Production Web App** | [https://moodline-pro.onrender.com/](https://moodline-pro.onrender.com/) | 🟢 **Live** |
| **⚡ Interactive Swagger API Docs** | [https://moodline-pro.onrender.com/docs](https://moodline-pro.onrender.com/docs) | 🟢 **Live** |
| **🩺 Health & Telemetry Probe** | [https://moodline-pro.onrender.com/health](https://moodline-pro.onrender.com/health) | 🟢 **Live** |

---

## 📑 Table of Contents

1. [🌟 Platform Highlights](#-platform-highlights)
2. [🧠 Deep Learning Architecture](#-deep-learning-architecture)
3. [🚀 Quick Start & Installation](#-quick-start--installation)
4. [⚡ Production API Reference](#-production-api-reference)
5. [🖥️ Interactive Web UI](#️-interactive-web-ui)
6. [📊 Persistent SQLite Analytics](#-persistent-sqlite-analytics)
7. [🧪 Testing & Benchmarks](#-testing--benchmarks)
8. [🐳 Docker Deployment](#-docker-deployment)
9. [📁 Project File Structure](#-project-file-structure)

---

## 🌟 Platform Highlights

* **Ultra-Fast Vectorized Inference Engine**: Pure mathematical NumPy/h5py forward pass executing trained BiGRU weights directly without heavy TensorFlow dependencies (<1.5ms latency).
* **Word Attribution Heatmap (Explainable AI)**: Saliency map computing leave-one-out word perturbation impact to highlight specific keywords driving predictions.
* **Paragraph Narrative Flow Arc (`/flow`)**: Sentence-by-sentence emotion timeline tracing emotional transitions (e.g. *Fear* $\rightarrow$ *Surprise* $\rightarrow$ *Joy*) over long text.
* **CSV & Document Batch Upload (`/upload`)**: Upload CSV, TXT, or JSON files to process hundreds of customer reviews or survey entries with distribution breakdown reports.
* **Real-Time Voice Dictation**: Hands-free voice speech-to-emotion analysis powered by the browser Web Speech API.
* **Persistent SQLite Analytics**: Logs queries, confidence scores, latencies, and sentiment metrics with 1-click historical replay.

---

## 🧠 Deep Learning Architecture

The emotion classification model evaluates text across **6 fundamental emotions**:
`joy` 😊, `sadness` 😢, `anger` 😡, `fear` 😨, `love` ❤️, `surprise` 😲.

```
Input Tokens (Max 50)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│ Embedding Layer (Vocab: 10,000 | Dimension: 300)        │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stacked BiGRU Layer 1 (Units: 128 | Return Sequences)   │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Dropout Layer (Rate: 0.5)                               │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Stacked BiGRU Layer 2 (Units: 64 | Concat Hidden State) │
└────────────────────────┬────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Dense Output Layer (Units: 6 | Softmax Activation)      │
└─────────────────────────────────────────────────────────┘
```

* **Training Accuracy**: 92.3% on `dair-ai/emotion` benchmark dataset.
* **Loss**: 0.243 (Sparse Categorical Crossentropy).

---

## 🚀 Quick Start & Installation

### 1. Clone Repository
```bash
git clone https://github.com/asm59-345/Emotion_Prediction_Model.git
cd Emotion_Prediction_Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Production Server Locally
```bash
python -m uvicorn src.main_prod:app --host 127.0.0.1 --port 8500 --reload
```

* **Interactive Web Interface**: [http://127.0.0.1:8500/](http://127.0.0.1:8500/)
* **Interactive OpenAPI Swagger Docs**: [http://127.0.0.1:8500/docs](http://127.0.0.1:8500/docs)

---

## ⚡ Production API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/predict` | Single text prediction with sentiment metrics and word attribution |
| `POST` | `/api/v1/predict/batch` | High-throughput vectorized batch prediction |
| `POST` | `/api/v1/predict/flow` | Paragraph narrative emotional arc & timeline |
| `POST` | `/api/v1/predict/upload` | Multipart file upload (CSV, TXT, JSON) with distribution report |
| `GET` | `/api/v1/analytics/summary` | Aggregated SQLite statistics & emotion breakdown |
| `GET` | `/api/v1/analytics/history` | Recent query log history |
| `GET` | `/health` / `/health/live` | Application and inference readiness telemetry |

### Sample Request: Single Prediction with Saliency
```bash
curl -X POST "https://moodline-pro.onrender.com/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so grateful and overjoyed with this support!", "explain": true}'
```

### Sample Response:
```json
{
  "text": "I am so grateful and overjoyed with this support!",
  "predicted_emotion": "joy",
  "emoji": "😊",
  "confidence": 0.9986,
  "sentiment": {
    "polarity": "positive",
    "intensity": 1.0
  },
  "all_probabilities": {
    "sadness": 0.0001,
    "joy": 0.9986,
    "love": 0.0008,
    "anger": 0.0002,
    "fear": 0.0001,
    "surprise": 0.0002
  },
  "latency_ms": 1.25,
  "attributions": [
    {"word": "grateful", "impact_score": 0.42},
    {"word": "overjoyed", "impact_score": 0.58}
  ]
}
```

---

## 🖥️ Interactive Web UI

The user interface provides:
* **Mode Tabs**: Switch seamlessly between *Sentence & Saliency*, *Paragraph Flow Arc*, and *File / CSV Batch*.
* **Dynamic Chromatic Glow**: Ambient background gradient reacts dynamically to detected emotions.
* **Word Heatmap**: Hoverable chips highlighting keyword impact percentages.
* **Voice Dictation**: Dictate sentences directly into the model.

---

## 📊 Persistent SQLite Analytics

Predictions and performance telemetry are logged to SQLite (`data/moodline_analytics.db`):
* Query counts and latency tracking.
* Emotion distribution charts.
* Average confidence metrics.
* 1-Click history query replay.

---

## 🧪 Testing & Benchmarks

Run the complete automated test suite:

```bash
python -m pytest tests/ -v
```

```
tests/test_api.py::test_health_endpoints PASSED
tests/test_api.py::test_predict_single_endpoint PASSED
tests/test_api.py::test_predict_batch_endpoint PASSED
tests/test_api.py::test_analytics_endpoints PASSED
tests/test_inference.py::test_tokenizer_loading PASSED
tests/test_inference.py::test_text_preprocessing PASSED
tests/test_inference.py::test_sequence_conversion PASSED
tests/test_inference.py::test_inference_engine PASSED
tests/test_inference.py::test_batch_inference PASSED
tests/test_inference.py::test_explainability PASSED
tests/test_inference.py::test_sentiment_analysis PASSED
tests/test_new_features.py::test_paragraph_flow_endpoint PASSED
tests/test_new_features.py::test_upload_csv_endpoint PASSED
tests/test_new_features.py::test_upload_txt_endpoint PASSED
======================== 14 passed in 1.16s ========================
```

---

## 🐳 Docker Deployment

### Run with Docker Compose
```bash
cd docker
docker-compose up --build -d
```

### Build Container Directly
```bash
docker build -t moodline-pro -f docker/Dockerfile .
docker run -p 8500:8000 moodline-pro
```

---

## 📁 Project File Structure

```
Emotion_Prediction_Model/
├── Artifacts/                     # Saved Keras BiGRU model and tokenizer
│   ├── BiGRU_Modle.keras
│   └── tokenizer.pkl
├── docker/                        # Production containerization
│   ├── Dockerfile
│   └── docker-compose.yml
├── src/                           # Production source package
│   ├── core/                      # Configuration, DB, and Logging
│   │   ├── config.py
│   │   ├── database.py
│   │   └── logger.py
│   ├── models/                    # Pydantic & SQLAlchemy Models
│   │   ├── db_models.py
│   │   ├── request.py
│   │   └── response.py
│   ├── services/                  # Business & Inference Logic
│   │   ├── analytics.py
│   │   ├── explainability.py
│   │   ├── inference.py
│   │   ├── sentiment.py
│   │   └── tokenizer.py
│   ├── api/                       # API Routers & Controllers
│   │   ├── v1/
│   │   │   ├── analytics.py
│   │   │   ├── health.py
│   │   │   └── predict.py
│   │   └── router.py
│   └── main_prod.py               # Production FastAPI App Entrypoint
├── static/                        # Frontend UI (HTML, CSS, JS)
│   ├── index.html
│   ├── script.js
│   └── style.css
├── tests/                         # Automated Pytest Suite
│   ├── test_api.py
│   ├── test_inference.py
│   └── test_new_features.py
├── .gitignore
├── architecture.md                # System Architecture & Technical Specifications
├── prompt.md                      # Prompt Engineering & LLM Integration Guide
├── SKILL.md                       # Antigravity/Agent Skill Definition
├── render.yaml                    # Render Deployment Blueprint
├── vercel.json                    # Vercel Deployment Configuration
├── main.py                        # Original legacy server
└── requirements.txt               # All production dependencies
```

---

## 📜 License
This project is licensed under the MIT License.
