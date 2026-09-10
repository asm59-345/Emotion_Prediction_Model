---
name: emotion-intelligence-service
description: High-performance Emotion & Sentiment Intelligence skill powered by stacked BiGRU neural networks, featuring explainability, paragraph flow analysis, and SQLite analytics.
---

# Emotion & Sentiment Intelligence Skill

This skill provides comprehensive instructions, schemas, and best practices for interacting with the **Moodline Pro** deep learning emotion classification system.

## When to Use This Skill
- Classifying emotional tone (`joy`, `sadness`, `anger`, `fear`, `love`, `surprise`) in user queries, conversational text, and reviews.
- Explaining ML decisions using word-level attribution scores (saliency maps).
- Analyzing multi-sentence stories, chat logs, or customer feedback to extract sentiment shift over time.
- Processing bulk CSV, TXT, or JSON files for sentiment distribution reporting.

---

## 1. Direct Python Service Usage

```python
from src.services.inference import inference_engine
from src.services.sentiment import analyze_sentiment
from src.services.explainability import explain_prediction

# Single prediction
text = "I feel incredibly thrilled and grateful for this opportunity!"
all_probs, top_emotion, confidence, latency_ms = inference_engine.predict_single(text)

# Sentiment polarity & intensity
sentiment = analyze_sentiment(top_emotion, confidence, all_probs)

# Word attribution (Explainability)
attributions = explain_prediction(text, top_emotion)

print(f"Emotion: {top_emotion} ({confidence:.2%})")
print(f"Sentiment: {sentiment.polarity} (Intensity: {sentiment.intensity})")
for attr in attributions:
    if attr.impact_score > 0.1:
        print(f" - Key contributor: '{attr.word}' (Impact: {attr.impact_score:.2f})")
```

---

## 2. API Endpoints Quick Reference

| Action | HTTP Endpoint | Payload Example |
| :--- | :--- | :--- |
| **Predict Single** | `POST /api/v1/predict` | `{"text": "I feel happy", "explain": true}` |
| **Predict Batch** | `POST /api/v1/predict/batch` | `{"texts": ["I feel happy", "I feel angry"]}` |
| **Analyze Flow** | `POST /api/v1/predict/flow` | `{"text": "I was scared. But then I felt overjoyed!"}` |
| **Upload Document** | `POST /api/v1/predict/upload` | Multipart Form (`file`: `.csv`/`.txt`/`.json`) |
| **Analytics Summary** | `GET /api/v1/analytics/summary` | Query parameters: None |
| **Health Probe** | `GET /health/ready` | Query parameters: None |

---

## 3. Emotion Label Taxonomy & Valences

| Emotion | Emoji | Sentiment Valence | Typical Triggers |
| :--- | :--- | :--- | :--- |
| **`joy`** | 😊 | Positive | Happiness, excitement, gratitude, accomplishment |
| **`love`** | ❤️ | Positive | Affection, warmth, deep appreciation, care |
| **`surprise`** | 😲 | Neutral / Variable | Shock, unexpected events, amazement, realization |
| **`sadness`** | 😢 | Negative | Grief, loneliness, despair, regret, disappointment |
| **`anger`** | 😡 | Negative | Frustration, fury, resentment, perceived injustice |
| **`fear`** | 😨 | Negative | Dread, panic, anxiety, uncertainty, terror |

---

## 4. Integration Best Practices
1. **Explainability**: Always enable `"explain": true` when generating audit reports or customer sentiment breakdowns to visualize keyword attribution.
2. **Batching**: Use `/api/v1/predict/batch` or `/api/v1/predict/upload` for arrays > 5 sentences to leverage vectorized matrix multiplication.
3. **Database Telemetry**: Predictions are logged in SQLite (`data/moodline_analytics.db`) and queryable via `/api/v1/analytics/history`.
