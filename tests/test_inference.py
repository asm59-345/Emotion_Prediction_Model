import pytest
import numpy as np
from src.services.tokenizer import tokenizer_service
from src.services.inference import inference_engine
from src.services.explainability import explain_prediction
from src.services.sentiment import analyze_sentiment

def test_tokenizer_loading():
    assert tokenizer_service.is_loaded
    assert len(tokenizer_service.word_index) > 0

def test_text_preprocessing():
    raw = "I can't believe it's 100% true!  "
    clean = tokenizer_service.preprocess_text(raw)
    assert clean == "i cant believe its true"

def test_sequence_conversion():
    seq = tokenizer_service.texts_to_sequences(["I feel happy"])
    assert seq.shape == (1, 50)
    assert seq[0, 0] != 0 # first word index
    assert seq[0, -1] == 0 # post-padding

def test_inference_engine():
    assert inference_engine.is_loaded
    all_probs, top_emotion, confidence, latency_ms = inference_engine.predict_single("I feel so happy and overjoyed today!")
    assert top_emotion == "joy"
    assert confidence > 0.8
    assert latency_ms > 0.0
    assert sum(all_probs.values()) == pytest.approx(1.0, rel=1e-3)

def test_batch_inference():
    texts = [
        "I feel so happy and excited today!",
        "I feel terrified and scared of the dark",
        "I feel totally depressed and lonely and sad"
    ]
    probs, latency_ms = inference_engine.predict_batch(texts)
    assert probs.shape == (3, 6)
    assert np.argmax(probs[0]) == 1 # joy
    assert np.argmax(probs[1]) == 4 # fear
    assert np.argmax(probs[2]) == 0 # sadness

def test_explainability():
    attributions = explain_prediction("I feel so happy today", "joy")
    assert len(attributions) > 0
    words = [a.word for a in attributions]
    assert "happy" in words
    # 'happy' should have positive impact score
    happy_attr = next(a for a in attributions if a.word == "happy")
    assert happy_attr.impact_score > 0.0

def test_sentiment_analysis():
    sentiment = analyze_sentiment("joy", 0.98, {"joy": 0.98, "sadness": 0.01})
    assert sentiment.polarity == "positive"
    assert 0.0 <= sentiment.intensity <= 1.0
