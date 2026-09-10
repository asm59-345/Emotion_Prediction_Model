from typing import Dict
from src.models.response import SentimentMetrics

# Mapping emotion categories to baseline sentiment valence
EMOTION_VALENCE = {
    "joy": "positive",
    "love": "positive",
    "surprise": "neutral",
    "sadness": "negative",
    "anger": "negative",
    "fear": "negative"
}

def analyze_sentiment(emotion: str, confidence: float, probabilities: Dict[str, float]) -> SentimentMetrics:
    """Derives compound polarity and emotional arousal intensity from emotion probabilities."""
    polarity = EMOTION_VALENCE.get(emotion, "neutral")
    
    # Calculate positive vs negative score distribution
    pos_score = probabilities.get("joy", 0.0) + probabilities.get("love", 0.0)
    neg_score = probabilities.get("sadness", 0.0) + probabilities.get("anger", 0.0) + probabilities.get("fear", 0.0)
    
    # Intensity measures the sharpness/entropy of the emotion distribution
    intensity = float(min(1.0, max(0.1, confidence * 1.1 - (1.0 - confidence) * 0.2)))
    
    if abs(pos_score - neg_score) < 0.15 and emotion == "surprise":
        polarity = "neutral"
        
    return SentimentMetrics(
        polarity=polarity,
        intensity=round(intensity, 4)
    )
