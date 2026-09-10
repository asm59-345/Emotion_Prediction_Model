from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class WordAttribution(BaseModel):
    word: str
    impact_score: float = Field(
        ...,
        description="Relative contribution of the word to the predicted emotion (-1.0 to 1.0)"
    )

class SentimentMetrics(BaseModel):
    polarity: str = Field(..., description="Overall sentiment polarity: positive, negative, or neutral")
    intensity: float = Field(..., description="Emotional arousal/intensity metric between 0.0 and 1.0")

class SinglePredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    sentiment: SentimentMetrics
    all_probabilities: Dict[str, float]
    latency_ms: float
    attributions: Optional[List[WordAttribution]] = None

class BatchPredictionResponse(BaseModel):
    total_samples: int
    predictions: List[SinglePredictionResponse]
    total_latency_ms: float

class FlowStep(BaseModel):
    step_index: int
    sentence: str
    emotion: str
    emoji: str
    confidence: float
    sentiment: SentimentMetrics

class ParagraphFlowResponse(BaseModel):
    total_sentences: int
    dominant_emotion: str
    narrative_arc: List[FlowStep]
    emotion_transitions: List[str]
    total_latency_ms: float

class FileUploadResponse(BaseModel):
    filename: str
    total_rows_processed: int
    emotion_distribution: Dict[str, int]
    dominant_emotion: str
    average_confidence: float
    sample_predictions: List[SinglePredictionResponse]
    processing_time_ms: float

class AnalyticsSummaryResponse(BaseModel):
    total_predictions: int
    emotion_distribution: Dict[str, int]
    average_confidence: float
    most_frequent_emotion: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    runtime: str
    memory_usage_mb: float
