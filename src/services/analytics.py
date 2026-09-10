from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Dict, Any, List
from datetime import datetime, timezone
from src.models.db_models import PredictionRecord
from src.models.response import AnalyticsSummaryResponse
from src.core.logger import logger

def log_prediction(
    db: Session,
    input_text: str,
    predicted_emotion: str,
    confidence: float,
    sentiment_polarity: str,
    emotional_intensity: float,
    all_probabilities: Dict[str, float],
    latency_ms: float,
    client_ip: str = None
) -> PredictionRecord:
    """Persists a prediction record to the SQLite SQL database."""
    try:
        record = PredictionRecord(
            input_text=input_text,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            sentiment_polarity=sentiment_polarity,
            emotional_intensity=emotional_intensity,
            all_probabilities=all_probabilities,
            latency_ms=latency_ms,
            client_ip=client_ip,
            created_at=datetime.now(timezone.utc)
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log prediction to SQLite: {e}")
        return None

def get_analytics_summary(db: Session) -> AnalyticsSummaryResponse:
    """Computes summary statistics directly using SQL queries."""
    total = db.query(func.count(PredictionRecord.id)).scalar() or 0
    if total == 0:
        return AnalyticsSummaryResponse(
            total_predictions=0,
            emotion_distribution={},
            average_confidence=0.0,
            most_frequent_emotion=None
        )
        
    avg_conf = db.query(func.avg(PredictionRecord.confidence)).scalar() or 0.0
    
    # Emotion counts
    distribution_query = (
        db.query(PredictionRecord.predicted_emotion, func.count(PredictionRecord.id))
        .group_by(PredictionRecord.predicted_emotion)
        .all()
    )
    dist = {emotion: count for emotion, count in distribution_query}
    most_frequent = max(dist.items(), key=lambda x: x[1])[0] if dist else None
    
    return AnalyticsSummaryResponse(
        total_predictions=total,
        emotion_distribution=dist,
        average_confidence=round(float(avg_conf), 4),
        most_frequent_emotion=most_frequent
    )

def get_recent_history(db: Session, limit: int = 20) -> List[PredictionRecord]:
    """Fetches recent prediction records."""
    return db.query(PredictionRecord).order_by(desc(PredictionRecord.created_at)).limit(limit).all()
