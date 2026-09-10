from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from src.core.database import get_db
from src.models.response import AnalyticsSummaryResponse
from src.services.analytics import get_analytics_summary, get_recent_history

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/summary", response_model=AnalyticsSummaryResponse)
def fetch_analytics_summary(db: Session = Depends(get_db)):
    """Returns aggregated prediction metrics, emotion distributions, and confidence averages."""
    return get_analytics_summary(db)

@router.get("/history")
def fetch_recent_history(
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Fetches the latest prediction history records from SQLite."""
    records = get_recent_history(db, limit=limit)
    return [
        {
            "id": r.id,
            "input_text": r.input_text,
            "predicted_emotion": r.predicted_emotion,
            "confidence": r.confidence,
            "sentiment_polarity": r.sentiment_polarity,
            "emotional_intensity": r.emotional_intensity,
            "latency_ms": r.latency_ms,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in records
    ]
