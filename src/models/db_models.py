from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from datetime import datetime, timezone
from src.core.database import Base

def utc_now():
    return datetime.now(timezone.utc)

class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    input_text = Column(Text, nullable=False)
    predicted_emotion = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    sentiment_polarity = Column(String(20), nullable=True) # positive, negative, neutral
    emotional_intensity = Column(Float, nullable=True) # 0.0 to 1.0
    all_probabilities = Column(JSON, nullable=False)
    latency_ms = Column(Float, nullable=False)
    client_ip = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=utc_now, index=True)
