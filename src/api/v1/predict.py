from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.orm import Session
import time
import re
import csv
import io
import json
import numpy as np
from src.core.database import get_db
from src.core.config import settings
from src.models.request import SingleTextInput, BatchTextInput, ParagraphFlowInput
from src.models.response import (
    SinglePredictionResponse,
    BatchPredictionResponse,
    ParagraphFlowResponse,
    FlowStep,
    FileUploadResponse
)
from src.services.inference import inference_engine
from src.services.sentiment import analyze_sentiment
from src.services.explainability import explain_prediction
from src.services.analytics import log_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.post("", response_model=SinglePredictionResponse)
def predict_single(
    payload: SingleTextInput,
    request: Request,
    db: Session = Depends(get_db)
):
    """Predicts emotion, sentiment polarity, intensity, and optional word-level attribution."""
    if not inference_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Inference engine is not loaded.")
        
    all_probs, top_emotion, confidence, latency_ms = inference_engine.predict_single(payload.text)
    sentiment = analyze_sentiment(top_emotion, confidence, all_probs)
    
    attributions = None
    if payload.explain:
        attributions = explain_prediction(payload.text, top_emotion)
        
    client_ip = request.client.host if request.client else "unknown"
    
    log_prediction(
        db=db,
        input_text=payload.text,
        predicted_emotion=top_emotion,
        confidence=confidence,
        sentiment_polarity=sentiment.polarity,
        emotional_intensity=sentiment.intensity,
        all_probabilities=all_probs,
        latency_ms=latency_ms,
        client_ip=client_ip
    )
    
    return SinglePredictionResponse(
        text=payload.text,
        predicted_emotion=top_emotion,
        emoji=settings.EMOTION_EMOJIS.get(top_emotion, "✨"),
        confidence=confidence,
        sentiment=sentiment,
        all_probabilities=all_probs,
        latency_ms=round(latency_ms, 2),
        attributions=attributions
    )

@router.post("/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchTextInput,
    request: Request,
    db: Session = Depends(get_db)
):
    """High-throughput batch inference over multiple text inputs."""
    if not inference_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Inference engine is not loaded.")
        
    batch_start = time.perf_counter()
    probs_matrix, _ = inference_engine.predict_batch(payload.texts)
    
    predictions = []
    client_ip = request.client.host if request.client else "unknown"
    
    for i, text in enumerate(payload.texts):
        probs = probs_matrix[i]
        top_idx = int(np.argmax(probs))
        top_emotion = settings.EMOTION_LABELS[top_idx]
        confidence = float(probs[top_idx])
        all_probs = {label: float(p) for label, p in zip(settings.EMOTION_LABELS, probs)}
        sentiment = analyze_sentiment(top_emotion, confidence, all_probs)
        
        attributions = None
        if payload.explain:
            attributions = explain_prediction(text, top_emotion)
            
        single_latency = round((time.perf_counter() - batch_start) * 1000.0 / len(payload.texts), 2)
        
        log_prediction(
            db=db,
            input_text=text,
            predicted_emotion=top_emotion,
            confidence=confidence,
            sentiment_polarity=sentiment.polarity,
            emotional_intensity=sentiment.intensity,
            all_probabilities=all_probs,
            latency_ms=single_latency,
            client_ip=client_ip
        )
        
        predictions.append(
            SinglePredictionResponse(
                text=text,
                predicted_emotion=top_emotion,
                emoji=settings.EMOTION_EMOJIS.get(top_emotion, "✨"),
                confidence=confidence,
                sentiment=sentiment,
                all_probabilities=all_probs,
                latency_ms=single_latency,
                attributions=attributions
            )
        )
        
    total_latency_ms = round((time.perf_counter() - batch_start) * 1000.0, 2)
    return BatchPredictionResponse(
        total_samples=len(payload.texts),
        predictions=predictions,
        total_latency_ms=total_latency_ms
    )

@router.post("/flow", response_model=ParagraphFlowResponse)
def predict_paragraph_flow(
    payload: ParagraphFlowInput,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Deconstructs a paragraph into constituent sentences and tracks the emotional arc / narrative progression.
    """
    if not inference_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Inference engine is not loaded.")
        
    start_time = time.perf_counter()
    
    # Robust sentence splitting that preserves short fragments
    raw_sentences = [s.strip() for s in re.split(r'[.!?\n]+', payload.text) if s.strip()]
    if not raw_sentences:
        raw_sentences = [payload.text.strip()]
        
    probs_matrix, _ = inference_engine.predict_batch(raw_sentences)
    
    flow_steps: list[FlowStep] = []
    emotions: list[str] = []
    client_ip = request.client.host if request.client else "unknown"
    
    for i, sent in enumerate(raw_sentences):
        probs = probs_matrix[i]
        top_idx = int(np.argmax(probs))
        top_emotion = settings.EMOTION_LABELS[top_idx]
        confidence = float(probs[top_idx])
        all_probs = {label: float(p) for label, p in zip(settings.EMOTION_LABELS, probs)}
        sentiment = analyze_sentiment(top_emotion, confidence, all_probs)
        
        emotions.append(top_emotion)
        flow_steps.append(
            FlowStep(
                step_index=i + 1,
                sentence=sent,
                emotion=top_emotion,
                emoji=settings.EMOTION_EMOJIS.get(top_emotion, "✨"),
                confidence=round(confidence, 4),
                sentiment=sentiment
            )
        )
        
    # Determine dominant emotion
    dominant = max(set(emotions), key=emotions.count)
    
    # Track transition states
    transitions = []
    for i in range(len(emotions) - 1):
        if emotions[i] != emotions[i + 1]:
            transitions.append(f"{emotions[i]} -> {emotions[i + 1]}")
            
    total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    
    # Save flow summary record to SQLite
    log_prediction(
        db=db,
        input_text=payload.text,
        predicted_emotion=dominant,
        confidence=flow_steps[0].confidence if flow_steps else 0.9,
        sentiment_polarity=flow_steps[0].sentiment.polarity if flow_steps else "neutral",
        emotional_intensity=flow_steps[0].sentiment.intensity if flow_steps else 0.5,
        all_probabilities={dominant: 1.0},
        latency_ms=total_latency_ms,
        client_ip=client_ip
    )
    
    return ParagraphFlowResponse(
        total_sentences=len(raw_sentences),
        dominant_emotion=dominant,
        narrative_arc=flow_steps,
        emotion_transitions=transitions,
        total_latency_ms=total_latency_ms
    )

@router.post("/upload", response_model=FileUploadResponse)
async def upload_and_process_file(file: UploadFile = File(...)):
    """
    Processes uploaded CSV, TXT, or JSON files containing text batches and returns aggregated emotional intelligence.
    """
    if not inference_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Inference engine is not loaded.")
        
    start_time = time.perf_counter()
    content_bytes = await file.read()
    
    try:
        text_content = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text_content = content_bytes.decode("latin-1")
        
    lines: list[str] = []
    filename = file.filename or "uploaded_file"
    
    if filename.endswith(".csv"):
        reader = csv.reader(io.StringIO(text_content))
        for row in reader:
            if row:
                line = " ".join(row).strip()
                if len(line) > 1:
                    lines.append(line)
    elif filename.endswith(".json"):
        try:
            data = json.loads(text_content)
            if isinstance(data, list):
                lines = [str(item) for item in data if len(str(item).strip()) > 1]
            elif isinstance(data, dict):
                lines = [str(v) for v in data.values() if len(str(v).strip()) > 1]
        except Exception:
            lines = [l.strip() for l in text_content.splitlines() if len(l.strip()) > 1]
    else:
        lines = [l.strip() for l in text_content.splitlines() if len(l.strip()) > 1]
        
    if not lines:
        raise HTTPException(status_code=400, detail="No readable text lines found in file.")
        
    # Limit batch to first 200 lines for quick responsive processing
    lines = lines[:200]
    
    probs_matrix, _ = inference_engine.predict_batch(lines)
    
    emotion_counts: dict[str, int] = {e: 0 for e in settings.EMOTION_LABELS}
    confidences: list[float] = []
    samples: list[SinglePredictionResponse] = []
    
    for i, line in enumerate(lines):
        probs = probs_matrix[i]
        top_idx = int(np.argmax(probs))
        top_emotion = settings.EMOTION_LABELS[top_idx]
        confidence = float(probs[top_idx])
        all_probs = {label: float(p) for label, p in zip(settings.EMOTION_LABELS, probs)}
        sentiment = analyze_sentiment(top_emotion, confidence, all_probs)
        
        emotion_counts[top_emotion] += 1
        confidences.append(confidence)
        
        if i < 5:  # Keep first 5 as samples
            samples.append(
                SinglePredictionResponse(
                    text=line[:120] + ("..." if len(line) > 120 else ""),
                    predicted_emotion=top_emotion,
                    emoji=settings.EMOTION_EMOJIS.get(top_emotion, "✨"),
                    confidence=round(confidence, 4),
                    sentiment=sentiment,
                    all_probabilities=all_probs,
                    latency_ms=0.0
                )
            )
            
    dominant = max(emotion_counts.items(), key=lambda x: x[1])[0]
    avg_conf = round(float(np.mean(confidences)), 4) if confidences else 0.0
    total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    
    return FileUploadResponse(
        filename=filename,
        total_rows_processed=len(lines),
        emotion_distribution=emotion_counts,
        dominant_emotion=dominant,
        average_confidence=avg_conf,
        sample_predictions=samples,
        processing_time_ms=total_latency_ms
    )
