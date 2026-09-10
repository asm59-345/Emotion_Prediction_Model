from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import time
import uuid

from src.core.config import settings
from src.core.database import Base, engine
from src.core.logger import logger
from src.api.router import main_router
from src.models.request import SingleTextInput
from src.models.response import SinglePredictionResponse
from src.services.inference import inference_engine
from src.services.sentiment import analyze_sentiment
from src.services.analytics import log_prediction
from src.core.database import SessionLocal

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully")
    
    # Check Inference engine
    if inference_engine.is_loaded:
        logger.info("Production BiGRU Engine loaded and ready for inference")
    else:
        logger.warning("Inference engine could not be initialized")
        
    yield
    logger.info("Application shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade Emotion & Sentiment Intelligence Platform with high-performance BiGRU inference.",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing & Correlation ID Middleware
@app.middleware("http")
async def add_process_time_and_trace(request: Request, call_next):
    start_time = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    response.headers["X-Request-ID"] = request_id
    return response

# Include Main Routers
app.include_router(main_router)

# Mount Static UI files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serves the frontend interface."""
    return FileResponse("static/index.html")

# Backwards-compatibility root endpoints matching legacy contract
@app.post("/predict", include_in_schema=False)
def legacy_predict(payload: SingleTextInput, request: Request):
    """Legacy compatibility endpoint returning exact shape expected by original frontend."""
    all_probs, top_emotion, confidence, latency_ms = inference_engine.predict_single(payload.text)
    sentiment = analyze_sentiment(top_emotion, confidence, all_probs)
    
    # Save to SQLite
    db = SessionLocal()
    try:
        log_prediction(
            db=db,
            input_text=payload.text,
            predicted_emotion=top_emotion,
            confidence=confidence,
            sentiment_polarity=sentiment.polarity,
            emotional_intensity=sentiment.intensity,
            all_probabilities=all_probs,
            latency_ms=latency_ms,
            client_ip=request.client.host if request.client else "unknown"
        )
    finally:
        db.close()
        
    return {
        "text": payload.text,
        "predicted_emotion": top_emotion,
        "confidence": confidence,
        "all_probabilites": all_probs, # matching legacy frontend key
        "all_probabilities": all_probs
    }
