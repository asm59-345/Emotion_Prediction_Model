from fastapi import APIRouter
import sys
import psutil
import os
from src.core.config import settings
from src.models.response import HealthResponse
from src.services.inference import inference_engine

router = APIRouter(tags=["Health & Telemetry"])

@router.get("/health", response_model=HealthResponse)
@router.get("/health/live", response_model=HealthResponse)
def health_live():
    """Live probe to check application responsiveness."""
    process = psutil.Process(os.getpid())
    memory_mb = round(process.memory_info().rss / (1024 * 1024), 2)
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=inference_engine.is_loaded,
        runtime=f"Python {sys.version.split()[0]}",
        memory_usage_mb=memory_mb
    )

@router.get("/health/ready")
def health_ready():
    """Readiness probe verifying the model weights are loaded and ready for inference."""
    if not inference_engine.is_loaded:
        return {"ready": False, "reason": "Model weights are not loaded."}
    return {"ready": True, "model": "BiGRU (Phase 2)", "vocab_size": settings.VOCAB_SIZE}
