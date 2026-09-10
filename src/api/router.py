from fastapi import APIRouter
from src.api.v1.predict import router as predict_router
from src.api.v1.analytics import router as analytics_router
from src.api.v1.health import router as health_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(predict_router)
api_v1_router.include_router(analytics_router)

main_router = APIRouter()
main_router.include_router(health_router)
main_router.include_router(api_v1_router)
