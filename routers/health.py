from fastapi import APIRouter
from src.predict import get_model
from config import MLFLOW_MODEL_NAME, MODEL_VERSION

router = APIRouter()

@router.get("/health")
def health():
    try:
        get_model()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "model": MLFLOW_MODEL_NAME,
        "version": MODEL_VERSION,
        "model_status": model_status,
    }