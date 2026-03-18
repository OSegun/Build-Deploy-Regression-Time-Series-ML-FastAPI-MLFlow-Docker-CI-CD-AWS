from fastapi import APIRouter
from src.api.schemas import PredictionRequest, PredictionResponse

router = APIRouter()


@router.post("/", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # TD: load model and run inference
    return PredictionResponse(
        predicted_price=0.0,
        confidence_interval={"lower": 0.0, "upper": 0.0},
    )
