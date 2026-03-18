from fastapi import FastAPI
from src.api.routers import predict

app = FastAPI(title="Housing Price Prediction API", version="0.1.0")

app.include_router(predict.router, prefix="/predict", tags=["predictions"])


@app.get("/health")
def health_check():
    return {"status": "ok"}
