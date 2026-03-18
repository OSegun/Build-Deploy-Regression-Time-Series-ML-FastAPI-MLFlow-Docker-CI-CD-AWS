from pydantic import BaseModel


class PredictionRequest(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    yr_built: int
    zipcode: str
    lat: float
    long: float


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_interval: dict
