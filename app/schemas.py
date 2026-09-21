from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]