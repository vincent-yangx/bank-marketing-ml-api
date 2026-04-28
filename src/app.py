from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.infer import load_model, predict_one

app = FastAPI(
    title = "Bank Marketing Prediction API",
    description = "A FastAPI service for predicting whether a client will subscribe to a term deposit.",
    version = "1.0.0"
)

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    
model = load_model()

@app.get("/")
def root():
    return {
        "message": "Bank Marketing Prediction API is running."
    }

@app.get("/health") 
def health_check():
    return {
        "status": "ok"
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_one(model, request.features)
        return result
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"Prediction failed: {str(e)}"
        )