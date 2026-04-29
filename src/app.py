from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.infer import load_model, predict_one

app = FastAPI(
    title = "Bank Marketing Prediction API",
    description = "A FastAPI service for predicting whether a client will subscribe to a term deposit.",
    version = "1.0.0"
)

REQUIRED_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day_of_week",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]

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
    missing_features = {
        feature for feature in REQUIRED_FEATURES
        if feature not in request.features
    }

    if missing_features:
        raise HTTPException(
            status_code = 400,
            detail = f"Missing required features: {', '.join(missing_features)}"
        )
    

    try:
        result = predict_one(model, request.features)
        return result
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"Prediction failed: {str(e)}"
        )