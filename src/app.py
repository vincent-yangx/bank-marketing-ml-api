from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.infer import load_model, predict_one
from src.db import init_db, insert_prediction_log, get_recent_prediction_logs

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
init_db()

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
        insert_prediction_log(request.features, result["prediction"], result["probability"])
        return result
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"Prediction failed: {str(e)}"
        )
    
@app.get("/log")
def get_logs(limit: int = 10):
    try:
        logs = get_recent_prediction_logs(limit = limit)
        return {
            "count": len(logs),
            "logs": logs
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch prediction logs: {str(e)}"
        )
