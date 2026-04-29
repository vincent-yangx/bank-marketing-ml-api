import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_success():
    sample_input = {
        "features": {
            "age": 30,
            "job": "admin.",
            "marital": "married",
            "education": "secondary",
            "default": "no",
            "balance": 1000,
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "day_of_week": "1",
            "month": "may",
            "duration": 300,
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }
    }

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()