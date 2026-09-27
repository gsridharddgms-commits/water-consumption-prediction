"""
Integration tests for FastAPI endpoints (/health and /predict).
"""

import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert "status" in json_resp
    assert json_resp["target_unit"] == "litres/hour"

def test_predict_endpoint():
    payload = {
        "temperature_2m": 22.5,
        "relative_humidity_2m": 60.0,
        "precipitation": 0.0,
        "surface_pressure": 1013.25,
        "wind_speed_10m": 10.0,
        "hour": 14,
        "day_of_week": 2,
        "day_of_month": 15,
        "month": 8,
        "year": 2023,
        "lag_1": 1500.0,
        "lag_24": 1800.0,
        "rolling_mean_24": 1650.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    json_resp = response.json()
    assert "prediction" in json_resp
    assert json_resp["unit"] == "litres/hour"
    assert json_resp["status"] == "success"
    assert json_resp["prediction"] >= 0.0
