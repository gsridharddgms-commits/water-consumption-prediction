"""
FastAPI REST Service for Water Consumption Prediction
Provides /health status and /predict real-time inference endpoints.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODELS_DIR = "models"
XGB_PATH = os.path.join(MODELS_DIR, "xgboost_model.pkl")

# Load model globally on module import
xgb_model = None
feature_names = []

if os.path.exists(XGB_PATH):
    data = joblib.load(XGB_PATH)
    xgb_model = data["model"]
    feature_names = data["feature_names"]
    print(f"FastAPI successfully loaded model from {XGB_PATH}")

app = FastAPI(
    title="Water Consumption Prediction API",
    description="FastAPI REST API for model serving and academic demonstration.",
    version="1.0.0"
)

# Input Pydantic Schema for Prediction Payload
class WaterPredictionRequest(BaseModel):
    temperature_2m: float = Field(..., example=22.5, description="Temperature in °C")
    relative_humidity_2m: float = Field(..., example=65.0, description="Relative Humidity in %")
    precipitation: float = Field(default=0.0, example=0.0, description="Precipitation in mm")
    surface_pressure: float = Field(default=1013.25, example=1013.25, description="Surface pressure in hPa")
    wind_speed_10m: float = Field(default=10.0, example=10.0, description="Wind speed in km/h")
    hour: int = Field(..., ge=0, le=23, example=14, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, example=2, description="Day of week (0=Mon, 6=Sun)")
    day_of_month: int = Field(default=15, ge=1, le=31, example=15)
    month: int = Field(..., ge=1, le=12, example=8, description="Month (1-12)")
    year: int = Field(default=2023, example=2023)
    lag_1: float = Field(..., example=1500.0, description="Previous hour water consumption in L/h")
    lag_24: float = Field(..., example=1800.0, description="Yesterday same hour consumption in L/h")
    rolling_mean_24: float = Field(..., example=1650.0, description="24h rolling mean consumption in L/h")

class WaterPredictionResponse(BaseModel):
    prediction: float
    unit: str = "litres/hour"
    model_version: str = "xgboost_v1.0"
    status: str = "success"

@app.get("/health")
def health_check():
    """Health check endpoint returning system status and model readiness."""
    return {
        "status": "healthy" if xgb_model is not None else "degraded",
        "model_loaded": xgb_model is not None,
        "target_unit": "litres/hour",
        "model_version": "xgboost_v1.0"
    }

@app.post("/predict", response_model=WaterPredictionResponse)
def predict_water_consumption(request: WaterPredictionRequest):
    """Generate live water consumption prediction from input features."""
    if xgb_model is None:
        raise HTTPException(status_code=500, detail="Model artifact is not loaded.")

    is_weekend = 1 if request.day_of_week >= 5 else 0

    feat_dict = {
        'temperature_2m': request.temperature_2m,
        'relative_humidity_2m': request.relative_humidity_2m,
        'precipitation': request.precipitation,
        'surface_pressure': request.surface_pressure,
        'wind_speed_10m': request.wind_speed_10m,
        'hour': request.hour,
        'day_of_week': request.day_of_week,
        'day_of_month': request.day_of_month,
        'month': request.month,
        'year': request.year,
        'is_weekend': is_weekend,
        'sin_hour': float(np.sin(2 * np.pi * request.hour / 24.0)),
        'cos_hour': float(np.cos(2 * np.pi * request.hour / 24.0)),
        'sin_day_of_week': float(np.sin(2 * np.pi * request.day_of_week / 7.0)),
        'cos_day_of_week': float(np.cos(2 * np.pi * request.day_of_week / 7.0)),
        'sin_month': float(np.sin(2 * np.pi * request.month / 12.0)),
        'cos_month': float(np.cos(2 * np.pi * request.month / 12.0)),
        'lag_1': request.lag_1,
        'lag_2': request.lag_1,
        'lag_24': request.lag_24,
        'lag_168': request.lag_24,
        'rolling_mean_6': request.lag_1,
        'rolling_mean_24': request.rolling_mean_24,
        'rolling_std_24': request.rolling_mean_24 * 0.2,
        'rolling_mean_168': request.rolling_mean_24,
        'temp_humidity_index': request.temperature_2m * (request.relative_humidity_2m / 100.0)
    }

    try:
        input_array = np.array([[feat_dict[col] for col in feature_names]])
        raw_pred = float(xgb_model.predict(input_array)[0])
        pred_val = round(max(0.0, raw_pred), 2)
        
        return WaterPredictionResponse(
            prediction=pred_val,
            unit="litres/hour",
            model_version="xgboost_v1.0",
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
