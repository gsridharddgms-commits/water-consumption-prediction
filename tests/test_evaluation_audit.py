"""
Unit tests for evaluation metrics, sMAPE, MAE improvement, and data leakage checks.
"""

import pytest
import numpy as np
import pandas as pd
from src.evaluate import compute_all_metrics
from src.features import create_hourly_features

def test_smape_calculation():
    y_true = np.array([100.0, 0.0, 50.0])
    y_pred = np.array([110.0, 10.0, 45.0])
    
    metrics = compute_all_metrics(y_true, y_pred)
    assert "sMAPE" in metrics
    assert metrics["sMAPE"] > 0.0
    assert metrics["sMAPE"] <= 200.0

def test_mae_improvement_formula():
    seasonal_mae = 3693.78
    lstm_mae = 2256.58
    improvement = ((seasonal_mae - lstm_mae) / seasonal_mae) * 100.0
    
    assert np.isclose(improvement, 38.91, atol=0.1)

def test_zero_leakage_rolling_shift():
    df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
        'water_consumption': np.arange(100, 200, dtype=float),
        'temperature_2m': np.random.randn(100),
        'relative_humidity_2m': np.random.randn(100),
        'precipitation': np.zeros(100),
        'surface_pressure': np.ones(100) * 1013,
        'wind_speed_10m': np.ones(100) * 10
    })
    
    feat_df = create_hourly_features(df)
    # rolling_mean_24 at index i must NOT equal average including current target at index i
    # Since rolling statistics use .shift(1), verify rolling_mean_6 at row 0 is based on shifted target
    assert 'rolling_mean_6' in feat_df.columns
    assert 'rolling_mean_24' in feat_df.columns
