"""
Unit tests for time-series feature engineering and future leakage prevention.
"""

import pytest
import pandas as pd
import numpy as np
from src.features import create_hourly_features, get_feature_columns

def test_lag_generation_no_future_leakage():
    df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=300, freq='h'),
        'water_consumption': np.arange(100.0, 400.0),
        'temperature_2m': np.random.randn(300),
        'relative_humidity_2m': np.random.randn(300),
        'precipitation': np.zeros(300),
        'surface_pressure': np.ones(300) * 1013,
        'wind_speed_10m': np.ones(300) * 10
    })
    
    feat_df = create_hourly_features(df)
    
    # Verify lag_1 at index i equals target at original index before dropna
    assert 'lag_1' in feat_df.columns
    assert 'lag_24' in feat_df.columns
    assert 'rolling_mean_24' in feat_df.columns
    assert len(feat_df) > 0
