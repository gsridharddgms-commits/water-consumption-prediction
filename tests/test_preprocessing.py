"""
Unit tests for data preprocessing and chronological splitting.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import chronological_split, detect_outliers, TimeSeriesScaler

def test_chronological_split():
    df = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
        'water_consumption': np.random.rand(100) * 100
    })
    train, val, test = chronological_split(df, train_ratio=0.70, val_ratio=0.15)
    
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert train['datetime'].max() < val['datetime'].min()
    assert val['datetime'].max() < test['datetime'].min()

def test_scaler_no_leakage():
    train_df = pd.DataFrame({'feat': [1.0, 2.0, 3.0, 4.0, 5.0]})
    test_df = pd.DataFrame({'feat': [10.0, 20.0]})
    
    scaler = TimeSeriesScaler(['feat'])
    scaled_train = scaler.fit_transform_train(train_df)
    scaled_test = scaler.transform(test_df)
    
    assert scaler.is_fitted
    assert np.isclose(scaled_train['feat'].mean(), 0.0, atol=1e-5)
