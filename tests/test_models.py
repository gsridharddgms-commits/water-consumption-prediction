"""
Unit tests for XGBoost and PyTorch LSTM model instantiation, training, and inference.
"""

import os
import pytest
import numpy as np
import pandas as pd
import torch

from src.models import XGBoostWaterRegressor, PyTorchLSTMRegressor, TimeSeriesSequenceDataset

def test_xgboost_fit_predict():
    X_train = np.random.randn(100, 10)
    y_train = np.random.rand(100) * 50
    X_test = np.random.randn(20, 10)
    
    xgb = XGBoostWaterRegressor(n_estimators=10)
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    
    assert len(preds) == 20
    assert not np.isnan(preds).any()

def test_lstm_forward_pass():
    model = PyTorchLSTMRegressor(input_dim=10, hidden_dim=16, num_layers=1)
    dummy_input = torch.randn(8, 24, 10)  # batch=8, seq_len=24, input_dim=10
    output = model(dummy_input)
    
    assert output.shape == (8,)
    assert not torch.isnan(output).any()
