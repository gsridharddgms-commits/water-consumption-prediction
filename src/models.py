"""
Models Module for Water Consumption Prediction
Implements classical ML (XGBoost / HistGradientBoosting) and Deep Learning
(PyTorch LSTM and Temporal Transformer) model architectures.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

MODELS_DIR = "models"

# ---------------------------------------------------------
# 1. Classical Machine Learning Model: XGBoost Regressor
# ---------------------------------------------------------
class XGBoostWaterRegressor:
    """XGBoost Regressor model wrapper for time-series forecasting."""
    def __init__(self, **params):
        default_params = {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': 42,
            'objective': 'reg:absoluteerror',  # MAE loss
            'n_jobs': -1
        }
        default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_names = []

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, feature_names: List[str] = None):
        self.feature_names = feature_names or []
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            return dict(zip(self.feature_names, self.model.feature_importances_.astype(float)))
        return {}

    def save(self, path: str = os.path.join(MODELS_DIR, "xgboost_model.pkl")):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names, "params": self.params}, path)
        print(f"Saved XGBoost model to {path}")

    @classmethod
    def load(cls, path: str = os.path.join(MODELS_DIR, "xgboost_model.pkl")):
        data = joblib.load(path)
        instance = cls(**data["params"])
        instance.model = data["model"]
        instance.feature_names = data["feature_names"]
        return instance

# ---------------------------------------------------------
# 2. PyTorch Dataset for Sequence Modeling
# ---------------------------------------------------------
class TimeSeriesSequenceDataset(Dataset):
    """Dataset for PyTorch sequence models (LSTM / Transformer)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 24):
        self.seq_len = seq_len
        self.sequences = []
        self.targets = []
        
        for i in range(len(X) - seq_len):
            self.sequences.append(X[i : i + seq_len])
            self.targets.append(y[i + seq_len])
            
        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

# ---------------------------------------------------------
# 3. Deep Learning Sequence Model: PyTorch LSTM Regressor
# ---------------------------------------------------------
class PyTorchLSTMRegressor(nn.Module):
    """LSTM Sequence Network for Time-Series Water Consumption Forecasting."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(PyTorchLSTMRegressor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Use last timestep output
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)

# ---------------------------------------------------------
# 4. Temporal Transformer Regressor
# ---------------------------------------------------------
class TemporalTransformerRegressor(nn.Module):
    """Transformer Model with Multi-Head Attention for Time-Series Forecasting."""
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super(TemporalTransformerRegressor, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        h = self.input_proj(x)
        h = self.transformer_encoder(h)
        # Average pooling across sequence dimension
        out = h.mean(dim=1)
        out = self.fc(out)
        return out.squeeze(-1)
