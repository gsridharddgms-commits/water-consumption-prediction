"""
Preprocessing Module for Water Consumption Prediction
Handles dataset cleaning, time-series splitting (70/15/15 chronological),
outlier detection (IQR & Rolling Z-score), and feature scaling fitted strictly on training data.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import StandardScaler

MODELS_DIR = "models"

class TimeSeriesScaler:
    """Scaler wrapper that ensures fitting strictly on training set."""
    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_transform_train(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler on train_df feature_cols and return scaled DataFrame."""
        scaled_df = train_df.copy()
        scaled_features = self.scaler.fit_transform(train_df[self.feature_cols])
        scaled_df[self.feature_cols] = scaled_features
        self.is_fitted = True
        return scaled_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform df feature_cols using fitted scaler parameters."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted on training data before transforming validation/test sets.")
        scaled_df = df.copy()
        scaled_features = self.scaler.transform(df[self.feature_cols])
        scaled_df[self.feature_cols] = scaled_features
        return scaled_df

    def save(self, path: str = os.path.join(MODELS_DIR, "scaler.pkl")):
        """Save fitted scaler to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved fitted scaler to {path}")

    @classmethod
    def load(cls, path: str = os.path.join(MODELS_DIR, "scaler.pkl")):
        """Load fitted scaler from disk."""
        return joblib.load(path)

def detect_outliers(df: pd.DataFrame, column: str = 'water_consumption') -> pd.DataFrame:
    """
    Detect outliers using IQR and Rolling Z-Score.
    Adds indicator columns 'is_iqr_outlier' and 'is_z_outlier' without dropping rows.
    """
    res_df = df.copy()
    
    # 1. IQR Outlier Detection
    q25 = res_df[column].quantile(0.25)
    q75 = res_df[column].quantile(0.75)
    iqr = q75 - q25
    upper_bound = q75 + 3.0 * iqr  # Using 3.0 * IQR for extreme consumption spikes
    res_df['is_iqr_outlier'] = res_df[column] > upper_bound
    
    # 2. Rolling Z-score (24-hour window)
    rolling_mean = res_df[column].rolling(window=24, min_periods=1).mean()
    rolling_std = res_df[column].rolling(window=24, min_periods=1).std().fillna(1.0)
    z_score = (res_df[column] - rolling_mean) / (rolling_std + 1e-5)
    res_df['is_z_outlier'] = z_score.abs() > 3.5
    
    return res_df

def chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform strict chronological splitting without shuffling.
    Train: oldest train_ratio (70%)
    Val: next val_ratio (15%)
    Test: remaining (15%)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    print(f"Chronological Split Completed: Total={n}, Train={len(train_df)} ({len(train_df)/n:.1%}), Val={len(val_df)} ({len(val_df)/n:.1%}), Test={len(test_df)} ({len(test_df)/n:.1%})")
    return train_df, val_df, test_df

if __name__ == "__main__":
    from src.data_loader import load_and_merge_datasets
    hourly, _, _ = load_and_merge_datasets()
    hourly_outliers = detect_outliers(hourly)
    print("IQR Outliers count:", hourly_outliers['is_iqr_outlier'].sum())
    print("Z Outliers count:", hourly_outliers['is_z_outlier'].sum())
    train, val, test = chronological_split(hourly)
