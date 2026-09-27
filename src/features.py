"""
Feature Engineering Module for Water Consumption Prediction
Generates time-series lag features, rolling statistics, calendar context,
and cyclical encodings strictly without future data leakage.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List

PROCESSED_DIR = os.path.join("data", "processed")

def create_hourly_features(df: pd.DataFrame, target_col: str = 'water_consumption') -> pd.DataFrame:
    """
    Constructs comprehensive time-series features for hourly data.
    Ensures zero future leakage by applying .shift(1) on all rolling operations.
    """
    feat_df = df.copy()
    feat_df['datetime'] = pd.to_datetime(feat_df['datetime'], utc=True)
    feat_df = feat_df.sort_values('datetime').reset_index(drop=True)

    # 1. Calendar & Temporal Features
    feat_df['hour'] = feat_df['datetime'].dt.hour
    feat_df['day_of_week'] = feat_df['datetime'].dt.dayofweek
    feat_df['day_of_month'] = feat_df['datetime'].dt.day
    feat_df['month'] = feat_df['datetime'].dt.month
    feat_df['year'] = feat_df['datetime'].dt.year
    feat_df['is_weekend'] = (feat_df['day_of_week'] >= 5).astype(int)

    # 2. Cyclical Sin/Cos Encodings
    feat_df['sin_hour'] = np.sin(2 * np.pi * feat_df['hour'] / 24.0)
    feat_df['cos_hour'] = np.cos(2 * np.pi * feat_df['hour'] / 24.0)
    feat_df['sin_day_of_week'] = np.sin(2 * np.pi * feat_df['day_of_week'] / 7.0)
    feat_df['cos_day_of_week'] = np.cos(2 * np.pi * feat_df['day_of_week'] / 7.0)
    feat_df['sin_month'] = np.sin(2 * np.pi * feat_df['month'] / 12.0)
    feat_df['cos_month'] = np.cos(2 * np.pi * feat_df['month'] / 12.0)

    # 3. Lag Features (Strict past observations)
    # lag_1: t-1 (previous hour)
    # lag_2: t-2 (2 hours ago)
    # lag_24: t-24 (same hour yesterday)
    # lag_168: t-168 (same hour last week)
    feat_df['lag_1'] = feat_df[target_col].shift(1)
    feat_df['lag_2'] = feat_df[target_col].shift(2)
    feat_df['lag_24'] = feat_df[target_col].shift(24)
    feat_df['lag_168'] = feat_df[target_col].shift(168)

    # 4. Rolling Statistics (Shifted by 1 to prevent taking current target into rolling mean)
    shifted_target = feat_df[target_col].shift(1)
    feat_df['rolling_mean_6'] = shifted_target.rolling(window=6, min_periods=1).mean()
    feat_df['rolling_mean_24'] = shifted_target.rolling(window=24, min_periods=1).mean()
    feat_df['rolling_std_24'] = shifted_target.rolling(window=24, min_periods=1).std().fillna(0)
    feat_df['rolling_mean_168'] = shifted_target.rolling(window=168, min_periods=1).mean()

    # 5. Weather Features & Interactions (if present)
    if 'temperature_2m' in feat_df.columns and 'relative_humidity_2m' in feat_df.columns:
        feat_df['temp_humidity_index'] = feat_df['temperature_2m'] * (feat_df['relative_humidity_2m'] / 100.0)

    # Drop initial rows with NaNs resulting from max lag (168 hours = 7 days)
    # To keep exact alignment and clean dataset
    feat_df = feat_df.dropna().reset_index(drop=True)
    return feat_df

def get_feature_columns() -> List[str]:
    """Return the list of engineered predictor feature names."""
    return [
        'temperature_2m', 'relative_humidity_2m', 'precipitation', 'surface_pressure', 'wind_speed_10m',
        'hour', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend',
        'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week', 'sin_month', 'cos_month',
        'lag_1', 'lag_2', 'lag_24', 'lag_168',
        'rolling_mean_6', 'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168',
        'temp_humidity_index'
    ]

if __name__ == "__main__":
    hourly_path = os.path.join(PROCESSED_DIR, "water_hourly_processed.csv")
    df = pd.read_csv(hourly_path)
    feat_df = create_hourly_features(df)
    out_path = os.path.join(PROCESSED_DIR, "water_hourly_featured.csv")
    feat_df.to_csv(out_path, index=False)
    print(f"Engineered features shape: {feat_df.shape}")
    print(f"Saved featured dataset to {out_path}")
