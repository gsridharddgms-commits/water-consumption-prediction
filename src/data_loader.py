"""
Data Loader Module for Water Consumption Prediction
Fetches real smart meter time-series data, merges historical weather data,
profiles the raw/merged datasets, and performs initial validation.
"""

import os
import json
import urllib.request
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

# Data URLs and parameters
WATER_DATA_URL = "https://raw.githubusercontent.com/LukaLike/water-consumption-data/main/minutely-water-consumption-data.csv"
# Zagreb coordinates matching dataset collection region
LATITUDE = 45.815
LONGITUDE = 15.9819

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")

def download_water_data(target_path: str = os.path.join(RAW_DIR, "water_consumption_raw.csv")) -> str:
    """Download real minutely water consumption dataset if not present."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        print(f"Downloading raw water dataset from {WATER_DATA_URL}...")
        req = urllib.request.Request(WATER_DATA_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(target_path, 'wb') as f:
            f.write(resp.read())
        print(f"Saved raw water dataset to {target_path}")
    else:
        print(f"Raw water dataset already exists at {target_path}")
    return target_path

def download_weather_data(start_date: str, end_date: str, target_path: str = os.path.join(RAW_DIR, "weather_raw.csv")) -> str:
    """Download matching hourly historical weather data from Open-Meteo API."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path):
        print(f"Downloading matching weather data from Open-Meteo ({start_date} to {end_date})...")
        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={LATITUDE}&longitude={LONGITUDE}&start_date={start_date}&end_date={end_date}&"
            f"hourly=temperature_2m,relative_humidity_2m,precipitation,surface_pressure,wind_speed_10m"
        )
        req = urllib.request.Request(weather_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
        
        hourly = data['hourly']
        weather_df = pd.DataFrame({
            'time': pd.to_datetime(hourly['time'], utc=True),
            'temperature_2m': hourly['temperature_2m'],
            'relative_humidity_2m': hourly['relative_humidity_2m'],
            'precipitation': hourly['precipitation'],
            'surface_pressure': hourly['surface_pressure'],
            'wind_speed_10m': hourly['wind_speed_10m']
        })
        weather_df.to_csv(target_path, index=False)
        print(f"Saved raw weather dataset to {target_path}")
    else:
        print(f"Raw weather dataset already exists at {target_path}")
    return target_path

def profile_dataset(df: pd.DataFrame, time_col: str, target_col: str) -> Dict[str, Any]:
    """Generate comprehensive dataset profile statistics."""
    profile = {}
    profile["num_rows"] = len(df)
    profile["num_cols"] = len(df.columns)
    profile["columns"] = list(df.columns)
    profile["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    profile["missing_values"] = df.isnull().sum().to_dict()
    profile["duplicate_rows"] = int(df.duplicated().sum())
    profile["duplicate_timestamps"] = int(df[time_col].duplicated().sum()) if time_col in df else 0
    
    if time_col in df:
        ts = pd.to_datetime(df[time_col])
        profile["min_timestamp"] = str(ts.min())
        profile["max_timestamp"] = str(ts.max())
        diffs = ts.sort_values().diff()
        profile["sampling_frequency_mode"] = str(diffs.mode().iloc[0]) if not diffs.empty else "N/A"
        profile["is_sorted"] = bool(ts.is_monotonic_increasing)
    
    if target_col in df:
        profile["target_summary"] = {
            "mean": float(df[target_col].mean()),
            "std": float(df[target_col].std()),
            "min": float(df[target_col].min()),
            "25%": float(df[target_col].quantile(0.25)),
            "50%": float(df[target_col].median()),
            "75%": float(df[target_col].quantile(0.75)),
            "max": float(df[target_col].max()),
            "zero_count": int((df[target_col] == 0).sum())
        }
    return profile

def load_and_merge_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Downloads water & weather raw data, resamples water dataset to hourly,
    merges with weather data, and saves initial processed dataset.
    Returns:
        hourly_df: DataFrame resampled to hourly resolution
        daily_df: DataFrame resampled to daily resolution
        profile_info: Profiling dictionary
    """
    water_path = download_water_data()
    raw_water = pd.read_csv(water_path)
    
    # Parse timestamp: epoch seconds to UTC datetime
    raw_water['datetime'] = pd.to_datetime(raw_water['created_at'], unit='s', utc=True)
    raw_water = raw_water.sort_values('datetime').reset_index(drop=True)
    
    # Profile raw water dataset
    raw_profile = profile_dataset(raw_water, 'datetime', 'sensor_value')
    print("--- Raw Water Dataset Profile ---")
    print(f"Total Raw Records: {raw_profile['num_rows']}")
    print(f"Date Range: {raw_profile['min_timestamp']} to {raw_profile['max_timestamp']}")
    
    start_date = pd.to_datetime(raw_profile['min_timestamp']).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(raw_profile['max_timestamp']).strftime('%Y-%m-%d')
    
    weather_path = download_weather_data(start_date, end_date)
    weather_df = pd.read_csv(weather_path)
    weather_df['datetime'] = pd.to_datetime(weather_df['time'], utc=True)
    
    # Resample raw minutely water dataset to hourly sum (litres/hour)
    raw_water_idx = raw_water.set_index('datetime')
    hourly_water = raw_water_idx['sensor_value'].resample('1h').sum().reset_index()
    hourly_water.rename(columns={'sensor_value': 'water_consumption'}, inplace=True)
    
    # Merge with weather data on hourly datetime
    merged_hourly = pd.merge(hourly_water, weather_df[['datetime', 'temperature_2m', 'relative_humidity_2m', 'precipitation', 'surface_pressure', 'wind_speed_10m']], on='datetime', how='left')
    
    # Resample to daily sum as well
    daily_water = raw_water_idx['sensor_value'].resample('1D').sum().reset_index()
    daily_water.rename(columns={'sensor_value': 'water_consumption'}, inplace=True)
    daily_weather = weather_df.set_index('datetime').resample('1D').agg({
        'temperature_2m': 'mean',
        'relative_humidity_2m': 'mean',
        'precipitation': 'sum',
        'surface_pressure': 'mean',
        'wind_speed_10m': 'mean'
    }).reset_index()
    merged_daily = pd.merge(daily_water, daily_weather, on='datetime', how='left')
    
    # Save processed datasets
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    hourly_out = os.path.join(PROCESSED_DIR, "water_hourly_processed.csv")
    daily_out = os.path.join(PROCESSED_DIR, "water_daily_processed.csv")
    
    merged_hourly.to_csv(hourly_out, index=False)
    merged_daily.to_csv(daily_out, index=False)
    print(f"Saved hourly dataset to {hourly_out} (Shape: {merged_hourly.shape})")
    print(f"Saved daily dataset to {daily_out} (Shape: {merged_daily.shape})")
    
    profile_info = profile_dataset(merged_hourly, 'datetime', 'water_consumption')
    return merged_hourly, merged_daily, profile_info

if __name__ == "__main__":
    hourly, daily, prof = load_and_merge_datasets()
    print("Hourly Dataset Head:\n", hourly.head())
    print("\nHourly Profiling Summary:\n", json.dumps(prof, indent=2))
