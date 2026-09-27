"""
Unit tests for data loading and profiling.
"""

import os
import pytest
import pandas as pd
from src.data_loader import profile_dataset, PROCESSED_DIR

def test_dataset_files_exist():
    hourly_path = os.path.join(PROCESSED_DIR, "water_hourly_processed.csv")
    assert os.path.exists(hourly_path), f"Processed hourly dataset missing at {hourly_path}"

def test_dataset_profiling():
    hourly_path = os.path.join(PROCESSED_DIR, "water_hourly_processed.csv")
    df = pd.read_csv(hourly_path)
    prof = profile_dataset(df, time_col='datetime', target_col='water_consumption')
    
    assert prof['num_rows'] > 1000
    assert prof['missing_values']['water_consumption'] == 0
    assert prof['duplicate_timestamps'] == 0
    assert 'min_timestamp' in prof
