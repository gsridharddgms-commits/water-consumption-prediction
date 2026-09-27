"""
Baseline Models Module
Implements Naive Baseline (lag_1) and Seasonal Naive Baseline (lag_24 / lag_168)
and measures baseline metrics for comparison against machine learning models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

def evaluate_baseline(df: pd.DataFrame, y_true_col: str = 'water_consumption', pred_col: str = 'lag_1') -> Dict[str, float]:
    """Calculate evaluation metrics for a baseline prediction column."""
    y_true = df[y_true_col].values
    y_pred = df[pred_col].values

    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # R2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))

    # MAPE (unstable with near-zero targets; kept for transparency)
    non_zero_mask = y_true > 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0
    else:
        mape = 0.0

    # sMAPE (symmetric MAPE)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-5
    smape = np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0

    # Peak Demand Error (top 5% consumption hours)
    peak_threshold = np.quantile(y_true, 0.95)
    peak_mask = y_true >= peak_threshold
    actual_peak_mean = np.mean(y_true[peak_mask])
    pred_peak_mean = np.mean(y_pred[peak_mask])
    peak_abs_error = np.abs(actual_peak_mean - pred_peak_mean)
    peak_pct_error = (peak_abs_error / actual_peak_mean) * 100.0 if actual_peak_mean > 0 else 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
        "sMAPE": float(smape),
        "Actual_Peak_Mean": float(actual_peak_mean),
        "Pred_Peak_Mean": float(pred_peak_mean),
        "Peak_Abs_Error": float(peak_abs_error),
        "Peak_Pct_Error": float(peak_pct_error)
    }

def run_baselines(test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluates Naive (lag_1) and Seasonal Naive (lag_24) baselines on test_df."""
    results = {}
    results["Naive Baseline (lag_1)"] = evaluate_baseline(test_df, pred_col='lag_1')
    results["Seasonal Naive Baseline (lag_24)"] = evaluate_baseline(test_df, pred_col='lag_24')
    results["Seasonal Naive Baseline (lag_168)"] = evaluate_baseline(test_df, pred_col='lag_168')
    return results

if __name__ == "__main__":
    from src.preprocessing import chronological_split
    df = pd.read_csv("data/processed/water_hourly_featured.csv")
    _, val_df, test_df = chronological_split(df)
    
    print("\n--- Validation Baselines ---")
    val_baselines = run_baselines(val_df)
    for k, v in val_baselines.items():
        print(f"{k}: MAE={v['MAE']:.2f}, RMSE={v['RMSE']:.2f}, R2={v['R2']:.4f}, sMAPE={v['sMAPE']:.2f}%, Peak Pct Err={v['Peak_Pct_Error']:.2f}%")
        
    print("\n--- Test Baselines ---")
    test_baselines = run_baselines(test_df)
    for k, v in test_baselines.items():
        print(f"{k}: MAE={v['MAE']:.2f}, RMSE={v['RMSE']:.2f}, R2={v['R2']:.4f}, sMAPE={v['sMAPE']:.2f}%, Peak Pct Err={v['Peak_Pct_Error']:.2f}%")
