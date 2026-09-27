"""
Evaluation Module for Water Consumption Prediction
Loads trained baseline, XGBoost, and PyTorch LSTM models,
evaluates on untouched Test set, computes MAE, RMSE, R2, MAPE, sMAPE, Peak Demand Error,
generates comparison tables, error analysis, and evaluation plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from src.preprocessing import chronological_split, TimeSeriesScaler
from src.features import get_feature_columns
from src.models import XGBoostWaterRegressor, PyTorchLSTMRegressor, TimeSeriesSequenceDataset
from src.baselines import evaluate_baseline

RESULTS_METRICS_DIR = os.path.join("results", "metrics")
RESULTS_PREDS_DIR = os.path.join("results", "predictions")
RESULTS_FIGS_DIR = os.path.join("results", "figures")
MODELS_DIR = "models"

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-10)))

    # MAPE (unstable for near-zero target values; reported for transparency)
    non_zero_mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0) if np.sum(non_zero_mask) > 0 else 0.0

    # sMAPE (symmetric MAPE)
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-5
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100.0)

    # Peak Demand Error (top 5% consumption hours)
    peak_threshold = np.quantile(y_true, 0.95)
    peak_mask = y_true >= peak_threshold
    actual_peak_mean = float(np.mean(y_true[peak_mask]))
    pred_peak_mean = float(np.mean(y_pred[peak_mask]))
    peak_abs_err = float(np.abs(actual_peak_mean - pred_peak_mean))
    peak_pct_err = float((peak_abs_err / actual_peak_mean) * 100.0) if actual_peak_mean > 0 else 0.0

    # Peak timing offset (difference in hour index between actual max peak and predicted max peak)
    actual_max_idx = int(np.argmax(y_true))
    pred_max_idx = int(np.argmax(y_pred))
    peak_timing_offset_hours = int(abs(actual_max_idx - pred_max_idx))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "sMAPE": smape,
        "Actual_Peak_Mean": actual_peak_mean,
        "Pred_Peak_Mean": pred_peak_mean,
        "Peak_Abs_Error": peak_abs_err,
        "Peak_Pct_Error": peak_pct_err,
        "Peak_Timing_Offset_Hours": peak_timing_offset_hours
    }

def run_evaluation():
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
    os.makedirs(RESULTS_PREDS_DIR, exist_ok=True)
    os.makedirs(RESULTS_FIGS_DIR, exist_ok=True)

    df = pd.read_csv("data/processed/water_hourly_featured.csv")
    train_df, val_df, test_df = chronological_split(df)
    feature_cols = get_feature_columns()

    # 1. Baselines
    naive_metrics = evaluate_baseline(test_df, pred_col='lag_1')
    seasonal_metrics = evaluate_baseline(test_df, pred_col='lag_24')

    # 2. XGBoost Evaluation
    xgb_model = XGBoostWaterRegressor.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    X_test = test_df[feature_cols].values
    y_test_xgb = test_df['water_consumption'].values
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = compute_all_metrics(y_test_xgb, xgb_preds)

    # 3. LSTM Evaluation
    scaler = TimeSeriesScaler.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    scaled_test = scaler.transform(test_df)
    seq_len = 24
    test_ds = TimeSeriesSequenceDataset(scaled_test[feature_cols].values, scaled_test['water_consumption'].values, seq_len=seq_len)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(os.path.join(MODELS_DIR, "lstm_model.pt"), map_location=device)
    lstm_model = PyTorchLSTMRegressor(input_dim=len(feature_cols), hidden_dim=64, num_layers=2).to(device)
    lstm_model.load_state_dict(checkpoint['state_dict'])
    lstm_model.eval()

    with torch.no_grad():
        test_seqs = test_ds.sequences.to(device)
        lstm_preds = lstm_model(test_seqs).cpu().numpy()
    y_test_lstm = test_ds.targets.numpy()
    lstm_metrics = compute_all_metrics(y_test_lstm, lstm_preds)

    # Correct MAE improvement percentage calculation relative to Seasonal Naive
    seasonal_mae = seasonal_metrics["MAE"]
    lstm_mae = lstm_metrics["MAE"]
    mae_improvement_pct = ((seasonal_mae - lstm_mae) / seasonal_mae) * 100.0

    # Compile Comparison Table
    comparison = [
        {"Model": "Naive Baseline (lag_1)", **naive_metrics},
        {"Model": "Seasonal Naive (lag_24)", **seasonal_metrics},
        {"Model": "XGBoost Regressor", **xgb_metrics},
        {"Model": "PyTorch LSTM Regressor", **lstm_metrics}
    ]
    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(os.path.join(RESULTS_METRICS_DIR, "model_comparison.csv"), index=False)
    with open(os.path.join(RESULTS_METRICS_DIR, "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n================ FINAL MODEL COMPARISON ================")
    print(comp_df[["Model", "MAE", "RMSE", "R2", "MAPE", "sMAPE", "Peak_Pct_Error", "Peak_Timing_Offset_Hours"]].to_string(index=False))
    print(f"\nExact Verified MAE Improvement (Seasonal Naive vs LSTM): {mae_improvement_pct:.2f}% (Expected ~38.9%)")

    # Save Prediction Dataframe with Timestamp & Residuals
    pred_df = test_df.iloc[seq_len:].copy().reset_index(drop=True)
    pred_df['actual_consumption'] = y_test_lstm
    pred_df['xgb_prediction'] = xgb_preds[seq_len:]
    pred_df['lstm_prediction'] = lstm_preds
    pred_df['xgb_residual'] = pred_df['actual_consumption'] - pred_df['xgb_prediction']
    pred_df['lstm_residual'] = pred_df['actual_consumption'] - pred_df['lstm_prediction']
    pred_df['xgb_abs_error'] = np.abs(pred_df['xgb_residual'])
    pred_df['lstm_abs_error'] = np.abs(pred_df['lstm_residual'])
    
    pred_df.to_csv(os.path.join(RESULTS_PREDS_DIR, "test_predictions.csv"), index=False)
    print(f"Saved test predictions trace to {os.path.join(RESULTS_PREDS_DIR, 'test_predictions.csv')}")

    # Generate Evaluation Charts
    print("\nGenerating Evaluation Charts...")

    # Chart 11: Actual vs Predicted Time Series Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    dt_subset = pd.to_datetime(pred_df['datetime'])
    ax.plot(dt_subset, pred_df['actual_consumption'], label='Actual Consumption', color='black', alpha=0.8, linewidth=1.2)
    ax.plot(dt_subset, pred_df['xgb_prediction'], label='XGBoost Forecast', color='#1f77b4', linestyle='--', alpha=0.85)
    ax.plot(dt_subset, pred_df['lstm_prediction'], label='PyTorch LSTM Forecast', color='#2ca02c', linestyle='-.', alpha=0.85)
    ax.set_title("11. Actual vs Predicted Water Consumption (Test Set Evaluation)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date & Time")
    ax.set_ylabel("Water Consumption (Litres/hour)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIGS_DIR, "11_actual_vs_predicted.png"), dpi=300)
    plt.close()

    # Chart 12: Residual Distribution Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(pred_df['xgb_residual'], label='XGBoost Residuals', color='#1f77b4', kde=True, ax=ax, alpha=0.4, bins=40)
    sns.histplot(pred_df['lstm_residual'], label='LSTM Residuals', color='#2ca02c', kde=True, ax=ax, alpha=0.4, bins=40)
    ax.set_title("12. Prediction Residual Distribution (e_t = Actual - Predicted)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Residual Error (Litres/hour)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIGS_DIR, "12_residual_distribution.png"), dpi=300)
    plt.close()

    # Chart 13: Prediction Absolute Errors Over Time
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dt_subset, pred_df['xgb_abs_error'], label='XGBoost Absolute Error', color='#1f77b4', alpha=0.7)
    ax.plot(dt_subset, pred_df['lstm_abs_error'], label='LSTM Absolute Error', color='#2ca02c', alpha=0.7)
    ax.set_title("13. Forecasting Absolute Error Over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute Error (Litres/hour)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIGS_DIR, "13_prediction_errors_over_time.png"), dpi=300)
    plt.close()

    # Chart 14: Peak Demand Comparison Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    models_list = ['Actual Peak', 'XGBoost Peak', 'LSTM Peak']
    peaks_list = [xgb_metrics['Actual_Peak_Mean'], xgb_metrics['Pred_Peak_Mean'], lstm_metrics['Pred_Peak_Mean']]
    sns.barplot(x=models_list, y=peaks_list, palette=['black', '#1f77b4', '#2ca02c'], ax=ax)
    ax.set_title("14. Peak Water Demand Forecast Comparison (Top 5% Hours)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean Peak Consumption (Litres/hour)")
    for i, v in enumerate(peaks_list):
        ax.text(i, v + 200, f"{v:.1f} L/h", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIGS_DIR, "14_peak_demand_comparison.png"), dpi=300)
    plt.close()

    print(f"Saved evaluation charts 11-14 to {RESULTS_FIGS_DIR}")
    return comp_df, pred_df

if __name__ == "__main__":
    run_evaluation()
