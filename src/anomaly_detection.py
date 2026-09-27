"""
Anomaly Detection Module for Water Consumption Prediction
Analyzes prediction residuals (Actual - Predicted) using Rolling Z-Score
and Isolation Forest to identify potential abnormal consumption patterns.
Outputs results framed strictly as 'Potential abnormal consumption requiring human investigation'.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

PRED_FILE = os.path.join("results", "predictions", "test_predictions.csv")
RESULTS_FIGS_DIR = os.path.join("results", "figures")
ANOMALIES_OUT_FILE = os.path.join("results", "predictions", "flagged_anomalies.csv")

def run_anomaly_detection(z_threshold: float = 3.0) -> pd.DataFrame:
    os.makedirs(RESULTS_FIGS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ANOMALIES_OUT_FILE), exist_ok=True)

    df = pd.read_csv(PRED_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

    # Use best model residuals (LSTM / XGBoost)
    df['residual'] = df['actual_consumption'] - df['lstm_prediction']
    
    # 1. Dynamic Rolling Z-Score on Residuals (24-hour window)
    rolling_mean_res = df['residual'].rolling(window=24, min_periods=1).mean()
    rolling_std_res = df['residual'].rolling(window=24, min_periods=1).std().fillna(1.0)
    df['residual_zscore'] = (df['residual'] - rolling_mean_res) / (rolling_std_res + 1e-5)
    df['is_z_anomaly'] = df['residual_zscore'].abs() > z_threshold

    # 2. Isolation Forest on Residuals & Actual Consumption
    iso = IsolationForest(contamination=0.03, random_state=42)
    features_iso = df[['actual_consumption', 'residual']].values
    iso_preds = iso.fit_predict(features_iso)
    df['is_iso_anomaly'] = iso_preds == -1

    # Combined Anomaly Flag
    df['is_abnormal_consumption'] = df['is_z_anomaly'] | df['is_iso_anomaly']
    
    anomalies_df = df[df['is_abnormal_consumption']].copy().reset_index(drop=True)
    anomalies_df.to_csv(ANOMALIES_OUT_FILE, index=False)
    
    print("\n================ ANOMALY DETECTION SUMMARY ================")
    print(f"Total Test Observations Analyzed : {len(df)}")
    print(f"Flagged Abnormal Events           : {len(anomalies_df)} ({len(anomalies_df)/len(df):.2%})")
    print(f"Z-score Anomalies (|Z| > {z_threshold})    : {df['is_z_anomaly'].sum()}")
    print(f"Isolation Forest Anomalies        : {df['is_iso_anomaly'].sum()}")

    # Display Top 10 Anomalies Requiring Investigation
    print("\n--- Top 10 Flagged Abnormal Consumption Timestamps ---")
    cols_to_show = ['datetime', 'actual_consumption', 'lstm_prediction', 'residual', 'residual_zscore']
    print(anomalies_df.sort_values('residual', ascending=False)[cols_to_show].head(10).to_string(index=False))

    # Chart 17: Anomaly Timeline Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    ax1.plot(df['datetime'], df['actual_consumption'], color='black', alpha=0.6, label='Actual Consumption')
    ax1.plot(df['datetime'], df['lstm_prediction'], color='#2ca02c', alpha=0.8, linestyle='--', label='LSTM Prediction')
    ax1.scatter(anomalies_df['datetime'], anomalies_df['actual_consumption'], color='red', s=35, zorder=5, label='Potential Abnormal Consumption')
    ax1.set_title("17a. Water Consumption & Flagged Abnormal Periods", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Litres/hour")
    ax1.legend(loc='upper right')

    ax2.plot(df['datetime'], df['residual'], color='#1f77b4', alpha=0.7, label='Residual (Actual - Predicted)')
    ax2.axhline(0, color='black', linestyle=':', linewidth=1)
    ax2.scatter(anomalies_df['datetime'], anomalies_df['residual'], color='red', s=35, zorder=5, label='Unexplained Large Residual')
    ax2.set_title("17b. Prediction Residuals & Anomalous Spikes", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Date & Time")
    ax2.set_ylabel("Residual (Litres/hour)")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIGS_DIR, "17_anomaly_timeline.png"), dpi=300)
    plt.close()

    print(f"Saved anomaly timeline figure 17 to {RESULTS_FIGS_DIR}")
    return df, anomalies_df

if __name__ == "__main__":
    run_anomaly_detection()
