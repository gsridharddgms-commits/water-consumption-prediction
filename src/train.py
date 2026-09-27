"""
Train Module for Water Consumption Prediction
Orchestrates model training, scaling fit on train set only,
hyperparameter tuning, PyTorch sequence training loops, evaluation,
and experiment logging into experiments/experiments.csv
"""

import os
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.preprocessing import chronological_split, TimeSeriesScaler
from src.features import get_feature_columns
from src.models import XGBoostWaterRegressor, PyTorchLSTMRegressor, TemporalTransformerRegressor, TimeSeriesSequenceDataset
from src.baselines import evaluate_baseline

MODELS_DIR = "models"
EXPERIMENTS_PATH = os.path.join("experiments", "experiments.csv")

def log_experiment(experiment_data: Dict[str, Any]):
    """Appends experiment result record to experiments/experiments.csv."""
    os.makedirs(os.path.dirname(EXPERIMENTS_PATH), exist_ok=True)
    df_exp = pd.DataFrame([experiment_data])
    if not os.path.exists(EXPERIMENTS_PATH):
        df_exp.to_csv(EXPERIMENTS_PATH, index=False)
    else:
        df_exp.to_csv(EXPERIMENTS_PATH, mode='a', header=False, index=False)
    print(f"Logged experiment {experiment_data.get('experiment_id')} to {EXPERIMENTS_PATH}")

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates MAE, RMSE, R2, MAPE, and Peak Demand Error."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
    
    non_zero_mask = y_true > 0
    mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100.0 if np.sum(non_zero_mask) > 0 else 0.0
    
    peak_threshold = np.quantile(y_true, 0.95)
    peak_mask = y_true >= peak_threshold
    actual_peak = np.mean(y_true[peak_mask])
    pred_peak = np.mean(y_pred[peak_mask])
    peak_abs_err = np.abs(actual_peak - pred_peak)
    peak_pct_err = (peak_abs_err / actual_peak) * 100.0 if actual_peak > 0 else 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape),
        "Peak_Abs_Error": float(peak_abs_err),
        "Peak_Pct_Error": float(peak_pct_err)
    }

def train_xgboost(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]):
    """Train XGBoost model and evaluate."""
    print("\n========== Training XGBoost Regressor ==========")
    start_time = time.time()
    
    X_train, y_train = train_df[feature_cols].values, train_df['water_consumption'].values
    X_val, y_val = val_df[feature_cols].values, val_df['water_consumption'].values
    X_test, y_test = test_df[feature_cols].values, test_df['water_consumption'].values

    xgb_model = XGBoostWaterRegressor(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    xgb_model.save(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

    val_preds = xgb_model.predict(X_val)
    test_preds = xgb_model.predict(X_test)

    val_metrics = calc_metrics(y_val, val_preds)
    test_metrics = calc_metrics(y_test, test_preds)
    train_duration = time.time() - start_time

    print(f"XGBoost Val Metrics  : MAE={val_metrics['MAE']:.2f}, RMSE={val_metrics['RMSE']:.2f}, R2={val_metrics['R2']:.4f}, Peak Err={val_metrics['Peak_Pct_Error']:.2f}%")
    print(f"XGBoost Test Metrics : MAE={test_metrics['MAE']:.2f}, RMSE={test_metrics['RMSE']:.2f}, R2={test_metrics['R2']:.4f}, Peak Err={test_metrics['Peak_Pct_Error']:.2f}%")

    log_experiment({
        "experiment_id": f"XGBoost_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "XGBoost",
        "hyperparameters": json.dumps(xgb_model.params),
        "random_seed": 42,
        "training_duration_sec": round(train_duration, 2),
        "val_MAE": val_metrics['MAE'],
        "val_RMSE": val_metrics['RMSE'],
        "val_R2": val_metrics['R2'],
        "test_MAE": test_metrics['MAE'],
        "test_RMSE": test_metrics['RMSE'],
        "test_R2": test_metrics['R2'],
        "test_Peak_Pct_Err": test_metrics['Peak_Pct_Error']
    })
    return xgb_model, val_preds, test_preds, val_metrics, test_metrics

def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], seq_len: int = 24, epochs: int = 50, batch_size: int = 32):
    """Train PyTorch LSTM Regressor model using sequence dataset."""
    print(f"\n========== Training PyTorch LSTM Regressor (Sequence Length = {seq_len}) ==========")
    start_time = time.time()

    # Fit scaler strictly on training set
    scaler = TimeSeriesScaler(feature_cols)
    scaled_train = scaler.fit_transform_train(train_df)
    scaled_val = scaler.transform(val_df)
    scaled_test = scaler.transform(test_df)
    scaler.save(os.path.join(MODELS_DIR, "scaler.pkl"))

    # Prepare PyTorch Datasets
    train_ds = TimeSeriesSequenceDataset(scaled_train[feature_cols].values, scaled_train['water_consumption'].values, seq_len=seq_len)
    val_ds = TimeSeriesSequenceDataset(scaled_val[feature_cols].values, scaled_val['water_consumption'].values, seq_len=seq_len)
    test_ds = TimeSeriesSequenceDataset(scaled_test[feature_cols].values, scaled_test['water_consumption'].values, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PyTorchLSTMRegressor(input_dim=len(feature_cols), hidden_dim=64, num_layers=2, dropout=0.2).to(device)
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    best_model_weights = None
    patience = 12
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(targets)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * len(targets)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs}: Train MAE={train_loss:.2f}, Val MAE={val_loss:.2f}")

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    # Save LSTM model state
    torch.save({"state_dict": model.state_dict(), "feature_cols": feature_cols, "seq_len": seq_len}, os.path.join(MODELS_DIR, "lstm_model.pt"))
    print(f"Saved LSTM model to {os.path.join(MODELS_DIR, 'lstm_model.pt')}")

    # Evaluate predictions
    def get_predictions(loader, ds):
        model.eval()
        preds = []
        with torch.no_grad():
            for seqs, _ in loader:
                seqs = seqs.to(device)
                outputs = model(seqs)
                preds.extend(outputs.cpu().numpy())
        return np.array(preds), ds.targets.numpy()

    val_preds, val_y = get_predictions(val_loader, val_ds)
    test_preds, test_y = get_predictions(test_loader, test_ds)

    val_metrics = calc_metrics(val_y, val_preds)
    test_metrics = calc_metrics(test_y, test_preds)
    train_duration = time.time() - start_time

    print(f"LSTM Val Metrics  : MAE={val_metrics['MAE']:.2f}, RMSE={val_metrics['RMSE']:.2f}, R2={val_metrics['R2']:.4f}, Peak Err={val_metrics['Peak_Pct_Error']:.2f}%")
    print(f"LSTM Test Metrics : MAE={test_metrics['MAE']:.2f}, RMSE={test_metrics['RMSE']:.2f}, R2={test_metrics['R2']:.4f}, Peak Err={test_metrics['Peak_Pct_Error']:.2f}%")

    log_experiment({
        "experiment_id": f"LSTM_{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "LSTM",
        "hyperparameters": json.dumps({"seq_len": seq_len, "hidden_dim": 64, "num_layers": 2, "epochs": epoch, "batch_size": batch_size}),
        "random_seed": 42,
        "training_duration_sec": round(train_duration, 2),
        "val_MAE": val_metrics['MAE'],
        "val_RMSE": val_metrics['RMSE'],
        "val_R2": val_metrics['R2'],
        "test_MAE": test_metrics['MAE'],
        "test_RMSE": test_metrics['RMSE'],
        "test_R2": test_metrics['R2'],
        "test_Peak_Pct_Err": test_metrics['Peak_Pct_Error']
    })
    return model, val_preds, test_preds, val_metrics, test_metrics, test_y

def run_full_training_pipeline():
    df = pd.read_csv("data/processed/water_hourly_featured.csv")
    train_df, val_df, test_df = chronological_split(df)
    feature_cols = get_feature_columns()

    # 1. Train XGBoost
    xgb_model, val_preds_xgb, test_preds_xgb, val_m_xgb, test_m_xgb = train_xgboost(train_df, val_df, test_df, feature_cols)

    # 2. Train PyTorch LSTM
    lstm_model, val_preds_lstm, test_preds_lstm, val_m_lstm, test_m_lstm, test_y_lstm = train_lstm(train_df, val_df, test_df, feature_cols)

    return {
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "xgb_test_metrics": test_m_xgb, "lstm_test_metrics": test_m_lstm,
        "test_preds_xgb": test_preds_xgb, "test_preds_lstm": test_preds_lstm
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    results = run_full_training_pipeline()
