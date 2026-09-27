"""
Explainability Module for Water Consumption Prediction
Computes XGBoost feature importances and SHAP values to explain
which historical consumption, weather, and calendar features drive predictions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.preprocessing import chronological_split
from src.features import get_feature_columns
from src.models import XGBoostWaterRegressor

MODELS_DIR = "models"
RESULTS_FIGS_DIR = os.path.join("results", "figures")
RESULTS_METRICS_DIR = os.path.join("results", "metrics")

def run_explainability():
    os.makedirs(RESULTS_FIGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)

    df = pd.read_csv("data/processed/water_hourly_featured.csv")
    _, _, test_df = chronological_split(df)
    feature_cols = get_feature_columns()

    xgb_model = XGBoostWaterRegressor.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    importances = xgb_model.get_feature_importances()
    
    imp_df = pd.DataFrame(list(importances.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False).reset_index(drop=True)
    imp_df.to_csv(os.path.join(RESULTS_METRICS_DIR, "feature_importances.csv"), index=False)
    
    print("\n--- Top 10 Most Important Features (XGBoost Gain) ---")
    print(imp_df.head(10).to_string(index=False))

    # Chart 15: Feature Importance Barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp_df.head(15), x='importance', y='feature', palette='mako', ax=ax)
    ax.set_title("15. Top 15 Feature Importances (XGBoost Model)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Relative Importance Score")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIGS_DIR, "15_feature_importance.png"), dpi=300)
    plt.close()

    # SHAP Explanation Analysis
    print("\nComputing SHAP Values for Model Explainability...")
    X_test = test_df[feature_cols].values
    explainer = shap.TreeExplainer(xgb_model.model)
    shap_values = explainer.shap_values(X_test)

    # Chart 16: SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_df[feature_cols], max_display=12, show=False)
    plt.title("16. SHAP Summary Plot (Feature Impact Distribution)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FIGS_DIR, "16_shap_summary.png"), dpi=300)
    plt.close()

    print(f"Saved explainability plots 15 and 16 to {RESULTS_FIGS_DIR}")
    return imp_df

if __name__ == "__main__":
    run_explainability()
