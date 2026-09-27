# Water Consumption Prediction System 💧

> An academic machine learning system for residential and municipal water consumption forecasting, explainability, residual anomaly investigation, and model serving.

---

## 📌 Project Overview
This project predicts future water demand using historical smart-meter consumption records, matched historical weather observations (temperature, relative humidity, precipitation, pressure, wind speed), calendar context, and cyclical time encodings.

### Key Highlights
- **Real Smart-Meter Dataset**: 4,831 hourly smart-meter observations matched with Open-Meteo historical weather data (temperature, relative humidity, precipitation, pressure, wind speed) covering August 13, 2022 – March 2, 2023.
- **Leakage Prevention**: The implemented pipeline was designed and verified to prevent temporal data leakage. Rolling statistics use previous observations only (`.shift(1)` is applied where required), scalers are fitted exclusively on the training set, the dataset split is strictly chronological (70% Train, 15% Validation, 15% Test), and the final test set is not used for hyperparameter tuning.
- **Model Comparison**: Evaluates Naive & Seasonal Naive baselines, XGBoost Regressor, and PyTorch LSTM Sequence Regressor.
- **Explainability**: XGBoost feature importances & SHAP TreeExplainer analysis.
- **Residual Anomaly Detection**: Identifies 28 potential abnormal consumption events using rolling Z-scores ($|Z| > 3.0$) and Isolation Forest.
- **Deployment Layer**: Multi-page Streamlit Dashboard and FastAPI REST API for model serving and academic demonstration.

---

## 🏆 Verified Experimental Results (Untouched Test Set)

| Model Architecture | MAE (Litres/hour) | RMSE (Litres/hour) | $R^2$ Score | MAPE (%) | sMAPE (%) | Peak Demand Error (%) | Peak Timing Offset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (`lag_1`)** | 3,879.01 | 7,597.19 | -0.9155 | 10,433.71% | 132.74% | 85.27% | N/A |
| **Seasonal Naive (`lag_24`)** | 3,693.78 | 7,611.54 | -0.9228 | 11,695.88% | 136.31% | 88.05% | N/A |
| **XGBoost Regressor** | 2,543.77 | 5,585.45 | -0.0354 | 4,608.41% | 166.31% | 94.08% | 229 hrs |
| **PyTorch LSTM Regressor** | **2,256.58** | **5,927.60** | **-0.1604** | **205.77%** | **178.06%** | **99.70%** | **366 hrs** |

---

## 📝 Academic Interpretation of Results

- **MAE Improvement (38.9%)**: Calculated directly from the untouched test-set results. Using Seasonal Naive MAE = 3693.78 L/hour and LSTM MAE = 2256.58 L/hour:

  `((3693.78 − 2256.58) / 3693.78) × 100 = 38.91%`

  Therefore, among the evaluated models, the PyTorch LSTM achieved the lowest MAE and reduced MAE by approximately **38.9%** compared with the Seasonal Naive baseline.

- **Unstable MAPE**: Standard MAPE is extremely large because very low or near-zero actual consumption values can produce very large percentage errors. Therefore, MAPE is not used as the primary model-selection metric. MAE and RMSE are emphasized for comparison, while sMAPE is additionally reported because it is less sensitive to very small actual values than standard MAPE.

- **Negative R²**: A negative R² indicates that the model's predictions have higher squared error than a constant prediction based on the mean target value on the evaluated test set. The LSTM achieved the lowest MAE among the evaluated models, but its negative R² indicates that substantial variation in test-set consumption remains unexplained.

- **Peak Demand Error**: Although the LSTM reduced overall MAE, peak-demand prediction remained challenging, with substantial peak magnitude and timing errors. Therefore, the model should not be interpreted as a highly reliable peak-demand forecasting system based on the current evaluation.

### Dataset Duration Limitation
The dataset covers August 13, 2022 to March 2, 2023, which is less than one full calendar year. Therefore, the current evaluation does not fully represent annual seasonal variation. Generalization to unseen years, unusual weather conditions, or substantially different consumption patterns requires additional data and validation.

---

## 📁 Repository Structure

```text
water-consumption-prediction/
│
├── data/
│   ├── raw/                  # Original raw water and weather logs
│   ├── processed/            # Cleaned, resampled, and engineered datasets
│   └── README.md             # Dataset documentation & lineage
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── data_loader.py        # Dataset acquisition & Open-Meteo weather integration
│   ├── generate_eda.py       # EDA chart generation
│   ├── preprocessing.py      # Chronological 70/15/15 split & train-only scaling
│   ├── features.py           # Zero-leakage time-series feature engineering
│   ├── baselines.py          # Naive & Seasonal Naive baseline evaluation
│   ├── models.py             # XGBoost & PyTorch LSTM model classes
│   ├── train.py              # Training loops, early stopping & experiment logger
│   ├── evaluate.py           # Evaluation metrics & figure generators
│   ├── explain.py            # Feature importance & SHAP explainability
│   └── anomaly_detection.py  # Residual rolling Z-score & Isolation Forest anomaly flagger
│
├── models/                   # Saved fitted scalers and model checkpoints
├── experiments/              # Experiment logs (experiments.csv)
│
├── results/
│   ├── figures/              # 17 high-resolution evaluation & EDA charts (PNG)
│   ├── metrics/              # Model comparison summary (JSON/CSV)
│   └── predictions/          # Test set prediction trace & flagged anomalies
│
├── app/
│   ├── dashboard.py          # Interactive Streamlit dashboard
│   └── api.py                # FastAPI REST API for model serving and academic demonstration
│
├── tests/                    # Automated unit & integration pytest suite
│
├── requirements.txt
├── README.md
└── MODEL_CARD.md
```

---

## ⚙️ Quickstart & Reproduction

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Complete ML Pipeline
Execute the pipeline components sequentially:
```bash
python src/data_loader.py

python src/generate_eda.py

python src/features.py

python -m src.baselines

python -m src.train

python -m src.evaluate

python -m src.explain

python -m src.anomaly_detection
```

### 3. Run Automated Tests
```bash
pytest
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```

### 5. Run FastAPI REST API for Model Serving
```bash
uvicorn app.api:app --reload --port 8000
```
- Interactive Swagger documentation available at `http://127.0.0.1:8000/docs`.

---

## ⚠️ Limitations

### Dataset Coverage
The current dataset covers less than one full year, limiting evaluation of complete annual seasonal patterns.

### Model Performance
Although the LSTM achieved the lowest MAE among the evaluated models, its negative R² and high peak-demand errors indicate that substantial prediction limitations remain.

### MAPE
MAPE is unstable when actual consumption values are very small or near zero and therefore is not used as the primary model-selection metric.

### Peak Demand
The current models show substantial peak-demand magnitude and timing errors.

### Anomaly Detection
Detected anomalies are potential abnormal consumption events based on prediction residuals. They are not confirmed leaks or infrastructure failures.

### Generalization
Performance on other locations, years, weather conditions, household populations, or infrastructure configurations requires additional data and validation.

### Operational Use
The system is an academic forecasting and analysis prototype. Predictions should not be treated as guaranteed operational decisions without additional validation and monitoring.

---

## 🔒 Governance & Ethics Notice
The residual-based anomaly detection pipeline identified 28 potential abnormal consumption events requiring further investigation. These represent unusually large prediction residuals and do not confirm physical leaks, pipe failures, meter faults, or illegal usage without independent ground-truth evidence.
