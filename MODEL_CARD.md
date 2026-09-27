# Model Card — Water Consumption Prediction System

## Model Details
- **Model Name**: Water Consumption Forecasting System (PyTorch LSTM & XGBoost Regressor)
- **Model Version**: 1.0.0 (Academic Demonstration)
- **Model Types**: 
  1. **Primary Deep Learning Model**: PyTorch 2-Layer LSTM Sequence Regressor (64 hidden units, dropout=0.2, sequence length = 24 hours)
  2. **Primary Classical ML Model**: XGBoost Regressor (350 estimators, learning_rate=0.03, max_depth=6)
- **Developer**: Machine Learning Systems Engineering Team
- **Date**: September 2026

## Intended Use
- **Primary Domain**: Residential and municipal hourly water consumption forecasting.
- **Intended Users**: Water resource managers, municipal utility analysts, and academic researchers.
- **Primary Application**: Short-term (hourly/daily) demand forecasting prototype for municipal pump scheduling, storage tank capacity planning, and demand management research.

## Out-of-Scope Uses
> [!CAUTION]
> The model MUST NOT be used for:
> 1. Autonomous physical water shutoff or automated valve control without human engineer oversight.
> 2. Definitive physical leak detection (flagged anomalies indicate *unusually large prediction residuals requiring human investigation*, not verified physical leaks).
> 3. Individual customer billing or legal enforcement.
> 4. Utility regions significantly outside the geographical and climate distribution represented in the training data.

## Training & Validation Methodology
- **Dataset**: 4,831 hourly observations combining real smart-meter water demand logs with matched Open-Meteo historical weather observations (temperature, relative humidity, precipitation, pressure, wind speed).
- **Time-Series Splitting**: Strict chronological split without shuffling:
  - **Training Set (70%)**: 3,264 hourly observations
  - **Validation Set (15%)**: 699 hourly observations
  - **Untouched Test Set (15%)**: 700 hourly observations
- **Data Leakage Verification**: The implemented pipeline was designed and verified to prevent temporal data leakage. All rolling window statistics and lag features use `.shift(1)` past observations only. Preprocessing scalers are fitted strictly on the Training set. The final test set was never used during hyperparameter selection.

## Verified Measured Performance (Untouched Test Set)
| Model Architecture | MAE (Litres/hour) | RMSE (Litres/hour) | $R^2$ Score | MAPE (%) | sMAPE (%) | Peak Demand Error (%) | Peak Timing Offset |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive Baseline (`lag_1`)** | 3,879.01 | 7,597.19 | -0.9155 | 10,433.71% | 132.74% | 85.27% | N/A |
| **Seasonal Naive (`lag_24`)** | 3,693.78 | 7,611.54 | -0.9228 | 11,695.88% | 136.31% | 88.05% | N/A |
| **XGBoost Regressor** | 2,543.77 | 5,585.45 | -0.0354 | 4,608.41% | 166.31% | 94.08% | 229 hrs |
| **PyTorch LSTM Regressor** | **2,256.58** | **5,927.60** | **-0.1604** | **205.77%** | **178.06%** | **99.70%** | **366 hrs** |

### Rigorous Academic Interpretation of Metrics:
- **MAE Improvement (38.9%)**: Calculated directly from test results:
  $$\text{MAE Improvement} = \frac{3693.78 - 2256.58}{3693.78} \times 100\% = \mathbf{38.91\%}$$
  Among the evaluated models, the PyTorch LSTM achieved the lowest MAE on the untouched test set, reducing MAE by **38.9%** compared with Seasonal Naive.
- **Unstable MAPE**: Standard MAPE is extremely large because near-zero actual consumption values inflate percentage-based errors. Therefore, MAE and RMSE are emphasized as the primary error metrics, and sMAPE is provided for bounded comparison.
- **Negative $R^2$**: A negative $R^2$ indicates that the model performs worse than a baseline that predicts the mean target value ($\bar{y}$) for the test set. The LSTM achieved the lowest MAE among the evaluated models, but its negative $R^2$ indicates that substantial variance in the test-set consumption remains unexplained.
- **Peak Demand Error**: Although the LSTM reduced overall MAE, peak-demand prediction remained challenging, with substantial peak magnitude and timing errors.

## Anomaly Detection
The residual-based anomaly detection pipeline identified **28 potential abnormal consumption events** requiring further investigation. These represent unusually large prediction residuals, not confirmed physical leaks or infrastructure faults.

## Regulatory and Privacy Considerations
- **Data Anonymization**: Smart meter values are aggregated at the household/service area level; no personally identifiable information (PII) or customer account numbers are stored or processed.
- **Applicable Governance**: Incorporates principles from Indian Personal Data Protection Act (DPDP), Bureau of Indian Standards (BIS) water supply guidelines, and global municipal data governance standards.
