# Dataset Documentation & Lineage

## Dataset Description

This project utilizes real smart-meter water consumption measurements matched with real historical weather observations.

### Data Sources
1. **Smart Meter Water Consumption Data**:
   - **Source Repository**: `LukaLike/water-consumption-data` (GitHub)
   - **Original Sampling Frequency**: Minutely resolution (263,811 records)
   - **Aggregated Granularity**: Hourly total demand (`water_hourly_processed.csv`, 4,831 records) & Daily total demand (`water_daily_processed.csv`, 202 records)
   - **Target Variable**: `water_consumption` (Litres/hour or Litres/day)
   - **Temporal Range**: August 13, 2022 to March 2, 2023

2. **Matched Weather Data**:
   - **Source**: Open-Meteo Historical Weather Archive API
   - **Geographic Location**: Zagreb Region (Latitude 45.815, Longitude 15.9819)
   - **Variables**: `temperature_2m` (°C), `relative_humidity_2m` (%), `precipitation` (mm), `surface_pressure` (hPa), `wind_speed_10m` (km/h)

## Dataset Directory Structure

```text
data/
├── raw/
│   ├── water_consumption_raw.csv    # Original minutely sensor logs
│   └── weather_raw.csv              # Hourly weather measurements
├── processed/
│   ├── water_hourly_processed.csv   # Hourly merged water & weather dataset
│   └── water_daily_processed.csv    # Daily merged water & weather dataset
└── README.md
```

## Data Quality Summary
- **Duplicates**: 0 exact duplicate rows after hourly resampling.
- **Timestamp Continuity**: Strict monotonic increasing hourly index (`sampling_frequency_mode` = 1 hour).
- **Missing Values**: 0 missing values across water consumption and weather parameters.
