"""
EDA Generator Module
Generates all 10 required EDA figures from actual water and weather data
and saves them to results/figures/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

FIGURES_DIR = os.path.join("results", "figures")
PROCESSED_HOURLY_PATH = os.path.join("data", "processed", "water_hourly_processed.csv")

def generate_eda_figures():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_HOURLY_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month_name()
    df['month_num'] = df['datetime'].dt.month
    df['is_weekend'] = df['datetime'].dt.dayofweek >= 5
    
    # Order for days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    print("Generating EDA Figures...")

    # 1. Water Consumption Over Time
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['datetime'], df['water_consumption'], color='#1f77b4', alpha=0.7, linewidth=0.8)
    ax.set_title("1. Water Consumption Time Series (Hourly Resolution)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Consumption (Litres/hour)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "01_consumption_over_time.png"), dpi=300)
    plt.close()

    # 2. Consumption Distribution (Histogram + Log Scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['water_consumption'], bins=50, kde=True, ax=ax1, color='#2ca02c')
    ax1.set_title("2a. Target Distribution (Litres/hour)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Water Consumption (Litres/hour)")
    
    log_vals = np.log1p(df['water_consumption'])
    sns.histplot(log_vals, bins=50, kde=True, ax=ax2, color='#9467bd')
    ax2.set_title("2b. Log1p Target Distribution", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Log(1 + Water Consumption)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "02_consumption_distribution.png"), dpi=300)
    plt.close()

    # 3. Boxplot of Consumption
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=df, x='hour', y='water_consumption', ax=ax1, palette='Blues_r', fliersize=1)
    ax1.set_title("3a. Hourly Consumption Boxplot", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Litres/hour")
    
    sns.boxplot(data=df, x='is_weekend', y='water_consumption', ax=ax2, palette='Set2')
    ax2.set_title("3b. Weekday vs Weekend Boxplot", fontsize=12, fontweight='bold')
    ax2.set_xticklabels(['Weekday', 'Weekend'])
    ax2.set_xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "03_consumption_boxplot.png"), dpi=300)
    plt.close()

    # 4. Monthly / Seasonal Pattern
    fig, ax = plt.subplots(figsize=(10, 5))
    monthly_agg = df.groupby('month_num')['water_consumption'].mean().reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_agg['month_name'] = monthly_agg['month_num'].apply(lambda m: month_names[m-1])
    sns.barplot(data=monthly_agg, x='month_name', y='water_consumption', ax=ax, palette='crest')
    ax.set_title("4. Average Hourly Water Consumption by Month", fontsize=14, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Consumption (Litres/hour)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "04_monthly_consumption.png"), dpi=300)
    plt.close()

    # 5. Day-of-Week Pattern
    fig, ax = plt.subplots(figsize=(10, 5))
    weekly_agg = df.groupby('day_of_week')['water_consumption'].mean().reindex(day_order).reset_index()
    sns.barplot(data=weekly_agg, x='day_of_week', y='water_consumption', ax=ax, palette='viridis')
    ax.set_title("5. Average Water Consumption by Day of Week", fontsize=14, fontweight='bold')
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Mean Consumption (Litres/hour)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "05_weekly_consumption.png"), dpi=300)
    plt.close()

    # 6. Hourly Diurnal Profile
    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_profile = df.groupby(['hour', 'is_weekend'])['water_consumption'].mean().reset_index()
    hourly_profile['day_type'] = hourly_profile['is_weekend'].map({False: 'Weekday', True: 'Weekend'})
    sns.lineplot(data=hourly_profile, x='hour', y='water_consumption', hue='day_type', marker='o', ax=ax, linewidth=2)
    ax.set_title("6. Diurnal Hourly Consumption Profile (Weekday vs Weekend)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Mean Consumption (Litres/hour)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "06_hourly_pattern.png"), dpi=300)
    plt.close()

    # 7. Temperature vs Consumption
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hexbin(df['temperature_2m'], df['water_consumption'], gridsize=40, cmap='YlOrRd', mincnt=1)
    ax.set_title("7. Temperature vs Water Consumption Density", fontsize=14, fontweight='bold')
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Water Consumption (Litres/hour)")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("Observation Density")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "07_temperature_vs_consumption.png"), dpi=300)
    plt.close()

    # 8. Rainfall vs Consumption
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(data=df, x='precipitation', y='water_consumption', alpha=0.5, color='#1f77b4', ax=ax)
    ax.set_title("8. Precipitation vs Water Consumption", fontsize=14, fontweight='bold')
    ax.set_xlabel("Precipitation (mm)")
    ax.set_ylabel("Water Consumption (Litres/hour)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "08_rainfall_vs_consumption.png"), dpi=300)
    plt.close()

    # 9. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    num_cols = ['water_consumption', 'temperature_2m', 'relative_humidity_2m', 'precipitation', 'surface_pressure', 'wind_speed_10m', 'hour']
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("9. Feature Correlation Heatmap", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "09_correlation_heatmap.png"), dpi=300)
    plt.close()

    # 10. Peak Demand Periods Analysis
    fig, ax = plt.subplots(figsize=(14, 5))
    top_99 = df['water_consumption'].quantile(0.99)
    peaks = df[df['water_consumption'] >= top_99]
    ax.plot(df['datetime'], df['water_consumption'], color='gray', alpha=0.5, label='Normal Demand')
    ax.scatter(peaks['datetime'], peaks['water_consumption'], color='red', s=25, label=f'Peak Demand (>= 99th percentile: {top_99:.1f} L/h)', zorder=5)
    ax.set_title("10. Peak Water Demand Identification (Top 1% Consumption Events)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Water Consumption (Litres/hour)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "10_peak_demand_periods.png"), dpi=300)
    plt.close()

    print(f"Successfully generated and saved all 10 figures to {FIGURES_DIR}")

if __name__ == "__main__":
    generate_eda_figures()
