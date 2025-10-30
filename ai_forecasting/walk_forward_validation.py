#!/usr/bin/env python3
"""
Walk-Forward Validation (Rolling Forecast Origin)

This is the PROPER way to evaluate time series forecasting models.

Method:
1. Start with initial training window (e.g., first 2 years)
2. Forecast next 48 hours
3. Observe actual prices for those 48 hours
4. Add actual data to training set
5. Optionally retrain model (or just expand window)
6. Repeat for entire test period (e.g., last 3 years)

This gives us:
- Realistic out-of-sample performance
- Many test samples across different market conditions
- Ability to compare models fairly
- Revenue impact quantification
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle

from train_unified_da_rt_quantile import prepare_data, UnifiedDART_Forecaster

DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced_with_ordc_load_2019_2025.parquet"
MODEL_PATH = Path("models/unified_da_rt_best.pth")
OUTPUT_DIR = Path("results/walk_forward")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("WALK-FORWARD VALIDATION")
print("="*80)
print()

# ============================================================================
# Configuration
# ============================================================================

# Initial training window
INITIAL_TRAIN_YEARS = 2

# How often to retrain (in days)
# Options: 1 (daily), 7 (weekly), 30 (monthly), None (never, just expand window)
RETRAIN_FREQUENCY_DAYS = 30  # Retrain monthly

# Forecast horizon
FORECAST_HORIZON = 48  # hours

# Step size (how far to move forward each iteration)
STEP_SIZE_HOURS = 48  # Move forward 48 hours each time

print("Configuration:")
print(f"  Initial training: {INITIAL_TRAIN_YEARS} years")
print(f"  Retrain frequency: {RETRAIN_FREQUENCY_DAYS} days")
print(f"  Forecast horizon: {FORECAST_HORIZON} hours")
print(f"  Step size: {STEP_SIZE_HOURS} hours")
print()

# ============================================================================
# Load and prepare data
# ============================================================================

print("Loading data...")
df_pl = pl.read_parquet(DATA_FILE)
df, feature_cols = prepare_data(df_pl)
print(f"Dataset: {len(df):,} samples")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Features: {len(feature_cols)}")
print()

# ============================================================================
# Define walk-forward periods
# ============================================================================

# Initial training period
initial_train_end = df.index[0] + pd.Timedelta(days=365*INITIAL_TRAIN_YEARS)
initial_train_data = df[df.index < initial_train_end]

# Test period (everything after initial training)
test_start = initial_train_end
test_end = df.index[-1] - pd.Timedelta(hours=FORECAST_HORIZON)  # Need 48h of actuals

print(f"Walk-forward setup:")
print(f"  Initial training: {df.index[0]} to {initial_train_end} ({len(initial_train_data):,} samples)")
print(f"  Test period: {test_start} to {test_end}")
print(f"  Test duration: {(test_end - test_start).days} days")
print()

# Generate forecast origins (times when we make forecasts)
forecast_origins = pd.date_range(test_start, test_end, freq=f'{STEP_SIZE_HOURS}h')
print(f"Number of forecast origins: {len(forecast_origins)}")
print(f"  → Generating {len(forecast_origins) * FORECAST_HORIZON:,} individual hourly forecasts")
print()

# ============================================================================
# Helper functions
# ============================================================================

def train_random_forest(df_train, horizon=24):
    """Train Random Forest on recent price history"""
    X = []
    y_da = []
    y_rt = []

    for i in range(48, len(df_train) - horizon):
        # Features: last 48h of prices + temporal
        features = [
            df_train.index[i].hour,
            df_train.index[i].dayofweek,
            df_train.index[i].month,
            df_train.index[i].dayofyear,
        ]
        features.extend(df_train['price_da'].iloc[i-48:i].values)
        features.extend(df_train['price_rt'].iloc[i-48:i].values)

        X.append(features)

        # Target: average price over forecast horizon
        y_da.append(df_train['price_da'].iloc[i+1:i+1+horizon].mean())
        y_rt.append(df_train['price_rt'].iloc[i+1:i+1+horizon].mean())

    X = np.array(X)
    y_da = np.array(y_da)
    y_rt = np.array(y_rt)

    # Train two RFs
    rf_da = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_da.fit(X, y_da)

    rf_rt = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_rt.fit(X, y_rt)

    return rf_da, rf_rt

def predict_rf(rf_da, rf_rt, df, origin_idx, horizon=48):
    """Make RF prediction for next 48 hours"""
    # Get features
    features = [
        df.index[origin_idx].hour,
        df.index[origin_idx].dayofweek,
        df.index[origin_idx].month,
        df.index[origin_idx].dayofyear,
    ]
    features.extend(df['price_da'].iloc[origin_idx-48:origin_idx].values)
    features.extend(df['price_rt'].iloc[origin_idx-48:origin_idx].values)

    X = np.array(features).reshape(1, -1)

    # Predict (returns single value - average over horizon)
    pred_da = rf_da.predict(X)[0]
    pred_rt = rf_rt.predict(X)[0]

    # Broadcast to all 48 hours (simple approach)
    return np.full(horizon, pred_da), np.full(horizon, pred_rt)

def predict_persistence(df, origin_idx, horizon=48):
    """Persistence baseline: use current price for all 48h"""
    current_da = df['price_da'].iloc[origin_idx]
    current_rt = df['price_rt'].iloc[origin_idx]
    return np.full(horizon, current_da), np.full(horizon, current_rt)

def predict_moving_average(df, origin_idx, horizon=48, window_hours=168):
    """Moving average baseline: use 7-day average"""
    ma_da = df['price_da'].iloc[max(0, origin_idx-window_hours):origin_idx].mean()
    ma_rt = df['price_rt'].iloc[max(0, origin_idx-window_hours):origin_idx].mean()
    return np.full(horizon, ma_da), np.full(horizon, ma_rt)

# ============================================================================
# Walk-forward validation
# ============================================================================

print("="*80)
print("RUNNING WALK-FORWARD VALIDATION")
print("="*80)
print()

# Storage for results
results = {
    'timestamps': [],
    'horizon_hours': [],
    'actual_da': [],
    'actual_rt': [],
    'pred_rf_da': [],
    'pred_rf_rt': [],
    'pred_persistence_da': [],
    'pred_persistence_rt': [],
    'pred_ma_da': [],
    'pred_ma_rt': [],
}

# Train initial models
print("Training initial Random Forest models...")
rf_da, rf_rt = train_random_forest(initial_train_data, horizon=24)
print(f"  ✓ Initial RF trained on {len(initial_train_data):,} samples")
print()

# Track when we last retrained
last_retrain_date = initial_train_end
retrain_count = 1

# Walk forward
print(f"Starting walk-forward loop ({len(forecast_origins)} iterations)...")
print()

for i, origin_time in enumerate(tqdm(forecast_origins, desc="Walk-forward")):

    # Get index in dataframe
    origin_idx = df.index.get_loc(origin_time)

    # Check if we should retrain
    if RETRAIN_FREQUENCY_DAYS and (origin_time - last_retrain_date).days >= RETRAIN_FREQUENCY_DAYS:
        # Retrain on all data up to current origin
        train_data = df[df.index < origin_time]
        rf_da, rf_rt = train_random_forest(train_data, horizon=24)
        last_retrain_date = origin_time
        retrain_count += 1
        tqdm.write(f"  Retrained #{retrain_count} at {origin_time} ({len(train_data):,} samples)")

    # Make predictions for next 48 hours
    pred_rf_da, pred_rf_rt = predict_rf(rf_da, rf_rt, df, origin_idx)
    pred_pers_da, pred_pers_rt = predict_persistence(df, origin_idx)
    pred_ma_da, pred_ma_rt = predict_moving_average(df, origin_idx)

    # Get actual values
    actual_da = df['price_da'].iloc[origin_idx+1:origin_idx+1+FORECAST_HORIZON].values
    actual_rt = df['price_rt'].iloc[origin_idx+1:origin_idx+1+FORECAST_HORIZON].values

    # Check if we have enough actuals (might be at end of dataset)
    if len(actual_da) < FORECAST_HORIZON:
        break

    # Store results for each horizon
    for h in range(FORECAST_HORIZON):
        forecast_time = origin_time + pd.Timedelta(hours=h+1)
        results['timestamps'].append(forecast_time)
        results['horizon_hours'].append(h+1)
        results['actual_da'].append(actual_da[h])
        results['actual_rt'].append(actual_rt[h])
        results['pred_rf_da'].append(pred_rf_da[h])
        results['pred_rf_rt'].append(pred_rf_rt[h])
        results['pred_persistence_da'].append(pred_pers_da[h])
        results['pred_persistence_rt'].append(pred_pers_rt[h])
        results['pred_ma_da'].append(pred_ma_da[h])
        results['pred_ma_rt'].append(pred_ma_rt[h])

print()
print(f"✓ Walk-forward complete!")
print(f"  Total retrains: {retrain_count}")
print(f"  Total forecasts generated: {len(results['timestamps']):,}")
print()

# ============================================================================
# Analyze results
# ============================================================================

print("="*80)
print("RESULTS")
print("="*80)
print()

results_df = pd.DataFrame(results)

# Overall performance
print("Overall Performance (all horizons):")
print(f"{'Model':<25} {'DA MAE':<15} {'RT MAE':<15}")
print("-"*55)

for model_name, pred_col_da, pred_col_rt in [
    ('Random Forest', 'pred_rf_da', 'pred_rf_rt'),
    ('Persistence', 'pred_persistence_da', 'pred_persistence_rt'),
    ('7-day MA', 'pred_ma_da', 'pred_ma_rt'),
]:
    mae_da = mean_absolute_error(results_df['actual_da'], results_df[pred_col_da])
    mae_rt = mean_absolute_error(results_df['actual_rt'], results_df[pred_col_rt])
    print(f"{model_name:<25} ${mae_da:<14.2f} ${mae_rt:<14.2f}")

print()

# Performance by horizon
print("Performance by Forecast Horizon (DA prices):")
print(f"{'Horizon':<12} {'RF MAE':<12} {'Persistence':<12} {'7-day MA':<12}")
print("-"*48)

for h in [1, 6, 12, 24, 36, 48]:
    horizon_data = results_df[results_df['horizon_hours'] == h]

    mae_rf = mean_absolute_error(horizon_data['actual_da'], horizon_data['pred_rf_da'])
    mae_pers = mean_absolute_error(horizon_data['actual_da'], horizon_data['pred_persistence_da'])
    mae_ma = mean_absolute_error(horizon_data['actual_da'], horizon_data['pred_ma_da'])

    print(f"{h}h{'':<9} ${mae_rf:<11.2f} ${mae_pers:<11.2f} ${mae_ma:<11.2f}")

print()

# Performance over time (by month)
print("Performance Over Time (monthly average DA MAE):")
results_df['month'] = pd.to_datetime(results_df['timestamps']).dt.to_period('M')
monthly_perf = results_df.groupby('month').apply(
    lambda x: pd.Series({
        'rf_mae': mean_absolute_error(x['actual_da'], x['pred_rf_da']),
        'pers_mae': mean_absolute_error(x['actual_da'], x['pred_persistence_da']),
        'n_forecasts': len(x)
    })
).reset_index()

print(f"{'Month':<12} {'RF MAE':<12} {'Persistence':<12} {'# Forecasts':<12}")
print("-"*48)
for _, row in monthly_perf.head(12).iterrows():
    print(f"{row['month']}{'':<4} ${row['rf_mae']:<11.2f} ${row['pers_mae']:<11.2f} {int(row['n_forecasts']):<12}")

print()

# Save results
results_file = OUTPUT_DIR / f"walk_forward_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
results_df.to_csv(results_file, index=False)
print(f"✓ Results saved: {results_file}")
print()

# ============================================================================
# Revenue impact analysis
# ============================================================================

print("="*80)
print("REVENUE IMPACT (100 MW Battery)")
print("="*80)
print()

# Simple strategy: charge when price forecast is low, discharge when high
# Use median price as threshold

median_price = results_df['actual_da'].median()
print(f"Median DA price: ${median_price:.2f}/MWh")
print()

def calculate_revenue(forecast_col, battery_mw=100):
    """Calculate revenue using perfect hindsight vs forecast"""
    revenue_perfect = 0
    revenue_forecast = 0

    for _, row in results_df.iterrows():
        actual = row['actual_da']
        forecast = row[forecast_col]

        # Perfect hindsight: buy low, sell high
        if actual < median_price:
            revenue_perfect += battery_mw * (median_price - actual)
        else:
            revenue_perfect += battery_mw * (actual - median_price)

        # Using forecast
        if forecast < median_price:  # Forecast says charge
            if actual < median_price:  # Correct prediction
                revenue_forecast += battery_mw * (median_price - actual)
            else:  # Wrong - charged when should have discharged
                revenue_forecast -= battery_mw * abs(actual - median_price) * 0.5
        else:  # Forecast says discharge
            if actual >= median_price:  # Correct
                revenue_forecast += battery_mw * (actual - median_price)
            else:  # Wrong
                revenue_forecast -= battery_mw * abs(actual - median_price) * 0.5

    return revenue_perfect, revenue_forecast

print("Revenue comparison (simple arbitrage strategy):")
print(f"{'Model':<25} {'Annual Revenue':<20} {'vs Perfect':<15}")
print("-"*60)

perfect_rev, _ = calculate_revenue('pred_rf_da')  # Perfect revenue is same for all

for model_name, pred_col in [
    ('Perfect Hindsight', 'actual_da'),
    ('Random Forest', 'pred_rf_da'),
    ('Persistence', 'pred_persistence_da'),
    ('7-day MA', 'pred_ma_da'),
]:
    _, revenue = calculate_revenue(pred_col)

    # Annualize
    days_in_data = (results_df['timestamps'].max() - results_df['timestamps'].min()).days
    annual_revenue = revenue * (365 / days_in_data)
    pct_of_perfect = (annual_revenue / (perfect_rev * 365 / days_in_data)) * 100

    print(f"{model_name:<25} ${annual_revenue/1e6:<19.2f}M {pct_of_perfect:<14.1f}%")

print()
print("="*80)
print("✓ Walk-forward validation complete!")
print("="*80)
