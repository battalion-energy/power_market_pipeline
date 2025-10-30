#!/usr/bin/env python3
"""
Deep dive analysis of model performance:
1. Quantile calibration - are the confidence intervals accurate?
2. Feature importance - what's actually being used?
3. Baseline comparison - Random Forest, persistence, naive forecasts
4. Temporal analysis - is recent data weighted properly?
5. Data gaps - what are we missing?
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from train_unified_da_rt_quantile import prepare_data, UnifiedDART_Forecaster, UnifiedDataset

DATA_FILE = "/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced_with_ordc_load_2019_2025.parquet"
MODEL_PATH = Path("models/unified_da_rt_best.pth")

print("="*80)
print("MODEL PERFORMANCE DEEP DIVE ANALYSIS")
print("="*80)
print()

# Load data
print("Loading data...")
df_pl = pl.read_parquet(DATA_FILE)
df, feature_cols = prepare_data(df_pl)
print(f"Dataset: {len(df):,} samples, {len(feature_cols)} features")
print()

# Split
n = len(df)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

df_train = df.iloc[:train_end].copy()
df_val = df.iloc[train_end:val_end].copy()
df_test = df.iloc[val_end:].copy()

print(f"Split:")
print(f"  Train: {len(df_train):,} ({df_train.index[0]} to {df_train.index[-1]})")
print(f"  Val:   {len(df_val):,} ({df_val.index[0]} to {df_val.index[-1]})")
print(f"  Test:  {len(df_test):,} ({df_test.index[0]} to {df_test.index[-1]})")
print()

# ============================================================================
# 1. CHECK QUANTILE CALIBRATION
# ============================================================================
print("="*80)
print("1. QUANTILE CALIBRATION ANALYSIS")
print("="*80)
print()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UnifiedDART_Forecaster(input_dim=len(feature_cols), n_quantiles=5).to(device)
checkpoint = torch.load(MODEL_PATH, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get predictions on test set
test_dataset = UnifiedDataset(df_test, feature_cols, lookback=168, horizon=48)
predictions_da = []
predictions_rt = []
actuals_da = []
actuals_rt = []

print("Generating test predictions...")
with torch.no_grad():
    for i in range(min(len(test_dataset), 500)):  # Sample 500 for speed
        x_hist = test_dataset.df[test_dataset.feature_cols].iloc[
            test_dataset.valid_indices[i]-168:test_dataset.valid_indices[i]
        ].values.astype(np.float32)
        x_hist = torch.from_numpy(x_hist).unsqueeze(0).to(device)

        # Get actuals
        idx = test_dataset.valid_indices[i]
        y_da_actual = test_dataset.df['price_da'].iloc[idx+1:idx+1+48].values
        y_rt_actual = test_dataset.df['price_rt'].iloc[idx+1:idx+1+48].values

        # Get predictions
        da_pred, rt_pred = model(x_hist)
        da_pred = da_pred.cpu().numpy()[0]
        rt_pred = rt_pred.cpu().numpy()[0]

        predictions_da.append(da_pred)
        predictions_rt.append(rt_pred)
        actuals_da.append(y_da_actual)
        actuals_rt.append(y_rt_actual)

predictions_da = np.array(predictions_da)  # (n_samples, 48, 5)
predictions_rt = np.array(predictions_rt)
actuals_da = np.array(actuals_da)  # (n_samples, 48)
actuals_rt = np.array(actuals_rt)

print(f"Collected {len(predictions_da)} test samples")
print()

# Check quantile calibration
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
print("Quantile Calibration (% of actuals below each quantile):")
print(f"{'Quantile':<12} {'Expected':<12} {'DA Actual':<12} {'RT Actual':<12} {'DA Error':<12} {'RT Error':<12}")
print("-"*80)

for q_idx, q in enumerate(quantiles):
    # Check how many actuals fall below this quantile prediction
    da_below = (actuals_da < predictions_da[:, :, q_idx]).mean() * 100
    rt_below = (actuals_rt < predictions_rt[:, :, q_idx]).mean() * 100
    expected = q * 100

    da_error = da_below - expected
    rt_error = rt_below - expected

    print(f"P{int(q*100):<10} {expected:<12.1f} {da_below:<12.1f} {rt_below:<12.1f} {da_error:<+12.1f} {rt_error:<+12.1f}")

print()
print("📊 Interpretation:")
print("  - Well-calibrated: Error near 0%")
print("  - Overconfident: Positive error (too many actuals above prediction)")
print("  - Underconfident: Negative error (too many actuals below prediction)")
print()

# ============================================================================
# 2. BASELINE COMPARISON
# ============================================================================
print("="*80)
print("2. BASELINE COMPARISON")
print("="*80)
print()

# Baseline 1: Persistence (use current price)
print("Baseline 1: Persistence (use current DA price for all 48h)")
persistence_mae_da = mean_absolute_error(
    actuals_da.flatten(),
    np.repeat(df_test['price_da'].iloc[[idx for idx in test_dataset.valid_indices[:len(actuals_da)]]].values[:, None], 48, axis=1).flatten()
)
print(f"  MAE: ${persistence_mae_da:.2f}/MWh")
print()

# Baseline 2: Simple moving average
print("Baseline 2: 7-day moving average")
df_test_copy = df_test.copy()
df_test_copy['price_da_ma7'] = df_test_copy['price_da'].rolling(24*7, min_periods=1).mean()
ma7_predictions = []
for idx in test_dataset.valid_indices[:len(actuals_da)]:
    ma7_val = df_test_copy.loc[df_test_copy.index[idx], 'price_da_ma7']
    ma7_predictions.append(np.full(48, ma7_val))
ma7_predictions = np.array(ma7_predictions)
ma7_mae = mean_absolute_error(actuals_da.flatten(), ma7_predictions.flatten())
print(f"  MAE: ${ma7_mae:.2f}/MWh")
print()

# Baseline 3: Random Forest (simple features only)
print("Baseline 3: Random Forest (hour, day_of_week, month, recent prices)")
print("  Training RF model...")

# Prepare simple features for RF
def prepare_rf_features(df_subset):
    features = []
    targets_da = []
    targets_rt = []

    for i in range(168, len(df_subset) - 48):
        # Simple temporal features
        row_features = [
            df_subset.index[i].hour,
            df_subset.index[i].dayofweek,
            df_subset.index[i].month,
            df_subset.index[i].dayofyear,
        ]

        # Recent price history (last 24h)
        row_features.extend(df_subset['price_da'].iloc[i-24:i].values)
        row_features.extend(df_subset['price_rt'].iloc[i-24:i].values)

        features.append(row_features)
        targets_da.append(df_subset['price_da'].iloc[i+1:i+25].mean())  # 24h ahead avg
        targets_rt.append(df_subset['price_rt'].iloc[i+1:i+25].mean())

    return np.array(features), np.array(targets_da), np.array(targets_rt)

X_train, y_train_da, y_train_rt = prepare_rf_features(df_train)
X_test, y_test_da, y_test_rt = prepare_rf_features(df_test)

rf_da = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_da.fit(X_train, y_train_da)
rf_pred_da = rf_da.predict(X_test)
rf_mae_da = mean_absolute_error(y_test_da, rf_pred_da)

rf_rt = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_rt.fit(X_train, y_train_rt)
rf_pred_rt = rf_rt.predict(X_test)
rf_mae_rt = mean_absolute_error(y_test_rt, rf_pred_rt)

print(f"  DA MAE: ${rf_mae_da:.2f}/MWh")
print(f"  RT MAE: ${rf_mae_rt:.2f}/MWh")
print()

# Compare to our model
our_mae_da = mean_absolute_error(actuals_da[:, :24].mean(axis=1), predictions_da[:, :24, 2].mean(axis=1))
our_mae_rt = mean_absolute_error(actuals_rt[:, :24].mean(axis=1), predictions_rt[:, :24, 2].mean(axis=1))

print("-"*80)
print("COMPARISON (24h average forecast):")
print("-"*80)
print(f"{'Model':<30} {'DA MAE':<15} {'RT MAE':<15}")
print("-"*80)
print(f"{'Persistence':<30} ${persistence_mae_da:<14.2f} {'N/A':<15}")
print(f"{'7-day Moving Average':<30} ${ma7_mae:<14.2f} {'N/A':<15}")
print(f"{'Random Forest':<30} ${rf_mae_da:<14.2f} ${rf_mae_rt:<14.2f}")
print(f"{'Our Transformer Model':<30} ${our_mae_da:<14.2f} ${our_mae_rt:<14.2f}")
print("-"*80)
print()

# ============================================================================
# 3. FEATURE IMPORTANCE (from Random Forest)
# ============================================================================
print("="*80)
print("3. FEATURE IMPORTANCE (Random Forest as proxy)")
print("="*80)
print()

feature_names = ['hour', 'day_of_week', 'month', 'day_of_year'] + \
                [f'da_price_lag_{i}h' for i in range(24, 0, -1)] + \
                [f'rt_price_lag_{i}h' for i in range(24, 0, -1)]

importances = rf_da.feature_importances_
top_indices = np.argsort(importances)[::-1][:15]

print("Top 15 Most Important Features (DA Price Prediction):")
print(f"{'Rank':<6} {'Feature':<30} {'Importance':<12}")
print("-"*60)
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank:<6} {feature_names[idx]:<30} {importances[idx]:<12.4f}")
print()

# ============================================================================
# 4. TEMPORAL ANALYSIS
# ============================================================================
print("="*80)
print("4. TEMPORAL WEIGHTING ANALYSIS")
print("="*80)
print()

print("Checking if model performance degrades with forecast horizon...")
mae_by_horizon_da = []
mae_by_horizon_rt = []

for h in range(48):
    mae_da_h = mean_absolute_error(actuals_da[:, h], predictions_da[:, h, 2])
    mae_rt_h = mean_absolute_error(actuals_rt[:, h], predictions_rt[:, h, 2])
    mae_by_horizon_da.append(mae_da_h)
    mae_by_horizon_rt.append(mae_rt_h)

print(f"{'Horizon':<12} {'DA MAE':<15} {'RT MAE':<15}")
print("-"*42)
for h in [0, 5, 11, 23, 35, 47]:
    print(f"{h+1}h{'':<9} ${mae_by_horizon_da[h]:<14.2f} ${mae_by_horizon_rt[h]:<14.2f}")
print()

if mae_by_horizon_da[47] > mae_by_horizon_da[0] * 1.5:
    print("⚠️  Performance degrades significantly at longer horizons")
    print("   → Consider: separate models for short/long horizons")
else:
    print("✓ Performance relatively stable across horizons")
print()

# ============================================================================
# 5. MISSING DATA ANALYSIS
# ============================================================================
print("="*80)
print("5. DATA GAPS & MISSING SOURCES")
print("="*80)
print()

print("Current data sources:")
print("  ✓ Historical prices (RT, DA)")
print("  ✓ ORDC scarcity pricing (2018-2025)")
print("  ✓ Load forecasts (2022-2025, 93.8% coverage)")
print("  ✓ Weather data (NASA POWER)")
print("  ✓ Ancillary services prices")
print("  ✓ Temporal features")
print()

print("Potentially missing data sources:")
print("  ⚠️  Wind/solar generation forecasts")
print("  ⚠️  Outage schedules (planned maintenance)")
print("  ⚠️  Gas prices / fuel costs")
print("  ⚠️  Transmission constraints")
print("  ⚠️  Renewable curtailment data")
print("  ⚠️  Real-time generation mix")
print("  ⚠️  Hub-level spatial data (using aggregates)")
print()

# Check feature coverage
print("Feature coverage in test set:")
missing_pct = df_test[feature_cols].isna().mean() * 100
high_missing = missing_pct[missing_pct > 5].sort_values(ascending=False)

if len(high_missing) > 0:
    print("\nFeatures with >5% missing data:")
    for feat, pct in high_missing.items():
        print(f"  {feat}: {pct:.1f}% missing")
else:
    print("  ✓ All features have <5% missing data")
print()

print("="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)
print()

# Generate recommendations based on analysis
recommendations = []

if our_mae_da > rf_mae_da * 1.2:
    recommendations.append("⚠️  Transformer underperforming Random Forest - consider simpler model or more data")

if abs((actuals_da < predictions_da[:, :, 2]).mean() * 100 - 50) > 10:
    recommendations.append("⚠️  P50 quantile poorly calibrated - adjust loss function weights")

if mae_by_horizon_da[47] > mae_by_horizon_da[0] * 1.5:
    recommendations.append("⚠️  Long horizon performance poor - train separate short/long models")

if len(recommendations) == 0:
    print("✓ Model performance is reasonable given the data")
else:
    print("Issues identified:")
    for rec in recommendations:
        print(f"  {rec}")

print()
print("✓ Analysis complete!")
