# ERCOT Datasets for Price Forecasting Model

## Overview

This document summarizes all ERCOT datasets available via the Web Service API for building a neural network price forecasting model. All datasets can be downloaded using the `download_all_forecast_data.py` script.

## Available Datasets

### 1. Renewable Generation (Actual + Forecasts)

#### Wind Power Production (NP4-732-CD)
- **Dataset**: `wind`
- **Frequency**: Hourly
- **Content**:
  - Actual wind generation (GEN)
  - STWPF (Short-Term Wind Power Forecast) - 168-hour rolling forecast
  - WGRPP (Wind Generation Resource Power Potential)
  - COP HSLs for Online WGRs
- **History**: 48 hours historical + 168 hours forecast
- **Regions**: System-wide and by load zone
- **Use for ML**: Critical for capturing renewable variability and forecast errors

#### Solar Power Production (NP4-745-CD)
- **Dataset**: `solar`
- **Frequency**: Hourly
- **Content**:
  - Actual solar generation (GEN)
  - STPPF (Short-Term PhotoVoltaic Power Forecast) - 168-hour rolling forecast
  - PVGRPP (PhotoVoltaic Generation Resource Power Potential)
  - COP HSLs for Online PVGRs
- **History**: 48 hours historical + 168 hours forecast
- **Regions**: System-wide and by geographical region
- **Use for ML**: Solar forecast accuracy is key driver of RT price volatility

### 2. Load Forecasts

#### Load Forecast by Forecast Zone (NP3-565-CD)
- **Dataset**: `load_fzone`
- **Frequency**: Hourly (posted hourly with new forecast)
- **Content**: 7-day (168-hour) rolling load forecast by forecast zone
- **Use for ML**: Captures forecast vintages - each hour gets a new forecast

#### Load Forecast by Weather Zone (NP3-566-CD)
- **Dataset**: `load_wzone`
- **Frequency**: Hourly (posted hourly with new forecast)
- **Content**: 7-day (168-hour) rolling load forecast by weather zone
- **Use for ML**: Weather-adjusted load forecasts, important for scarcity pricing

### 3. Actual Load

#### Actual System Load by Weather Zone (NP6-345-CD)
- **Dataset**: `actual_load_wzone`
- **Frequency**: 5-minute
- **Content**: Actual system load by weather zone
- **Use for ML**: High-resolution actual load for RT price correlations

#### Actual System Load by Forecast Zone (NP6-346-CD)
- **Dataset**: `actual_load_fzone`
- **Frequency**: Hourly
- **Content**: Actual system load by forecast zone
- **Use for ML**: Hourly actual load for DA price modeling

### 4. Generation Mix

#### Fuel Mix (NP6-787-CD)
- **Dataset**: `fuel_mix`
- **Frequency**: 15-minute
- **Content**: Actual generation by fuel type
  - Wind
  - Solar
  - Natural Gas
  - Coal
  - Nuclear
  - Hydro
  - Other
- **Use for ML**: Fuel mix affects marginal cost and price formation

#### System-Wide Demand (NP6-322-CD)
- **Dataset**: `system_demand`
- **Frequency**: 5-minute
- **Content**:
  - Total system-wide demand
  - Net load after renewables
- **Use for ML**: Key indicator of system stress and scarcity pricing

### 5. System Metrics & Constraints

#### DAM System Lambda (NP4-191-CD)
- **Dataset**: `lambda`
- **Frequency**: Hourly (DA)
- **Content**: Day-ahead system lambda (shadow price of energy balance constraint)
- **Use for ML**: Direct indicator of system-wide scarcity and binding constraints

#### Unplanned Resource Outages (NP3-233-CD)
- **Dataset**: `outages`
- **Frequency**: Event-based
- **Content**:
  - Resource name
  - Capacity (MW)
  - Start/end times
  - Nature of outage
- **Use for ML**: Unexpected supply reductions cause price spikes

### 6. Prices (Target Variables & Features)

#### DAM Settlement Point Prices (NP4-190-CD)
- **Dataset**: `dam_prices`
- **Frequency**: Hourly
- **Content**: Day-ahead LMP at all settlement points
- **Use for ML**: Primary target for DA price forecasting

#### RTM Settlement Point Prices (NP6-785-CD)
- **Dataset**: `rtm_prices`
- **Frequency**: 5-minute
- **Content**: Real-time LMP at all settlement points
- **Use for ML**: Primary target for RT price forecasting

#### Ancillary Services Prices (NP4-188-CD)
- **Dataset**: `as_prices`
- **Frequency**: Hourly (DA clearing)
- **Content**: MCPC for all AS products:
  - Regulation Up (REGUP)
  - Regulation Down (REGDN)
  - Responsive Reserve Service (RRS)
  - Non-Spinning Reserve (NSPIN)
  - ERCOT Contingency Reserve Service (ECRS)
- **Use for ML**: AS scarcity affects energy prices

## Battery Charge/Discharge Data

### Available via 60-Day Disclosure Data (Already Implemented)

**Important**: ERCOT batteries have TWO resources:
1. **Gen Resource** (for discharging) - in `60d_SCED_Gen_Resources`
2. **Load Resource** (for charging) - in `60d_SCED_Load_Resources`

#### 60-Day SCED Gen Resource Data (NP3-965-ER)
- Contains actual discharge data for batteries
- 5-minute telemetered output
- Base points and HSLs
- **Lag**: 60 days

#### 60-Day SCED Load Resource Data (NP3-965-ER)
- Contains actual charging data for batteries
- 5-minute telemetered input
- **Lag**: 60 days

### Real-Time Battery Status

The ERCOT dashboard shows real-time battery charge/discharge, but this data is **not currently available via the Public API**. It may be available through:
- Internal ERCOT systems
- Private data feeds
- Derived from SCED Gen/Load disclosure data (60-day lag)

## Reserve Margins

**Capacity, Demand and Reserves (CDR) Report**: This is a periodic published PDF report, not available via API. It contains:
- Planning reserve margins
- Resource adequacy forecasts
- Peak demand projections
- 5-10 year outlook

You would need to manually download the CDR reports from ERCOT's website if needed.

## What Other Datasets Are Useful for Price Forecasting?

### Currently Available (via this implementation):
✅ Wind generation + forecasts (with forecast vintages)
✅ Solar generation + forecasts (with forecast vintages)
✅ Load forecasts (with forecast vintages)
✅ Actual load (5-min and hourly)
✅ Fuel mix
✅ System demand
✅ Outages
✅ DAM/RTM prices (target variables)
✅ AS prices

### Additional Features to Consider:

#### Weather Data (External Source)
- Temperature (by weather zone)
- Wind speed
- Cloud cover
- Humidity
- Historical weather vs forecast
- **Recommended source**: NASA POWER API, NOAA, or commercial weather APIs

#### Calendar/Time Features
- Hour of day
- Day of week
- Month
- Season
- Holidays
- On-peak/off-peak indicators
- DST transitions

#### Historical Price Features
- Lagged prices (t-1, t-24, t-168)
- Rolling averages
- Price volatility
- Spread between hubs
- DA vs RT basis (congestion + uncertainty)

#### Forecast Error Features
- Load forecast error (actual - forecast)
- Wind forecast error
- Solar forecast error
- These are powerful predictors of RT price spikes

#### Constraint/Congestion Features (Advanced)
- DAM shadow prices (transmission constraints)
- Binding constraint frequencies
- Line loading indicators
- Available transfer capability

### Not Available via Public API:
❌ Real-time battery SOC (state of charge) - 60-day lag only
❌ Reserve margins (CDR report) - periodic PDF only
❌ Transmission line flows - may be available through other reports
❌ Weather forecasts - need external source
❌ Natural gas prices - need external source (EIA, etc.)

## Usage

### List All Available Datasets
```bash
uv run python download_all_forecast_data.py --list
```

### Download Last 30 Days of All Datasets
```bash
uv run python download_all_forecast_data.py --days 30
```

### Download Specific Datasets
```bash
# Just renewable generation
uv run python download_all_forecast_data.py --datasets wind solar --days 90

# Load and generation data
uv run python download_all_forecast_data.py --datasets wind solar load_fzone actual_load_fzone fuel_mix --days 180
```

### Download Specific Date Range
```bash
uv run python download_all_forecast_data.py --start-date 2024-01-01 --end-date 2024-12-31
```

### Download and Convert to Parquet
```bash
uv run python download_all_forecast_data.py --days 30 --convert-to-parquet
```

## Data Availability Window

**Important**: ERCOT Web Service API only has data from **December 11, 2023** onwards.

For historical data before this date, you must use:
- Selenium web scraping (already implemented for historical data)
- Manual downloads from ERCOT Data Portal
- Historical archive files

## Parquet Conversion

The script can automatically convert downloaded CSV files to Parquet format for:
- 70-90% storage savings
- 100x faster query performance
- Better integration with data science tools (pandas, polars, DuckDB, etc.)

## Next Steps

1. **Download test dataset** (30 days) to verify all endpoints work
2. **Download full historical data** (Dec 2023 - present)
3. **Convert to Parquet** for efficient storage and querying
4. **Extract features** from raw data for ML model
5. **Build forecast error features** by comparing forecasts to actuals
6. **Integrate external weather data** for additional predictive power
7. **Train neural network model** with all features

## ML Model Architecture Recommendations

### Input Features (Examples)
- Hour of day, day of week, month (cyclical encoding)
- Actual load (current, lagged)
- Load forecast (multiple vintages)
- Wind generation (actual, forecast, forecast error)
- Solar generation (actual, forecast, forecast error)
- Fuel mix percentages
- Previous prices (lagged 1h, 24h, 168h)
- AS prices (scarcity indicators)
- Outage capacity (MW offline)
- System lambda (constraint indicator)

### Target Variables
- DA LMP (hourly ahead)
- RT LMP (5-min ahead or hourly average)
- Price volatility

### Model Types to Consider
- LSTM (for time series dependencies)
- Transformer (for attention across multiple time scales)
- Gradient boosting (XGBoost, LightGBM) for baseline
- Ensemble of multiple models

## Support

For issues or questions:
- Check ERCOT API documentation: https://apiexplorer.ercot.com/
- Review code in `ercot_ws_downloader/` directory
- See `ERCOT_WEB_SERVICE_DOWNLOAD_README.md` for more details
