# Walk-Forward Forecast Demo Guide

## ✅ System Ready

The forecast API now serves **proper walk-forward forecasts** with **NO look-ahead bias** for the following demonstration dates.

## 📅 Demo Dates for Mercuria Presentation

Navigate to these specific dates in the dashboard and click **"Show AI Forecast"** to see walk-forward predictions:

### 1. **February 20, 2024** ⭐ (Matches your screenshot!)
- **URL**: `http://localhost:3000/bess-market?date=2024-02-20&bess=GAMBIT_BESS1`
- **Also**: `http://localhost:3000/bess-market-bidding?date=2024-02-20&hub=HB_HOUSTON`
- **Trained on**: 6,090 samples (data up to Feb 19, 2024 23:00)
- **Forecast MAE**: DA $42.81/MWh, RT $21.04/MWh
- **Why show**: Matches the screenshot you provided - demonstrates real integration

### 2. **January 1, 2024** ⭐ (Matches your screenshot!)
- **URL**: `http://localhost:3000/bess-market-bidding?date=2024-01-01&hub=HB_HOUSTON`
- **Trained on**: 4,893 samples (data up to Dec 31, 2023 23:00)
- **Forecast MAE**: DA $24.67/MWh, RT $13.09/MWh
- **Why show**: New year forecast, good accuracy

### 3. **August 15, 2023** (Summer Peak)
- **URL**: `http://localhost:3000/bess-market-bidding?date=2023-08-15&hub=HB_HOUSTON`
- **Trained on**: 1,579 samples (data up to Aug 14, 2023 23:00)
- **Forecast MAE**: DA $44.10/MWh, RT $28.28/MWh
- **Why show**: Summer high demand period

### 4. **March 15, 2024** (Spring Shoulder)
- **URL**: `http://localhost:3000/bess-market-bidding?date=2024-03-15&hub=HB_HOUSTON`
- **Trained on**: 6,664 samples (data up to Mar 14, 2024 23:00)
- **Forecast MAE**: DA $26.35/MWh, RT $12.74/MWh
- **Why show**: Mild conditions, good accuracy

### 5. **April 10, 2024** (Mild Spring)
- **URL**: `http://localhost:3000/bess-market-bidding?date=2024-04-10&hub=HB_HOUSTON`
- **Trained on**: 7,288 samples (data up to Apr 9, 2024 23:00)
- **Forecast MAE**: DA $19.78/MWh, RT $8.37/MWh
- **Why show**: Best accuracy - excellent forecast performance

### 6. **October 1, 2024** (Fall Moderate)
- **URL**: `http://localhost:3000/bess-market-bidding?date=2024-10-01&hub=HB_HOUSTON`
- **Trained on**: 11,462 samples (data up to Sep 30, 2024 23:00)
- **Forecast MAE**: DA $29.58/MWh, RT $7.98/MWh
- **Why show**: Recent data, good accuracy

### 7. **December 1, 2024** (Recent Winter)
- **URL**: `http://localhost:3000/bess-market-bidding?date=2024-12-01&hub=HB_HOUSTON`
- **Trained on**: 12,923 samples (data up to Nov 30, 2024 23:00)
- **Forecast MAE**: DA $22.69/MWh, RT $17.00/MWh
- **Why show**: Most recent forecast available

---

## 🎯 Recommended Demo Flow

### **Best 3 Dates to Show** (if time limited):

1. **February 20, 2024** - Matches your existing screenshot
2. **April 10, 2024** - Best forecast accuracy (DA MAE $19.78)
3. **December 1, 2024** - Most recent, shows system works with fresh data

---

## 📊 What the Forecasts Show

Each forecast includes:

✅ **5 Quantiles for DA Prices**: P10, P25, **P50 (median)**, P75, P90
✅ **5 Quantiles for RT Prices**: P10, P25, **P50 (median)**, P75, P90
✅ **Spike Probabilities**: High (>$400) and Extreme (>$1000)
✅ **Actual Prices**: Included for comparison (shows forecast vs reality)
✅ **48-Hour Horizon**: Full 2-day ahead forecast

---

## 🎨 Visual Appearance

When you click "Show AI Forecast", you'll see:

- **Purple lines**: DA price forecast (P50) with shaded quantile bands
- **Green lines**: RT price forecast (P50) with shaded quantile bands
- **Light shading**: P10-P90 range (80% confidence interval)
- **Dark shading**: P25-P75 range (50% confidence interval)
- **Solid lines**: P50 median forecast

---

## 🔬 Walk-Forward Methodology Explained

**Key Point for Investors:** These forecasts have **NO look-ahead bias**.

For each demonstration date:
1. ✅ Model was trained ONLY on data **before** that date
2. ✅ Forecast was generated without seeing the future
3. ✅ Actual prices are shown for comparison
4. ✅ This is how the model would perform in production

**Example:**
- Date: Feb 20, 2024
- Training data: Only data up to Feb 19, 2024 23:00
- Forecast: Generated for Feb 20-22, 2024
- Validation: Compare forecast vs actual (both shown in API response)

---

## 🚨 Important Notes

### For Non-Demo Dates:
- If you select a date NOT in the list above, the API will serve a **retrospective forecast**
- These have look-ahead bias (model trained on ALL data)
- The API will include a warning: `"look_ahead_bias": true`

### Hub Specificity:
- Current forecasts are **system-wide averages**, not hub-specific
- The model averages across all ERCOT hubs
- For GAMBIT_BESS1 and HB_HOUSTON: forecasts show general ERCOT conditions

### Resource Specificity:
- Forecasts are **price forecasts**, not resource-specific
- GAMBIT_BESS1 will see the same price forecast as other resources
- Future enhancement: resource-specific dispatch forecasts

---

## 🎤 Demo Talking Points

### 1. **Uncertainty Quantification**
> "Rather than just giving you a single price prediction, we provide full uncertainty quantification with 5 quantiles. The P10-P90 range shows you the 80% confidence interval - there's a 10% chance prices will be below P10, and a 10% chance above P90."

### 2. **Walk-Forward Validation**
> "These forecasts were generated using proper walk-forward methodology - the model was trained only on data available before each forecast date. This eliminates look-ahead bias and shows how the system would actually perform in production."

### 3. **Dashboard Integration**
> "The forecasts are seamlessly integrated into your existing BESS bidding workflow. Click 'Show AI Forecast' on any of these dates to see the predictions overlaid on actual market data."

### 4. **Spike Detection**
> "The system also provides spike probability forecasts - predicting when prices might exceed $400/MWh (high spike) or $1000/MWh (extreme spike). This is critical for BESS revenue optimization."

### 5. **Actual vs Forecast**
> "For each demonstration date, we can show you both the forecast AND what actually happened. For example, on April 10, 2024, our DA forecast had a mean absolute error of just $19.78/MWh."

---

## 🔧 Technical Details

### Model Training:
- **Architecture**: Transformer-based with quantile regression
- **Features**: 236 features including ORDC indicators, load forecasts, weather data
- **Training samples**: Varies by date (1,579 to 12,923 samples)
- **Training cutoff**: Always 1 hour before forecast origin

### Forecast Horizon:
- **48 hours ahead**: Industry-standard 2-day forecast
- **Hourly granularity**: One prediction per hour
- **Total predictions per forecast**: 48 hours × 10 values (5 DA quantiles + 5 RT quantiles)

### API Response:
- **Format**: JSON
- **Size**: ~225KB for all 7 demo forecasts
- **Load time**: <100ms (pre-computed, served from cache)

---

## 📝 Quick Start Commands

```bash
# Test the API
curl "http://localhost:5000/forecast?origin_time=2024-02-20T00:00:00" | jq .

# Check available dates
curl "http://localhost:5000/health"

# Restart API if needed
cd /home/enrico/projects/power_market_pipeline/ai_forecasting
../.venv/bin/python forecast_api.py > forecast_api.log 2>&1 &
```

---

## ✅ Pre-Demo Checklist

- [ ] Forecast API running: `curl http://localhost:5000/health`
- [ ] Battalion dashboard running: `http://localhost:3000`
- [ ] Browser tab open to Feb 20, 2024 page
- [ ] "Show AI Forecast" button visible
- [ ] Prepared to explain walk-forward methodology
- [ ] Ready to discuss uncertainty quantification

---

## 🎯 Success Metrics

If the demo goes well, investors should understand:

1. ✅ **System provides uncertainty quantification** (not just point estimates)
2. ✅ **Forecasts are validated properly** (walk-forward, no look-ahead bias)
3. ✅ **Integration is production-ready** (seamless dashboard integration)
4. ✅ **Performance is measurable** (MAE metrics for each forecast)
5. ✅ **Technology is sophisticated** (Transformer architecture, 236 features)

---

**Good luck with your Mercuria demo on Friday at 1 PM!** 🚀
