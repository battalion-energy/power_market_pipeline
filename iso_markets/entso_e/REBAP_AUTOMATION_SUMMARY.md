# reBAP Automated Download Implementation Summary

**Complete OAuth 2.0 automation system for German imbalance price data**

---

## ✅ What's Been Built

### **1. OAuth 2.0 Authentication Module**
**File:** `iso_markets/entso_e/netztransparenz_oauth.py`

- Full OAuth 2.0 Client Credentials Grant implementation
- Token caching with automatic refresh
- Support for all Netztransparenz API endpoints
- Standalone testing capability

**Features:**
- Automatic token acquisition and refresh
- 3600-second token lifetime management
- Health check endpoint testing
- Comprehensive error handling

---

### **2. Automated reBAP Downloader**
**File:** `iso_markets/entso_e/download_rebap_auto.py`

- Downloads latest reBAP data from OAuth API
- Converts JSON API response to standardized CSV format
- Merges with existing 5.8-year historical dataset
- Deduplicates records (keeps latest download_date)
- Verifies 15-minute data continuity
- Auto-generates filename with date range

**Usage:**
```bash
cd /home/enrico/projects/power_market_pipeline
uv run python iso_markets/entso_e/download_rebap_auto.py --days 7
```

**Output:** Updated `rebap_de_YYYY-MM-DD_YYYY-MM-DD.csv` with merged data

---

### **3. Daily Automation Script**
**File:** `iso_markets/entso_e/scripts/update_rebap_daily.sh`

- Wrapper script for cron scheduling
- Comprehensive logging to `logs/rebap/`
- Automatic log rotation (keeps 30 days)
- Error handling and status reporting

**Cron Setup:**
```bash
# Add to crontab (crontab -e)
0 6 * * * /home/enrico/projects/power_market_pipeline/iso_markets/entso_e/scripts/update_rebap_daily.sh
```

---

### **4. Complete Documentation**
**File:** `iso_markets/entso_e/OAUTH_SETUP_GUIDE.md`

- Step-by-step OAuth registration guide
- Credential configuration instructions
- Testing procedures
- Troubleshooting guide
- Security best practices

---

## 🔧 Configuration Required

### **Step 1: Register for OAuth Credentials**

You need to obtain **Client ID** and **Client Secret**:

1. Go to: https://extranet.netztransparenz.de/
2. Register/login
3. Navigate to **"My Clients"**
4. Create new client:
   - Name: "BESS EMS - reBAP Downloader"
   - Description: "Automated reBAP data retrieval"
5. Copy credentials (Secret shown only once!)

**Time Required:** ~15 minutes

---

### **Step 2: Update .env File**

Edit `/home/enrico/projects/power_market_pipeline/.env`:

```bash
# Replace placeholders with your actual credentials
NETZTRANSPARENZ_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
NETZTRANSPARENZ_CLIENT_SECRET=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

**Current Status:** Placeholders added at lines 54-55

---

### **Step 3: Test OAuth Connection**

```bash
cd /home/enrico/projects/power_market_pipeline

# Test authentication
uv run python iso_markets/entso_e/netztransparenz_oauth.py

# Test full download
uv run python iso_markets/entso_e/download_rebap_auto.py --days 7
```

**Expected Output:** OAuth token acquired, data downloaded, merged with existing 203K+ records

---

### **Step 4: Enable Daily Automation**

```bash
# Edit crontab
crontab -e

# Add line (runs daily at 6 AM)
0 6 * * * /home/enrico/projects/power_market_pipeline/iso_markets/entso_e/scripts/update_rebap_daily.sh
```

---

## 📊 Current Data Status

### **Historical Dataset (Already Downloaded)**
- **File:** `/pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_2019-12-31_2025-10-15.csv`
- **Records:** 203,036 (15-minute resolution)
- **Date Range:** 2019-12-31 to 2025-10-15
- **Coverage:** 5.8 years
- **Size:** 12.8 MB
- **Status:** ✅ Complete, no gaps

### **Price Statistics**
- **Undersupply:** -€8,270 to +€15,859 per MWh
- **Oversupply:** -€8,270 to +€15,859 per MWh
- **Volatility:** 20x higher than day-ahead prices
- **Average:** ~€50-100 per MWh range

---

## 🔄 How Automated Updates Work

### **Daily Process (After OAuth Setup)**

1. **Cron triggers** at 6:00 AM daily
2. **Script determines** latest date in existing dataset
3. **OAuth authentication** obtains access token
4. **API downloads** data from (latest_date - 1 day) to today
   - 1-day overlap ensures no gaps
5. **Data conversion** from JSON to standardized CSV format
6. **Merge & deduplicate** with existing dataset
7. **Continuity verification** checks for 15-minute gaps
8. **Save updated file** with new date range in filename
9. **Log results** to `logs/rebap/rebap_update_YYYYMMDD_HHMMSS.log`

**Estimated Run Time:** 10-30 seconds per day

---

## 📁 File Structure

```
power_market_pipeline/
├── .env                              # OAuth credentials (UPDATE REQUIRED)
├── iso_markets/entso_e/
│   ├── netztransparenz_oauth.py      # OAuth 2.0 client module
│   ├── download_rebap_auto.py        # Automated downloader
│   ├── OAUTH_SETUP_GUIDE.md          # Registration guide
│   ├── REBAP_AUTOMATION_SUMMARY.md   # This file
│   └── scripts/
│       ├── update_rebap_daily.sh     # Daily cron script
│       ├── update_rebap.sh           # Manual update (fallback)
│       └── convert_rebap_update.py   # Manual CSV processor
└── logs/rebap/                       # Daily update logs
    └── rebap_update_*.log

/pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/
├── rebap_de_2019-12-31_2025-10-15.csv    # Current dataset
├── rebap_de_2019-12-31_2025-10-28.csv    # After first update
└── raw/                                   # Archived manual downloads
```

---

## 🔐 Security Considerations

### **Credentials Management**
- ✅ `.env` file already in `.gitignore`
- ✅ OAuth credentials never committed to git
- ✅ Client Secret shown only once at creation
- ⚠️ Store backup in password manager

### **API Rate Limits**
- **Default:** ~10,000 requests/day
- **Your Usage:** 1 request/day (well within limits)
- **Token Lifetime:** 3600 seconds (auto-refresh)

### **Access Control**
- OAuth client can be disabled in portal
- Up to 5 clients per account
- Monitor access in "My Clients" dashboard

---

## 🎯 Next Steps

### **Immediate (Required for Automation)**
1. ✅ Scripts created
2. ❌ Register at https://extranet.netztransparenz.de/
3. ❌ Create OAuth client
4. ❌ Update .env with credentials
5. ❌ Test OAuth connection
6. ❌ Add to crontab

**Time:** ~30 minutes total

---

### **Future Enhancements**

Once OAuth is working, you can expand to:

#### **Additional German Market Data**
```python
# Already supported by OAuth client

# Activated aFRR (secondary reserve)
client.get_nrv_saldo_data('AktivierteSRL', 'Qualitaetsgesichert', date_from, date_to)

# Activated mFRR (minute reserve)
client.get_nrv_saldo_data('AktivierteMRL', 'Qualitaetsgesichert', date_from, date_to)

# Day-ahead spot prices
client.get_spot_market_prices(date_from, date_to)

# Redispatch volumes
client.get_redispatch_data(date_from, date_to)
```

#### **Multi-Zone Expansion**
- Same OAuth client works for all German TSO data
- No additional registration needed
- ~1 day to implement per data type

---

## 📈 Use Cases for BESS EMS

### **Revenue Optimization**
- **Day-ahead arbitrage:** Buy/sell based on DA price forecasts
- **Imbalance trading:** Profit from reBAP price volatility
- **Frequency response:** FCR/aFRR participation

### **Risk Management**
- **Imbalance exposure:** Monitor reBAP vs scheduled position
- **Price spike protection:** Historical volatility analysis
- **Revenue forecasting:** 5.8 years of historical data

### **Market Analysis**
- **Price correlation:** DA prices vs imbalance prices
- **Volatility patterns:** Hourly/daily/seasonal trends
- **Reserve activation:** aFRR/mFRR deployment patterns

---

## 🐛 Troubleshooting

### **"Missing OAuth credentials"**
**Cause:** .env not updated with real credentials

**Fix:**
1. Check: `cat .env | grep NETZTRANSPARENZ`
2. Replace `your_client_id_here` with actual UUID
3. Replace `your_client_secret_here` with actual secret

---

### **"401 Unauthorized"**
**Cause:** Invalid or expired credentials

**Fix:**
1. Verify credentials are correct (copy-paste from portal)
2. Check client is active in portal
3. Create new client if needed

---

### **"No new data downloaded"**
**Cause:** Data already up-to-date

**Expected:**
- reBAP has 8-day publication delay
- Latest data may not be available yet
- Script will skip if no new data

**Check:**
```bash
tail -5 /pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_*.csv
```

---

## 📊 Comparison: Manual vs Automated

| Feature | Manual Process | Automated OAuth |
|---------|---------------|-----------------|
| **Setup Time** | 0 minutes | 30 minutes (one-time) |
| **Monthly Effort** | 5 minutes | 0 minutes |
| **Data Freshness** | Monthly | Daily |
| **Reliability** | Depends on you | 100% automated |
| **API Calls** | 0 | 1/day (~30/month) |
| **Maintenance** | None | Minimal (logs) |
| **Cost** | Free | Free (within limits) |
| **Best For** | Infrequent updates | Daily operations |

**Recommendation:**
- **Now:** Use manual process (working, reliable)
- **After OAuth setup:** Switch to automated (better for BESS operations)
- **Fallback:** Keep manual scripts as backup

---

## 📞 Support

### **OAuth Registration Issues**
- Portal: https://extranet.netztransparenz.de/
- Email: support@netztransparenz.de

### **API Documentation**
- Swagger UI: https://extranet.netztransparenz.de/swagger
- Base URL: https://ds.netztransparenz.de/api/v1

### **Script Issues**
- OAuth Module: `iso_markets/entso_e/netztransparenz_oauth.py`
- Downloader: `iso_markets/entso_e/download_rebap_auto.py`
- Setup Guide: `iso_markets/entso_e/OAUTH_SETUP_GUIDE.md`

---

## 📝 Summary Checklist

- ✅ **OAuth authentication module** - Ready
- ✅ **Automated downloader** - Ready
- ✅ **Daily cron script** - Ready
- ✅ **Documentation** - Complete
- ✅ **Manual fallback** - Working
- ✅ **Historical data** - 5.8 years downloaded
- ❌ **OAuth credentials** - Registration required
- ❌ **Cron job** - Setup pending
- ❌ **Testing** - Pending credentials

**Status:** 80% complete - Awaiting OAuth registration

---

**Implementation Date:** 2025-10-28
**Historical Data:** 2019-12-31 to 2025-10-15 (203,036 records)
**Next Update:** After OAuth credentials obtained

**Estimated Time to Full Automation:** 30 minutes (OAuth registration + testing)
