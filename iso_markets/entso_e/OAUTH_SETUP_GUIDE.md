# Netztransparenz OAuth 2.0 Setup Guide

**Complete guide to register for API access and enable automated reBAP downloads**

---

## Prerequisites

✅ You already have:
- 5.8 years of historical reBAP data (2019-12-31 to 2025-10-15)
- Automated download scripts ready to use
- Manual update process as fallback

❌ You need to obtain:
- OAuth 2.0 Client ID
- OAuth 2.0 Client Secret

---

## Step 1: Register for API Access

### **1.1 Create Account**
1. Go to: https://extranet.netztransparenz.de/
2. Click "Register" or "Registrieren"
3. Fill out registration form:
   - Company name
   - Contact information
   - Email address
   - Purpose: "BESS Energy Management System - Data Analysis"

4. Verify email and activate account

**Time Required:** 10-15 minutes

---

### **1.2 Create OAuth Client**

Once logged in:

1. Navigate to **"My Clients"** or **"Meine Clients"** in the menu
2. Click **"Create New Client"** button
3. Fill out client information:
   - **Client Name:** "BESS EMS - reBAP Downloader"
   - **Description:** "Automated reBAP imbalance price data retrieval"
   - **Redirect URI:** (Not required for Client Credentials Grant - leave blank or use `http://localhost`)

4. Click **"Create"** or **"Erstellen"**

5. **IMPORTANT:** Copy your credentials immediately:
   ```
   Client ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   Client Secret: yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
   ```

   ⚠️ **WARNING:** Client Secret is shown only ONCE! Save it securely.

6. You can create up to **5 clients** per account

**Time Required:** 5 minutes

---

## Step 2: Configure Credentials

### **2.1 Add to .env File**

Edit `/home/enrico/projects/power_market_pipeline/.env`:

```bash
# Netztransparenz OAuth 2.0 Credentials
NETZTRANSPARENZ_CLIENT_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
NETZTRANSPARENZ_CLIENT_SECRET=yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy
```

**Format Notes:**
- Client ID is typically a UUID format (with dashes)
- Client Secret is a long alphanumeric string
- No quotes needed in .env file
- Keep .env file private (already in .gitignore)

### **2.2 Remove Old API Key**

The old `NETZTRANSPARENZ_KEY=ntp_...` is not used. You can remove it or leave it (scripts will ignore it).

---

## Step 3: Test OAuth Connection

### **3.1 Test Authentication**

Run the OAuth test:

```bash
cd /home/enrico/projects/power_market_pipeline
uv run python iso_markets/entso_e/netztransparenz_oauth.py
```

**Expected Output:**
```
Testing Netztransparenz OAuth Client
============================================================

Requesting new OAuth access token...
✅ Token acquired (expires in 3600s)
✅ API connection successful

Testing reBAP data retrieval...
✅ Retrieved 192 records

Sample data:
[
  {
    "datum": "2025-10-26T00:00:00",
    "rebapUnterdeckt": -45.67,
    "rebapUeberdeckt": -45.67
  },
  ...
]
```

### **3.2 Test Full Download**

Test the complete download and merge process:

```bash
cd /home/enrico/projects/power_market_pipeline
uv run python iso_markets/entso_e/download_rebap_auto.py --days 7
```

**Expected Output:**
```
================================================================================
Automated reBAP Data Download
================================================================================
Date: 2025-10-28 ...

Requesting new OAuth access token...
✅ Token acquired (expires in 3600s)
✅ API connection successful

Download range:
  From: 2025-10-14 23:00:00
  To: 2025-10-28 ...
  Days: 14

Downloading reBAP data from API...
✅ Downloaded 1344 records from API

Converting API data to standardized format...
  New records: 1,344
  Date range: 2025-10-14 23:00:00+00:00 to 2025-10-28 ...

Loading existing data: rebap_de_2019-12-31_2025-10-15.csv
  Existing records: 203,036
  Date range: 2019-12-31 23:00:00+00:00 to 2025-10-15 22:45:00+00:00

Merging data...
  Combined records: 204,380
  Duplicates removed: 0

Verifying data continuity...
✅ No gaps - continuous 15-minute intervals

Saving to: /pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_2019-12-31_2025-10-28.csv
  File size: 12.9 MB

Summary:
  Total records: 204,380
  Date range: 2019-12-31 23:00:00+00:00 to 2025-10-28 ...
  Price range (undersupply): €-8270.00 to €15859.00/MWh

================================================================================
✅ reBAP update complete!
================================================================================
```

---

## Step 4: Enable Daily Automation

### **4.1 Add to Crontab**

Edit crontab:
```bash
crontab -e
```

Add this line to run daily at 6:00 AM:
```bash
# reBAP daily update (runs at 6:00 AM)
0 6 * * * /home/enrico/projects/power_market_pipeline/iso_markets/entso_e/scripts/update_rebap_daily.sh
```

**Alternative Schedules:**
```bash
# Run twice daily (6 AM and 6 PM)
0 6,18 * * * /home/enrico/projects/power_market_pipeline/iso_markets/entso_e/scripts/update_rebap_daily.sh

# Run every 6 hours
0 */6 * * * /home/enrico/projects/power_market_pipeline/iso_markets/entso_e/scripts/update_rebap_daily.sh

# Run weekly on Mondays at 6 AM
0 6 * * 1 /home/enrico/projects/power_market_pipeline/iso_markets/entso_e/scripts/update_rebap_daily.sh
```

### **4.2 Verify Cron Job**

List your cron jobs:
```bash
crontab -l
```

### **4.3 Check Logs**

Logs are saved to:
```bash
/home/enrico/projects/power_market_pipeline/logs/rebap/
```

View latest log:
```bash
ls -lt ~/projects/power_market_pipeline/logs/rebap/ | head -5
tail -50 ~/projects/power_market_pipeline/logs/rebap/rebap_update_*.log
```

---

## Step 5: Verify Automated Updates

### **5.1 Check Data Freshness**

```bash
# View latest records
tail -10 /pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_*.csv

# Check file modification time
ls -lh /pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_*.csv
```

### **5.2 Monitor for Gaps**

Run this check periodically:
```python
import pandas as pd
df = pd.read_csv('/pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_2019-12-31_2025-10-28.csv')
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], utc=True)
df['time_diff'] = df['datetime_utc'].diff()
gaps = df[df['time_diff'] > pd.Timedelta(minutes=15)]
print(f"Gaps found: {len(gaps)}")
```

---

## Troubleshooting

### **Error: "Missing OAuth credentials"**

**Cause:** .env file missing credentials

**Fix:**
1. Check .env file exists: `cat .env | grep NETZTRANSPARENZ`
2. Verify both CLIENT_ID and CLIENT_SECRET are set
3. No quotes around values
4. Reload environment: `cd /home/enrico/projects/power_market_pipeline`

---

### **Error: "401 Unauthorized"**

**Cause:** Invalid credentials or expired client

**Fix:**
1. Verify credentials are correct (copy-paste from portal)
2. Check if client is still active in "My Clients" portal
3. Create new client if needed
4. Update .env with new credentials

---

### **Error: "404 Not Found"**

**Cause:** API endpoint changed or wrong URL

**Fix:**
1. Check API documentation: https://extranet.netztransparenz.de/swagger
2. Verify endpoint structure matches documentation
3. Contact Netztransparenz support if persistent

---

### **Error: "Rate limit exceeded"**

**Cause:** Too many API requests

**Fix:**
1. Reduce cron frequency (daily instead of hourly)
2. Download larger time periods less frequently
3. Contact Netztransparenz for rate limit increase

---

### **No New Data Downloaded**

**Cause:** Data already up-to-date or publication delay

**Check:**
1. reBAP has 8-day publication delay
2. Latest data may not be available yet
3. View latest date in dataset
4. Compare with portal: https://www.netztransparenz.de/de-de/Regelenergie/Ausgleichsenergiepreis/reBAP

---

## Manual Fallback

If OAuth API has issues, you can always use the manual process:

1. Download CSV from https://www.netztransparenz.de/de-de/Regelenergie/Ausgleichsenergiepreis/reBAP
2. Save to `/home/enrico/Downloads/GermanPowerMarket/`
3. Run: `./iso_markets/entso_e/scripts/update_rebap.sh`

---

## API Rate Limits

**Default Limits** (check with Netztransparenz):
- Requests per day: ~10,000
- Requests per hour: ~1,000
- Data retention: 10+ years

**Your Usage:**
- Daily update: ~1 API call/day
- Weekly update: ~1 API call/week
- Well within limits

---

## Security Best Practices

1. **Keep credentials private:**
   - Never commit .env to git (already in .gitignore)
   - Don't share Client Secret
   - Rotate credentials annually

2. **Monitor access:**
   - Review "My Clients" portal for unauthorized access
   - Check API usage logs
   - Disable unused clients

3. **Backup credentials:**
   - Store Client Secret in password manager
   - Document in secure location
   - Keep screenshot of portal

---

## Next Steps

Once OAuth is working:

1. **Expand Data Collection:**
   - Activated aFRR: `NrvSaldo/AktivierteSRL/Qualitaetsgesichert`
   - Activated mFRR: `NrvSaldo/AktivierteMRL/Qualitaetsgesichert`
   - Day-ahead spot prices: `Spotmarktpreise`

2. **Multi-Zone Expansion:**
   - Same OAuth client works for all German TSO data
   - No additional registration needed

3. **Real-Time Integration:**
   - Integrate into BESS EMS
   - Use for revenue optimization
   - Price forecasting models

---

## Support

**Netztransparenz Support:**
- Email: support@netztransparenz.de
- Portal: https://extranet.netztransparenz.de/
- API Docs: https://extranet.netztransparenz.de/swagger

**Scripts Location:**
- OAuth Client: `iso_markets/entso_e/netztransparenz_oauth.py`
- Downloader: `iso_markets/entso_e/download_rebap_auto.py`
- Daily Script: `iso_markets/entso_e/scripts/update_rebap_daily.sh`
- This Guide: `iso_markets/entso_e/OAUTH_SETUP_GUIDE.md`

---

**Setup Time Estimate:**
- Registration: 10-15 minutes
- OAuth client creation: 5 minutes
- Testing: 5 minutes
- Cron setup: 2 minutes
- **Total: ~30 minutes**

**Maintenance:**
- Daily automated updates: 0 minutes (automated)
- Credential rotation: 5 minutes/year
- Monitoring: 5 minutes/month

---

**Last Updated:** 2025-10-28
**Status:** Ready for OAuth registration
