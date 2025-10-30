# Manual reBAP Update Process

**Frequency:** Monthly (or as needed)
**Time Required:** ~5 minutes
**Why Manual:** Netztransparenz API requires full OAuth setup; manual download is simpler for infrequent updates

---

## Quick Update Process

### **Step 1: Download Latest Data**
1. Go to: https://www.netztransparenz.de/de-de/Regelenergie/Ausgleichsenergiepreis/reBAP
2. Click download/export button
3. Select date range: **Last downloaded date to today**
4. Download CSV to `/home/enrico/Downloads/GermanPowerMarket/`

### **Step 2: Process & Move Data**
```bash
cd /home/enrico/projects/power_market_pipeline

# Run the update script (automatically finds new file and processes it)
./iso_markets/entso_e/scripts/update_rebap.sh
```

That's it! The script will:
- Find the new CSV in Downloads
- Append to existing data
- Remove duplicates
- Verify data quality
- Move to correct location

---

## Detailed Instructions

### **1. Download from Netztransparenz**

**Portal:** https://www.netztransparenz.de/de-de/Regelenergie/Ausgleichsenergiepreis/reBAP

**What to Download:**
- Product: reBAP (unterdeckt/überdeckt)
- Date Range: From last update to present
- Format: CSV

**Current Data Coverage:**
- Last Update: 2025-10-15
- Next Update Due: ~2025-11-15 (monthly)

### **2. Run Update Script**

The script will automatically:
1. Find latest CSV in Downloads folder
2. Read new data
3. Load existing data
4. Merge and deduplicate
5. Verify continuity
6. Save updated file
7. Update documentation

### **3. Verify Update**

```bash
# Check last date in dataset
tail -5 /pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_*.csv

# Check record count
wc -l /pool/ssd8tb/data/iso/ENTSO_E/csv_files/rebap/rebap_de_*.csv
```

---

## Update Script (Created Below)

**Location:** `iso_markets/entso_e/scripts/update_rebap.sh`

**Usage:**
```bash
./iso_markets/entso_e/scripts/update_rebap.sh
```

---

## Alternative: Automated API Access

If you need automated updates, you'll need to:

### **Option A: OAuth API (Complex)**
1. Register at https://extranet.netztransparenz.de/
2. Create OAuth client (get client_id + client_secret)
3. Implement OAuth token flow
4. Build API client
5. Integrate into cronjob

**Time:** ~1 week development
**Complexity:** High
**Benefit:** Fully automated

### **Option B: Web Scraper (Medium)**
1. Build Selenium/Playwright scraper
2. Automate browser downloads
3. Process downloaded files

**Time:** ~2-3 days development
**Complexity:** Medium
**Benefit:** Automated but fragile

### **Option C: Manual Updates (Simple)** ✅ RECOMMENDED
1. Download CSV monthly
2. Run update script
3. Done!

**Time:** 5 minutes/month
**Complexity:** Very Low
**Benefit:** Simple, reliable, low maintenance

For your use case (BESS EMS with historical analysis), **monthly manual updates are sufficient and most practical**.

---

## Update Schedule Recommendation

**Frequency:** Monthly
**Best Time:** First week of each month
**Why:** reBAP data has 8-day publication delay, so monthly updates capture all finalized data

**Calendar Reminder:**
```
Monthly - 1st of month
Task: Update reBAP data
Command: cd ~/projects/power_market_pipeline && ./iso_markets/entso_e/scripts/update_rebap.sh
Time: 5 minutes
```

---

## Future: If You Need Full Automation

If your EMS operations require daily/weekly reBAP updates, let me know and I'll:
1. Set up full OAuth2.0 client
2. Build automated downloader
3. Integrate into daily cronjob

**Current Assessment:** Not needed yet. You have 5.8 years of historical data, and monthly updates are sufficient for analysis and backtesting.

---

**Last Manual Update:** 2025-10-28
**Data Coverage:** 2019-12-31 to 2025-10-15
**Next Update Due:** ~2025-11-15
