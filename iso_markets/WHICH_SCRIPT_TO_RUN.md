# Which Script Should I Run? 🤔

Quick reference guide for ISO converter scripts.

---

## 🧪 **Testing / First Time Setup**

### `test_converters_safe.sh`
**Use when:** Testing converters, verifying everything works

```bash
./test_converters_safe.sh
```

**What it does:**
- Runs all 7 converters with **2024 data only** (small test)
- One at a time with memory protection
- Individual log files for each
- Shows success/failure status

**Output:** Test logs in `logs/*_safe_test_*.log`

**Time:** ~5-10 minutes

---

## 🚀 **Production: Process ALL Data**

### `run_all_converters_safe.sh`
**Use when:** Initial data load, processing all historical data

```bash
# Process ALL years for all ISOs
./run_all_converters_safe.sh

# Or process specific year
./run_all_converters_safe.sh --year 2024
```

**What it does:**
- Runs all 7 converters sequentially
- Processes ALL available data (2019-2025+)
- Day-Ahead markets only (fastest)
- Full memory protection (80GB cap)
- Progress tracking and statistics

**Output:** Production parquet files in `/pool/ssd8tb/data/iso/unified_iso_data/parquet/`

**Time:** 2-6 hours (depends on data volume)

**Recommended for:**
- ✅ First-time setup
- ✅ Bulk processing
- ✅ When you have 2+ hours available

---

## 🔄 **Daily/Weekly: Rebuild Current Year**

### `rebuild_2025_parquet.sh` ⭐ **← Use This Regularly**
**Use when:** Updating current year data, daily/weekly data refresh

```bash
# Rebuild 2025 DA data (recommended)
./rebuild_2025_parquet.sh

# Or rebuild RT data
./rebuild_2025_parquet.sh --rt-only

# Or rebuild everything
./rebuild_2025_parquet.sh --all
```

**What it does:**
- Rebuilds **2025 parquet files only**
- Uses atomic file moves (safe during reads!)
- One ISO at a time
- Memory protected

**Atomic Update Process:**
1. Creates temp file: `da_energy_hourly_2025.parquet.tmp`
2. Writes all 2025 data to temp file
3. **Atomic move:** `mv .tmp → .parquet` (instant replace)
4. **Readers never see corrupted data!** ✅

**Output:** Updates files in place atomically

**Time:** 15-30 minutes

**Recommended schedule:**
- 📅 **Daily:** Rebuild 2025 DA data to get latest prices
- 📅 **Weekly:** Full rebuild with `--all` flag
- 📅 **Monthly:** Run full `run_all_converters_safe.sh` to update all years

**Safe to run while:**
- ✅ Other processes are reading parquet files
- ✅ Analysis notebooks are open
- ✅ Dashboards are running

---

## 🎯 **Single ISO: Run One Converter**

### `run_converter_safe.sh`
**Use when:** Processing one specific ISO

```bash
# Examples:
./run_converter_safe.sh miso --year 2024 --da-only
./run_converter_safe.sh ercot --all
./run_converter_safe.sh pjm --year 2023
```

**What it does:**
- Runs ONE converter with memory protection
- Flexible arguments
- Faster than running all

**Time:** 5-30 minutes (depends on ISO and year)

**Use cases:**
- 🔧 Debugging one ISO
- 🔧 Reprocessing specific year
- 🔧 Testing after code changes

---

## 📊 **Comparison Table**

| Script | Speed | Scope | Use Case | Atomic? |
|--------|-------|-------|----------|---------|
| **test_converters_safe.sh** | ⚡ Fast (5-10m) | 2024 only | Testing | ✅ |
| **run_all_converters_safe.sh** | 🐌 Slow (2-6h) | All years | Initial load | ✅ |
| **rebuild_2025_parquet.sh** | ⚡⚡ Fastest (15-30m) | 2025 only | **Daily updates** ⭐ | ✅ |
| **run_converter_safe.sh** | ⚡ Fast (5-30m) | Custom | One-off runs | ✅ |

---

## 🗓️ **Recommended Workflow**

### **Initial Setup (Do Once):**
```bash
# 1. Test everything works
./test_converters_safe.sh

# 2. Process all historical data
./run_all_converters_safe.sh
# Wait 2-6 hours...

# 3. Verify output
ls -lh /pool/ssd8tb/data/iso/unified_iso_data/parquet/*/
```

### **Daily Operations:**
```bash
# Every morning: rebuild current year (2025)
./rebuild_2025_parquet.sh
# Takes 15-30 minutes

# Safe to run in cron:
0 6 * * * cd /home/enrico/projects/power_market_pipeline/iso_markets && ./rebuild_2025_parquet.sh >> /pool/ssd8tb/data/iso/unified_iso_data/logs/cron_rebuild.log 2>&1
```

### **Monthly Maintenance:**
```bash
# Once a month: update all years
./run_all_converters_safe.sh
```

### **Fix One ISO:**
```bash
# If MISO failed, rerun just MISO
./run_converter_safe.sh miso --year 2025 --da-only
```

---

## ⚙️ **Configuration**

### Change Memory Limits:
```bash
# For big datasets
export MEMORY_CAP_GB=120
export ISO_CONVERTER_MEMORY_LIMIT_GB=100
./rebuild_2025_parquet.sh

# For constrained systems
export MEMORY_CAP_GB=40
export ISO_CONVERTER_MEMORY_LIMIT_GB=30
./rebuild_2025_parquet.sh
```

### Market Types:
```bash
# Day-Ahead only (fastest, recommended)
./rebuild_2025_parquet.sh --da-only

# Real-Time only
./rebuild_2025_parquet.sh --rt-only

# Ancillary Services only
./rebuild_2025_parquet.sh --as-only

# Everything (slow)
./rebuild_2025_parquet.sh --all
```

---

## 🆘 **Troubleshooting**

### "Which script for daily updates?"
➡️ **`rebuild_2025_parquet.sh`** - Fast, safe, atomic

### "Which script for initial setup?"
➡️ **`run_all_converters_safe.sh`** - Processes all data

### "How do I test changes?"
➡️ **`test_converters_safe.sh`** - Quick validation

### "Process failed, how to retry?"
➡️ **`run_converter_safe.sh <iso> --year YYYY`** - Rerun one ISO

### "Is it safe to run while reading files?"
➡️ **YES!** All scripts use atomic writes. Readers never see partial data.

### "Can I run in parallel?"
➡️ **NO!** These scripts are designed to run sequentially to prevent memory issues. Run one script at a time.

---

## 📝 **Summary**

**For 99% of use cases:**

```bash
# Initial setup (once)
./run_all_converters_safe.sh

# Daily updates (automated)
./rebuild_2025_parquet.sh
```

**That's it!** 🎉

---

## 📚 **More Information**

- Memory protection details: `MEMORY_PROTECTION_SYSTEM.md`
- Converter implementation: `ALL_ISO_CONVERTERS_README.md`
- Troubleshooting: Check `logs/` directory
