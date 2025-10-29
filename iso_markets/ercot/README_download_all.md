# ERCOT Historical Data Downloader

This system downloads ALL historical data from the ERCOT data archive automatically.

## Components

1. **scrape_ercot_catalog.py** - Scrapes the ERCOT website to create a catalog of all available datasets
2. **ercot_datasets_catalog.csv** - CSV file containing all dataset URLs and names (44 datasets)
3. **ercot_download_all_historical.py** - Main script that downloads all datasets with tracking
4. **ercot_download_tracking.json** - JSON file tracking download progress for each dataset

## Setup

All data will be stored in: `/Users/enrico/data/ERCOT_data_clean_archive`

Each dataset gets its own subdirectory based on the dataset name.

## Usage

### Update Mode (Default)
Downloads missing data from the last downloaded date to today. This is the mode you'll use regularly to keep data up-to-date.

```bash
# Update all datasets with latest data
python ercot_download_all_historical.py

# Update specific datasets only
python ercot_download_all_historical.py --datasets NP4-732-CD NP4-737-CD
```

### Extend Mode
Downloads historical data backwards from your earliest downloaded date to a historical cutoff (default: 2019-01-01).

```bash
# Extend all datasets back to 2019-01-01 (default)
python ercot_download_all_historical.py --mode extend

# Extend back to a different date
python ercot_download_all_historical.py --mode extend --historical-cutoff 2010-01-01

# Extend specific datasets only
python ercot_download_all_historical.py --mode extend --datasets NP4-732-CD
```

## How It Works

### First Run (Update Mode)
- For each dataset in the catalog:
  - Downloads data from 2019-01-01 to today
  - Creates a subdirectory in the base download path
  - Updates tracking JSON with date ranges

### Subsequent Runs (Update Mode)
- Reads the tracking JSON to see what was last downloaded
- For each dataset, downloads data from the last end_date to today
- Updates tracking JSON

### Extend Mode
- Reads the tracking JSON
- For each dataset, downloads data from historical_cutoff to the earliest start_date
- Extends the date range backwards
- Updates tracking JSON

## Tracking JSON Format

```json
{
  "NP4-732-CD": {
    "name": "Wind Power Production - Hourly Averaged Actual and Forecasted Values",
    "url": "https://data.ercot.com/data-product-archive/NP4-732-CD",
    "start_date": "2019-01-01",
    "end_date": "2025-10-28",
    "last_updated": "2025-10-28T10:30:00"
  }
}
```

## Available Datasets

The catalog contains 44 datasets including:
- Wind and Solar Power Production (actual and forecasted)
- Load forecasts and actual loads
- LMPs and Settlement Point Prices (DAM and Real-Time)
- Shadow Prices and Congestion data
- Ancillary Services
- Resource Outages
- And many more...

See `ercot_datasets_catalog.csv` for the complete list.

## Tips

1. **Start small**: Test with a single dataset first:
   ```bash
   python ercot_download_all_historical.py --datasets NP4-732-CD
   ```

2. **Run overnight**: Downloading all 44 datasets can take many hours. Consider running overnight or in the background.

3. **Regular updates**: Set up a cron job to run the update mode daily:
   ```bash
   0 2 * * * cd /path/to/ercot && python ercot_download_all_historical.py
   ```

4. **Check tracking**: Review `ercot_download_tracking.json` to see what's been downloaded

5. **Resume after failure**: If a download fails, just run the script again. It will skip datasets that are already up-to-date.

## Refreshing the Catalog

If new datasets are added to ERCOT, re-run the catalog scraper:

```bash
python scrape_ercot_catalog.py
```

This will update `ercot_datasets_catalog.csv` with any new datasets.
