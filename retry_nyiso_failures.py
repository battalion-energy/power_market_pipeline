#!/usr/bin/env python3
"""
Retry failed NYISO downloads.

Retries specific dates that failed during the initial download.
"""

import sys
from datetime import datetime
from pathlib import Path
import logging

# Add parent directory to path to import the downloader
sys.path.insert(0, str(Path(__file__).parent))

# Import after path modification
from download_nyiso_gridstatus import NYISOGridstatusDownloader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nyiso_retry.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Failed downloads to retry
FAILED_DOWNLOADS = [
    # Timeout errors - should work on retry
    ('2019-11-13', 'lmp_day_ahead_hourly'),
    ('2019-12-19', 'as_day_ahead_hourly'),
    ('2022-09-15', 'lmp_day_ahead_hourly'),
    ('2022-11-09', 'lmp_real_time_5_min'),
    ('2024-10-02', 'lmp_real_time_5_min'),

    # No objects to concatenate - may be missing data
    ('2021-04-01', 'load'),
    ('2021-04-02', 'load'),
    ('2021-04-03', 'load'),
    ('2021-04-04', 'load'),
    ('2021-04-05', 'load'),
    ('2021-04-06', 'load'),
    ('2024-03-27', 'load'),
    ('2025-08-01', 'lmp_real_time_5_min'),
]


def main():
    output_dir = Path("/pool/ssd8tb/data/iso")
    downloader = NYISOGridstatusDownloader(output_dir)

    logger.info(f"\n{'='*80}")
    logger.info(f"NYISO FAILED DOWNLOADS RETRY")
    logger.info(f"Retrying {len(FAILED_DOWNLOADS)} failed downloads")
    logger.info(f"{'='*80}\n")

    successes = 0
    failures = 0

    for date_str, data_type in FAILED_DOWNLOADS:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        logger.info(f"\nRetrying {data_type} for {date_str}...")

        try:
            if data_type == 'lmp_day_ahead_hourly':
                result = downloader.download_lmp_day(date, 'DAY_AHEAD_HOURLY')
            elif data_type == 'lmp_real_time_5_min':
                result = downloader.download_lmp_day(date, 'REAL_TIME_5_MIN')
            elif data_type == 'as_day_ahead_hourly':
                result = downloader.download_as_day(date, 'DAY_AHEAD_HOURLY')
            elif data_type == 'load':
                result = downloader.download_load_day(date)
            else:
                logger.error(f"  Unknown data type: {data_type}")
                failures += 1
                continue

            if result:
                successes += 1
                logger.info(f"  ✓ Success!")
            else:
                failures += 1
                logger.warning(f"  ✗ Still failed")

        except Exception as e:
            failures += 1
            logger.error(f"  ✗ Exception: {e}")

    logger.info(f"\n{'='*80}")
    logger.info(f"RETRY COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successes: {successes}")
    logger.info(f"Failures: {failures}")
    logger.info(f"{'='*80}\n")


if __name__ == "__main__":
    main()
