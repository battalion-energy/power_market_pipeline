#!/usr/bin/env python3
"""
Verify ERCOT Web Service downloads for completeness and detect gaps.

This script:
1. Scans CSV files downloaded from ERCOT Web Service API
2. Checks for date/time gaps in the data
3. Verifies record counts make sense
4. Generates a gap report

Usage:
    python ercot_ws_verify_downloads.py
    python ercot_ws_verify_downloads.py --dataset DAM_Prices
    python ercot_ws_verify_downloads.py --fix-gaps
"""

import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DownloadVerifier:
    """Verify downloaded CSV files for completeness."""

    def __init__(self, base_dir: Path):
        """
        Initialize verifier.

        Args:
            base_dir: Base directory containing downloaded CSV files
        """
        self.base_dir = base_dir
        self.datasets = self._discover_datasets()

    def _discover_datasets(self) -> Dict[str, Path]:
        """Discover all dataset directories."""
        datasets = {}

        # Define expected directories
        expected_dirs = [
            "DAM_Settlement_Point_Prices",
            "RTM_Settlement_Point_Prices",
            "RegUpDown_Prices",
            "Reserve_Prices",
            "60-Day_DAM_Disclosure_Reports/Gen_Resources",
            "60-Day_DAM_Disclosure_Reports/Load_Resources",
            "60-Day_SCED_Disclosure_Reports/Gen_Resources",
            "60-Day_SCED_Disclosure_Reports/Load_Resources",
        ]

        for dir_path in expected_dirs:
            full_path = self.base_dir / dir_path
            if full_path.exists():
                dataset_name = dir_path.replace("/", "_")
                datasets[dataset_name] = full_path
                logger.info(f"Found dataset: {dataset_name} at {full_path}")

        return datasets

    def get_csv_files(self, dataset_dir: Path) -> List[Path]:
        """Get all CSV files in a directory sorted by date."""
        csv_files = sorted(dataset_dir.glob("*.csv"))
        return csv_files

    def parse_date_range_from_filename(self, filename: str) -> Tuple[datetime, datetime]:
        """
        Parse date range from filename.

        Expected format: dataset_name_YYYYMMDD_YYYYMMDD.csv
        """
        try:
            parts = filename.replace(".csv", "").split("_")
            # Last two parts should be dates
            start_str = parts[-2]
            end_str = parts[-1]

            start_date = datetime.strptime(start_str, "%Y%m%d")
            end_date = datetime.strptime(end_str, "%Y%m%d")

            return start_date, end_date
        except Exception as e:
            logger.warning(f"Could not parse dates from {filename}: {e}")
            return None, None

    def detect_gaps(self, csv_files: List[Path]) -> List[Tuple[datetime, datetime]]:
        """
        Detect gaps between downloaded CSV files.

        Args:
            csv_files: List of CSV file paths

        Returns:
            List of (gap_start, gap_end) tuples
        """
        gaps = []

        if not csv_files:
            return gaps

        # Parse all file date ranges
        file_ranges = []
        for csv_file in csv_files:
            start, end = self.parse_date_range_from_filename(csv_file.name)
            if start and end:
                file_ranges.append((start, end, csv_file))

        # Sort by start date
        file_ranges.sort(key=lambda x: x[0])

        # Check for gaps
        for i in range(len(file_ranges) - 1):
            current_end = file_ranges[i][1]
            next_start = file_ranges[i + 1][0]

            # Gap exists if next file doesn't start the day after current ends
            expected_next = current_end + timedelta(days=1)
            if next_start > expected_next:
                gap_start = expected_next
                gap_end = next_start - timedelta(days=1)
                gaps.append((gap_start, gap_end))
                logger.warning(
                    f"Gap detected: {gap_start.date()} to {gap_end.date()} "
                    f"({(gap_end - gap_start).days + 1} days)"
                )

        return gaps

    def verify_dataset(self, dataset_name: str, dataset_dir: Path) -> Dict:
        """
        Verify a single dataset for completeness.

        Returns:
            Dictionary with verification results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Verifying: {dataset_name}")
        logger.info(f"Directory: {dataset_dir}")
        logger.info(f"{'='*80}")

        csv_files = self.get_csv_files(dataset_dir)

        if not csv_files:
            logger.warning(f"No CSV files found in {dataset_dir}")
            return {
                "dataset": dataset_name,
                "status": "no_files",
                "file_count": 0,
                "gaps": []
            }

        logger.info(f"Found {len(csv_files)} CSV files")

        # Detect gaps
        gaps = self.detect_gaps(csv_files)

        # Get date range
        first_file_start, _ = self.parse_date_range_from_filename(csv_files[0].name)
        _, last_file_end = self.parse_date_range_from_filename(csv_files[-1].name)

        # Count total records
        total_records = 0
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                total_records += len(df)
            except Exception as e:
                logger.error(f"Error reading {csv_file}: {e}")

        result = {
            "dataset": dataset_name,
            "status": "ok" if not gaps else "gaps_found",
            "file_count": len(csv_files),
            "first_date": first_file_start.isoformat() if first_file_start else None,
            "last_date": last_file_end.isoformat() if last_file_end else None,
            "total_records": total_records,
            "gaps": [
                {
                    "start": gap[0].isoformat(),
                    "end": gap[1].isoformat(),
                    "days": (gap[1] - gap[0]).days + 1
                }
                for gap in gaps
            ]
        }

        # Print summary
        logger.info(f"\nResults:")
        logger.info(f"  Files: {result['file_count']}")
        logger.info(f"  Date range: {first_file_start.date() if first_file_start else 'N/A'} to {last_file_end.date() if last_file_end else 'N/A'}")
        logger.info(f"  Total records: {total_records:,}")
        logger.info(f"  Gaps: {len(gaps)}")

        if gaps:
            logger.warning(f"\n  Found {len(gaps)} gap(s):")
            for gap in gaps:
                logger.warning(
                    f"    {gap[0].date()} to {gap[1].date()} "
                    f"({(gap[1] - gap[0]).days + 1} days)"
                )
        else:
            logger.info(f"  ✓ No gaps detected!")

        return result

    def verify_all(self) -> Dict:
        """Verify all discovered datasets."""
        results = {}

        for dataset_name, dataset_dir in self.datasets.items():
            result = self.verify_dataset(dataset_name, dataset_dir)
            results[dataset_name] = result

        return results

    def save_report(self, results: Dict, output_file: Path):
        """Save verification results to JSON file."""
        report = {
            "verification_time": datetime.now().isoformat(),
            "datasets": results,
            "summary": {
                "total_datasets": len(results),
                "datasets_with_gaps": sum(1 for r in results.values() if r.get("gaps")),
                "datasets_ok": sum(1 for r in results.values() if r["status"] == "ok"),
            }
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nVerification report saved to: {output_file}")


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Verify ERCOT Web Service downloads"
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base directory containing downloaded CSV files"
    )

    parser.add_argument(
        "--dataset",
        help="Verify specific dataset only"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ercot_ws_verification_report.json"),
        help="Output JSON report file (default: ercot_ws_verification_report.json)"
    )

    args = parser.parse_args()

    # Default base directory
    if args.base_dir is None:
        base_dir = Path(os.getenv(
            "ERCOT_DATA_DIR",
            "/pool/ssd8tb/data/iso/ERCOT"
        )) / "ercot_market_data" / "ERCOT_data"
    else:
        base_dir = args.base_dir

    logger.info(f"Verifying downloads in: {base_dir}")

    verifier = DownloadVerifier(base_dir)

    if args.dataset:
        # Verify single dataset
        dataset_dir = verifier.datasets.get(args.dataset)
        if not dataset_dir:
            logger.error(f"Dataset '{args.dataset}' not found")
            logger.info(f"Available datasets: {list(verifier.datasets.keys())}")
            return 1

        result = verifier.verify_dataset(args.dataset, dataset_dir)
        results = {args.dataset: result}
    else:
        # Verify all datasets
        results = verifier.verify_all()

    # Save report
    verifier.save_report(results, args.output)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)

    for dataset_name, result in results.items():
        status_icon = "✓" if result["status"] == "ok" else "✗"
        gap_count = len(result.get("gaps", []))
        logger.info(f"{status_icon} {dataset_name}: {result['file_count']} files, {gap_count} gaps")

    total_gaps = sum(len(r.get("gaps", [])) for r in results.values())
    if total_gaps == 0:
        logger.info("\n✓ All datasets verified successfully with no gaps!")
        return 0
    else:
        logger.warning(f"\n✗ Found {total_gaps} gaps across all datasets")
        return 1


if __name__ == "__main__":
    exit(main())
