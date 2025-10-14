#!/usr/bin/env python3
"""
Convert Web Service API CSV files to ZIP CSV format.

Web Service API format:
  - Headers: 0,1,2,3,4,5,...
  - Filename: 60d_dam_gen_resources_20241129_20241201.csv
  - Date format: 2024-12-01

ZIP CSV format:
  - Headers: "Delivery Date","Hour Ending","QSE","DME",...
  - Filename: 60d_DAM_Gen_Resource_Data-01-DEC-24.csv
  - Date format: 12/01/2024
"""

import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Column headers in correct order (from ZIP CSV files)
ZIP_HEADERS = [
    "Delivery Date",
    "Hour Ending",
    "QSE",
    "DME",
    "Resource Name",
    "Resource Type",
    "QSE submitted Curve-MW1",
    "QSE submitted Curve-Price1",
    "QSE submitted Curve-MW2",
    "QSE submitted Curve-Price2",
    "QSE submitted Curve-MW3",
    "QSE submitted Curve-Price3",
    "QSE submitted Curve-MW4",
    "QSE submitted Curve-Price4",
    "QSE submitted Curve-MW5",
    "QSE submitted Curve-Price5",
    "QSE submitted Curve-MW6",
    "QSE submitted Curve-Price6",
    "QSE submitted Curve-MW7",
    "QSE submitted Curve-Price7",
    "QSE submitted Curve-MW8",
    "QSE submitted Curve-Price8",
    "QSE submitted Curve-MW9",
    "QSE submitted Curve-Price9",
    "QSE submitted Curve-MW10",
    "QSE submitted Curve-Price10",
    "Start Up Hot",
    "Start Up Inter",
    "Start Up Cold",
    "Min Gen Cost",
    "HSL",
    "LSL",
    "Resource Status",
    "Awarded Quantity",
    "Settlement Point Name",
    "Energy Settlement Point Price",
    "RegUp Awarded",
    "RegUp MCPC",
    "RegDown Awarded",
    "RegDown MCPC",
    "RRSPFR Awarded",
    "RRSFFR Awarded",
    "RRSUFR Awarded",
    "RRS MCPC",
    "ECRSSD Awarded",
    "ECRS MCPC",
    "NonSpin Awarded",
    "NonSpin MCPC",
]


def convert_date_format(date_str: str) -> str:
    """Convert 2024-12-01 to 12/01/2024"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%m/%d/%Y")


def get_zip_filename(date_str: str) -> str:
    """
    Convert date to ZIP filename format.
    2024-12-01 -> 60d_DAM_Gen_Resource_Data-01-DEC-24.csv
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return f"60d_DAM_Gen_Resource_Data-{dt.strftime('%d-%b-%y').upper()}.csv"


def convert_api_file_to_zip_format(
    api_file: Path, output_dir: Path, dry_run: bool = False
) -> Tuple[bool, str]:
    """
    Convert a single API CSV file to ZIP format.

    Returns: (success, message)
    """
    try:
        # Read API file (numeric headers)
        df = pd.read_csv(api_file)

        # Check if it has the numeric header format
        if list(df.columns)[0] != '0':
            return False, f"Already has proper headers, skipping: {api_file.name}"

        # Verify column count matches
        if len(df.columns) != len(ZIP_HEADERS):
            return False, f"Column count mismatch: {len(df.columns)} vs {len(ZIP_HEADERS)}"

        # Replace numeric headers with proper names
        df.columns = ZIP_HEADERS

        # Convert date format: 2024-12-01 -> 12/01/2024
        df["Delivery Date"] = df["Delivery Date"].apply(convert_date_format)

        # Group by delivery date and save one file per date
        grouped = df.groupby("Delivery Date")

        converted_files = []
        for delivery_date_str, group_df in grouped:
            # Get the date in YYYY-MM-DD format for filename generation
            dt = datetime.strptime(delivery_date_str, "%m/%d/%Y")
            date_for_filename = dt.strftime("%Y-%m-%d")

            output_filename = get_zip_filename(date_for_filename)
            output_path = output_dir / output_filename

            if not dry_run:
                # Write with proper quoting (matching ZIP format)
                group_df.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL
                converted_files.append(output_filename)
            else:
                converted_files.append(f"[DRY RUN] {output_filename}")

        return True, f"Converted {api_file.name} -> {len(converted_files)} files: {', '.join(converted_files[:3])}{'...' if len(converted_files) > 3 else ''}"

    except Exception as e:
        return False, f"Error converting {api_file.name}: {e}"


def convert_all_api_files(
    api_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    backup: bool = True
) -> None:
    """
    Convert all Web Service API files to ZIP format.
    """
    print(f"Converting API files from: {api_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Dry run: {dry_run}")
    print(f"Backup originals: {backup}")
    print("=" * 80)

    # Find all API files
    api_files = sorted(api_dir.glob("60d_dam_gen_resources_*.csv"))

    if not api_files:
        print("❌ No API files found!")
        return

    print(f"Found {len(api_files)} API files to convert\n")

    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create backup directory if requested
        if backup:
            backup_dir = api_dir / "api_originals_backup"
            backup_dir.mkdir(exist_ok=True)
            print(f"Backup directory: {backup_dir}\n")

    # Convert each file
    success_count = 0
    skip_count = 0
    error_count = 0

    for api_file in api_files:
        success, message = convert_api_file_to_zip_format(api_file, output_dir, dry_run)

        if success:
            print(f"✅ {message}")
            success_count += 1

            # Move original to backup
            if backup and not dry_run:
                backup_path = backup_dir / api_file.name
                shutil.move(str(api_file), str(backup_path))
        elif "skipping" in message.lower():
            print(f"⏭️  {message}")
            skip_count += 1
        else:
            print(f"❌ {message}")
            error_count += 1

    print("\n" + "=" * 80)
    print(f"Conversion complete!")
    print(f"  ✅ Converted: {success_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"  ❌ Errors: {error_count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert API CSV files to ZIP format")
    parser.add_argument(
        "--api-dir",
        type=Path,
        default=Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/Gen_Resources"),
        help="Directory containing API CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/60-Day_DAM_Disclosure_Reports/csv"),
        help="Output directory for converted files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't backup original API files"
    )

    args = parser.parse_args()

    convert_all_api_files(
        api_dir=args.api_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
