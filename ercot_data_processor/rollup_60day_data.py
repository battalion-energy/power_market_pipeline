#!/usr/bin/env python3
"""
Roll up 60-day ERCOT data files (DAM Disclosure and COP All Updates) into annual CSV files.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import re
import logging
from typing import Dict, List, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base paths
BASE_DIR = Path("/Users/enrico/data/ERCOT_data")

class SixtyDayRollup:
    """Handler for rolling up 60-day ERCOT data files."""
    
    def __init__(self, base_dir: Path = BASE_DIR):
        self.base_dir = base_dir
        self.rollup_dir = base_dir / "rollup_60day"
        self.rollup_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from filename - handles multiple formats."""
        
        # Format 1: DD-MMM-YY (e.g., '15-AUG-25' in newer files)
        match = re.search(r'(\d{2})-([A-Z]{3})-(\d{2})', filename)
        if match:
            day = int(match.group(1))
            month_str = match.group(2)
            year = int(match.group(3))
            
            # Convert 2-digit year to 4-digit
            year = 2000 + year if year < 50 else 1900 + year
            
            # Month mapping
            months = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month = months.get(month_str, 1)
            
            try:
                return datetime(year, month, day)
            except:
                return None
        
        # Format 2: MMDDYYYY (e.g., 'CompleteCOP_01012022.csv')
        match = re.search(r'CompleteCOP_(\d{2})(\d{2})(\d{4})', filename)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            
            try:
                return datetime(year, month, day)
            except:
                return None
        
        return None
    
    def rollup_cop_all_updates(self):
        """Roll up 60-day COP All Updates files."""
        logger.info("Rolling up 60-Day COP All Updates...")
        
        cop_dir = self.base_dir / "60-Day_COP_All_Updates" / "csv"
        if not cop_dir.exists():
            cop_dir = self.base_dir / "60-Day_COP_All_Updates"
        
        if not cop_dir.exists():
            logger.warning(f"COP directory not found: {cop_dir}")
            return
        
        # Find all COP files
        cop_files = list(cop_dir.glob("60d_COP_All_Updates-*.csv"))
        logger.info(f"Found {len(cop_files)} COP All Updates files")
        
        if not cop_files:
            return
        
        # Group by year
        files_by_year = {}
        for file_path in cop_files:
            file_date = self.parse_date_from_filename(file_path.name)
            if file_date:
                year = file_date.year
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append((file_date, file_path))
        
        # Process each year
        output_dir = self.rollup_dir / "COP_All_Updates"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for year, file_list in sorted(files_by_year.items()):
            # Sort by date
            file_list.sort(key=lambda x: x[0])
            files = [fp for _, fp in file_list]
            
            logger.info(f"  Processing COP {year}: {len(files)} files")
            
            # Read and combine files
            dfs = []
            for file_path in files:
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    df['source_file'] = file_path.name
                    df['file_date'] = self.parse_date_from_filename(file_path.name)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file_path.name}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Sort by file_date to maintain chronological order
                combined_df = combined_df.sort_values('file_date')
                
                # Save to CSV
                output_file = output_dir / f"COP_All_Updates_{year}.csv"
                combined_df.to_csv(output_file, index=False)
                logger.info(f"    Saved {len(combined_df):,} rows to {output_file.name}")
                
                # Also save parquet
                parquet_file = output_dir / f"COP_All_Updates_{year}.parquet"
                combined_df.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
    
    def rollup_dam_disclosures(self):
        """Roll up all 60-day DAM Disclosure files."""
        logger.info("Rolling up 60-Day DAM Disclosures...")
        
        dam_dir = self.base_dir / "60-Day_DAM_Disclosure_Reports" / "csv"
        if not dam_dir.exists():
            dam_dir = self.base_dir / "60-Day_DAM_Disclosure_Reports"
        
        if not dam_dir.exists():
            logger.warning(f"DAM directory not found: {dam_dir}")
            return
        
        # Define DAM file patterns
        dam_patterns = {
            'EnergyBidAwards': r'60d_DAM_EnergyBidAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'EnergyBids': r'60d_DAM_EnergyBids-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'EnergyOnlyOfferAwards': r'60d_DAM_EnergyOnlyOfferAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'EnergyOnlyOffers': r'60d_DAM_EnergyOnlyOffers-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'Gen_Resource_Data': r'60d_DAM_Gen_Resource_Data-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'Generation_Resource_ASOffers': r'60d_DAM_Generation_Resource_ASOffers-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'Load_Resource_ASOffers': r'60d_DAM_Load_Resource_ASOffers-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'Load_Resource_Data': r'60d_DAM_Load_Resource_Data-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'PTP_Obligation_Option': r'60d_DAM_PTP_Obligation_Option-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'PTPObligationBidAwards': r'60d_DAM_PTPObligationBidAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'QSE_Self_Arranged_AS': r'60d_DAM_QSE_Self_Arranged_AS-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'PTP_Obligation_OptionAwards': r'60d_DAM_PTP_Obligation_OptionAwards-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'PTPObligationBids': r'60d_DAM_PTPObligationBids-\d{2}-[A-Z]{3}-\d{2}\.csv',
        }
        
        for file_type, pattern in dam_patterns.items():
            logger.info(f"  Processing {file_type}...")
            
            # Find matching files
            matching_files = []
            for file_path in dam_dir.glob("*.csv"):
                if re.match(pattern, file_path.name):
                    matching_files.append(file_path)
            
            if not matching_files:
                logger.info(f"    No files found for {file_type}")
                continue
            
            # Group by year
            files_by_year = {}
            for file_path in matching_files:
                file_date = self.parse_date_from_filename(file_path.name)
                if file_date:
                    year = file_date.year
                    if year not in files_by_year:
                        files_by_year[year] = []
                    files_by_year[year].append((file_date, file_path))
            
            # Process each year
            output_dir = self.rollup_dir / "DAM_Disclosures" / file_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for year, file_list in sorted(files_by_year.items()):
                # Sort by date
                file_list.sort(key=lambda x: x[0])
                files = [fp for _, fp in file_list]
                
                logger.info(f"    Processing {year}: {len(files)} files")
                
                # Read and combine files
                dfs = []
                for file_path in files:
                    try:
                        # Read with appropriate dtypes
                        df = pd.read_csv(file_path, low_memory=False)
                        df['source_file'] = file_path.name
                        df['file_date'] = self.parse_date_from_filename(file_path.name)
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file_path.name}: {e}")
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    
                    # Sort by file_date
                    combined_df = combined_df.sort_values('file_date')
                    
                    # Save to CSV
                    output_file = output_dir / f"{file_type}_{year}.csv"
                    combined_df.to_csv(output_file, index=False)
                    logger.info(f"      Saved {len(combined_df):,} rows to {output_file.name}")
                    
                    # Also save parquet
                    parquet_file = output_dir / f"{file_type}_{year}.parquet"
                    combined_df.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
    
    def generate_summary(self):
        """Generate a summary of all rollup files created."""
        summary_file = self.rollup_dir / "rollup_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("60-Day Data Rollup Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.rollup_dir}\n\n")
            
            # List all created files
            for subdir in sorted(self.rollup_dir.iterdir()):
                if subdir.is_dir():
                    f.write(f"\n{subdir.name}:\n")
                    f.write("-" * 40 + "\n")
                    
                    csv_files = list(subdir.rglob("*.csv"))
                    for csv_file in sorted(csv_files):
                        size_mb = csv_file.stat().st_size / (1024 * 1024)
                        f.write(f"  {csv_file.name}: {size_mb:.1f} MB\n")
        
        logger.info(f"Summary saved to {summary_file}")
    
    def rollup_sasm_disclosures(self):
        """Roll up 60-day SASM Disclosure files."""
        logger.info("Rolling up 60-Day SASM Disclosures...")
        
        sasm_dir = self.base_dir / "60-Day_SASM_Disclosure_Reports" / "csv"
        if not sasm_dir.exists():
            sasm_dir = self.base_dir / "60-Day_SASM_Disclosure_Reports"
        
        if not sasm_dir.exists():
            logger.warning(f"SASM directory not found: {sasm_dir}")
            return
        
        # Define SASM file patterns
        sasm_patterns = {
            'Generation_Resource_AS_Offers': r'60d_SASM_Generation_Resource_AS_Offers-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'Generation_Resource_AS_Offer_Awards': r'60d_SASM_Generation_Resource_AS_Offer_Awards-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'Load_Resource_AS_Offers': r'60d_SASM_Load_Resource_AS_Offers-\d{2}-[A-Z]{3}-\d{2}\.csv',
            'Load_Resource_AS_Offer_Awards': r'60d_SASM_Load_Resource_AS_Offer_Awards-\d{2}-[A-Z]{3}-\d{2}\.csv',
        }
        
        for file_type, pattern in sasm_patterns.items():
            logger.info(f"  Processing SASM {file_type}...")
            
            # Find matching files
            matching_files = []
            for file_path in sasm_dir.glob("*.csv"):
                if re.match(pattern, file_path.name):
                    matching_files.append(file_path)
            
            if not matching_files:
                logger.info(f"    No files found for {file_type}")
                continue
            
            # Group by year
            files_by_year = {}
            for file_path in matching_files:
                file_date = self.parse_date_from_filename(file_path.name)
                if file_date:
                    year = file_date.year
                    if year not in files_by_year:
                        files_by_year[year] = []
                    files_by_year[year].append((file_date, file_path))
            
            # Process each year
            output_dir = self.rollup_dir / "SASM_Disclosures" / file_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for year, file_list in sorted(files_by_year.items()):
                # Sort by date
                file_list.sort(key=lambda x: x[0])
                files = [fp for _, fp in file_list]
                
                logger.info(f"    Processing {year}: {len(files)} files")
                
                # Read and combine files
                dfs = []
                for file_path in files:
                    try:
                        df = pd.read_csv(file_path, low_memory=False)
                        df['source_file'] = file_path.name
                        df['file_date'] = self.parse_date_from_filename(file_path.name)
                        dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading {file_path.name}: {e}")
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    
                    # Sort by file_date
                    combined_df = combined_df.sort_values('file_date')
                    
                    # Save to CSV
                    output_file = output_dir / f"SASM_{file_type}_{year}.csv"
                    combined_df.to_csv(output_file, index=False)
                    logger.info(f"      Saved {len(combined_df):,} rows to {output_file.name}")
                    
                    # Also save parquet
                    parquet_file = output_dir / f"SASM_{file_type}_{year}.parquet"
                    combined_df.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
    
    def rollup_cop_adjustment_snapshots(self):
        """Roll up COP Adjustment Period Snapshot files (handles format change in 2022)."""
        logger.info("Rolling up COP Adjustment Period Snapshots...")
        
        cop_snapshot_dir = self.base_dir / "60-Day_COP_Adjustment_Period_Snapshot" / "csv"
        if not cop_snapshot_dir.exists():
            cop_snapshot_dir = self.base_dir / "60-Day_COP_Adjustment_Period_Snapshot"
        
        if not cop_snapshot_dir.exists():
            logger.warning(f"COP Snapshot directory not found: {cop_snapshot_dir}")
            return
        
        # Find all COP snapshot files (both formats)
        # Format 1: CompleteCOP_MMDDYYYY.csv (before Dec 2022)
        old_format_files = list(cop_snapshot_dir.glob("CompleteCOP_*.csv"))
        
        # Format 2: 60d_COP_Adjustment_Period_Snapshot-DD-MMM-YY.csv (after Dec 2022)
        new_format_files = list(cop_snapshot_dir.glob("60d_COP_Adjustment_Period_Snapshot-*.csv"))
        
        all_files = old_format_files + new_format_files
        logger.info(f"Found {len(old_format_files)} CompleteCOP files and {len(new_format_files)} 60d_COP files")
        logger.info(f"Total: {len(all_files)} COP Adjustment Snapshot files")
        
        if not all_files:
            return
        
        # Group by year
        files_by_year = {}
        for file_path in all_files:
            file_date = self.parse_date_from_filename(file_path.name)
            if file_date:
                year = file_date.year
                if year not in files_by_year:
                    files_by_year[year] = []
                files_by_year[year].append((file_date, file_path))
        
        # Process each year
        output_dir = self.rollup_dir / "COP_Adjustment_Snapshots"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for year, file_list in sorted(files_by_year.items()):
            # Sort by date
            file_list.sort(key=lambda x: x[0])
            files = [fp for _, fp in file_list]
            
            logger.info(f"  Processing COP Snapshots {year}: {len(files)} files")
            
            # Count format types for this year
            old_format_count = sum(1 for f in files if 'CompleteCOP' in f.name)
            new_format_count = sum(1 for f in files if '60d_COP' in f.name)
            if old_format_count > 0 and new_format_count > 0:
                logger.info(f"    Mixed formats: {old_format_count} CompleteCOP, {new_format_count} 60d_COP")
            
            # Read and combine files
            dfs = []
            for file_path in files:
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    df['source_file'] = file_path.name
                    df['file_date'] = self.parse_date_from_filename(file_path.name)
                    df['file_format'] = 'CompleteCOP' if 'CompleteCOP' in file_path.name else '60d_COP'
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error reading {file_path.name}: {e}")
            
            if dfs:
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Sort by file_date to maintain chronological order
                combined_df = combined_df.sort_values('file_date')
                
                # Save to CSV
                output_file = output_dir / f"COP_Adjustment_Snapshot_{year}.csv"
                combined_df.to_csv(output_file, index=False)
                logger.info(f"    Saved {len(combined_df):,} rows to {output_file.name}")
                
                # Also save parquet
                parquet_file = output_dir / f"COP_Adjustment_Snapshot_{year}.parquet"
                combined_df.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
    
    def run(self):
        """Run the complete rollup process."""
        logger.info("Starting 60-Day Data Rollup Process")
        logger.info(f"Base directory: {self.base_dir}")
        logger.info(f"Output directory: {self.rollup_dir}")
        
        # Process COP Adjustment Period Snapshots
        self.rollup_cop_adjustment_snapshots()
        
        # Process COP All Updates
        self.rollup_cop_all_updates()
        
        # Process DAM Disclosures
        self.rollup_dam_disclosures()
        
        # Process SASM Disclosures
        self.rollup_sasm_disclosures()
        
        # Generate summary
        self.generate_summary()
        
        logger.info("Rollup process complete!")

def main():
    """Main entry point."""
    rollup = SixtyDayRollup()
    rollup.run()

if __name__ == "__main__":
    main()