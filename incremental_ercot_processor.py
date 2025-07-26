#!/usr/bin/env python3
"""Incremental ERCOT data processor with centralized extraction management."""

import os
import sys
import zipfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Set
import pandas as pd
import logging
from dataclasses import dataclass, asdict
import hashlib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import init_db, get_db
from database.models_v2 import LMP, AncillaryServices
from sqlalchemy import text, func
from process_ercot_historical_batch import (
    infer_location_type, store_lmp_batch, store_ancillary_batch
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Base directories
ERCOT_DATA_DIR = "/Users/enrico/data/ERCOT_data"
WORK_DIR = Path("./data/ercot_processing")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Processing state file
PROCESSING_STATE_FILE = WORK_DIR / "processing_state.json"


@dataclass
class ProcessedFile:
    """Track processed file information."""
    zip_path: str
    zip_hash: str
    csv_path: str
    processed_date: str
    record_count: int
    date_range: Optional[Dict[str, str]] = None
    

class ProcessingState:
    """Manage incremental processing state."""
    
    def __init__(self):
        self.processed_files: Dict[str, ProcessedFile] = {}
        self.load_state()
        
    def load_state(self):
        """Load processing state from disk."""
        if PROCESSING_STATE_FILE.exists():
            try:
                with open(PROCESSING_STATE_FILE, 'r') as f:
                    data = json.load(f)
                    for zip_path, file_data in data.items():
                        self.processed_files[zip_path] = ProcessedFile(**file_data)
                logger.info(f"Loaded state for {len(self.processed_files)} processed files")
            except Exception as e:
                logger.error(f"Error loading state: {e}")
                self.processed_files = {}
        else:
            logger.info("No existing state file found")
            
    def save_state(self):
        """Save processing state to disk."""
        data = {k: asdict(v) for k, v in self.processed_files.items()}
        with open(PROCESSING_STATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
            
    def is_processed(self, zip_path: Path) -> bool:
        """Check if file has been processed."""
        zip_str = str(zip_path)
        if zip_str not in self.processed_files:
            return False
            
        # Check if file has changed
        current_hash = self.get_file_hash(zip_path)
        stored_hash = self.processed_files[zip_str].zip_hash
        
        if current_hash != stored_hash:
            logger.info(f"File {zip_path.name} has changed, will reprocess")
            return False
            
        return True
        
    def mark_processed(self, zip_path: Path, csv_path: Path, record_count: int, 
                       date_range: Optional[Dict[str, str]] = None):
        """Mark file as processed."""
        self.processed_files[str(zip_path)] = ProcessedFile(
            zip_path=str(zip_path),
            zip_hash=self.get_file_hash(zip_path),
            csv_path=str(csv_path),
            processed_date=datetime.now().isoformat(),
            record_count=record_count,
            date_range=date_range
        )
        self.save_state()
        
    @staticmethod
    def get_file_hash(filepath: Path) -> str:
        """Get MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class IncrementalProcessor:
    """Process ERCOT data incrementally."""
    
    def __init__(self):
        self.state = ProcessingState()
        init_db()
        
    def extract_zip(self, zip_path: Path, extract_dir: Path) -> Optional[Path]:
        """Extract zip file and return path to CSV."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # List all files in zip
                namelist = zf.namelist()
                
                # Find CSV files
                csv_files = [f for f in namelist if f.endswith('.csv')]
                
                if not csv_files:
                    logger.warning(f"No CSV files found in {zip_path.name}")
                    return None
                    
                # Extract just the CSV file(s)
                for csv_file in csv_files:
                    zf.extract(csv_file, extract_dir)
                    
                # Return path to first CSV
                return extract_dir / csv_files[0]
                
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return None
            
    def process_dam_lmp_file(self, zip_path: Path) -> int:
        """Process a single DAM LMP file."""
        # Check if already processed
        if self.state.is_processed(zip_path):
            logger.info(f"Skipping already processed: {zip_path.name}")
            return 0
            
        # Extract to temporary directory
        extract_dir = WORK_DIR / "temp" / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = self.extract_zip(zip_path, extract_dir)
        if not csv_path:
            return 0
            
        try:
            # Read and process CSV
            df = pd.read_csv(csv_path)
            
            # Parse date and hour
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
            df['hour'] = df['HourEnding'].str.extract(r'(\d+)').astype(int)
            df['interval_start'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
            df['interval_end'] = df['interval_start'] + timedelta(hours=1)
            
            # Rename and format
            df = df.rename(columns={
                'BusName': 'location',
                'LMP': 'lmp'
            })
            
            # Add metadata
            df['iso'] = 'ERCOT'
            df['market'] = 'DAM'
            df['location_type'] = df['location'].apply(infer_location_type)
            df['energy'] = None
            df['congestion'] = None
            df['loss'] = None
            
            # Prepare for storage
            lmp_df = df[['interval_start', 'interval_end', 'iso', 'location', 
                        'location_type', 'market', 'lmp', 'energy', 'congestion', 'loss']]
            
            # Store in database
            records_stored = store_lmp_batch(lmp_df)
            
            # Get date range
            date_range = {
                "start": df['interval_start'].min().isoformat(),
                "end": df['interval_start'].max().isoformat()
            }
            
            # Mark as processed
            self.state.mark_processed(zip_path, csv_path, records_stored, date_range)
            
            logger.info(f"Processed {zip_path.name}: {records_stored} records stored")
            
            # Clean up temporary files
            csv_path.unlink()
            
            return records_stored
            
        except Exception as e:
            logger.error(f"Error processing {zip_path}: {e}")
            return 0
            
    def process_rt_spp_file(self, zip_path: Path) -> int:
        """Process a single RT SPP file."""
        if self.state.is_processed(zip_path):
            logger.info(f"Skipping already processed: {zip_path.name}")
            return 0
            
        extract_dir = WORK_DIR / "temp" / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = self.extract_zip(zip_path, extract_dir)
        if not csv_path:
            return 0
            
        try:
            df = pd.read_csv(csv_path)
            
            # Parse timestamp
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
            df['minutes'] = (df['DeliveryHour'] - 1) * 60 + (df['DeliveryInterval'] - 1) * 15
            df['interval_start'] = df['DeliveryDate'] + pd.to_timedelta(df['minutes'], unit='m')
            df['interval_end'] = df['interval_start'] + timedelta(minutes=15)
            
            # Rename columns
            df = df.rename(columns={
                'SettlementPointName': 'location',
                'SettlementPointPrice': 'lmp'
            })
            
            # Add metadata
            df['iso'] = 'ERCOT'
            df['market'] = 'RT15M'
            df['location_type'] = df.apply(
                lambda row: row.get('SettlementPointType', '').lower() 
                if pd.notna(row.get('SettlementPointType')) 
                else infer_location_type(row['location']), 
                axis=1
            )
            df['energy'] = None
            df['congestion'] = None
            df['loss'] = None
            
            lmp_df = df[['interval_start', 'interval_end', 'iso', 'location',
                        'location_type', 'market', 'lmp', 'energy', 'congestion', 'loss']]
            
            records_stored = store_lmp_batch(lmp_df)
            
            date_range = {
                "start": df['interval_start'].min().isoformat(),
                "end": df['interval_start'].max().isoformat()
            }
            
            self.state.mark_processed(zip_path, csv_path, records_stored, date_range)
            logger.info(f"Processed {zip_path.name}: {records_stored} records stored")
            
            csv_path.unlink()
            return records_stored
            
        except Exception as e:
            logger.error(f"Error processing {zip_path}: {e}")
            return 0
            
    def process_ancillary_file(self, zip_path: Path) -> int:
        """Process a single ancillary services file."""
        if self.state.is_processed(zip_path):
            logger.info(f"Skipping already processed: {zip_path.name}")
            return 0
            
        extract_dir = WORK_DIR / "temp" / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = self.extract_zip(zip_path, extract_dir)
        if not csv_path:
            return 0
            
        try:
            df = pd.read_csv(csv_path)
            
            # Parse date and hour
            df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%m/%d/%Y')
            df['hour'] = df['HourEnding'].str.extract(r'(\d+)').astype(int)
            df['interval_start'] = df['DeliveryDate'] + pd.to_timedelta(df['hour'] - 1, unit='h')
            df['interval_end'] = df['interval_start'] + timedelta(hours=1)
            
            # Pivot and process
            df_pivot = df.pivot_table(
                index=['interval_start', 'interval_end'],
                columns='AncillaryType',
                values='MCPC',
                aggfunc='first'
            ).reset_index()
            
            # Convert to long format
            records = []
            service_mapping = {
                'REGUP': 'REGUP',
                'REGDN': 'REGDOWN',
                'RRS': 'RRS',
                'NSPIN': 'NON_SPIN',
                'ECRS': 'ECRS'
            }
            
            for service, standard_name in service_mapping.items():
                if service in df_pivot.columns:
                    service_df = df_pivot[['interval_start', 'interval_end', service]].copy()
                    service_df = service_df.dropna(subset=[service])
                    service_df['product'] = standard_name
                    service_df['clearing_price'] = service_df[service]
                    service_df['iso'] = 'ERCOT'
                    service_df['region'] = 'ERCOT'
                    service_df['market'] = 'DAM'
                    service_df['clearing_quantity'] = None
                    service_df['requirement'] = None
                    records.append(service_df[['interval_start', 'interval_end', 'iso', 
                                              'region', 'market', 'product', 
                                              'clearing_price', 'clearing_quantity', 'requirement']])
                    
            if records:
                as_df = pd.concat(records, ignore_index=True)
                records_stored = store_ancillary_batch(as_df)
                
                date_range = {
                    "start": as_df['interval_start'].min().isoformat(),
                    "end": as_df['interval_start'].max().isoformat()
                }
                
                self.state.mark_processed(zip_path, csv_path, records_stored, date_range)
                logger.info(f"Processed {zip_path.name}: {records_stored} records stored")
                
                csv_path.unlink()
                return records_stored
            
            return 0
            
        except Exception as e:
            logger.error(f"Error processing {zip_path}: {e}")
            return 0
            
    def process_directory(self, directory: str, pattern: str, processor_method: str) -> Dict[str, int]:
        """Process new files in a directory."""
        source_dir = Path(ERCOT_DATA_DIR) / directory
        if not source_dir.exists():
            logger.error(f"Directory not found: {source_dir}")
            return {"files": 0, "records": 0}
            
        # Get all zip files
        zip_files = sorted(source_dir.glob(pattern))
        
        # Filter to only unprocessed files
        unprocessed = [f for f in zip_files if not self.state.is_processed(f)]
        
        logger.info(f"Found {len(unprocessed)} new files (out of {len(zip_files)} total) in {directory}")
        
        if not unprocessed:
            return {"files": 0, "records": 0}
            
        total_records = 0
        files_processed = 0
        
        # Process files one by one
        for i, zip_file in enumerate(unprocessed):
            logger.info(f"\nProcessing file {i+1}/{len(unprocessed)}: {zip_file.name}")
            
            # Call the appropriate processor method
            if processor_method == "dam_lmp":
                records = self.process_dam_lmp_file(zip_file)
            elif processor_method == "rt_spp":
                records = self.process_rt_spp_file(zip_file)
            elif processor_method == "ancillary":
                records = self.process_ancillary_file(zip_file)
            else:
                logger.error(f"Unknown processor method: {processor_method}")
                continue
                
            if records > 0:
                total_records += records
                files_processed += 1
                
        return {"files": files_processed, "records": total_records}
        
    def get_processing_summary(self) -> Dict[str, any]:
        """Get summary of processing state."""
        summary = {
            "total_files_processed": len(self.state.processed_files),
            "total_records": sum(f.record_count for f in self.state.processed_files.values()),
            "date_ranges": {},
            "last_processed": None
        }
        
        # Group by directory
        by_directory = {}
        for file_path, info in self.state.processed_files.items():
            # Extract directory from path
            parts = Path(file_path).parts
            for i, part in enumerate(parts):
                if part == "ERCOT_data" and i + 1 < len(parts):
                    directory = parts[i + 1]
                    if directory not in by_directory:
                        by_directory[directory] = {
                            "count": 0,
                            "records": 0,
                            "earliest": None,
                            "latest": None
                        }
                    
                    by_directory[directory]["count"] += 1
                    by_directory[directory]["records"] += info.record_count
                    
                    if info.date_range:
                        start = info.date_range.get("start")
                        end = info.date_range.get("end")
                        
                        if start and (not by_directory[directory]["earliest"] or 
                                     start < by_directory[directory]["earliest"]):
                            by_directory[directory]["earliest"] = start
                            
                        if end and (not by_directory[directory]["latest"] or 
                                   end > by_directory[directory]["latest"]):
                            by_directory[directory]["latest"] = end
                    
                    break
                    
        summary["by_directory"] = by_directory
        
        # Get last processed time
        if self.state.processed_files:
            last_times = [f.processed_date for f in self.state.processed_files.values()]
            summary["last_processed"] = max(last_times)
            
        return summary


def main():
    """Main function."""
    processor = IncrementalProcessor()
    
    # Show current state
    summary = processor.get_processing_summary()
    logger.info("\n📊 Current Processing State:")
    logger.info(f"Total files processed: {summary['total_files_processed']}")
    logger.info(f"Total records: {summary['total_records']:,}")
    
    if summary["by_directory"]:
        logger.info("\nBy directory:")
        for directory, info in summary["by_directory"].items():
            logger.info(f"  {directory}:")
            logger.info(f"    Files: {info['count']}")
            logger.info(f"    Records: {info['records']:,}")
            if info['earliest']:
                logger.info(f"    Date range: {info['earliest'][:10]} to {info['latest'][:10]}")
    
    # Process each data type
    results = {}
    
    logger.info("\n📊 Processing DAM Hourly LMPs...")
    results["DAM LMP"] = processor.process_directory(
        "DAM_Hourly_LMPs",
        "*DAMHRLMPNP4183_csv.zip",
        "dam_lmp"
    )
    
    logger.info("\n📊 Processing RT Settlement Point Prices...")
    results["RT SPP"] = processor.process_directory(
        "Settlement_Point_Prices_at_Resource_Nodes,_Hubs_and_Load_Zones",
        "*SPPHLZNP6905*csv.zip",
        "rt_spp"
    )
    
    logger.info("\n📊 Processing DAM Ancillary Services...")
    results["Ancillary"] = processor.process_directory(
        "DAM_Clearing_Prices_for_Capacity",
        "*DAMCPCNP4188_csv.zip",
        "ancillary"
    )
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 PROCESSING COMPLETE")
    logger.info("=" * 60)
    
    total_files = 0
    total_records = 0
    for data_type, info in results.items():
        logger.info(f"{data_type}: {info['files']} new files, {info['records']:,} records")
        total_files += info['files']
        total_records += info['records']
        
    logger.info(f"\nTotal: {total_files} files, {total_records:,} records")
    
    # Show final state
    final_summary = processor.get_processing_summary()
    logger.info(f"\n📊 Final State:")
    logger.info(f"Total files in state: {final_summary['total_files_processed']}")
    logger.info(f"Total records in state: {final_summary['total_records']:,}")


if __name__ == "__main__":
    main()