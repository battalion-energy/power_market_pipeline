#!/usr/bin/env python3
"""
BESS Revenue Leaderboard - Parquet Migration
Phase 1: Schema Registry and BESS Resource Identification
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, date
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Base paths
BASE_DATA_DIR = Path("/Users/enrico/data/ERCOT_data")
ROLLUP_DIR = BASE_DATA_DIR / "rollup_files"
OUTPUT_DIR = BASE_DATA_DIR / "bess_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

@dataclass
class BESSResource:
    """BESS Resource metadata"""
    resource_name: str
    settlement_point: str
    capacity_mw: float
    duration_hours: float
    first_active_date: date
    last_active_date: Optional[date]
    resource_type: str = "PWRSTR"
    
class SchemaRegistry:
    """Manages and validates Parquet schemas"""
    
    def __init__(self, rollup_dir: Path):
        self.rollup_dir = rollup_dir
        self.schemas = {}
        self.load_all_schemas()
    
    def load_all_schemas(self):
        """Load all schema.json files from rollup directories"""
        schema_files = {
            'RT_prices': self.rollup_dir / 'RT_prices' / 'schema.json',
            'DA_prices': self.rollup_dir / 'DA_prices' / 'schema.json',
            'AS_prices': self.rollup_dir / 'AS_prices' / 'schema.json',
            'DAM_Gen_Resources': self.rollup_dir / 'DAM_Gen_Resources' / 'schema.json',
            'SCED_Gen_Resources': self.rollup_dir / 'SCED_Gen_Resources' / 'schema.json',
            'COP_Snapshots': self.rollup_dir / 'COP_Snapshots' / 'schema.json',
            'COP_60day': self.rollup_dir / 'COP_60day' / 'schema.json',
        }
        
        for name, path in schema_files.items():
            if path.exists():
                with open(path, 'r') as f:
                    self.schemas[name] = json.load(f)
                    logger.info(f"Loaded schema for {name}: {len(self.schemas[name].get('columns', {}))} columns")
            else:
                logger.warning(f"Schema file not found: {path}")
    
    def validate_parquet_file(self, file_path: Path, expected_schema_name: str) -> bool:
        """Validate a Parquet file against expected schema"""
        try:
            # Read Parquet metadata
            parquet_file = pq.ParquetFile(file_path)
            parquet_schema = parquet_file.schema_arrow
            
            # Get expected schema
            if expected_schema_name not in self.schemas:
                logger.error(f"No schema found for {expected_schema_name}")
                return False
            
            expected = self.schemas[expected_schema_name].get('columns', {})
            
            # Check columns
            parquet_columns = set(parquet_schema.names)
            expected_columns = set(expected.keys())
            
            missing = expected_columns - parquet_columns
            extra = parquet_columns - expected_columns
            
            if missing:
                logger.warning(f"Missing columns in {file_path.name}: {missing}")
            if extra:
                logger.info(f"Extra columns in {file_path.name}: {extra}")
            
            return len(missing) == 0  # File is valid if no required columns are missing
            
        except Exception as e:
            logger.error(f"Error validating {file_path}: {e}")
            return False
    
    def get_column_availability(self) -> pd.DataFrame:
        """Create matrix of column availability by year"""
        availability = []
        
        for dataset in ['DAM_Gen_Resources', 'SCED_Gen_Resources', 'COP_Snapshots']:
            dataset_dir = self.rollup_dir / dataset
            if not dataset_dir.exists():
                continue
                
            for year in range(2014, 2026):
                parquet_file = dataset_dir / f"{year}.parquet"
                if parquet_file.exists():
                    pf = pq.ParquetFile(parquet_file)
                    columns = pf.schema_arrow.names
                    
                    for col in columns:
                        availability.append({
                            'dataset': dataset,
                            'year': year,
                            'column': col,
                            'available': True
                        })
        
        if availability:
            df = pd.DataFrame(availability)
            pivot = df.pivot_table(
                index='column',
                columns=['dataset', 'year'],
                values='available',
                fill_value=False
            )
            return pivot
        return pd.DataFrame()

class BESSRegistry:
    """Extract and manage BESS resource registry from Parquet files"""
    
    def __init__(self, rollup_dir: Path):
        self.rollup_dir = rollup_dir
        self.resources = {}
        
    def extract_bess_resources(self, years: List[int] = None) -> Dict[str, BESSResource]:
        """Extract all BESS resources from COP and DAM files"""
        if years is None:
            years = list(range(2019, 2025))  # BESS started appearing in 2019
        
        all_resources = {}
        
        for year in years:
            logger.info(f"Scanning year {year} for BESS resources...")
            
            # Try COP snapshots first (most comprehensive)
            cop_resources = self._extract_from_cop(year)
            for name, resource in cop_resources.items():
                if name not in all_resources:
                    all_resources[name] = resource
                else:
                    # Update last active date
                    all_resources[name].last_active_date = resource.last_active_date
            
            # Also check DAM files for additional info
            dam_resources = self._extract_from_dam(year)
            for name, resource in dam_resources.items():
                if name not in all_resources:
                    all_resources[name] = resource
        
        self.resources = all_resources
        logger.info(f"Found {len(all_resources)} unique BESS resources")
        return all_resources
    
    def _extract_from_cop(self, year: int) -> Dict[str, BESSResource]:
        """Extract BESS resources from COP snapshot files"""
        resources = {}
        
        # Try both COP directories
        for cop_dir in ['COP_Snapshots', 'COP_60day']:
            cop_file = self.rollup_dir / cop_dir / f"{year}.parquet"
            if not cop_file.exists():
                continue
            
            try:
                # Read Parquet file to check columns first
                pf = pq.ParquetFile(cop_file)
                available_columns = pf.schema_arrow.names
                
                # Map expected to actual column names
                column_mapping = {
                    'ResourceName': 'Resource Name',
                    'ResourceType': 'Resource Type', 
                    'MaxSOC': 'Maximum SOC',
                    'MinSOC': 'Minimum SOC',
                    'DeliveryDate': 'Delivery Date',
                    'HSL': 'High Sustained Limit'
                }
                
                # Select columns that exist
                columns_to_read = []
                for actual_col in available_columns:
                    if actual_col in ['ResourceName', 'ResourceType', 'MaxSOC', 'MinSOC', 'DeliveryDate', 'HSL', 'Status']:
                        columns_to_read.append(actual_col)
                
                if 'ResourceName' not in columns_to_read:
                    continue  # Can't process without resource names
                
                table = pq.read_table(cop_file, columns=columns_to_read)
                df = table.to_pandas()
                
                # Rename columns to expected names
                rename_map = {actual: expected for actual, expected in column_mapping.items() if actual in df.columns}
                df = df.rename(columns=rename_map)
                
                # Filter for BESS (PWRSTR type if column exists)
                if 'Resource Type' in df.columns:
                    bess_df = df[df['Resource Type'] == 'PWRSTR'].copy()
                else:
                    # Identify by name pattern
                    bess_df = df[df['Resource Name'].str.contains('BESS|BES[0-9]|ESS|BATTERY|BATCAVE|_STOR', na=False, case=False)].copy()
                
                if len(bess_df) == 0:
                    continue
                
                # Extract unique resources
                for resource_name in bess_df['Resource Name'].unique():
                    if pd.isna(resource_name):
                        continue
                    
                    resource_data = bess_df[bess_df['Resource Name'] == resource_name].iloc[0]
                    
                    # Extract capacity from HSL or SOC
                    capacity = 0
                    if 'High Sustained Limit' in resource_data:
                        capacity = float(resource_data['High Sustained Limit']) if not pd.isna(resource_data['High Sustained Limit']) else 0
                    elif 'Maximum SOC' in resource_data:
                        # Estimate from SOC (assume 4-hour battery)
                        max_soc = float(resource_data['Maximum SOC']) if not pd.isna(resource_data['Maximum SOC']) else 0
                        capacity = max_soc / 4.0
                    
                    # Get dates
                    dates = pd.to_datetime(bess_df[bess_df['Resource Name'] == resource_name]['Delivery Date'])
                    first_date = dates.min().date() if not dates.empty else date(year, 1, 1)
                    last_date = dates.max().date() if not dates.empty else date(year, 12, 31)
                    
                    # Extract settlement point from resource name
                    settlement_point = self._extract_settlement_point(resource_name)
                    
                    resources[resource_name] = BESSResource(
                        resource_name=resource_name,
                        settlement_point=settlement_point,
                        capacity_mw=capacity,
                        duration_hours=4.0,  # Default assumption
                        first_active_date=first_date,
                        last_active_date=last_date
                    )
                    
            except Exception as e:
                logger.error(f"Error processing COP {year}: {e}")
        
        return resources
    
    def _extract_from_dam(self, year: int) -> Dict[str, BESSResource]:
        """Extract BESS resources from DAM files"""
        resources = {}
        dam_file = self.rollup_dir / 'DAM_Gen_Resources' / f"{year}.parquet"
        
        if not dam_file.exists():
            return resources
        
        try:
            # Check actual column names
            pf = pq.ParquetFile(dam_file)
            available_columns = pf.schema_arrow.names
            
            # Map to actual column names
            column_map = {
                'ResourceName': 'Resource Name',
                'ResourceType': 'Resource Type',
                'SettlementPointName': 'Settlement Point',
                'DeliveryDate': 'Delivery Date'
            }
            
            # Select columns that exist
            columns_to_read = []
            for col in ['ResourceName', 'ResourceType', 'SettlementPointName', 'DeliveryDate']:
                if col in available_columns:
                    columns_to_read.append(col)
            
            if 'ResourceName' not in columns_to_read:
                return resources
                
            # Read with filter if ResourceType exists
            if 'ResourceType' in columns_to_read:
                table = pq.read_table(
                    dam_file,
                    columns=columns_to_read,
                    filters=[('ResourceType', '=', 'PWRSTR')]
                )
            else:
                table = pq.read_table(dam_file, columns=columns_to_read)
            
            df = table.to_pandas()
            
            # Rename columns
            rename_map = {actual: expected for actual, expected in column_map.items() if actual in df.columns}
            df = df.rename(columns=rename_map)
            
            for resource_name in df['Resource Name'].unique():
                if pd.isna(resource_name) or resource_name in resources:
                    continue
                
                resource_data = df[df['Resource Name'] == resource_name].iloc[0]
                settlement_point = resource_data['Settlement Point'] if 'Settlement Point' in resource_data else ''
                
                dates = pd.to_datetime(df[df['Resource Name'] == resource_name]['Delivery Date'])
                first_date = dates.min().date() if not dates.empty else date(year, 1, 1)
                last_date = dates.max().date() if not dates.empty else date(year, 12, 31)
                
                resources[resource_name] = BESSResource(
                    resource_name=resource_name,
                    settlement_point=settlement_point,
                    capacity_mw=100.0,  # Default if not found
                    duration_hours=4.0,
                    first_active_date=first_date,
                    last_active_date=last_date
                )
                
        except Exception as e:
            logger.error(f"Error processing DAM {year}: {e}")
        
        return resources
    
    def _extract_settlement_point(self, resource_name: str) -> str:
        """Extract settlement point from resource name pattern"""
        # Common patterns: SETTLEMENTPOINT_BESS1, SETTLEMENTPOINT_BES1
        parts = resource_name.split('_')
        if len(parts) > 1 and ('BESS' in parts[-1] or 'BES' in parts[-1] or 'ESS' in parts[-1]):
            return '_'.join(parts[:-1])
        return resource_name
    
    def save_registry(self, output_path: Path = None):
        """Save BESS registry to Parquet and JSON"""
        if output_path is None:
            output_path = OUTPUT_DIR / "bess_registry"
        
        # Convert to DataFrame
        records = [asdict(resource) for resource in self.resources.values()]
        df = pd.DataFrame(records)
        
        # Sort by capacity and name
        df = df.sort_values(['capacity_mw', 'resource_name'], ascending=[False, True])
        
        # Save as Parquet
        parquet_path = output_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved BESS registry to {parquet_path}")
        
        # Also save as JSON for reference
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(records, f, indent=2, default=str)
        logger.info(f"Saved BESS registry to {json_path}")
        
        # Save summary statistics
        self._save_summary_stats(df)
        
        return df
    
    def _save_summary_stats(self, df: pd.DataFrame):
        """Save summary statistics about BESS resources"""
        stats = {
            'total_resources': len(df),
            'total_capacity_mw': df['capacity_mw'].sum(),
            'avg_capacity_mw': df['capacity_mw'].mean(),
            'by_year': {}
        }
        
        # Group by first active year
        df['first_year'] = pd.to_datetime(df['first_active_date']).dt.year
        year_stats = df.groupby('first_year').agg({
            'resource_name': 'count',
            'capacity_mw': 'sum'
        }).rename(columns={'resource_name': 'count'})
        
        stats['by_year'] = year_stats.to_dict('index')
        
        # Save stats
        stats_path = OUTPUT_DIR / "bess_registry_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"BESS Registry Statistics:")
        logger.info(f"  Total Resources: {stats['total_resources']}")
        logger.info(f"  Total Capacity: {stats['total_capacity_mw']:.1f} MW")
        logger.info(f"  Average Capacity: {stats['avg_capacity_mw']:.1f} MW")

def main():
    """Run Phase 1 of migration: Schema validation and BESS identification"""
    
    logger.info("="*80)
    logger.info("BESS Parquet Migration - Phase 1: Schema & Resource Identification")
    logger.info("="*80)
    
    # Step 1: Validate schemas
    logger.info("\n1. Validating Parquet schemas...")
    schema_registry = SchemaRegistry(ROLLUP_DIR)
    
    # Save column availability matrix
    availability = schema_registry.get_column_availability()
    if not availability.empty:
        availability_path = OUTPUT_DIR / "column_availability.csv"
        availability.to_csv(availability_path)
        logger.info(f"Saved column availability matrix to {availability_path}")
    
    # Step 2: Extract BESS resources
    logger.info("\n2. Extracting BESS resources from Parquet files...")
    bess_registry = BESSRegistry(ROLLUP_DIR)
    resources = bess_registry.extract_bess_resources(years=list(range(2019, 2025)))
    
    # Step 3: Save registry
    logger.info("\n3. Saving BESS registry...")
    registry_df = bess_registry.save_registry()
    
    # Step 4: Validate key files for BESS analysis
    logger.info("\n4. Validating key Parquet files for BESS analysis...")
    critical_files = [
        ('RT_prices', 2024),
        ('DA_prices', 2024),
        ('AS_prices', 2024),
        ('DAM_Gen_Resources', 2024),
        ('SCED_Gen_Resources', 2024),
        ('COP_Snapshots', 2024),
    ]
    
    validation_results = []
    for dataset, year in critical_files:
        file_path = ROLLUP_DIR / dataset / f"{year}.parquet"
        if file_path.exists():
            is_valid = schema_registry.validate_parquet_file(file_path, dataset)
            validation_results.append({
                'dataset': dataset,
                'year': year,
                'exists': True,
                'valid': is_valid,
                'size_mb': file_path.stat().st_size / (1024*1024)
            })
        else:
            validation_results.append({
                'dataset': dataset,
                'year': year,
                'exists': False,
                'valid': False,
                'size_mb': 0
            })
    
    validation_df = pd.DataFrame(validation_results)
    validation_path = OUTPUT_DIR / "parquet_validation.csv"
    validation_df.to_csv(validation_path, index=False)
    logger.info(f"Saved validation results to {validation_path}")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("Phase 1 Complete - Summary:")
    logger.info(f"  ✓ Schemas loaded: {len(schema_registry.schemas)}")
    logger.info(f"  ✓ BESS resources found: {len(resources)}")
    logger.info(f"  ✓ Valid Parquet files: {validation_df['valid'].sum()}/{len(validation_df)}")
    logger.info(f"  ✓ Output directory: {OUTPUT_DIR}")
    logger.info("="*80)
    
    return schema_registry, bess_registry

if __name__ == "__main__":
    schema_registry, bess_registry = main()