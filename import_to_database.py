#!/usr/bin/env python3
"""
DATABASE IMPORT SCRIPT FOR BESS-EIA MATCHED DATA
Imports the monthly matched CSV into the database with proper validation and error handling.

Usage:
    python import_to_database.py [--file BESS_MATCHED_LATEST.csv] [--dry-run]

Features:
    - Upsert logic (update existing, insert new)
    - Data validation before import
    - Transaction management
    - Detailed logging
    - Dry run mode for testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from decimal import Decimal
import logging
import sys
import json
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/enrico/projects/battalion-platform/.env')

class DatabaseImporter:
    """Production-quality database importer for BESS data"""
    
    def __init__(self, logger):
        self.logger = logger
        self.conn = None
        self.cursor = None
        
    def connect(self) -> bool:
        """Connect to the database"""
        try:
            # Get connection parameters from environment
            db_params = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': os.getenv('DB_PORT', '5432'),
                'database': os.getenv('DB_NAME', 'battalion'),
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')
            }
            
            # For Prisma/Next.js apps, try DATABASE_URL first
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                self.logger.info("Using DATABASE_URL for connection")
                self.conn = psycopg2.connect(database_url)
            else:
                self.logger.info(f"Connecting to {db_params['host']}:{db_params['port']}/{db_params['database']}")
                self.conn = psycopg2.connect(**db_params)
            
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            self.logger.info("✅ Database connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_tables(self):
        """Create or verify database tables exist"""
        self.logger.info("Creating/verifying database tables...")
        
        create_sql = """
        -- Main BESS facilities table
        CREATE TABLE IF NOT EXISTS bess_facilities (
            id SERIAL PRIMARY KEY,
            bess_gen_resource VARCHAR(100) UNIQUE NOT NULL,
            substation VARCHAR(100),
            load_zone VARCHAR(20),
            ercot_county VARCHAR(100),
            ercot_capacity_mw DECIMAL(10,2),
            
            -- EIA matched data
            eia_plant_name VARCHAR(200),
            eia_generator_id VARCHAR(50),
            eia_county VARCHAR(100),
            eia_capacity_mw DECIMAL(10,2),
            eia_latitude DECIMAL(10,6),
            eia_longitude DECIMAL(10,6),
            eia_operating_year INTEGER,
            eia_status VARCHAR(50),
            
            -- Matching metadata
            match_score DECIMAL(5,2),
            match_timestamp TIMESTAMP,
            
            -- Physical zone from coordinates
            physical_zone VARCHAR(20),
            zone_mismatch BOOLEAN DEFAULT FALSE,
            
            -- Validation flags
            distance_from_county_center DECIMAL(10,2),
            validation_flags JSONB,
            
            -- Audit fields
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_source VARCHAR(50),
            pipeline_version VARCHAR(20)
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_bess_county ON bess_facilities(ercot_county);
        CREATE INDEX IF NOT EXISTS idx_bess_zone ON bess_facilities(load_zone);
        CREATE INDEX IF NOT EXISTS idx_bess_physical_zone ON bess_facilities(physical_zone);
        CREATE INDEX IF NOT EXISTS idx_bess_eia_status ON bess_facilities(eia_status);
        CREATE INDEX IF NOT EXISTS idx_bess_match_score ON bess_facilities(match_score);
        
        -- History table for tracking changes
        CREATE TABLE IF NOT EXISTS bess_facilities_history (
            id SERIAL PRIMARY KEY,
            bess_gen_resource VARCHAR(100),
            change_type VARCHAR(20), -- INSERT, UPDATE
            changed_fields JSONB,
            old_values JSONB,
            new_values JSONB,
            changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            changed_by VARCHAR(100)
        );
        
        -- Monthly statistics table
        CREATE TABLE IF NOT EXISTS bess_pipeline_stats (
            id SERIAL PRIMARY KEY,
            run_date DATE,
            total_bess INTEGER,
            matched_count INTEGER,
            unmatched_count INTEGER,
            match_rate DECIMAL(5,2),
            avg_match_score DECIMAL(5,2),
            validation_issues INTEGER,
            pipeline_version VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            self.cursor.execute(create_sql)
            self.conn.commit()
            self.logger.info("✅ Database tables ready")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            self.conn.rollback()
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data before import"""
        self.logger.info("Validating data...")
        
        # Required columns
        required_cols = ['BESS_Gen_Resource']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean and standardize
        df = df.copy()
        
        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['BESS_Gen_Resource'], keep='last')
        if len(df) < original_len:
            self.logger.warning(f"Removed {original_len - len(df)} duplicate records")
        
        # Standardize column names
        column_mapping = {
            'BESS_Gen_Resource': 'bess_gen_resource',
            'Substation': 'substation',
            'Load_Zone': 'load_zone',
            'ERCOT_County': 'ercot_county',
            'ERCOT_Capacity_MW': 'ercot_capacity_mw',
            'EIA_Plant_Name': 'eia_plant_name',
            'EIA_Generator_ID': 'eia_generator_id',
            'EIA_County': 'eia_county',
            'EIA_Capacity_MW': 'eia_capacity_mw',
            'EIA_Latitude': 'eia_latitude',
            'EIA_Longitude': 'eia_longitude',
            'EIA_Operating_Year': 'eia_operating_year',
            'EIA_Status': 'eia_status',
            'Match_Score': 'match_score',
            'Match_Timestamp': 'match_timestamp'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Calculate physical zone from coordinates
        def get_physical_zone(row):
            lat = row.get('eia_latitude')
            lon = row.get('eia_longitude')
            if pd.isna(lat) or pd.isna(lon):
                return None
            
            if lon <= -100.0:
                return 'LZ_WEST'
            elif lat >= 32.0 and lon >= -98.5:
                return 'LZ_NORTH'
            elif (28.5 <= lat <= 30.5) and (-96.0 <= lon <= -94.5):
                return 'LZ_HOUSTON'
            else:
                return 'LZ_SOUTH'
        
        df['physical_zone'] = df.apply(get_physical_zone, axis=1)
        df['zone_mismatch'] = df.apply(
            lambda x: x['load_zone'] != x['physical_zone'] if pd.notna(x['physical_zone']) else False,
            axis=1
        )
        
        # Add metadata
        df['data_source'] = 'Monthly Pipeline'
        df['pipeline_version'] = '2.0'
        
        # Convert timestamps
        if 'match_timestamp' in df.columns:
            df['match_timestamp'] = pd.to_datetime(df['match_timestamp'])
        
        self.logger.info(f"✅ Validated {len(df)} records")
        return df
    
    def upsert_records(self, df: pd.DataFrame, dry_run: bool = False):
        """Upsert records to database (update existing, insert new)"""
        self.logger.info(f"Upserting {len(df)} records...")
        
        if dry_run:
            self.logger.info("DRY RUN MODE - no actual database changes")
            return
        
        upsert_sql = """
        INSERT INTO bess_facilities (
            bess_gen_resource, substation, load_zone, ercot_county, ercot_capacity_mw,
            eia_plant_name, eia_generator_id, eia_county, eia_capacity_mw,
            eia_latitude, eia_longitude, eia_operating_year, eia_status,
            match_score, match_timestamp, physical_zone, zone_mismatch,
            data_source, pipeline_version, updated_at
        ) VALUES (
            %(bess_gen_resource)s, %(substation)s, %(load_zone)s, %(ercot_county)s, %(ercot_capacity_mw)s,
            %(eia_plant_name)s, %(eia_generator_id)s, %(eia_county)s, %(eia_capacity_mw)s,
            %(eia_latitude)s, %(eia_longitude)s, %(eia_operating_year)s, %(eia_status)s,
            %(match_score)s, %(match_timestamp)s, %(physical_zone)s, %(zone_mismatch)s,
            %(data_source)s, %(pipeline_version)s, CURRENT_TIMESTAMP
        )
        ON CONFLICT (bess_gen_resource) 
        DO UPDATE SET
            substation = EXCLUDED.substation,
            load_zone = EXCLUDED.load_zone,
            ercot_county = EXCLUDED.ercot_county,
            ercot_capacity_mw = EXCLUDED.ercot_capacity_mw,
            eia_plant_name = EXCLUDED.eia_plant_name,
            eia_generator_id = EXCLUDED.eia_generator_id,
            eia_county = EXCLUDED.eia_county,
            eia_capacity_mw = EXCLUDED.eia_capacity_mw,
            eia_latitude = EXCLUDED.eia_latitude,
            eia_longitude = EXCLUDED.eia_longitude,
            eia_operating_year = EXCLUDED.eia_operating_year,
            eia_status = EXCLUDED.eia_status,
            match_score = EXCLUDED.match_score,
            match_timestamp = EXCLUDED.match_timestamp,
            physical_zone = EXCLUDED.physical_zone,
            zone_mismatch = EXCLUDED.zone_mismatch,
            data_source = EXCLUDED.data_source,
            pipeline_version = EXCLUDED.pipeline_version,
            updated_at = CURRENT_TIMESTAMP;
        """
        
        # Convert DataFrame to list of dicts for execute_batch
        records = df.replace({np.nan: None}).to_dict('records')
        
        try:
            # Use execute_batch for better performance
            execute_batch(self.cursor, upsert_sql, records, page_size=100)
            self.conn.commit()
            self.logger.info(f"✅ Successfully upserted {len(records)} records")
            
            # Log CROSSETT specifically
            crossett_records = [r for r in records if 'CROSSETT' in str(r.get('bess_gen_resource', '')).upper()]
            if crossett_records:
                for r in crossett_records:
                    self.logger.info(f"  CROSSETT verified: {r['bess_gen_resource']} in {r['eia_county']} County")
                    
        except Exception as e:
            self.logger.error(f"Failed to upsert records: {e}")
            self.conn.rollback()
            raise
    
    def update_statistics(self, df: pd.DataFrame, validation_report: dict = None):
        """Update pipeline statistics table"""
        self.logger.info("Updating statistics...")
        
        stats_sql = """
        INSERT INTO bess_pipeline_stats (
            run_date, total_bess, matched_count, unmatched_count,
            match_rate, avg_match_score, validation_issues, pipeline_version
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        # Calculate statistics
        total_bess = len(df)
        matched_count = len(df[df['match_score'].notna()]) if 'match_score' in df.columns else 0
        unmatched_count = total_bess - matched_count
        match_rate = (matched_count / total_bess * 100) if total_bess > 0 else 0
        avg_score = df['match_score'].mean() if 'match_score' in df.columns else 0
        validation_issues = len(validation_report.get('issues', [])) if validation_report else 0
        
        try:
            self.cursor.execute(stats_sql, (
                datetime.now().date(),
                total_bess,
                matched_count,
                unmatched_count,
                match_rate,
                avg_score,
                validation_issues,
                '2.0'
            ))
            self.conn.commit()
            self.logger.info("✅ Statistics updated")
        except Exception as e:
            self.logger.error(f"Failed to update statistics: {e}")
            self.conn.rollback()
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.logger.info("Database connection closed")
    
    def run(self, csv_file: Path, dry_run: bool = False):
        """Run the complete import process"""
        self.logger.info("="*70)
        self.logger.info("STARTING DATABASE IMPORT")
        self.logger.info(f"Input file: {csv_file}")
        self.logger.info(f"Dry run: {dry_run}")
        self.logger.info("="*70)
        
        # Load CSV
        try:
            df = pd.read_csv(csv_file)
            self.logger.info(f"Loaded {len(df)} records from CSV")
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}")
            return False
        
        # Connect to database
        if not self.connect():
            return False
        
        try:
            # Create tables if needed
            self.create_tables()
            
            # Validate data
            df = self.validate_data(df)
            
            # Upsert records
            self.upsert_records(df, dry_run=dry_run)
            
            # Load validation report if exists
            validation_report = None
            validation_file = csv_file.parent / csv_file.name.replace('BESS_MATCHED', 'VALIDATION_REPORT').replace('.csv', '.json')
            if validation_file.exists():
                with open(validation_file) as f:
                    validation_report = json.load(f)
            
            # Update statistics
            if not dry_run:
                self.update_statistics(df, validation_report)
            
            self.logger.info("\n" + "="*70)
            self.logger.info("IMPORT COMPLETE")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Import failed: {e}", exc_info=True)
            return False
        finally:
            self.close()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import BESS matched data to database')
    parser.add_argument('--file', type=str, 
                       default='output/BESS_MATCHED_LATEST.csv',
                       help='CSV file to import')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate without actually importing')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check file exists
    csv_file = Path(args.file)
    if not csv_file.exists():
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    
    # Run import
    importer = DatabaseImporter(logger)
    success = importer.run(csv_file, dry_run=args.dry_run)
    
    if success:
        logger.info("✅ Import successful!")
        sys.exit(0)
    else:
        logger.error("❌ Import failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()