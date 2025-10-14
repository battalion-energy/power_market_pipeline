#!/usr/bin/env python3
"""
Import BESS data to existing Prisma database using SQLite.
This version works with the existing Battalion Platform database.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
import json
import logging
import sys

class PrismaDBImporter:
    """Import BESS data to Prisma SQLite database"""
    
    def __init__(self, logger):
        self.logger = logger
        self.conn = None
        self.cursor = None
        
    def connect(self) -> bool:
        """Connect to the SQLite database"""
        try:
            # Find the Prisma database
            db_paths = [
                '/home/enrico/projects/battalion-platform/apps/neoweb/prisma/dev.db',
                '/home/enrico/projects/battalion-platform/apps/modeling/prisma/dev.db',
                '/home/enrico/projects/battalion-platform/prisma/dev.db'
            ]
            
            db_file = None
            for path in db_paths:
                if Path(path).exists():
                    db_file = path
                    break
            
            if not db_file:
                self.logger.error("Could not find Prisma database file")
                return False
            
            self.logger.info(f"Connecting to SQLite database: {db_file}")
            self.conn = sqlite3.connect(db_file)
            self.conn.row_factory = sqlite3.Row
            self.cursor = self.conn.cursor()
            
            self.logger.info("✅ Database connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def create_tables(self):
        """Create BESS tables if they don't exist"""
        self.logger.info("Creating/verifying BESS tables...")
        
        create_sql = """
        -- Create BESS facilities table
        CREATE TABLE IF NOT EXISTS bess_facilities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bess_gen_resource TEXT UNIQUE NOT NULL,
            substation TEXT,
            load_zone TEXT,
            ercot_county TEXT,
            ercot_capacity_mw REAL,
            
            -- EIA matched data
            eia_plant_name TEXT,
            eia_generator_id TEXT,
            eia_county TEXT,
            eia_capacity_mw REAL,
            eia_latitude REAL,
            eia_longitude REAL,
            eia_operating_year INTEGER,
            eia_status TEXT,
            
            -- Matching metadata
            match_score REAL,
            match_timestamp TEXT,
            
            -- Physical zone
            physical_zone TEXT,
            zone_mismatch INTEGER DEFAULT 0,
            
            -- Validation
            distance_from_county_center REAL,
            validation_flags TEXT,
            
            -- Audit
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            data_source TEXT,
            pipeline_version TEXT
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_bess_county ON bess_facilities(ercot_county);
        CREATE INDEX IF NOT EXISTS idx_bess_zone ON bess_facilities(load_zone);
        CREATE INDEX IF NOT EXISTS idx_bess_eia_status ON bess_facilities(eia_status);
        
        -- Statistics table
        CREATE TABLE IF NOT EXISTS bess_pipeline_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TEXT,
            total_bess INTEGER,
            matched_count INTEGER,
            match_rate REAL,
            avg_match_score REAL,
            validation_issues INTEGER,
            pipeline_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            self.cursor.executescript(create_sql)
            self.conn.commit()
            self.logger.info("✅ Database tables ready")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    def validate_and_import(self, csv_file: Path):
        """Validate and import CSV data"""
        # Load CSV
        df = pd.read_csv(csv_file)
        self.logger.info(f"Loaded {len(df)} records from CSV")
        
        # Clean column names
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
        
        # Calculate physical zone
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
            lambda x: 1 if x.get('load_zone') != x.get('physical_zone') and pd.notna(x.get('physical_zone')) else 0,
            axis=1
        )
        
        # Add metadata
        df['data_source'] = 'Monthly Pipeline'
        df['pipeline_version'] = '2.0'
        df['updated_at'] = datetime.now().isoformat()
        
        # Import records
        self.logger.info(f"Importing {len(df)} records...")
        
        for _, row in df.iterrows():
            # Prepare values
            values = row.replace({np.nan: None}).to_dict()
            
            # Check if exists
            self.cursor.execute(
                "SELECT id FROM bess_facilities WHERE bess_gen_resource = ?",
                (values['bess_gen_resource'],)
            )
            exists = self.cursor.fetchone()
            
            if exists:
                # Update existing
                update_sql = """
                UPDATE bess_facilities SET
                    substation = ?,
                    load_zone = ?,
                    ercot_county = ?,
                    ercot_capacity_mw = ?,
                    eia_plant_name = ?,
                    eia_generator_id = ?,
                    eia_county = ?,
                    eia_capacity_mw = ?,
                    eia_latitude = ?,
                    eia_longitude = ?,
                    eia_operating_year = ?,
                    eia_status = ?,
                    match_score = ?,
                    match_timestamp = ?,
                    physical_zone = ?,
                    zone_mismatch = ?,
                    data_source = ?,
                    pipeline_version = ?,
                    updated_at = ?
                WHERE bess_gen_resource = ?
                """
                
                self.cursor.execute(update_sql, (
                    values.get('substation'),
                    values.get('load_zone'),
                    values.get('ercot_county'),
                    values.get('ercot_capacity_mw'),
                    values.get('eia_plant_name'),
                    values.get('eia_generator_id'),
                    values.get('eia_county'),
                    values.get('eia_capacity_mw'),
                    values.get('eia_latitude'),
                    values.get('eia_longitude'),
                    values.get('eia_operating_year'),
                    values.get('eia_status'),
                    values.get('match_score'),
                    values.get('match_timestamp'),
                    values.get('physical_zone'),
                    values.get('zone_mismatch'),
                    values.get('data_source'),
                    values.get('pipeline_version'),
                    values.get('updated_at'),
                    values.get('bess_gen_resource')
                ))
            else:
                # Insert new
                insert_sql = """
                INSERT INTO bess_facilities (
                    bess_gen_resource, substation, load_zone, ercot_county, ercot_capacity_mw,
                    eia_plant_name, eia_generator_id, eia_county, eia_capacity_mw,
                    eia_latitude, eia_longitude, eia_operating_year, eia_status,
                    match_score, match_timestamp, physical_zone, zone_mismatch,
                    data_source, pipeline_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                
                self.cursor.execute(insert_sql, (
                    values.get('bess_gen_resource'),
                    values.get('substation'),
                    values.get('load_zone'),
                    values.get('ercot_county'),
                    values.get('ercot_capacity_mw'),
                    values.get('eia_plant_name'),
                    values.get('eia_generator_id'),
                    values.get('eia_county'),
                    values.get('eia_capacity_mw'),
                    values.get('eia_latitude'),
                    values.get('eia_longitude'),
                    values.get('eia_operating_year'),
                    values.get('eia_status'),
                    values.get('match_score'),
                    values.get('match_timestamp'),
                    values.get('physical_zone'),
                    values.get('zone_mismatch'),
                    values.get('data_source'),
                    values.get('pipeline_version'),
                    datetime.now().isoformat(),
                    values.get('updated_at')
                ))
        
        self.conn.commit()
        self.logger.info(f"✅ Successfully imported {len(df)} records")
        
        # Check CROSSETT specifically
        self.cursor.execute(
            "SELECT * FROM bess_facilities WHERE bess_gen_resource LIKE 'CROSSETT%'"
        )
        crossett = self.cursor.fetchall()
        if crossett:
            for row in crossett:
                self.logger.info(f"  CROSSETT verified: {row['bess_gen_resource']} in {row['eia_county']} County")
                self.logger.info(f"    Coordinates: ({row['eia_latitude']}, {row['eia_longitude']})")
        
        # Update statistics
        stats_sql = """
        INSERT INTO bess_pipeline_stats (
            run_date, total_bess, matched_count, match_rate, avg_match_score,
            validation_issues, pipeline_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        total = len(df)
        avg_score = df['match_score'].mean() if 'match_score' in df.columns else 0
        
        self.cursor.execute(stats_sql, (
            datetime.now().date().isoformat(),
            total,
            total,
            100.0,
            avg_score,
            0,
            '2.0'
        ))
        self.conn.commit()
        
        return len(df)
    
    def verify_data(self):
        """Verify imported data"""
        self.logger.info("\nVerifying imported data...")
        
        # Count total records
        self.cursor.execute("SELECT COUNT(*) as count FROM bess_facilities")
        total = self.cursor.fetchone()['count']
        self.logger.info(f"  Total BESS facilities: {total}")
        
        # Check CROSSETT
        self.cursor.execute("""
            SELECT bess_gen_resource, eia_county, eia_latitude, eia_longitude, physical_zone
            FROM bess_facilities 
            WHERE bess_gen_resource LIKE 'CROSSETT%'
        """)
        crossett = self.cursor.fetchall()
        
        if crossett:
            self.logger.info("\n  CROSSETT verification:")
            for row in crossett:
                self.logger.info(f"    {row['bess_gen_resource']}:")
                self.logger.info(f"      County: {row['eia_county']}")
                self.logger.info(f"      Coordinates: ({row['eia_latitude']}, {row['eia_longitude']})")
                self.logger.info(f"      Physical Zone: {row['physical_zone']}")
                
                if row['eia_county'] == 'Crane' and row['physical_zone'] == 'LZ_WEST':
                    self.logger.info("      ✅ CORRECTLY LOCATED IN CRANE COUNTY, WEST TEXAS!")
                else:
                    self.logger.error("      ❌ LOCATION ERROR!")
        
        # Zone mismatches
        self.cursor.execute("SELECT COUNT(*) as count FROM bess_facilities WHERE zone_mismatch = 1")
        mismatches = self.cursor.fetchone()['count']
        self.logger.info(f"\n  Zone mismatches: {mismatches}")
        
        # Statistics
        self.cursor.execute("""
            SELECT * FROM bess_pipeline_stats 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        stats = self.cursor.fetchone()
        if stats:
            self.logger.info("\n  Latest pipeline run:")
            self.logger.info(f"    Date: {stats['run_date']}")
            self.logger.info(f"    Total: {stats['total_bess']}")
            self.logger.info(f"    Match rate: {stats['match_rate']}%")
            self.logger.info(f"    Avg score: {stats['avg_match_score']:.1f}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
        self.logger.info("Database connection closed")
    
    def run(self, csv_file: Path):
        """Run the complete import"""
        self.logger.info("="*70)
        self.logger.info("STARTING SQLITE DATABASE IMPORT")
        self.logger.info(f"Input file: {csv_file}")
        self.logger.info("="*70)
        
        if not self.connect():
            return False
        
        try:
            self.create_tables()
            imported = self.validate_and_import(csv_file)
            self.verify_data()
            
            self.logger.info("\n" + "="*70)
            self.logger.info("IMPORT COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"✅ Successfully imported {imported} records")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Import failed: {e}", exc_info=True)
            return False
        finally:
            self.close()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Import BESS data to Prisma SQLite database')
    parser.add_argument('--file', type=str,
                       default='output/BESS_MATCHED_202509_FIXED.csv',
                       help='CSV file to import')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check file
    csv_file = Path(args.file)
    if not csv_file.exists():
        logger.error(f"File not found: {csv_file}")
        sys.exit(1)
    
    # Run import
    importer = PrismaDBImporter(logger)
    success = importer.run(csv_file)
    
    if success:
        logger.info("✅ Import successful!")
        sys.exit(0)
    else:
        logger.error("❌ Import failed!")
        sys.exit(1)

if __name__ == '__main__':
    main()