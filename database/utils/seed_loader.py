"""Database seed data loader for reference and lookup tables."""

import os
import json
import csv
import psycopg2
from pathlib import Path
from typing import Dict, Any, List
import yaml


class SeedLoader:
    """Loads seed data for reference tables, ISOs, and lookup data."""
    
    def __init__(self, db_params: Dict[str, Any]):
        self.db_params = db_params
        self.seeds_dir = Path(__file__).parent.parent / 'seeds'
    
    def load_all_seeds(self):
        """Load all seed data files."""
        if not self.seeds_dir.exists():
            print(f"Seeds directory not found: {self.seeds_dir}")
            return
        
        # Load in specific order due to dependencies
        seed_order = [
            'isos.yaml',
            'ercot_locations.csv',
            'caiso_locations.csv', 
            'isone_locations.csv',
            'nyiso_locations.csv',
            'data_catalog.yaml'
        ]
        
        for seed_file in seed_order:
            seed_path = self.seeds_dir / seed_file
            if seed_path.exists():
                self.load_seed_file(seed_path)
            else:
                print(f"⚠ Seed file not found: {seed_file}")
    
    def load_seed_file(self, file_path: Path):
        """Load a single seed file based on its extension."""
        print(f"Loading seed data: {file_path.name}")
        
        try:
            if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
                self.load_yaml_seed(file_path)
            elif file_path.suffix == '.csv':
                self.load_csv_seed(file_path)
            elif file_path.suffix == '.json':
                self.load_json_seed(file_path)
            else:
                print(f"⚠ Unknown seed file format: {file_path.suffix}")
                
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {e}")
    
    def load_yaml_seed(self, file_path: Path):
        """Load YAML seed data."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Determine table and data from filename
        if 'isos' in file_path.name:
            self.load_isos_data(data)
        elif 'data_catalog' in file_path.name:
            self.load_data_catalog(data)
        else:
            print(f"⚠ Unknown YAML seed file: {file_path.name}")
    
    def load_csv_seed(self, file_path: Path):
        """Load CSV seed data."""
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Determine table from filename
        if 'locations' in file_path.name:
            iso_code = file_path.name.split('_')[0].upper()
            self.load_locations_data(rows, iso_code)
        else:
            print(f"⚠ Unknown CSV seed file: {file_path.name}")
    
    def load_json_seed(self, file_path: Path):
        """Load JSON seed data."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"⚠ JSON seed loading not implemented for: {file_path.name}")
    
    def load_isos_data(self, data: Dict[str, Any]):
        """Load ISO reference data."""
        isos = data.get('isos', [])
        
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    for iso in isos:
                        cur.execute("""
                            INSERT INTO isos (code, name, timezone)
                            VALUES (%(code)s, %(name)s, %(timezone)s)
                            ON CONFLICT (code) DO UPDATE SET
                                name = EXCLUDED.name,
                                timezone = EXCLUDED.timezone
                        """, iso)
                
                conn.commit()
                print(f"✓ Loaded {len(isos)} ISO records")
                
        except psycopg2.Error as e:
            print(f"✗ Error loading ISO data: {e}")
    
    def load_locations_data(self, rows: List[Dict], iso_code: str):
        """Load location data for a specific ISO."""
        if not rows:
            print(f"⚠ No location data for {iso_code}")
            return
        
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Get ISO ID
                    cur.execute("SELECT id FROM isos WHERE code = %s", (iso_code,))
                    iso_result = cur.fetchone()
                    if not iso_result:
                        print(f"✗ ISO {iso_code} not found in database")
                        return
                    
                    iso_id = iso_result[0]
                    
                    # Insert locations
                    for row in rows:
                        location_data = {
                            'iso_id': iso_id,
                            'location_id': row.get('location_id'),
                            'location_name': row.get('location_name'),
                            'location_type': row.get('location_type'),
                            'latitude': float(row['latitude']) if row.get('latitude') else None,
                            'longitude': float(row['longitude']) if row.get('longitude') else None,
                            'state': row.get('state'),
                            'county': row.get('county'),
                            'is_active': row.get('is_active', 'true').lower() == 'true'
                        }
                        
                        cur.execute("""
                            INSERT INTO locations (
                                iso_id, location_id, location_name, location_type,
                                latitude, longitude, state, county, is_active
                            ) VALUES (
                                %(iso_id)s, %(location_id)s, %(location_name)s, %(location_type)s,
                                %(latitude)s, %(longitude)s, %(state)s, %(county)s, %(is_active)s
                            )
                            ON CONFLICT (iso_id, location_id) DO UPDATE SET
                                location_name = EXCLUDED.location_name,
                                location_type = EXCLUDED.location_type,
                                latitude = EXCLUDED.latitude,
                                longitude = EXCLUDED.longitude,
                                state = EXCLUDED.state,
                                county = EXCLUDED.county,
                                is_active = EXCLUDED.is_active
                        """, location_data)
                
                conn.commit()
                print(f"✓ Loaded {len(rows)} {iso_code} location records")
                
        except psycopg2.Error as e:
            print(f"✗ Error loading {iso_code} location data: {e}")
        except ValueError as e:
            print(f"✗ Data format error for {iso_code} locations: {e}")
    
    def load_data_catalog(self, data: Dict[str, Any]):
        """Load data catalog metadata."""
        datasets = data.get('datasets', [])
        
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    for dataset in datasets:
                        # Insert dataset
                        cur.execute("""
                            INSERT INTO data_catalog (
                                dataset_name, table_name, iso, description,
                                update_frequency, spatial_granularity, temporal_granularity,
                                is_public, requires_auth
                            ) VALUES (
                                %(dataset_name)s, %(table_name)s, %(iso)s, %(description)s,
                                %(update_frequency)s, %(spatial_granularity)s, %(temporal_granularity)s,
                                %(is_public)s, %(requires_auth)s
                            )
                            ON CONFLICT (dataset_name) DO UPDATE SET
                                table_name = EXCLUDED.table_name,
                                iso = EXCLUDED.iso,
                                description = EXCLUDED.description,
                                update_frequency = EXCLUDED.update_frequency,
                                spatial_granularity = EXCLUDED.spatial_granularity,
                                temporal_granularity = EXCLUDED.temporal_granularity,
                                is_public = EXCLUDED.is_public,
                                requires_auth = EXCLUDED.requires_auth
                        """, dataset)
                        
                        # Insert column definitions
                        columns = dataset.get('columns', [])
                        for col in columns:
                            col_data = {
                                'dataset_name': dataset['dataset_name'],
                                'column_name': col['column_name'],
                                'data_type': col['data_type'],
                                'unit': col.get('unit'),
                                'description': col.get('description'),
                                'is_required': col.get('is_required', False),
                                'display_order': col.get('display_order', 0)
                            }
                            
                            cur.execute("""
                                INSERT INTO data_catalog_columns (
                                    dataset_name, column_name, data_type, unit,
                                    description, is_required, display_order
                                ) VALUES (
                                    %(dataset_name)s, %(column_name)s, %(data_type)s, %(unit)s,
                                    %(description)s, %(is_required)s, %(display_order)s
                                )
                                ON CONFLICT (dataset_name, column_name) DO UPDATE SET
                                    data_type = EXCLUDED.data_type,
                                    unit = EXCLUDED.unit,
                                    description = EXCLUDED.description,
                                    is_required = EXCLUDED.is_required,
                                    display_order = EXCLUDED.display_order
                            """, col_data)
                
                conn.commit()
                print(f"✓ Loaded {len(datasets)} data catalog entries")
                
        except psycopg2.Error as e:
            print(f"✗ Error loading data catalog: {e}")
    
    def create_sample_seed_files(self):
        """Create sample seed files for reference."""
        self.seeds_dir.mkdir(exist_ok=True)
        
        # Sample ISOs file
        isos_data = {
            'isos': [
                {
                    'code': 'ERCOT',
                    'name': 'Electric Reliability Council of Texas',
                    'timezone': 'US/Central'
                },
                {
                    'code': 'CAISO', 
                    'name': 'California Independent System Operator',
                    'timezone': 'US/Pacific'
                },
                {
                    'code': 'ISONE',
                    'name': 'ISO New England',
                    'timezone': 'US/Eastern'
                },
                {
                    'code': 'NYISO',
                    'name': 'New York Independent System Operator', 
                    'timezone': 'US/Eastern'
                }
            ]
        }
        
        with open(self.seeds_dir / 'isos.yaml', 'w') as f:
            yaml.dump(isos_data, f, default_flow_style=False)
        
        # Sample ERCOT locations
        ercot_locations = [
            {
                'location_id': 'HB_BUSAVG',
                'location_name': 'Bus Average Hub',
                'location_type': 'hub',
                'latitude': 31.0,
                'longitude': -97.0,
                'state': 'TX',
                'is_active': 'true'
            },
            {
                'location_id': 'HB_HOUSTON',
                'location_name': 'Houston Hub',
                'location_type': 'hub',
                'latitude': 29.7604,
                'longitude': -95.3698,
                'state': 'TX',
                'is_active': 'true'
            }
        ]
        
        with open(self.seeds_dir / 'ercot_locations.csv', 'w', newline='') as f:
            if ercot_locations:
                writer = csv.DictWriter(f, fieldnames=ercot_locations[0].keys())
                writer.writeheader()
                writer.writerows(ercot_locations)
        
        print(f"✓ Created sample seed files in {self.seeds_dir}")