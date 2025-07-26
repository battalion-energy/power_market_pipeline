#!/usr/bin/env python3
"""
Database setup script for Power Market Pipeline.

This script handles:
- Database creation and validation
- Migration execution
- Seed data population
- Index creation and optimization
- Development vs production configurations
"""

import os
import sys
import argparse
import psycopg2
from pathlib import Path
from typing import List, Dict, Any
import yaml
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from database.utils.migration_runner import MigrationRunner
from database.utils.seed_loader import SeedLoader


class DatabaseSetup:
    """Comprehensive database setup and management."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.db_params = self._get_db_params()
        self.migration_runner = MigrationRunner(self.db_params)
        self.seed_loader = SeedLoader(self.db_params)
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load database configuration."""
        if config_path:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Parse DATABASE_URL if available
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            db_config = {
                'host': parsed.hostname or 'localhost',
                'port': parsed.port or 5432,
                'name': parsed.path.lstrip('/') if parsed.path else 'power_market',
                'user': parsed.username or os.getenv('USER', ''),
                'password': parsed.password or '',
                'timezone': 'UTC'
            }
        else:
            # Default configuration from environment
            db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', 5432)),
                'name': os.getenv('DB_NAME', 'power_market'),
                'user': os.getenv('DB_USER', os.getenv('USER', '')),
                'password': os.getenv('DB_PASSWORD', ''),
                'timezone': 'UTC'
            }
        
        return {
            'database': db_config,
            'setup': {
                'create_extensions': True,
                'run_migrations': True,
                'load_seeds': True,
                'create_indexes': True,
                'enable_compression': True
            }
        }
    
    def _get_db_params(self) -> Dict[str, Any]:
        """Extract database connection parameters."""
        db_config = self.config['database']
        return {
            'host': db_config['host'],
            'port': db_config['port'],
            'database': db_config['name'],
            'user': db_config['user'],
            'password': db_config['password']
        }
    
    def create_database_if_not_exists(self):
        """Create database if it doesn't exist."""
        db_name = self.db_params['database']
        
        # Connect to postgres database to create our database
        postgres_params = self.db_params.copy()
        postgres_params['database'] = 'postgres'
        
        try:
            with psycopg2.connect(**postgres_params) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    # Check if database exists
                    cur.execute(
                        "SELECT 1 FROM pg_database WHERE datname = %s",
                        (db_name,)
                    )
                    
                    if not cur.fetchone():
                        print(f"Creating database: {db_name}")
                        cur.execute(f'CREATE DATABASE "{db_name}"')
                        print(f"✓ Database {db_name} created successfully")
                    else:
                        print(f"✓ Database {db_name} already exists")
                        
        except psycopg2.Error as e:
            print(f"Error creating database: {e}")
            sys.exit(1)
    
    def validate_database_connection(self):
        """Validate database connection and basic setup."""
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    print(f"✓ Connected to PostgreSQL: {version}")
                    
                    # Check for TimescaleDB
                    cur.execute(
                        "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"
                    )
                    if cur.fetchone():
                        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
                        ts_version = cur.fetchone()[0]
                        print(f"✓ TimescaleDB extension found: {ts_version}")
                    else:
                        print("⚠ TimescaleDB extension not found - will attempt to install")
                        
        except psycopg2.Error as e:
            print(f"✗ Database connection failed: {e}")
            sys.exit(1)
    
    def setup_extensions(self):
        """Install required PostgreSQL extensions."""
        if not self.config['setup']['create_extensions']:
            return
            
        extensions = ['timescaledb', 'pg_stat_statements', 'pg_trgm']
        
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    for ext in extensions:
                        try:
                            cur.execute(f"CREATE EXTENSION IF NOT EXISTS {ext}")
                            print(f"✓ Extension {ext} enabled")
                        except psycopg2.Error as e:
                            if ext == 'timescaledb':
                                print(f"✗ Failed to install TimescaleDB: {e}")
                                print("  Please install TimescaleDB: https://docs.timescale.com/install/")
                                sys.exit(1)
                            else:
                                print(f"⚠ Warning: Could not install {ext}: {e}")
                
                conn.commit()
                
        except psycopg2.Error as e:
            print(f"Error setting up extensions: {e}")
            sys.exit(1)
    
    def run_migrations(self):
        """Execute all database migrations."""
        if not self.config['setup']['run_migrations']:
            return
            
        print("\n🔄 Running database migrations...")
        self.migration_runner.run_all_migrations()
        print("✓ All migrations completed successfully")
    
    def load_seed_data(self):
        """Load seed data into database."""
        if not self.config['setup']['load_seeds']:
            return
            
        print("\n🌱 Loading seed data...")
        self.seed_loader.load_all_seeds()
        print("✓ Seed data loaded successfully")
    
    def optimize_database(self):
        """Run database optimization tasks."""
        if not self.config['setup']['enable_compression']:
            return
            
        print("\n⚡ Optimizing database performance...")
        
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Update table statistics
                    cur.execute("ANALYZE")
                    print("✓ Updated table statistics")
                    
                    # Refresh materialized views
                    views = ['v_lmp_hourly', 'v_lmp_daily']
                    for view in views:
                        try:
                            cur.execute(f"REFRESH MATERIALIZED VIEW {view}")
                            print(f"✓ Refreshed materialized view: {view}")
                        except psycopg2.Error:
                            print(f"⚠ Could not refresh view {view} (may not exist yet)")
                
                conn.commit()
                
        except psycopg2.Error as e:
            print(f"Warning during optimization: {e}")
    
    def show_database_info(self):
        """Display database information and statistics."""
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    print("\n📊 Database Information:")
                    print("-" * 40)
                    
                    # Database size
                    cur.execute(
                        "SELECT pg_size_pretty(pg_database_size(current_database()))"
                    )
                    db_size = cur.fetchone()[0]
                    print(f"Database size: {db_size}")
                    
                    # Table counts
                    cur.execute("""
                        SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
                        FROM pg_stat_user_tables 
                        WHERE schemaname = 'public'
                        ORDER BY tablename
                    """)
                    
                    tables = cur.fetchall()
                    if tables:
                        print("\nTable statistics:")
                        for schema, table, inserts, updates, deletes in tables:
                            total_ops = inserts + updates + deletes
                            if total_ops > 0:
                                print(f"  {table}: {inserts:,} inserts, {updates:,} updates, {deletes:,} deletes")
                    
                    # Hypertable info
                    cur.execute("""
                        SELECT hypertable_name, num_chunks, compressed_chunks
                        FROM timescaledb_information.hypertables
                    """)
                    
                    hypertables = cur.fetchall()
                    if hypertables:
                        print("\nTimescaleDB hypertables:")
                        for table, chunks, compressed in hypertables:
                            print(f"  {table}: {chunks} chunks ({compressed} compressed)")
                            
        except psycopg2.Error as e:
            print(f"Could not retrieve database info: {e}")
    
    def full_setup(self):
        """Run complete database setup process."""
        print("🚀 Starting Power Market Pipeline Database Setup")
        print("=" * 50)
        
        self.create_database_if_not_exists()
        self.validate_database_connection()
        self.setup_extensions()
        self.run_migrations()
        self.load_seed_data()
        self.optimize_database()
        self.show_database_info()
        
        print("\n🎉 Database setup completed successfully!")
        print(f"Database: {self.db_params['database']} on {self.db_params['host']}")
        print("\nYou can now start using the Power Market Pipeline:")
        print("  python -m power_market_pipeline.cli download --iso ERCOT --market DAM")


def main():
    parser = argparse.ArgumentParser(description='Power Market Pipeline Database Setup')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--skip-create', action='store_true', help='Skip database creation')
    parser.add_argument('--skip-migrations', action='store_true', help='Skip migrations')
    parser.add_argument('--skip-seeds', action='store_true', help='Skip seed data')
    parser.add_argument('--info-only', action='store_true', help='Show database info only')
    
    args = parser.parse_args()
    
    setup = DatabaseSetup(args.config)
    
    # Override config based on arguments
    if args.skip_migrations:
        setup.config['setup']['run_migrations'] = False
    if args.skip_seeds:
        setup.config['setup']['load_seeds'] = False
    
    if args.info_only:
        setup.validate_database_connection()
        setup.show_database_info()
    else:
        if not args.skip_create:
            setup.create_database_if_not_exists()
        setup.full_setup()


if __name__ == '__main__':
    main()