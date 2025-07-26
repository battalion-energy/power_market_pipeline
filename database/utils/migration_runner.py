"""Database migration runner with version tracking."""

import os
import re
import psycopg2
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime


class MigrationRunner:
    """Handles database schema migrations with version tracking."""
    
    def __init__(self, db_params: Dict[str, Any]):
        self.db_params = db_params
        self.migrations_dir = Path(__file__).parent.parent / 'migrations'
        self.ensure_migration_table()
    
    def ensure_migration_table(self):
        """Create migration tracking table if it doesn't exist."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            version VARCHAR(50) UNIQUE NOT NULL,
            filename VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64),
            execution_time_ms INTEGER
        );
        
        CREATE INDEX IF NOT EXISTS idx_schema_migrations_version 
        ON schema_migrations(version);
        """
        
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    cur.execute(create_table_sql)
                conn.commit()
        except psycopg2.Error as e:
            print(f"Error creating migration table: {e}")
            raise
    
    def get_migration_files(self) -> List[Tuple[str, Path]]:
        """Get all migration files sorted by version."""
        migration_files = []
        
        if not self.migrations_dir.exists():
            print(f"Migration directory not found: {self.migrations_dir}")
            return migration_files
        
        # Pattern: 001_description.sql, 002_another_migration.sql
        pattern = re.compile(r'^(\d+)_.*\.sql$')
        
        for file_path in self.migrations_dir.glob('*.sql'):
            match = pattern.match(file_path.name)
            if match:
                version = match.group(1)
                migration_files.append((version, file_path))
        
        # Sort by version number
        migration_files.sort(key=lambda x: int(x[0]))
        return migration_files
    
    def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migration versions."""
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version FROM schema_migrations ORDER BY version")
                    return [row[0] for row in cur.fetchall()]
        except psycopg2.Error as e:
            print(f"Error getting applied migrations: {e}")
            return []
    
    def calculate_checksum(self, content: str) -> str:
        """Calculate simple checksum for migration content."""
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def apply_migration(self, version: str, file_path: Path) -> bool:
        """Apply a single migration file."""
        try:
            # Read migration file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            checksum = self.calculate_checksum(content)
            start_time = datetime.now()
            
            print(f"Applying migration {version}: {file_path.name}")
            
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    # Execute migration
                    cur.execute(content)
                    
                    # Record successful migration
                    execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    cur.execute("""
                        INSERT INTO schema_migrations (version, filename, checksum, execution_time_ms)
                        VALUES (%s, %s, %s, %s)
                    """, (version, file_path.name, checksum, execution_time))
                
                conn.commit()
                print(f"✓ Migration {version} applied successfully ({execution_time}ms)")
                return True
                
        except psycopg2.Error as e:
            print(f"✗ Error applying migration {version}: {e}")
            return False
        except Exception as e:
            print(f"✗ Unexpected error with migration {version}: {e}")
            return False
    
    def run_all_migrations(self) -> bool:
        """Run all pending migrations."""
        migration_files = self.get_migration_files()
        applied_migrations = set(self.get_applied_migrations())
        
        if not migration_files:
            print("No migration files found")
            return True
        
        pending_migrations = [
            (version, path) for version, path in migration_files
            if version not in applied_migrations
        ]
        
        if not pending_migrations:
            print("All migrations are up to date")
            return True
        
        print(f"Found {len(pending_migrations)} pending migrations")
        
        success_count = 0
        for version, file_path in pending_migrations:
            if self.apply_migration(version, file_path):
                success_count += 1
            else:
                print(f"Migration {version} failed - stopping")
                return False
        
        print(f"Successfully applied {success_count}/{len(pending_migrations)} migrations")
        return success_count == len(pending_migrations)
    
    def show_migration_status(self):
        """Display current migration status."""
        migration_files = self.get_migration_files()
        applied_migrations = set(self.get_applied_migrations())
        
        print("\nMigration Status:")
        print("-" * 50)
        
        for version, file_path in migration_files:
            status = "✓ Applied" if version in applied_migrations else "⏳ Pending"
            print(f"{version}: {file_path.name} - {status}")
        
        # Show migration history
        try:
            with psycopg2.connect(**self.db_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT version, filename, applied_at, execution_time_ms
                        FROM schema_migrations 
                        ORDER BY applied_at DESC 
                        LIMIT 5
                    """)
                    
                    recent_migrations = cur.fetchall()
                    if recent_migrations:
                        print("\nRecent Migrations:")
                        print("-" * 50)
                        for version, filename, applied_at, exec_time in recent_migrations:
                            print(f"{version}: {filename} - {applied_at} ({exec_time}ms)")
                            
        except psycopg2.Error as e:
            print(f"Could not retrieve migration history: {e}")
    
    def rollback_migration(self, target_version: str) -> bool:
        """Rollback to a specific migration version."""
        print(f"⚠ Rollback functionality not implemented yet")
        print(f"  Target version: {target_version}")
        print(f"  This would require down migrations or manual intervention")
        return False