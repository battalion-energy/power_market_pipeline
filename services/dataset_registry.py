"""Dataset registry service for managing dataset metadata."""

import json
from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from database.dataset_models import (
    Dataset,
    DatasetCategory,
    DatasetColumn,
    DatasetTag,
    DataQualityRule,
)


class DatasetRegistry:
    """Service for managing dataset metadata and catalog."""
    
    def __init__(self):
        self.dataset_definitions = self._load_dataset_definitions()
    
    def _load_dataset_definitions(self) -> Dict:
        """Load dataset definitions from configuration."""
        return {
            "ercot_lmp_by_bus_dam": {
                "name": "ERCOT Day-Ahead LMP by Bus",
                "description": "Day-ahead market locational marginal prices for all ERCOT electrical buses including energy, congestion, and loss components",
                "category": "energy_prices",
                "table_name": "energy_prices",
                "update_frequency": "daily",
                "typical_delay": "1 day",
                "spatial_resolution": "nodal",
                "temporal_resolution": "hourly",
                "data_format": "time-series",
                "columns": [
                    {
                        "column_name": "timestamp",
                        "display_name": "Timestamp",
                        "data_type": "timestamp",
                        "description": "Hour ending timestamp in ERCOT local time (CPT/CDT)",
                        "is_required": True,
                        "is_primary_key": True,
                        "display_order": 1
                    },
                    {
                        "column_name": "node_id",
                        "display_name": "Node ID",
                        "data_type": "string",
                        "description": "Electrical bus identifier",
                        "is_required": True,
                        "is_primary_key": True,
                        "is_indexed": True,
                        "display_order": 2
                    },
                    {
                        "column_name": "lmp",
                        "display_name": "LMP",
                        "data_type": "decimal",
                        "unit": "$/MWh",
                        "description": "Locational Marginal Price",
                        "min_value": -250,
                        "max_value": 9000,
                        "display_order": 3
                    },
                    {
                        "column_name": "energy_component",
                        "display_name": "Energy Component",
                        "data_type": "decimal",
                        "unit": "$/MWh",
                        "description": "Energy component of LMP",
                        "display_order": 4
                    },
                    {
                        "column_name": "congestion_component",
                        "display_name": "Congestion Component",
                        "data_type": "decimal",
                        "unit": "$/MWh",
                        "description": "Congestion component of LMP",
                        "display_order": 5
                    },
                    {
                        "column_name": "loss_component",
                        "display_name": "Loss Component",
                        "data_type": "decimal",
                        "unit": "$/MWh",
                        "description": "Loss component of LMP",
                        "display_order": 6
                    }
                ],
                "tags": ["dam", "energy", "prices", "nodal", "ercot"],
                "quality_rules": [
                    {
                        "rule_name": "lmp_range_check",
                        "rule_type": "range",
                        "column_name": "lmp",
                        "rule_definition": {"min": -250, "max": 9000},
                        "severity": "error"
                    },
                    {
                        "rule_name": "component_sum_check",
                        "rule_type": "calculation",
                        "rule_definition": {
                            "formula": "abs(lmp - (energy_component + congestion_component + loss_component)) < 0.01"
                        },
                        "severity": "warning"
                    }
                ]
            },
            "ercot_lmp_by_bus_rtm": {
                "name": "ERCOT Real-Time LMP by Bus",
                "description": "Real-time market locational marginal prices for all ERCOT electrical buses at 5-minute intervals",
                "category": "energy_prices",
                "table_name": "energy_prices",
                "update_frequency": "5-minute",
                "typical_delay": "5 minutes",
                "spatial_resolution": "nodal",
                "temporal_resolution": "5-minute",
                "data_format": "time-series",
                "columns": [
                    # Similar to DAM but with 5-minute intervals
                ],
                "tags": ["rtm", "energy", "prices", "nodal", "ercot", "real-time"]
            },
            # Add more dataset definitions...
        }
    
    def register_dataset(self, iso_code: str, dataset_key: str) -> Dataset:
        """Register a dataset in the catalog."""
        if dataset_key not in self.dataset_definitions:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        definition = self.dataset_definitions[dataset_key]
        
        with get_db() as db:
            # Get ISO
            from database.models import ISO
            iso = db.query(ISO).filter(ISO.code == iso_code).first()
            if not iso:
                raise ValueError(f"ISO {iso_code} not found")
            
            # Get or create category
            category = db.query(DatasetCategory).filter(
                DatasetCategory.name == definition["category"]
            ).first()
            
            # Check if dataset exists
            dataset_id = f"{iso_code.lower()}_{dataset_key.replace(iso_code.lower() + '_', '')}"
            dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            
            if not dataset:
                # Create dataset
                dataset = Dataset(
                    dataset_id=dataset_id,
                    iso_id=iso.id,
                    category_id=category.id if category else None,
                    name=definition["name"],
                    description=definition["description"],
                    table_name=definition["table_name"],
                    update_frequency=definition["update_frequency"],
                    typical_delay=definition["typical_delay"],
                    spatial_resolution=definition["spatial_resolution"],
                    temporal_resolution=definition["temporal_resolution"],
                    data_format=definition["data_format"]
                )
                db.add(dataset)
                db.flush()
                
                # Add columns
                for col_def in definition.get("columns", []):
                    column = DatasetColumn(
                        dataset_id=dataset.id,
                        **col_def
                    )
                    db.add(column)
                
                # Add tags
                for tag in definition.get("tags", []):
                    tag_obj = DatasetTag(
                        dataset_id=dataset.id,
                        tag=tag
                    )
                    db.add(tag_obj)
                
                # Add quality rules
                for rule_def in definition.get("quality_rules", []):
                    rule = DataQualityRule(
                        dataset_id=dataset.id,
                        rule_definition=rule_def.get("rule_definition"),
                        **{k: v for k, v in rule_def.items() if k != "rule_definition"}
                    )
                    db.add(rule)
                
                db.commit()
            
            return dataset
    
    def update_dataset_statistics(self, dataset_id: str):
        """Update dataset statistics from actual data."""
        with get_db() as db:
            dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
            if not dataset:
                return
            
            # Update based on table type
            if dataset.table_name == "energy_prices":
                from database.models import EnergyPrice
                
                # Get statistics
                stats = db.query(
                    func.count(EnergyPrice.timestamp).label("total_rows"),
                    func.min(EnergyPrice.timestamp).label("earliest"),
                    func.max(EnergyPrice.timestamp).label("latest")
                ).filter(
                    EnergyPrice.iso_id == dataset.iso_id
                ).first()
                
                if stats:
                    dataset.total_rows = stats.total_rows
                    dataset.earliest_data = stats.earliest.date() if stats.earliest else None
                    dataset.latest_data = stats.latest
                    dataset.last_updated = datetime.utcnow()
                    
                    # Calculate completeness
                    if dataset.earliest_data and dataset.latest_data:
                        expected_rows = self._calculate_expected_rows(
                            dataset.earliest_data,
                            dataset.latest_data,
                            dataset.temporal_resolution,
                            dataset.spatial_resolution
                        )
                        if expected_rows > 0:
                            dataset.completeness_pct = min(
                                100.0, 
                                (stats.total_rows / expected_rows) * 100
                            )
                
                # Update column statistics
                for column in dataset.columns:
                    if column.column_name == "lmp":
                        col_stats = db.query(
                            func.count(EnergyPrice.lmp).label("non_null_count"),
                            func.avg(EnergyPrice.lmp).label("avg_value"),
                            func.stddev(EnergyPrice.lmp).label("std_dev")
                        ).filter(
                            EnergyPrice.iso_id == dataset.iso_id
                        ).first()
                        
                        if col_stats:
                            column.null_count = stats.total_rows - (col_stats.non_null_count or 0)
                            column.avg_value = col_stats.avg_value
                            column.std_dev = col_stats.std_dev
                
                db.commit()
    
    def _calculate_expected_rows(
        self,
        start_date,
        end_date,
        temporal_resolution: str,
        spatial_resolution: str
    ) -> int:
        """Calculate expected number of rows based on resolution."""
        # This is a simplified calculation
        days = (end_date - start_date).days + 1
        
        # Intervals per day
        if temporal_resolution == "5-minute":
            intervals_per_day = 288  # 24 * 12
        elif temporal_resolution == "hourly":
            intervals_per_day = 24
        elif temporal_resolution == "daily":
            intervals_per_day = 1
        else:
            intervals_per_day = 24
        
        # Number of locations (approximate)
        if spatial_resolution == "nodal":
            locations = 5000  # Approximate for ERCOT
        elif spatial_resolution == "zonal":
            locations = 10
        else:
            locations = 1
        
        return days * intervals_per_day * locations
    
    def get_dataset_catalog(self, iso_code: Optional[str] = None) -> List[Dict]:
        """Get catalog of available datasets."""
        with get_db() as db:
            query = db.query(Dataset).filter(Dataset.is_active == True)
            
            if iso_code:
                from database.models import ISO
                iso = db.query(ISO).filter(ISO.code == iso_code).first()
                if iso:
                    query = query.filter(Dataset.iso_id == iso.id)
            
            datasets = query.all()
            
            catalog = []
            for dataset in datasets:
                catalog.append({
                    "dataset_id": dataset.dataset_id,
                    "name": dataset.name,
                    "description": dataset.description,
                    "iso": dataset.iso.code if dataset.iso else None,
                    "category": dataset.category.name if dataset.category else None,
                    "update_frequency": dataset.update_frequency,
                    "spatial_resolution": dataset.spatial_resolution,
                    "temporal_resolution": dataset.temporal_resolution,
                    "earliest_data": dataset.earliest_data.isoformat() if dataset.earliest_data else None,
                    "latest_data": dataset.latest_data.isoformat() if dataset.latest_data else None,
                    "completeness_pct": float(dataset.completeness_pct) if dataset.completeness_pct else None,
                    "total_rows": dataset.total_rows,
                    "tags": [tag.tag for tag in dataset.tags]
                })
            
            return catalog