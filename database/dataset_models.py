"""Dataset metadata models for power market data catalog."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Boolean,
    Column,
    Date,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship

from .models import Base


class DatasetCategory(Base):
    """Categories for organizing datasets."""
    
    __tablename__ = "dataset_categories"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    display_order = Column(Integer, default=0)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="category")


class Dataset(Base):
    """Dataset metadata and documentation."""
    
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(String(100), unique=True, nullable=False)
    iso_id = Column(Integer, ForeignKey("isos.id"))
    category_id = Column(Integer, ForeignKey("dataset_categories.id"))
    name = Column(String(200), nullable=False)
    description = Column(Text)
    table_name = Column(String(100))
    
    # Update tracking
    update_frequency = Column(String(50))  # '5-minute', 'hourly', 'daily'
    typical_delay = Column(String(50))
    earliest_data = Column(Date)
    latest_data = Column(TIMESTAMP(timezone=True))
    last_updated = Column(TIMESTAMP(timezone=True))
    
    # Data properties
    spatial_resolution = Column(String(50))  # 'nodal', 'zonal', 'system'
    temporal_resolution = Column(String(50))  # '5-minute', 'hourly', 'daily'
    data_format = Column(String(50))  # 'time-series', 'snapshot'
    
    # Quality metrics
    completeness_pct = Column(Numeric(5, 2))
    avg_daily_rows = Column(Integer)
    total_rows = Column(Integer)
    
    # Documentation
    notes = Column(Text)
    limitations = Column(Text)
    source_url = Column(Text)
    
    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    updated_at = Column(TIMESTAMP, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    iso = relationship("ISO")
    category = relationship("DatasetCategory", back_populates="datasets")
    columns = relationship("DatasetColumn", back_populates="dataset", order_by="DatasetColumn.display_order")
    tags = relationship("DatasetTag", back_populates="dataset")
    quality_rules = relationship("DataQualityRule", back_populates="dataset")


class DatasetColumn(Base):
    """Column metadata for datasets."""
    
    __tablename__ = "dataset_columns"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    column_name = Column(String(100), nullable=False)
    display_name = Column(String(100))
    data_type = Column(String(50), nullable=False)
    unit = Column(String(50))
    description = Column(Text)
    
    # Column properties
    is_required = Column(Boolean, default=False)
    is_primary_key = Column(Boolean, default=False)
    is_indexed = Column(Boolean, default=False)
    
    # Value constraints
    min_value = Column(Numeric)
    max_value = Column(Numeric)
    allowed_values = Column(ARRAY(String))
    
    # Statistics
    null_count = Column(Integer)
    distinct_count = Column(Integer)
    avg_value = Column(Numeric)
    std_dev = Column(Numeric)
    
    display_order = Column(Integer, default=0)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="columns")
    
    __table_args__ = (
        UniqueConstraint("dataset_id", "column_name"),
    )


class DatasetTag(Base):
    """Tags for dataset search and filtering."""
    
    __tablename__ = "dataset_tags"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    tag = Column(String(50), nullable=False)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="tags")
    
    __table_args__ = (
        UniqueConstraint("dataset_id", "tag"),
    )


class DataQualityRule(Base):
    """Data quality rules for validation."""
    
    __tablename__ = "data_quality_rules"
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    rule_name = Column(String(100), nullable=False)
    rule_type = Column(String(50))  # 'range', 'pattern', 'reference', 'completeness'
    column_name = Column(String(100))
    rule_definition = Column(JSONB)
    severity = Column(String(20))  # 'error', 'warning', 'info'
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="quality_rules")


class DatasetRelationship(Base):
    """Relationships between datasets for joins."""
    
    __tablename__ = "dataset_relationships"
    
    id = Column(Integer, primary_key=True)
    from_dataset_id = Column(Integer, ForeignKey("datasets.id"))
    to_dataset_id = Column(Integer, ForeignKey("datasets.id"))
    relationship_type = Column(String(50))  # 'one-to-many', 'many-to-many'
    from_column = Column(String(100))
    to_column = Column(String(100))
    description = Column(Text)
    
    # Relationships
    from_dataset = relationship("Dataset", foreign_keys=[from_dataset_id])
    to_dataset = relationship("Dataset", foreign_keys=[to_dataset_id])