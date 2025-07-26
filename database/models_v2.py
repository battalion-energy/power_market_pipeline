"""Standardized database models for power market data."""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ISO(Base):
    """ISO/RTO entity model."""
    
    __tablename__ = "isos"
    
    id = Column(Integer, primary_key=True)
    code = Column(String(10), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    timezone = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    locations = relationship("Location", back_populates="iso")


class Location(Base):
    """Unified location model for all ISOs."""
    
    __tablename__ = "locations"
    
    id = Column(Integer, primary_key=True)
    iso_id = Column(Integer, ForeignKey("isos.id"))
    location_id = Column(String(100), nullable=False)
    location_name = Column(String(200))
    location_type = Column(String(50))  # hub, zone, node, interface, generator
    latitude = Column(Numeric(10, 6))
    longitude = Column(Numeric(10, 6))
    state = Column(String(2))
    county = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    iso = relationship("ISO", back_populates="locations")
    
    __table_args__ = (
        UniqueConstraint("iso_id", "location_id"),
        Index("idx_locations_iso_location", "iso_id", "location_id"),
        Index("idx_locations_type", "location_type"),
    )


class LMP(Base):
    """Standardized LMP data model."""
    
    __tablename__ = "lmp"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    location = Column(String(100), primary_key=True)
    location_type = Column(String(50))
    market = Column(String(10), primary_key=True)  # DAM, RT5M, RT15M, HASP
    lmp = Column(Numeric(10, 2))
    energy = Column(Numeric(10, 2))
    congestion = Column(Numeric(10, 2))
    loss = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_lmp_iso_location_time", "iso", "location", "interval_start"),
        Index("idx_lmp_market", "market"),
        Index("idx_lmp_location_type", "location_type"),
    )


class AncillaryServices(Base):
    """Standardized ancillary services model."""
    
    __tablename__ = "ancillary_services"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    region = Column(String(100), primary_key=True)
    market = Column(String(10), primary_key=True)
    product = Column(String(50), primary_key=True)
    clearing_price = Column(Numeric(10, 2))
    clearing_quantity = Column(Numeric(10, 2))
    requirement = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_as_iso_product_time", "iso", "product", "interval_start"),
        Index("idx_as_region", "region"),
    )


class Load(Base):
    """Load actual and forecast data."""
    
    __tablename__ = "load"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    load_area = Column(String(100), primary_key=True)
    forecast_type = Column(String(50), primary_key=True)  # actual, forecast_1h, forecast_dam
    load_mw = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)


class GenerationFuelMix(Base):
    """Generation by fuel type."""
    
    __tablename__ = "generation_fuel_mix"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    fuel_type = Column(String(50), primary_key=True)
    generation_mw = Column(Numeric(10, 2))
    percentage = Column(Numeric(5, 2))
    created_at = Column(DateTime, default=datetime.utcnow)


class InterconnectionFlow(Base):
    """Power flow between ISOs."""
    
    __tablename__ = "interconnection_flow"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    from_iso = Column(String(10), primary_key=True)
    to_iso = Column(String(10), primary_key=True)
    interface_name = Column(String(100))
    flow_mw = Column(Numeric(10, 2))  # Positive = export, Negative = import
    limit_mw = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)


class DataCatalog(Base):
    """Data catalog for dataset metadata."""
    
    __tablename__ = "data_catalog"
    
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(100), unique=True, nullable=False)
    table_name = Column(String(100), nullable=False)
    iso = Column(String(10))
    description = Column(Text)
    
    # Update info
    update_frequency = Column(String(50))
    last_updated = Column(TIMESTAMP)
    earliest_data = Column(DateTime)
    latest_data = Column(DateTime)
    
    # Data properties
    spatial_granularity = Column(String(50))  # nodal, zonal, system
    temporal_granularity = Column(String(50))  # 5min, hourly, daily
    
    # Access info
    is_public = Column(Boolean, default=True)
    requires_auth = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    columns = relationship("DataCatalogColumn", back_populates="dataset")


class DataCatalogColumn(Base):
    """Column definitions for datasets."""
    
    __tablename__ = "data_catalog_columns"
    
    id = Column(Integer, primary_key=True)
    dataset_name = Column(String(100), ForeignKey("data_catalog.dataset_name"))
    column_name = Column(String(100), nullable=False)
    data_type = Column(String(50), nullable=False)
    unit = Column(String(50))
    description = Column(Text)
    is_required = Column(Boolean, default=False)
    display_order = Column(Integer, default=0)
    
    # Relationships
    dataset = relationship("DataCatalog", back_populates="columns")
    
    __table_args__ = (
        UniqueConstraint("dataset_name", "column_name"),
    )