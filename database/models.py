"""SQLAlchemy models for power market data."""

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
    nodes = relationship("Node", back_populates="iso")
    energy_prices = relationship("EnergyPrice", back_populates="iso")
    ancillary_prices = relationship("AncillaryPrice", back_populates="iso")
    download_history = relationship("DownloadHistory", back_populates="iso")


class Node(Base):
    """Node/Settlement point model."""
    
    __tablename__ = "nodes"
    
    id = Column(Integer, primary_key=True)
    iso_id = Column(Integer, ForeignKey("isos.id"))
    node_id = Column(String(100), nullable=False)
    node_name = Column(String(200))
    node_type = Column(String(50))  # HUB, ZONE, NODE, SETTLEMENT_POINT
    latitude = Column(Numeric(10, 6))
    longitude = Column(Numeric(10, 6))
    voltage_level = Column(Integer)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    iso = relationship("ISO", back_populates="nodes")
    energy_prices = relationship("EnergyPrice", back_populates="node")
    
    __table_args__ = (
        UniqueConstraint("iso_id", "node_id"),
        Index("idx_nodes_iso_node", "iso_id", "node_id"),
        Index("idx_nodes_type", "node_type"),
    )


class EnergyPrice(Base):
    """Energy price time series data."""
    
    __tablename__ = "energy_prices"
    
    timestamp = Column(TIMESTAMP(timezone=True), primary_key=True)
    iso_id = Column(Integer, ForeignKey("isos.id"), primary_key=True)
    node_id = Column(Integer, ForeignKey("nodes.id"), primary_key=True)
    market_type = Column(String(20), nullable=False, primary_key=True)  # DAM, RTM
    lmp = Column(Numeric(10, 2))
    energy_component = Column(Numeric(10, 2))
    congestion_component = Column(Numeric(10, 2))
    loss_component = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    iso = relationship("ISO", back_populates="energy_prices")
    node = relationship("Node", back_populates="energy_prices")
    
    __table_args__ = (
        Index("idx_energy_prices_iso_node_time", "iso_id", "node_id", "timestamp"),
        Index("idx_energy_prices_market_type", "market_type"),
    )


class AncillaryPrice(Base):
    """Ancillary services price data."""
    
    __tablename__ = "ancillary_prices"
    
    timestamp = Column(TIMESTAMP(timezone=True), primary_key=True)
    iso_id = Column(Integer, ForeignKey("isos.id"), primary_key=True)
    service_type = Column(String(50), nullable=False, primary_key=True)
    market_type = Column(String(20), nullable=False, primary_key=True)
    price = Column(Numeric(10, 2))
    quantity_mw = Column(Numeric(10, 2))
    zone = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    iso = relationship("ISO", back_populates="ancillary_prices")
    
    __table_args__ = (
        Index("idx_ancillary_prices_iso_service", "iso_id", "service_type", "timestamp"),
        Index("idx_ancillary_prices_zone", "zone"),
    )


class DownloadHistory(Base):
    """Track data download history and status."""
    
    __tablename__ = "download_history"
    
    id = Column(Integer, primary_key=True)
    iso_id = Column(Integer, ForeignKey("isos.id"))
    data_type = Column(String(50), nullable=False)
    start_timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    end_timestamp = Column(TIMESTAMP(timezone=True), nullable=False)
    download_started_at = Column(DateTime, nullable=False)
    download_completed_at = Column(DateTime)
    status = Column(String(20), nullable=False)  # PENDING, IN_PROGRESS, COMPLETED, FAILED
    error_message = Column(Text)
    file_path = Column(String(500))
    row_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    iso = relationship("ISO", back_populates="download_history")
    
    __table_args__ = (
        Index("idx_download_history_iso_type", "iso_id", "data_type"),
        Index("idx_download_history_status", "status"),
    )


class DataQualityCheck(Base):
    """Data quality monitoring."""
    
    __tablename__ = "data_quality_checks"
    
    id = Column(Integer, primary_key=True)
    check_timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    iso_id = Column(Integer, ForeignKey("isos.id"))
    data_type = Column(String(50))
    check_type = Column(String(100))  # MISSING_DATA, OUTLIER, DUPLICATE
    severity = Column(String(20))  # INFO, WARNING, ERROR
    description = Column(Text)
    affected_start_time = Column(TIMESTAMP(timezone=True))
    affected_end_time = Column(TIMESTAMP(timezone=True))
    resolved = Column(Boolean, default=False)