"""Extended database models for additional power market data types."""

from datetime import date, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP

from .models_v2 import Base


class TransmissionConstraints(Base):
    """Transmission constraints and congestion data."""
    
    __tablename__ = "transmission_constraints"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    constraint_id = Column(String(100), primary_key=True)
    constraint_name = Column(String(200))
    contingency_name = Column(String(200))
    monitored_element = Column(String(200))
    shadow_price = Column(Numeric(10, 2))
    binding_limit = Column(Numeric(10, 2))
    flow_mw = Column(Numeric(10, 2))
    limit_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class Weather(Base):
    """Weather observations data."""
    
    __tablename__ = "weather"
    
    timestamp = Column(TIMESTAMP(timezone=True), primary_key=True)
    iso = Column(String(10), primary_key=True)
    weather_station_id = Column(String(100), primary_key=True)
    location_name = Column(String(200))
    latitude = Column(Numeric(10, 6))
    longitude = Column(Numeric(10, 6))
    temperature_f = Column(Numeric(5, 2))
    dew_point_f = Column(Numeric(5, 2))
    humidity_pct = Column(Numeric(5, 2))
    wind_speed_mph = Column(Numeric(5, 2))
    wind_direction_deg = Column(Integer)
    cloud_cover_pct = Column(Numeric(5, 2))
    pressure_mb = Column(Numeric(6, 2))
    visibility_miles = Column(Numeric(5, 2))
    precipitation_in = Column(Numeric(5, 2))
    created_at = Column(DateTime, default=datetime.utcnow)


class RenewableGenerationForecast(Base):
    """Solar and wind generation forecasts."""
    
    __tablename__ = "renewable_generation_forecast"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    resource_type = Column(String(20), primary_key=True)  # 'solar', 'wind'
    forecast_type = Column(String(50), primary_key=True)  # 'day_ahead', 'hour_ahead', 'real_time'
    region = Column(String(100))
    forecast_mw = Column(Numeric(10, 2))
    capacity_mw = Column(Numeric(10, 2))
    capacity_factor = Column(Numeric(5, 2))
    created_at = Column(DateTime, default=datetime.utcnow)


class CapacityChanges(Base):
    """Generator capacity changes and outages."""
    
    __tablename__ = "capacity_changes"
    
    effective_date = Column(Date, primary_key=True)
    iso = Column(String(10), primary_key=True)
    unit_name = Column(String(200), primary_key=True)
    resource_type = Column(String(50))
    fuel_type = Column(String(50))
    change_type = Column(String(50))  # 'planned_outage', 'forced_outage', 'derate', 'return_to_service'
    capacity_change_mw = Column(Numeric(10, 2))
    total_capacity_mw = Column(Numeric(10, 2))
    expected_return_date = Column(Date)
    reason = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class DemandResponse(Base):
    """Demand response events and performance."""
    
    __tablename__ = "demand_response"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    program_name = Column(String(100))
    zone = Column(String(100))
    event_type = Column(String(50))  # 'economic', 'reliability', 'emergency'
    dispatched_mw = Column(Numeric(10, 2))
    actual_mw = Column(Numeric(10, 2))
    price_per_mwh = Column(Numeric(10, 2))
    participants = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class StorageOperations(Base):
    """Battery storage operations data."""
    
    __tablename__ = "storage_operations"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    resource_id = Column(String(100), primary_key=True)
    resource_name = Column(String(200))
    zone = Column(String(100))
    charging_mw = Column(Numeric(10, 2))
    discharging_mw = Column(Numeric(10, 2))
    state_of_charge_mwh = Column(Numeric(10, 2))
    capacity_mwh = Column(Numeric(10, 2))
    efficiency_pct = Column(Numeric(5, 2))
    created_at = Column(DateTime, default=datetime.utcnow)


class Emissions(Base):
    """Emissions and carbon intensity data."""
    
    __tablename__ = "emissions"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    zone = Column(String(100))
    co2_tons = Column(Numeric(12, 2))
    co2_intensity_lb_per_mwh = Column(Numeric(10, 2))
    nox_tons = Column(Numeric(10, 2))
    so2_tons = Column(Numeric(10, 2))
    marginal_fuel_type = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class Curtailment(Base):
    """Renewable energy curtailment data."""
    
    __tablename__ = "curtailment"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    resource_type = Column(String(20), primary_key=True)  # 'solar', 'wind', 'nuclear'
    zone = Column(String(100))
    curtailed_mw = Column(Numeric(10, 2))
    economic_curtailment_mw = Column(Numeric(10, 2))
    manual_curtailment_mw = Column(Numeric(10, 2))
    reason = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)


class VirtualPowerPlant(Base):
    """Aggregated distributed energy resources."""
    
    __tablename__ = "virtual_power_plant"
    
    interval_start = Column(TIMESTAMP(timezone=True), primary_key=True)
    interval_end = Column(TIMESTAMP(timezone=True), nullable=False)
    iso = Column(String(10), primary_key=True)
    aggregator_name = Column(String(200))
    zone = Column(String(100))
    resource_type = Column(String(50))  # 'residential_solar', 'ev_charging', 'battery', 'hvac'
    registered_capacity_mw = Column(Numeric(10, 2))
    available_capacity_mw = Column(Numeric(10, 2))
    dispatched_mw = Column(Numeric(10, 2))
    participant_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class SupplyStack(Base):
    """Market supply stack data."""
    
    __tablename__ = "supply_stack"
    
    snapshot_date = Column(Date, primary_key=True)
    iso = Column(String(10), primary_key=True)
    fuel_type = Column(String(50), primary_key=True)
    price_range_min = Column(Numeric(10, 2))
    price_range_max = Column(Numeric(10, 2))
    capacity_mw = Column(Numeric(10, 2))
    heat_rate_btu_per_kwh = Column(Numeric(10, 2))
    variable_om_cost = Column(Numeric(10, 2))
    created_at = Column(DateTime, default=datetime.utcnow)