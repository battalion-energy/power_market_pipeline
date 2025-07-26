-- Power Market Pipeline Extended Schema V3
-- Additional tables for comprehensive ISO data collection

-- Transmission constraints
CREATE TABLE IF NOT EXISTS transmission_constraints (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    constraint_id VARCHAR(100) NOT NULL,
    constraint_name VARCHAR(200),
    contingency_name VARCHAR(200),
    monitored_element VARCHAR(200),
    shadow_price DECIMAL(10, 2),
    binding_limit DECIMAL(10, 2),
    flow_mw DECIMAL(10, 2),
    limit_type VARCHAR(50), -- 'thermal', 'voltage', 'stability'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('transmission_constraints', 'interval_start', if_not_exists => TRUE);
CREATE INDEX idx_constraints_iso_time ON transmission_constraints(iso, interval_start DESC);

-- Weather data
CREATE TABLE IF NOT EXISTS weather (
    timestamp TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    weather_station_id VARCHAR(100) NOT NULL,
    location_name VARCHAR(200),
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    temperature_f DECIMAL(5, 2),
    dew_point_f DECIMAL(5, 2),
    humidity_pct DECIMAL(5, 2),
    wind_speed_mph DECIMAL(5, 2),
    wind_direction_deg INTEGER,
    cloud_cover_pct DECIMAL(5, 2),
    pressure_mb DECIMAL(6, 2),
    visibility_miles DECIMAL(5, 2),
    precipitation_in DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('weather', 'timestamp', if_not_exists => TRUE);
CREATE INDEX idx_weather_iso_station ON weather(iso, weather_station_id, timestamp DESC);

-- Solar and wind generation forecasts
CREATE TABLE IF NOT EXISTS renewable_generation_forecast (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    resource_type VARCHAR(20) NOT NULL, -- 'solar', 'wind'
    forecast_type VARCHAR(50) NOT NULL, -- 'day_ahead', 'hour_ahead', 'real_time'
    region VARCHAR(100),
    forecast_mw DECIMAL(10, 2),
    capacity_mw DECIMAL(10, 2),
    capacity_factor DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('renewable_generation_forecast', 'interval_start', if_not_exists => TRUE);

-- Capacity and outages
CREATE TABLE IF NOT EXISTS capacity_changes (
    effective_date DATE NOT NULL,
    iso VARCHAR(10) NOT NULL,
    unit_name VARCHAR(200) NOT NULL,
    resource_type VARCHAR(50),
    fuel_type VARCHAR(50),
    change_type VARCHAR(50), -- 'planned_outage', 'forced_outage', 'derate', 'return_to_service'
    capacity_change_mw DECIMAL(10, 2),
    total_capacity_mw DECIMAL(10, 2),
    expected_return_date DATE,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_capacity_changes_iso_date ON capacity_changes(iso, effective_date DESC);

-- Demand response
CREATE TABLE IF NOT EXISTS demand_response (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    program_name VARCHAR(100),
    zone VARCHAR(100),
    event_type VARCHAR(50), -- 'economic', 'reliability', 'emergency'
    dispatched_mw DECIMAL(10, 2),
    actual_mw DECIMAL(10, 2),
    price_per_mwh DECIMAL(10, 2),
    participants INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('demand_response', 'interval_start', if_not_exists => TRUE);

-- Storage (battery) operations
CREATE TABLE IF NOT EXISTS storage_operations (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    resource_id VARCHAR(100) NOT NULL,
    resource_name VARCHAR(200),
    zone VARCHAR(100),
    charging_mw DECIMAL(10, 2), -- Positive for charging
    discharging_mw DECIMAL(10, 2), -- Positive for discharging
    state_of_charge_mwh DECIMAL(10, 2),
    capacity_mwh DECIMAL(10, 2),
    efficiency_pct DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('storage_operations', 'interval_start', if_not_exists => TRUE);

-- Emissions and carbon intensity
CREATE TABLE IF NOT EXISTS emissions (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    zone VARCHAR(100),
    co2_tons DECIMAL(12, 2),
    co2_intensity_lb_per_mwh DECIMAL(10, 2),
    nox_tons DECIMAL(10, 2),
    so2_tons DECIMAL(10, 2),
    marginal_fuel_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('emissions', 'interval_start', if_not_exists => TRUE);

-- Curtailment data
CREATE TABLE IF NOT EXISTS curtailment (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    resource_type VARCHAR(20) NOT NULL, -- 'solar', 'wind', 'nuclear'
    zone VARCHAR(100),
    curtailed_mw DECIMAL(10, 2),
    economic_curtailment_mw DECIMAL(10, 2),
    manual_curtailment_mw DECIMAL(10, 2),
    reason VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('curtailment', 'interval_start', if_not_exists => TRUE);

-- Virtual power plant (aggregated DER)
CREATE TABLE IF NOT EXISTS virtual_power_plant (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    aggregator_name VARCHAR(200),
    zone VARCHAR(100),
    resource_type VARCHAR(50), -- 'residential_solar', 'ev_charging', 'battery', 'hvac'
    registered_capacity_mw DECIMAL(10, 2),
    available_capacity_mw DECIMAL(10, 2),
    dispatched_mw DECIMAL(10, 2),
    participant_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('virtual_power_plant', 'interval_start', if_not_exists => TRUE);

-- Market fundamentals (supply stack)
CREATE TABLE IF NOT EXISTS supply_stack (
    snapshot_date DATE NOT NULL,
    iso VARCHAR(10) NOT NULL,
    fuel_type VARCHAR(50) NOT NULL,
    price_range_min DECIMAL(10, 2),
    price_range_max DECIMAL(10, 2),
    capacity_mw DECIMAL(10, 2),
    heat_rate_btu_per_kwh DECIMAL(10, 2),
    variable_om_cost DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_supply_stack_iso_date ON supply_stack(iso, snapshot_date DESC);

-- Update data catalog with new datasets
INSERT INTO data_catalog (dataset_name, table_name, iso, description, update_frequency, spatial_granularity, temporal_granularity) VALUES
    ('ercot_load_forecast', 'load', 'ERCOT', 'ERCOT load forecast and actuals', '5min', 'system', '5min'),
    ('ercot_generation_fuel', 'generation_fuel_mix', 'ERCOT', 'ERCOT generation by fuel type', '5min', 'system', '5min'),
    ('ercot_transmission_constraints', 'transmission_constraints', 'ERCOT', 'ERCOT transmission constraints and shadow prices', '5min', 'constraint', '5min'),
    ('ercot_renewable_forecast', 'renewable_generation_forecast', 'ERCOT', 'ERCOT wind and solar generation forecasts', 'hourly', 'system', 'hourly'),
    ('ercot_storage', 'storage_operations', 'ERCOT', 'ERCOT battery storage operations', 'hourly', 'resource', 'hourly'),
    ('caiso_load_forecast', 'load', 'CAISO', 'CAISO load forecast and actuals', '5min', 'system', '5min'),
    ('caiso_generation_fuel', 'generation_fuel_mix', 'CAISO', 'CAISO generation by fuel type', '5min', 'system', '5min'),
    ('caiso_renewable_forecast', 'renewable_generation_forecast', 'CAISO', 'CAISO renewable generation forecasts', '5min', 'system', '5min'),
    ('caiso_emissions', 'emissions', 'CAISO', 'CAISO carbon emissions and intensity', '5min', 'system', '5min'),
    ('isone_load_forecast', 'load', 'ISONE', 'ISO-NE load forecast and actuals', '5min', 'system', '5min'),
    ('isone_generation_fuel', 'generation_fuel_mix', 'ISONE', 'ISO-NE generation by fuel type', '5min', 'system', '5min'),
    ('nyiso_load_forecast', 'load', 'NYISO', 'NYISO load forecast and actuals', '5min', 'zonal', '5min'),
    ('nyiso_generation_fuel', 'generation_fuel_mix', 'NYISO', 'NYISO generation by fuel type', '5min', 'system', '5min')
ON CONFLICT (dataset_name) DO NOTHING;

-- Add compression policies for new tables
SELECT add_compression_policy('transmission_constraints', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('weather', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_compression_policy('renewable_generation_forecast', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('demand_response', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('storage_operations', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('emissions', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('curtailment', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('virtual_power_plant', INTERVAL '7 days', if_not_exists => TRUE);