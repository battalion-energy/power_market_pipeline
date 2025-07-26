-- Power Market Pipeline Schema V2
-- Standardized schema with consistent column naming conventions

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ISO metadata (unchanged)
CREATE TABLE IF NOT EXISTS isos (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Unified locations table (replaces nodes)
CREATE TABLE IF NOT EXISTS locations (
    id SERIAL PRIMARY KEY,
    iso_id INTEGER REFERENCES isos(id),
    location_id VARCHAR(100) NOT NULL, -- Standard ID across all ISOs
    location_name VARCHAR(200),
    location_type VARCHAR(50), -- 'hub', 'zone', 'node', 'interface', 'generator'
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    state VARCHAR(2),
    county VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(iso_id, location_id)
);

CREATE INDEX idx_locations_iso_location ON locations(iso_id, location_id);
CREATE INDEX idx_locations_type ON locations(location_type);

-- Standardized LMP table (following GridStatus pattern)
CREATE TABLE IF NOT EXISTS lmp (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    location VARCHAR(100) NOT NULL,
    location_type VARCHAR(50),
    market VARCHAR(10) NOT NULL, -- 'DAM', 'RT5M', 'RT15M', 'HASP'
    lmp DECIMAL(10, 2),
    energy DECIMAL(10, 2),
    congestion DECIMAL(10, 2),
    loss DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Convert to hypertable
SELECT create_hypertable('lmp', 'interval_start', if_not_exists => TRUE);

-- Create optimized indexes for common query patterns
CREATE INDEX idx_lmp_iso_location_time ON lmp(iso, location, interval_start DESC);
CREATE INDEX idx_lmp_market ON lmp(market);
CREATE INDEX idx_lmp_location_type ON lmp(location_type);

-- Standardized ancillary services table
CREATE TABLE IF NOT EXISTS ancillary_services (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    region VARCHAR(100), -- Can be zone, system, or specific region
    market VARCHAR(10) NOT NULL, -- 'DAM', 'RTM'
    product VARCHAR(50) NOT NULL, -- 'REGUP', 'REGDOWN', 'SPIN', 'NON_SPIN', etc.
    clearing_price DECIMAL(10, 2),
    clearing_quantity DECIMAL(10, 2),
    requirement DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Convert to hypertable
SELECT create_hypertable('ancillary_services', 'interval_start', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX idx_as_iso_product_time ON ancillary_services(iso, product, interval_start DESC);
CREATE INDEX idx_as_region ON ancillary_services(region);

-- Load and forecast data
CREATE TABLE IF NOT EXISTS load (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    load_area VARCHAR(100),
    forecast_type VARCHAR(50), -- 'actual', 'forecast_1h', 'forecast_dam', etc.
    load_mw DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('load', 'interval_start', if_not_exists => TRUE);

-- Generation by fuel type
CREATE TABLE IF NOT EXISTS generation_fuel_mix (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    iso VARCHAR(10) NOT NULL,
    fuel_type VARCHAR(50), -- 'solar', 'wind', 'natural_gas', 'nuclear', etc.
    generation_mw DECIMAL(10, 2),
    percentage DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('generation_fuel_mix', 'interval_start', if_not_exists => TRUE);

-- Interconnection flow
CREATE TABLE IF NOT EXISTS interconnection_flow (
    interval_start TIMESTAMPTZ NOT NULL,
    interval_end TIMESTAMPTZ NOT NULL,
    from_iso VARCHAR(10) NOT NULL,
    to_iso VARCHAR(10) NOT NULL,
    interface_name VARCHAR(100),
    flow_mw DECIMAL(10, 2), -- Positive = export, Negative = import
    limit_mw DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

SELECT create_hypertable('interconnection_flow', 'interval_start', if_not_exists => TRUE);

-- Data catalog for dataset metadata
CREATE TABLE IF NOT EXISTS data_catalog (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) UNIQUE NOT NULL, -- e.g., 'ercot_lmp_dam'
    table_name VARCHAR(100) NOT NULL,
    iso VARCHAR(10),
    description TEXT,
    
    -- Update info
    update_frequency VARCHAR(50),
    last_updated TIMESTAMP,
    earliest_data DATE,
    latest_data DATE,
    
    -- Data properties
    spatial_granularity VARCHAR(50), -- 'nodal', 'zonal', 'system'
    temporal_granularity VARCHAR(50), -- '5min', 'hourly', 'daily'
    
    -- Access info
    is_public BOOLEAN DEFAULT true,
    requires_auth BOOLEAN DEFAULT false,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Column definitions for each dataset
CREATE TABLE IF NOT EXISTS data_catalog_columns (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) REFERENCES data_catalog(dataset_name),
    column_name VARCHAR(100) NOT NULL,
    data_type VARCHAR(50) NOT NULL,
    unit VARCHAR(50),
    description TEXT,
    is_required BOOLEAN DEFAULT false,
    display_order INTEGER DEFAULT 0,
    UNIQUE(dataset_name, column_name)
);

-- Insert ISO data
INSERT INTO isos (code, name, timezone) VALUES 
    ('ERCOT', 'Electric Reliability Council of Texas', 'US/Central'),
    ('CAISO', 'California Independent System Operator', 'US/Pacific'),
    ('ISONE', 'ISO New England', 'US/Eastern'),
    ('NYISO', 'New York Independent System Operator', 'US/Eastern'),
    ('MISO', 'Midcontinent Independent System Operator', 'US/Central'),
    ('PJM', 'PJM Interconnection', 'US/Eastern'),
    ('SPP', 'Southwest Power Pool', 'US/Central')
ON CONFLICT (code) DO NOTHING;

-- Insert data catalog entries
INSERT INTO data_catalog (dataset_name, table_name, iso, description, update_frequency, spatial_granularity, temporal_granularity) VALUES
    ('ercot_lmp_dam', 'lmp', 'ERCOT', 'ERCOT Day-Ahead Market LMP by location', 'daily', 'nodal', 'hourly'),
    ('ercot_lmp_rtm', 'lmp', 'ERCOT', 'ERCOT Real-Time Market LMP by location', '5min', 'nodal', '5min'),
    ('ercot_as_dam', 'ancillary_services', 'ERCOT', 'ERCOT Day-Ahead Ancillary Services', 'daily', 'system', 'hourly'),
    ('caiso_lmp_dam', 'lmp', 'CAISO', 'CAISO Day-Ahead Market LMP by node', 'daily', 'nodal', 'hourly'),
    ('caiso_lmp_rtm', 'lmp', 'CAISO', 'CAISO Real-Time Market LMP by node', '5min', 'nodal', '5min'),
    ('isone_lmp_dam', 'lmp', 'ISONE', 'ISO-NE Day-Ahead Energy Market LMP', 'daily', 'nodal', 'hourly'),
    ('isone_lmp_rtm', 'lmp', 'ISONE', 'ISO-NE Real-Time Energy Market LMP', '5min', 'nodal', '5min'),
    ('nyiso_lmp_dam', 'lmp', 'NYISO', 'NYISO Day-Ahead Market LBMP by zone', 'daily', 'zonal', 'hourly'),
    ('nyiso_lmp_rtm', 'lmp', 'NYISO', 'NYISO Real-Time Market LBMP by zone', '5min', 'zonal', '5min')
ON CONFLICT (dataset_name) DO NOTHING;

-- Insert column definitions for LMP datasets
INSERT INTO data_catalog_columns (dataset_name, column_name, data_type, unit, description, is_required, display_order) VALUES
    ('ercot_lmp_dam', 'interval_start', 'timestamp', NULL, 'Start of the hour interval', true, 1),
    ('ercot_lmp_dam', 'interval_end', 'timestamp', NULL, 'End of the hour interval', true, 2),
    ('ercot_lmp_dam', 'location', 'string', NULL, 'Settlement point or hub name', true, 3),
    ('ercot_lmp_dam', 'location_type', 'string', NULL, 'Type of location (hub, zone, node)', true, 4),
    ('ercot_lmp_dam', 'lmp', 'float', '$/MWh', 'Locational Marginal Price', true, 5),
    ('ercot_lmp_dam', 'energy', 'float', '$/MWh', 'Energy component of LMP', false, 6),
    ('ercot_lmp_dam', 'congestion', 'float', '$/MWh', 'Congestion component of LMP', false, 7),
    ('ercot_lmp_dam', 'loss', 'float', '$/MWh', 'Loss component of LMP', false, 8)
ON CONFLICT (dataset_name, column_name) DO NOTHING;

-- Views for easy data access
CREATE OR REPLACE VIEW v_lmp_latest AS
SELECT DISTINCT ON (iso, location, market)
    interval_start,
    interval_end,
    iso,
    location,
    location_type,
    market,
    lmp,
    energy,
    congestion,
    loss
FROM lmp
ORDER BY iso, location, market, interval_start DESC;

-- Hourly aggregation view
CREATE MATERIALIZED VIEW IF NOT EXISTS v_lmp_hourly AS
SELECT 
    time_bucket('1 hour', interval_start) AS hour,
    iso,
    location,
    location_type,
    market,
    AVG(lmp) as avg_lmp,
    MIN(lmp) as min_lmp,
    MAX(lmp) as max_lmp,
    AVG(energy) as avg_energy,
    AVG(congestion) as avg_congestion,
    AVG(loss) as avg_loss,
    COUNT(*) as data_points
FROM lmp
GROUP BY hour, iso, location, location_type, market;

-- Daily aggregation view
CREATE MATERIALIZED VIEW IF NOT EXISTS v_lmp_daily AS
SELECT 
    time_bucket('1 day', interval_start) AS day,
    iso,
    location,
    location_type,
    market,
    AVG(lmp) as avg_lmp,
    MIN(lmp) as min_lmp,
    MAX(lmp) as max_lmp,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lmp) as median_lmp,
    COUNT(*) as data_points
FROM lmp
GROUP BY day, iso, location, location_type, market;

-- Compression policies
SELECT add_compression_policy('lmp', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('ancillary_services', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('load', INTERVAL '7 days', if_not_exists => TRUE);