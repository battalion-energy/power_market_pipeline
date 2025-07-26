-- TimescaleDB schema for power market data
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ISO metadata table
CREATE TABLE IF NOT EXISTS isos (
    id SERIAL PRIMARY KEY,
    code VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    timezone VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert ISO data
INSERT INTO isos (code, name, timezone) VALUES 
    ('ERCOT', 'Electric Reliability Council of Texas', 'US/Central'),
    ('CAISO', 'California Independent System Operator', 'US/Pacific'),
    ('ISONE', 'ISO New England', 'US/Eastern'),
    ('NYISO', 'New York Independent System Operator', 'US/Eastern')
ON CONFLICT (code) DO NOTHING;

-- Nodes/Settlement Points table
CREATE TABLE IF NOT EXISTS nodes (
    id SERIAL PRIMARY KEY,
    iso_id INTEGER REFERENCES isos(id),
    node_id VARCHAR(100) NOT NULL,
    node_name VARCHAR(200),
    node_type VARCHAR(50), -- 'HUB', 'ZONE', 'NODE', 'SETTLEMENT_POINT'
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    voltage_level INTEGER,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(iso_id, node_id)
);

CREATE INDEX idx_nodes_iso_node ON nodes(iso_id, node_id);
CREATE INDEX idx_nodes_type ON nodes(node_type);

-- Energy prices table (Day-ahead and Real-time)
CREATE TABLE IF NOT EXISTS energy_prices (
    timestamp TIMESTAMPTZ NOT NULL,
    iso_id INTEGER REFERENCES isos(id),
    node_id INTEGER REFERENCES nodes(id),
    market_type VARCHAR(20) NOT NULL, -- 'DAM' (day-ahead), 'RTM' (real-time)
    lmp DECIMAL(10, 2), -- Locational Marginal Price
    energy_component DECIMAL(10, 2),
    congestion_component DECIMAL(10, 2),
    loss_component DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('energy_prices', 'timestamp', if_not_exists => TRUE);

-- Create indexes for performance
CREATE INDEX idx_energy_prices_iso_node_time ON energy_prices(iso_id, node_id, timestamp DESC);
CREATE INDEX idx_energy_prices_market_type ON energy_prices(market_type);

-- Ancillary services prices table
CREATE TABLE IF NOT EXISTS ancillary_prices (
    timestamp TIMESTAMPTZ NOT NULL,
    iso_id INTEGER REFERENCES isos(id),
    service_type VARCHAR(50) NOT NULL, -- 'SPIN', 'NON_SPIN', 'REGULATION_UP', 'REGULATION_DOWN', etc.
    market_type VARCHAR(20) NOT NULL, -- 'DAM', 'RTM'
    price DECIMAL(10, 2),
    quantity_mw DECIMAL(10, 2),
    zone VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Convert to hypertable
SELECT create_hypertable('ancillary_prices', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX idx_ancillary_prices_iso_service ON ancillary_prices(iso_id, service_type, timestamp DESC);
CREATE INDEX idx_ancillary_prices_zone ON ancillary_prices(zone);

-- Data download tracking table
CREATE TABLE IF NOT EXISTS download_history (
    id SERIAL PRIMARY KEY,
    iso_id INTEGER REFERENCES isos(id),
    data_type VARCHAR(50) NOT NULL, -- 'ENERGY_DAM', 'ENERGY_RTM', 'ANCILLARY_DAM', etc.
    start_timestamp TIMESTAMPTZ NOT NULL,
    end_timestamp TIMESTAMPTZ NOT NULL,
    download_started_at TIMESTAMP NOT NULL,
    download_completed_at TIMESTAMP,
    status VARCHAR(20) NOT NULL, -- 'PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED'
    error_message TEXT,
    file_path VARCHAR(500),
    row_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_download_history_iso_type ON download_history(iso_id, data_type);
CREATE INDEX idx_download_history_status ON download_history(status);

-- Aggregated hourly data for faster queries
CREATE MATERIALIZED VIEW energy_prices_hourly AS
SELECT 
    time_bucket('1 hour', timestamp) AS hour,
    iso_id,
    node_id,
    market_type,
    AVG(lmp) as avg_lmp,
    MIN(lmp) as min_lmp,
    MAX(lmp) as max_lmp,
    AVG(energy_component) as avg_energy,
    AVG(congestion_component) as avg_congestion,
    AVG(loss_component) as avg_loss,
    COUNT(*) as data_points
FROM energy_prices
GROUP BY hour, iso_id, node_id, market_type;

-- Create refresh policy for materialized view
CREATE OR REPLACE FUNCTION refresh_energy_prices_hourly()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY energy_prices_hourly;
END;
$$ LANGUAGE plpgsql;

-- Hub prices view for common queries
CREATE VIEW hub_prices AS
SELECT 
    ep.timestamp,
    i.code as iso_code,
    n.node_name as hub_name,
    ep.market_type,
    ep.lmp,
    ep.energy_component,
    ep.congestion_component,
    ep.loss_component
FROM energy_prices ep
JOIN nodes n ON ep.node_id = n.id
JOIN isos i ON ep.iso_id = i.id
WHERE n.node_type = 'HUB';

-- Data quality checks
CREATE TABLE IF NOT EXISTS data_quality_checks (
    id SERIAL PRIMARY KEY,
    check_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    iso_id INTEGER REFERENCES isos(id),
    data_type VARCHAR(50),
    check_type VARCHAR(100), -- 'MISSING_DATA', 'OUTLIER', 'DUPLICATE', etc.
    severity VARCHAR(20), -- 'INFO', 'WARNING', 'ERROR'
    description TEXT,
    affected_start_time TIMESTAMPTZ,
    affected_end_time TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT false
);

-- Compression policy for older data
SELECT add_compression_policy('energy_prices', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('ancillary_prices', INTERVAL '7 days', if_not_exists => TRUE);