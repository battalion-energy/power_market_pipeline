-- ISO-Specific Views for Easy Data Access
-- Makes querying energy markets as simple as "SELECT * FROM ercot_realtime"

-- ==========================================
-- ERCOT VIEWS  
-- ==========================================

-- ERCOT Real-time pricing (5-minute intervals)
CREATE OR REPLACE VIEW ercot_realtime AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as settlement_point,
    location_type,
    lmp as price_mwh,
    energy as energy_component,
    congestion as congestion_component, 
    loss as loss_component,
    (congestion + loss) as basis_mwh,  -- Useful calculated field
    created_at as data_inserted
FROM lmp
WHERE iso = 'ERCOT' 
  AND market = 'RT5M'
ORDER BY interval_start DESC, location;

-- ERCOT Day-ahead pricing (hourly intervals)
CREATE OR REPLACE VIEW ercot_dayahead AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as settlement_point,
    location_type,
    lmp as price_mwh,
    energy as energy_component,
    congestion as congestion_component,
    loss as loss_component,
    (congestion + loss) as basis_mwh,
    created_at as data_inserted
FROM lmp
WHERE iso = 'ERCOT'
  AND market = 'DAM'
ORDER BY interval_start DESC, location;

-- ERCOT Hubs only (most commonly requested)
CREATE OR REPLACE VIEW ercot_hubs_realtime AS
SELECT * FROM ercot_realtime 
WHERE location_type = 'hub'
ORDER BY timestamp DESC, settlement_point;

CREATE OR REPLACE VIEW ercot_hubs_dayahead AS
SELECT * FROM ercot_dayahead
WHERE location_type = 'hub' 
ORDER BY timestamp DESC, settlement_point;

-- ==========================================
-- CAISO VIEWS
-- ==========================================

-- CAISO Real-time pricing
CREATE OR REPLACE VIEW caiso_realtime AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as node_name,
    location_type,
    lmp as price_mwh,
    energy as energy_component,
    congestion as congestion_component,
    loss as loss_component,
    (congestion + loss) as basis_mwh,
    created_at as data_inserted
FROM lmp
WHERE iso = 'CAISO'
  AND market = 'RT5M'
ORDER BY interval_start DESC, location;

-- CAISO Day-ahead pricing  
CREATE OR REPLACE VIEW caiso_dayahead AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as node_name,
    location_type,
    lmp as price_mwh,
    energy as energy_component,
    congestion as congestion_component,
    loss as loss_component,
    (congestion + loss) as basis_mwh,
    created_at as data_inserted
FROM lmp
WHERE iso = 'CAISO'
  AND market = 'DAM'
ORDER BY interval_start DESC, location;

-- CAISO Trading Hubs only
CREATE OR REPLACE VIEW caiso_hubs_realtime AS
SELECT * FROM caiso_realtime
WHERE location_type = 'hub'
ORDER BY timestamp DESC, node_name;

CREATE OR REPLACE VIEW caiso_hubs_dayahead AS
SELECT * FROM caiso_dayahead
WHERE location_type = 'hub'
ORDER BY timestamp DESC, node_name;

-- ==========================================
-- ISO-NE VIEWS
-- ==========================================

-- ISO-NE Real-time pricing
CREATE OR REPLACE VIEW isone_realtime AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as node_name,
    location_type,
    lmp as price_mwh,
    energy as energy_component,
    congestion as congestion_component,
    loss as loss_component,
    (congestion + loss) as basis_mwh,
    created_at as data_inserted
FROM lmp
WHERE iso = 'ISONE'
  AND market = 'RT5M'
ORDER BY interval_start DESC, location;

-- ISO-NE Day-ahead pricing
CREATE OR REPLACE VIEW isone_dayahead AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as node_name,
    location_type,
    lmp as price_mwh,
    energy as energy_component,
    congestion as congestion_component,
    loss as loss_component,
    (congestion + loss) as basis_mwh,
    created_at as data_inserted
FROM lmp
WHERE iso = 'ISONE'
  AND market = 'DAM'
ORDER BY interval_start DESC, location;

-- ==========================================
-- NYISO VIEWS
-- ==========================================

-- NYISO Real-time pricing (LBMP = Locational Based Marginal Pricing)
CREATE OR REPLACE VIEW nyiso_realtime AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as zone_name,
    location_type,
    lmp as lbmp_mwh,  -- NYISO calls it LBMP
    energy as energy_component,
    congestion as congestion_component,
    loss as loss_component,
    (congestion + loss) as basis_mwh,
    created_at as data_inserted
FROM lmp
WHERE iso = 'NYISO'
  AND market = 'RT5M'
ORDER BY interval_start DESC, location;

-- NYISO Day-ahead pricing
CREATE OR REPLACE VIEW nyiso_dayahead AS
SELECT 
    interval_start as timestamp,
    interval_end,
    location as zone_name,
    location_type,
    lmp as lbmp_mwh,
    energy as energy_component,
    congestion as congestion_component,
    loss as loss_component,
    (congestion + loss) as basis_mwh,
    created_at as data_inserted
FROM lmp
WHERE iso = 'NYISO'
  AND market = 'DAM'
ORDER BY interval_start DESC, location;

-- ==========================================
-- CROSS-ISO COMPARISON VIEWS
-- ==========================================

-- All ISOs Real-time Hub Prices (for comparison dashboards)
CREATE OR REPLACE VIEW all_hubs_realtime AS
SELECT 
    iso,
    interval_start as timestamp,
    location as hub_name,
    lmp as price_mwh,
    CASE 
        WHEN iso = 'ERCOT' THEN 'Central'
        WHEN iso = 'CAISO' THEN 'Pacific' 
        WHEN iso = 'ISONE' THEN 'Eastern'
        WHEN iso = 'NYISO' THEN 'Eastern'
        ELSE 'Unknown'
    END as timezone_region
FROM lmp
WHERE market = 'RT5M' 
  AND location_type = 'hub'
ORDER BY timestamp DESC, iso, hub_name;

-- All ISOs Day-ahead Hub Prices
CREATE OR REPLACE VIEW all_hubs_dayahead AS
SELECT 
    iso,
    interval_start as timestamp,
    location as hub_name,
    lmp as price_mwh,
    CASE 
        WHEN iso = 'ERCOT' THEN 'Central'
        WHEN iso = 'CAISO' THEN 'Pacific'
        WHEN iso = 'ISONE' THEN 'Eastern' 
        WHEN iso = 'NYISO' THEN 'Eastern'
        ELSE 'Unknown'
    END as timezone_region
FROM lmp
WHERE market = 'DAM'
  AND location_type = 'hub'
ORDER BY timestamp DESC, iso, hub_name;

-- ==========================================
-- CONVENIENCE VIEWS FOR COMMON QUERIES
-- ==========================================

-- Latest prices across all ISOs (for live dashboards)
CREATE OR REPLACE VIEW latest_prices AS
SELECT DISTINCT ON (iso, location, market)
    iso,
    location,
    location_type,
    market,
    interval_start as latest_timestamp,
    lmp as current_price_mwh,
    (EXTRACT(EPOCH FROM (NOW() - interval_start)) / 60)::INTEGER as minutes_old
FROM lmp
WHERE interval_start >= NOW() - INTERVAL '4 hours'  -- Only recent data
ORDER BY iso, location, market, interval_start DESC;

-- Price volatility by ISO (last 24 hours)
CREATE OR REPLACE VIEW price_volatility_24h AS
SELECT 
    iso,
    location,
    location_type,
    market,
    COUNT(*) as data_points,
    AVG(lmp) as avg_price,
    MIN(lmp) as min_price,
    MAX(lmp) as max_price,
    (MAX(lmp) - MIN(lmp)) as price_range,
    STDDEV(lmp) as price_stddev,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lmp) as median_price
FROM lmp
WHERE interval_start >= NOW() - INTERVAL '24 hours'
GROUP BY iso, location, location_type, market
ORDER BY price_stddev DESC;

-- Today's price summary by ISO
CREATE OR REPLACE VIEW todays_price_summary AS
SELECT 
    iso,
    market,
    location_type,
    COUNT(DISTINCT location) as locations_count,
    AVG(lmp) as avg_price_mwh,
    MIN(lmp) as min_price_mwh,
    MAX(lmp) as max_price_mwh,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lmp) as median_price_mwh
FROM lmp
WHERE DATE(interval_start) = CURRENT_DATE
GROUP BY iso, market, location_type
ORDER BY iso, market, location_type;

-- Comment documentation for each view
COMMENT ON VIEW ercot_realtime IS 'ERCOT real-time market pricing (5-minute intervals) with calculated basis field';
COMMENT ON VIEW ercot_dayahead IS 'ERCOT day-ahead market pricing (hourly intervals) with calculated basis field';
COMMENT ON VIEW ercot_hubs_realtime IS 'ERCOT real-time pricing for trading hubs only';
COMMENT ON VIEW ercot_hubs_dayahead IS 'ERCOT day-ahead pricing for trading hubs only';
COMMENT ON VIEW caiso_realtime IS 'CAISO real-time market pricing (5-minute intervals)';
COMMENT ON VIEW caiso_dayahead IS 'CAISO day-ahead market pricing (hourly intervals)';
COMMENT ON VIEW isone_realtime IS 'ISO-NE real-time market pricing (5-minute intervals)';
COMMENT ON VIEW isone_dayahead IS 'ISO-NE day-ahead market pricing (hourly intervals)';
COMMENT ON VIEW nyiso_realtime IS 'NYISO real-time LBMP pricing (5-minute intervals)';
COMMENT ON VIEW nyiso_dayahead IS 'NYISO day-ahead LBMP pricing (hourly intervals)';
COMMENT ON VIEW all_hubs_realtime IS 'Cross-ISO comparison of real-time hub prices';
COMMENT ON VIEW all_hubs_dayahead IS 'Cross-ISO comparison of day-ahead hub prices';
COMMENT ON VIEW latest_prices IS 'Most recent price data across all ISOs with data freshness indicator';
COMMENT ON VIEW price_volatility_24h IS 'Price volatility metrics for the last 24 hours by ISO and location';
COMMENT ON VIEW todays_price_summary IS 'Daily price summary statistics by ISO and market type';