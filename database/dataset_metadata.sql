-- Dataset metadata tables for power market pipeline
-- Similar to GridStatus approach for documenting datasets

-- Dataset categories
CREATE TABLE IF NOT EXISTS dataset_categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    display_order INTEGER DEFAULT 0
);

INSERT INTO dataset_categories (name, description, display_order) VALUES
    ('energy_prices', 'Locational Marginal Prices (LMP) for energy', 1),
    ('ancillary_services', 'Ancillary services prices and quantities', 2),
    ('load_forecast', 'Load forecasts and actuals', 3),
    ('generation', 'Generation by fuel type and unit', 4),
    ('transmission', 'Transmission constraints and flows', 5),
    ('weather', 'Weather data for key locations', 6)
ON CONFLICT (name) DO NOTHING;

-- Dataset definitions
CREATE TABLE IF NOT EXISTS datasets (
    id SERIAL PRIMARY KEY,
    dataset_id VARCHAR(100) UNIQUE NOT NULL, -- e.g., 'ercot_lmp_by_bus_dam'
    iso_id INTEGER REFERENCES isos(id),
    category_id INTEGER REFERENCES dataset_categories(id),
    name VARCHAR(200) NOT NULL,
    description TEXT,
    table_name VARCHAR(100), -- Physical table/view name
    
    -- Update tracking
    update_frequency VARCHAR(50), -- '5-minute', 'hourly', 'daily'
    typical_delay VARCHAR(50), -- '5 minutes', '1 hour', etc.
    earliest_data DATE,
    latest_data TIMESTAMP,
    last_updated TIMESTAMP,
    
    -- Data properties
    spatial_resolution VARCHAR(50), -- 'nodal', 'zonal', 'system'
    temporal_resolution VARCHAR(50), -- '5-minute', 'hourly', 'daily'
    data_format VARCHAR(50), -- 'time-series', 'snapshot'
    
    -- Quality metrics
    completeness_pct DECIMAL(5,2),
    avg_daily_rows INTEGER,
    total_rows BIGINT,
    
    -- Documentation
    notes TEXT,
    limitations TEXT,
    source_url TEXT,
    
    -- Metadata
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dataset columns metadata
CREATE TABLE IF NOT EXISTS dataset_columns (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    column_name VARCHAR(100) NOT NULL,
    display_name VARCHAR(100),
    data_type VARCHAR(50) NOT NULL,
    unit VARCHAR(50), -- '$/MWh', 'MW', etc.
    description TEXT,
    
    -- Column properties
    is_required BOOLEAN DEFAULT false,
    is_primary_key BOOLEAN DEFAULT false,
    is_indexed BOOLEAN DEFAULT false,
    
    -- Value constraints
    min_value DECIMAL,
    max_value DECIMAL,
    allowed_values TEXT[], -- For categorical columns
    
    -- Statistics (updated periodically)
    null_count BIGINT,
    distinct_count BIGINT,
    avg_value DECIMAL,
    std_dev DECIMAL,
    
    display_order INTEGER DEFAULT 0,
    UNIQUE(dataset_id, column_name)
);

-- Dataset relationships (for joins)
CREATE TABLE IF NOT EXISTS dataset_relationships (
    id SERIAL PRIMARY KEY,
    from_dataset_id INTEGER REFERENCES datasets(id),
    to_dataset_id INTEGER REFERENCES datasets(id),
    relationship_type VARCHAR(50), -- 'one-to-many', 'many-to-many'
    from_column VARCHAR(100),
    to_column VARCHAR(100),
    description TEXT
);

-- Dataset tags for search/filtering
CREATE TABLE IF NOT EXISTS dataset_tags (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    tag VARCHAR(50) NOT NULL,
    UNIQUE(dataset_id, tag)
);

-- Data quality rules
CREATE TABLE IF NOT EXISTS data_quality_rules (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50), -- 'range', 'pattern', 'reference', 'completeness'
    column_name VARCHAR(100),
    rule_definition JSONB, -- Flexible rule storage
    severity VARCHAR(20), -- 'error', 'warning', 'info'
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dataset access logs for usage tracking
CREATE TABLE IF NOT EXISTS dataset_access_logs (
    id SERIAL PRIMARY KEY,
    dataset_id INTEGER REFERENCES datasets(id),
    access_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_type VARCHAR(50), -- 'api', 'download', 'query'
    user_id VARCHAR(100),
    row_count INTEGER,
    response_time_ms INTEGER
);

-- Example: Insert ERCOT DAM LMP dataset metadata
INSERT INTO datasets (
    dataset_id,
    iso_id,
    category_id,
    name,
    description,
    table_name,
    update_frequency,
    typical_delay,
    spatial_resolution,
    temporal_resolution,
    data_format
) VALUES (
    'ercot_lmp_by_bus_dam',
    (SELECT id FROM isos WHERE code = 'ERCOT'),
    (SELECT id FROM dataset_categories WHERE name = 'energy_prices'),
    'ERCOT Day-Ahead LMP by Bus',
    'Day-ahead market locational marginal prices for all ERCOT electrical buses including energy, congestion, and loss components',
    'energy_prices',
    'daily',
    '1 day',
    'nodal',
    'hourly',
    'time-series'
) ON CONFLICT (dataset_id) DO NOTHING;

-- Create views for easy dataset access
CREATE OR REPLACE VIEW v_dataset_catalog AS
SELECT 
    d.dataset_id,
    i.code as iso,
    dc.name as category,
    d.name,
    d.description,
    d.update_frequency,
    d.spatial_resolution,
    d.temporal_resolution,
    d.earliest_data,
    d.latest_data,
    d.completeness_pct,
    d.total_rows,
    array_agg(DISTINCT dt.tag) as tags
FROM datasets d
JOIN isos i ON d.iso_id = i.id
JOIN dataset_categories dc ON d.category_id = dc.id
LEFT JOIN dataset_tags dt ON d.id = dt.dataset_id
WHERE d.is_active = true
GROUP BY d.id, i.code, dc.name;

-- Function to update dataset statistics
CREATE OR REPLACE FUNCTION update_dataset_statistics(p_dataset_id VARCHAR)
RETURNS void AS $$
DECLARE
    v_dataset RECORD;
    v_row_count BIGINT;
    v_latest_timestamp TIMESTAMP;
    v_earliest_timestamp TIMESTAMP;
BEGIN
    -- Get dataset info
    SELECT * INTO v_dataset 
    FROM datasets 
    WHERE dataset_id = p_dataset_id;
    
    IF v_dataset.table_name = 'energy_prices' THEN
        -- Get statistics for energy prices
        SELECT 
            COUNT(*),
            MAX(timestamp),
            MIN(timestamp)
        INTO v_row_count, v_latest_timestamp, v_earliest_timestamp
        FROM energy_prices
        WHERE iso_id = v_dataset.iso_id;
        
        -- Update dataset record
        UPDATE datasets
        SET 
            total_rows = v_row_count,
            latest_data = v_latest_timestamp,
            earliest_data = v_earliest_timestamp::date,
            updated_at = CURRENT_TIMESTAMP
        WHERE dataset_id = p_dataset_id;
    END IF;
    
    -- Add similar logic for other tables
END;
$$ LANGUAGE plpgsql;

-- Trigger to track dataset access
CREATE OR REPLACE FUNCTION log_dataset_access()
RETURNS trigger AS $$
BEGIN
    -- This would be called from your application layer
    -- when datasets are accessed via API
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;