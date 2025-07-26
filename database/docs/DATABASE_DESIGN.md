# Power Market Pipeline Database Design

## 🏗️ **Design Philosophy: World-Class Time-Series Architecture**

This database schema represents a **world-class approach to energy market data** that prioritizes:
- **Temporal-first design** with consistent interval patterns
- **Cross-ISO standardization** while preserving domain terminology
- **Meta-programmable structure** enabling dynamic query generation
- **Scalable performance** through TimescaleDB hypertables
- **User-centric access patterns** via intuitive views

## 🎯 **Core Design Principles**

### 1. **Standardized Temporal Model**
Every table follows the same interval pattern:
```sql
interval_start TIMESTAMPTZ NOT NULL,
interval_end TIMESTAMPTZ NOT NULL,
```

**Why this matters:**
- Handles both instantaneous and period-based data
- Timezone-aware for global deployments
- Consistent query patterns across all data types
- Natural handling of market schedule changes (DST, holidays)

### 2. **Unified Column Naming Convention**
```sql
iso VARCHAR(10) NOT NULL,        -- Always 'iso' (ERCOT, CAISO, etc.)
location VARCHAR(100) NOT NULL,  -- Always 'location' (standardized IDs)
market VARCHAR(10) NOT NULL,     -- Always 'market' (DAM, RT5M, etc.)
```

This enables **powerful meta-programming**:
```python
# Single function works across ALL data types and ISOs
def query_market_data(table, iso, location, market, start, end):
    return f"""
    SELECT * FROM {table} 
    WHERE iso = '{iso}' AND location = '{location}' 
    AND market = '{market}'
    AND interval_start >= '{start}' AND interval_end <= '{end}'
    """
```

### 3. **Self-Documenting Data Catalog**
The `data_catalog` table provides rich metadata:
- Dataset descriptions and update frequencies
- Spatial/temporal granularity information  
- Column definitions with units and descriptions
- Data lineage and quality metrics

This enables **automatic API generation**, **validation**, and **documentation**.

### 4. **Performance-Optimized Indexing**
```sql
-- Optimized for common query patterns
CREATE INDEX idx_lmp_iso_location_time ON lmp(iso, location, interval_start DESC);
CREATE INDEX idx_lmp_market ON lmp(market);
```

Designed for **sub-second response times** even with billions of rows.

## 🚀 **TimescaleDB Superpower Features**

### Hypertables for Infinite Scale
```sql
SELECT create_hypertable('lmp', 'interval_start');
```
- **Automatic time-based partitioning**
- **Parallel query execution** across time chunks
- **Handles decades of high-frequency data** without performance degradation

### Intelligent Compression
```sql
SELECT add_compression_policy('lmp', INTERVAL '7 days');
```
- **Automatic compression** of older data
- **90%+ storage savings** for historical data
- **Transparent query access** - no application changes needed

### Built-in Aggregation Views
```sql
CREATE MATERIALIZED VIEW v_lmp_hourly AS
SELECT 
    time_bucket('1 hour', interval_start) AS hour,
    AVG(lmp), MIN(lmp), MAX(lmp),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lmp) as median_lmp
FROM lmp GROUP BY hour, iso, location;
```

**Dashboard queries run in milliseconds** instead of seconds.

## 🎨 **ISO-Specific Views: User Experience Excellence**

### The Problem We Solved
Traditional energy databases force users to remember:
- ISO codes (`'ERCOT'`, `'CAISO'`, `'ISONE'`, `'NYISO'`)
- Market codes (`'DAM'`, `'RT5M'`, `'RT15M'`)  
- Location hierarchies (nodes vs hubs vs zones)
- Column mappings across different ISOs

### Our Solution: Intention-Driven Data Access

**Instead of this complexity:**
```sql
SELECT interval_start, location, lmp, energy, congestion, loss
FROM lmp 
WHERE iso = 'ERCOT' AND market = 'RT5M' AND location_type = 'hub'
  AND interval_start >= NOW() - INTERVAL '1 week';
```

**Users simply write:**
```sql
SELECT * FROM ercot_hubs_realtime 
WHERE timestamp >= NOW() - INTERVAL '1 week';
```

### Domain-Specific Terminology
Each view uses the **actual terminology** that market participants use:

- **ERCOT**: `settlement_point`, `price_mwh`
- **NYISO**: `zone_name`, `lbmp_mwh` (Locational Based Marginal Price)
- **CAISO**: `node_name`, `price_mwh`
- **ISO-NE**: `node_name`, `price_mwh`

### Built-in Calculations
Common trader calculations are pre-computed:
```sql
-- Basis spread calculation automatically included
(congestion + loss) as basis_mwh
```

## 📊 **Real-World Query Examples**

### Trading Floor Queries
```sql
-- "What are current hub prices across all markets?"
SELECT iso, hub_name, current_price_mwh, minutes_old
FROM latest_prices 
WHERE location_type = 'hub' AND minutes_old < 10
ORDER BY current_price_mwh DESC;
```

### Risk Management
```sql
-- "Which markets are experiencing high volatility?"
SELECT iso, market, price_stddev, price_range
FROM price_volatility_24h
WHERE price_stddev > 20
ORDER BY price_stddev DESC;
```

### Cross-ISO Analysis
```sql
-- "Compare day-ahead vs real-time spreads"
SELECT 
    da.iso, da.hub_name,
    da.price_mwh as da_price,
    rt.price_mwh as rt_price,
    (rt.price_mwh - da.price_mwh) as spread
FROM all_hubs_dayahead da
JOIN all_hubs_realtime rt USING (iso, hub_name, timestamp)
WHERE da.timestamp >= CURRENT_DATE;
```

## 🔧 **Meta-Programming Capabilities**

### Dynamic Query Generation
```python
# Generic function works for ANY ISO/market combination
def get_price_data(iso, market_type, location=None, days=30):
    view_name = f"{iso.lower()}_{market_type}"
    query = f"SELECT * FROM {view_name}"
    if location:
        query += f" WHERE settlement_point = '{location}'"
    query += f" AND timestamp >= NOW() - INTERVAL '{days} days'"
    return execute_query(query)

# Works across all markets:
houston_rt = get_price_data('ERCOT', 'realtime', 'HB_HOUSTON', 7)
caiso_da = get_price_data('CAISO', 'dayahead', 'SP15_GEN_HUB', 30)
```

### Automatic API Generation
```python
# Views enable zero-config API endpoints
@app.route('/api/<iso>/<market_type>')
def get_market_data(iso, market_type):
    view_name = f"{iso.lower()}_{market_type}"
    return query_view(view_name, request.args)
```

### Data Quality Monitoring
```python
# Automatic gap detection across all datasets
for dataset in get_all_datasets():
    expected_intervals = generate_intervals(dataset.temporal_granularity)
    actual_intervals = query_intervals(dataset.table_name)
    gaps = expected_intervals - actual_intervals
    alert_if_gaps(dataset.dataset_name, gaps)
```

## 🌟 **Why This Design Will Stand the Test of Time**

### 1. **ISO-Agnostic Foundation**
The core schema works for **any electricity market globally**:
- ERCOT, CAISO, ISONE, NYISO (implemented)
- MISO, PJM, SPP (ready to add)
- European markets (NORDPOOL, EPEX) (future)
- Asian markets (AEMO, JEPX) (future)

### 2. **Data-Type Extensible**
New data types slot in seamlessly:
- Renewable generation forecasts
- Transmission constraints
- Ancillary service requirements
- Carbon pricing
- Weather correlations

### 3. **Technology Evolution Ready**
- **Database agnostic**: Core concepts work with PostgreSQL, ClickHouse, BigQuery
- **API evolution**: Views provide stable interfaces as schemas evolve
- **Scale transitions**: Starts with single server, scales to distributed clusters

### 4. **User Experience Focus**
As markets evolve and new users join:
- **Zero learning curve** for domain experts
- **Self-documenting** through intuitive naming
- **Progressive complexity** - simple queries are simple, complex analysis is possible

## 🎯 **The Meta-Programming Sweet Spot**

This schema achieves the rare combination of being:
- **Simple enough** for analysts to write ad-hoc queries
- **Structured enough** for robust application development
- **Flexible enough** for unknown future requirements  
- **Rich enough** for advanced analytics and ML workflows

**Result**: Your database becomes a **programmable energy data platform**, not just storage.

## 🏆 **Best Practices Demonstrated**

1. **Temporal data modeling** with consistent interval patterns
2. **Cross-domain standardization** while preserving terminology
3. **Performance optimization** through proper indexing and partitioning
4. **User experience design** applied to database schemas
5. **Meta-programming enablement** through consistent naming conventions
6. **Self-documentation** through catalog tables and view comments
7. **Future-proofing** through abstraction layers (views)
8. **Domain expertise integration** in schema and terminology choices

This represents **decades of energy market data experience** distilled into a cohesive, scalable, and maintainable design.