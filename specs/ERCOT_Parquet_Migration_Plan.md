# ERCOT Parquet Migration Plan for BESS Revenue Leaderboard

## Executive Summary

This document outlines the comprehensive migration plan to transition the BESS revenue calculation and leaderboard system from CSV-based processing to Parquet-based processing. The migration leverages the annual rollup Parquet files created by the Rust processor, providing 10-100x performance improvements and 90% storage reduction.

## Current State Assessment

### Existing Parquet Files (from Annual Rollup)

#### Price Data (Complete)
```
/Users/enrico/data/ERCOT_data/rollup_files/
├── RT_prices/               # Real-time Settlement Point Prices
│   ├── 2011-2025.parquet   # 15-minute intervals
│   └── schema.json
├── DA_prices/               # Day-Ahead Market Prices  
│   ├── 2010-2025.parquet   # Hourly
│   └── schema.json
└── AS_prices/               # Ancillary Services Prices
    ├── 2010-2025.parquet    # Hourly by service type
    └── schema.json
```

#### 60-Day Disclosure Data (In Progress)
```
├── DAM_Gen_Resources/       # Day-Ahead Awards
│   ├── 2019-2025.parquet   # Hourly awards
│   └── schema.json
├── SCED_Gen_Resources/      # Real-Time Dispatch
│   ├── 2019-2025.parquet   # 5-minute dispatch
│   └── schema.json
├── COP_Snapshots/           # Operating Plan & SOC
│   ├── 2014-2025.parquet   # Hourly snapshots
│   └── schema.json
└── SASM_AS_Offers/          # AS Market Offers
    ├── 2019-2025.parquet    # Daily AS offers
    └── schema.json
```

## Column Schema Evolution

### Critical BESS Columns by Year

#### 2019-2020: Early BESS Era
```json
{
  "Resource Name": "Utf8",
  "Resource Type": "Utf8",           // Filter: = 'PWRSTR'
  "Settlement Point": "Utf8",
  "Awarded Quantity": "Float64",     // DAM energy awards
  "RegUp": "Float64",                // Regulation Up AS
  "RegDown": "Float64",              // Regulation Down AS  
  "RRS": "Float64",                  // Responsive Reserve
  "NSPIN": "Float64"                 // Non-Spinning Reserve
}
```

#### 2021: Enhanced SOC Tracking
```json
{
  // Previous columns plus:
  "Maximum SOC": "Float64",          // Max state of charge (MWh)
  "Minimum SOC": "Float64",          // Min state of charge (MWh)
  "Hour Beginning Planned SOC": "Float64",  // Planned SOC
  "Beginning SOC": "Float64"         // Actual SOC at hour start
}
```

#### 2022: ECRS Introduction
```json
{
  // Previous columns plus:
  "ECRS": "Float64",                 // Emergency Contingency Reserve
  "ECRSM": "Float64",                // ECRS Make-Whole
  "ECRSS": "Float64",                // ECRS Self-Schedule
  "Hour Ending": "Utf8"              // Changed from Int64 to Utf8
}
```

#### 2023-2024: Market Maturation
```json
{
  // Previous columns plus:
  "Telemetered Output": "Float64",   // Actual MW output
  "Base Point": "Float64",           // Dispatch instruction
  "HDL": "Float64",                  // High Dispatch Limit
  "LDL": "Float64",                  // Low Dispatch Limit
  "Resource Status": "Utf8"          // ON/OFF/STARTUP/SHUTDOWN
}
```

## Migration Implementation Plan

### Phase 1: Schema Validation (Week 1)

1. **Create Schema Registry**
```python
# Generate comprehensive schema documentation
schemas = {
    'RT_prices': load_schema('/rollup_files/RT_prices/schema.json'),
    'DA_prices': load_schema('/rollup_files/DA_prices/schema.json'),
    'AS_prices': load_schema('/rollup_files/AS_prices/schema.json'),
    'DAM_Gen_Resources': load_schema('/rollup_files/DAM_Gen_Resources/schema.json'),
    'SCED_Gen_Resources': load_schema('/rollup_files/SCED_Gen_Resources/schema.json'),
    'COP_Snapshots': load_schema('/rollup_files/COP_Snapshots/schema.json')
}
```

2. **Document Column Availability**
- Create matrix of column availability by year
- Note when columns were added/removed
- Handle missing columns gracefully in queries

### Phase 2: BESS Resource Identification (Week 1)

1. **Extract BESS Registry from Parquet**
```python
import pyarrow.parquet as pq
import pyarrow.compute as pc

# Read COP snapshots to find all BESS resources
cop_2024 = pq.read_table('/rollup_files/COP_Snapshots/2024.parquet')
bess_filter = pc.equal(cop_2024['Resource Type'], 'PWRSTR')
bess_resources = cop_2024.filter(bess_filter)['Resource Name'].unique()

# Save BESS registry
bess_registry = {
    'resource_name': resource,
    'settlement_point': extract_settlement_point(resource),
    'capacity_mw': extract_capacity(resource),
    'duration_hours': 4,  # Default assumption
    'first_active_date': find_first_appearance(resource)
}
```

### Phase 3: Revenue Calculator Migration (Week 2)

1. **Port Energy Revenue Calculation**
```python
def calculate_energy_revenue_parquet(resource_name, year):
    # Read Parquet files directly
    dam_awards = pq.read_table(f'/DAM_Gen_Resources/{year}.parquet',
                               filters=[('Resource Name', '=', resource_name)])
    
    dam_prices = pq.read_table(f'/DA_prices/{year}.parquet',
                               filters=[('SettlementPoint', '=', settlement_point)])
    
    # Join on datetime and calculate revenue
    dam_revenue = dam_awards['Awarded Quantity'] * dam_prices['SettlementPointPrice']
    
    # Similar for RT with 15-minute settlement
    rt_dispatch = pq.read_table(f'/SCED_Gen_Resources/{year}.parquet',
                                filters=[('Resource Name', '=', resource_name)])
    
    rt_prices = pq.read_table(f'/RT_prices/{year}.parquet',
                              filters=[('SettlementPointName', '=', settlement_point)])
    
    # RT revenue = (Telemetered Output - DAM Award) * RT Price
    rt_revenue = (rt_dispatch['Telemetered Output'] - dam_award) * rt_prices['SettlementPointPrice']
    
    return dam_revenue + rt_revenue
```

2. **Port AS Revenue Calculation**
```python
def calculate_as_revenue_parquet(resource_name, year):
    # Read AS awards and prices
    as_awards = pq.read_table(f'/DAM_Gen_Resources/{year}.parquet',
                              filters=[('Resource Name', '=', resource_name)])
    
    as_prices = pq.read_table(f'/AS_prices/{year}.parquet')
    
    # Calculate revenue for each AS product
    revenues = {}
    for service in ['RegUp', 'RegDown', 'RRS', 'ECRS', 'NSPIN']:
        if service in as_awards.column_names:
            award = as_awards[service]
            price = as_prices.filter(pc.equal(as_prices['AncillaryType'], service))['MCPC']
            revenues[service] = (award * price).sum()
    
    return revenues
```

### Phase 4: Performance Leaderboard (Week 2)

1. **Create Aggregated Leaderboard Table**
```python
def generate_leaderboard_parquet(year, month=None):
    # Process all BESS resources in parallel
    results = []
    
    for resource in bess_registry:
        energy_rev = calculate_energy_revenue_parquet(resource, year)
        as_rev = calculate_as_revenue_parquet(resource, year)
        
        results.append({
            'resource_name': resource,
            'capacity_mw': bess_registry[resource]['capacity_mw'],
            'energy_revenue': energy_rev,
            'as_revenue': sum(as_rev.values()),
            'total_revenue': energy_rev + sum(as_rev.values()),
            'revenue_per_mw': (energy_rev + sum(as_rev.values())) / capacity_mw,
            'strategy': classify_strategy(energy_rev, as_rev)
        })
    
    # Sort by total revenue
    leaderboard = pd.DataFrame(results).sort_values('total_revenue', ascending=False)
    
    # Save as Parquet for future queries
    leaderboard.to_parquet(f'/bess_analysis/leaderboard_{year}.parquet')
    
    return leaderboard
```

### Phase 5: Optimization & Caching (Week 3)

1. **Pre-compute Common Aggregations**
```python
# Daily aggregations for faster queries
daily_summaries = {
    'dam_awards_by_resource': aggregate_daily(dam_gen_resources),
    'rt_dispatch_by_resource': aggregate_daily(sced_gen_resources),
    'price_summaries': aggregate_price_stats(rt_prices, da_prices)
}
```

2. **Implement Query Cache**
```python
@lru_cache(maxsize=1000)
def cached_revenue_query(resource, date_range):
    # Cache frequently accessed revenue calculations
    pass
```

## Migration Benefits

### Performance Improvements
| Operation | CSV Time | Parquet Time | Improvement |
|-----------|----------|--------------|-------------|
| Load Annual Data | 45-60s | 0.5-2s | 30-100x |
| Filter by Resource | 10-15s | 0.1-0.3s | 50x |
| Join Price & Awards | 20-30s | 0.5-1s | 30x |
| Generate Leaderboard | 5-10 min | 10-30s | 20x |

### Storage Optimization
| Dataset | CSV Size | Parquet Size | Reduction |
|---------|----------|--------------|-----------|
| RT Prices (2024) | 15 GB | 1.2 GB | 92% |
| DAM Awards (2024) | 8 GB | 600 MB | 93% |
| SCED Dispatch (2024) | 25 GB | 2 GB | 92% |
| Total Annual | ~100 GB | ~8 GB | 92% |

### Query Capabilities
- **Columnar Access**: Only load needed columns
- **Predicate Pushdown**: Filter at file level
- **Parallel Processing**: Multi-threaded reads
- **Time Range Queries**: Efficient datetime filtering
- **Statistical Functions**: Built-in aggregations

## Implementation Timeline

### Week 1: Foundation
- [ ] Validate all Parquet schemas
- [ ] Document column evolution
- [ ] Create BESS resource registry
- [ ] Set up Parquet reading infrastructure

### Week 2: Core Migration
- [ ] Port revenue calculations to Parquet
- [ ] Implement leaderboard generation
- [ ] Create comparison tests vs CSV
- [ ] Validate revenue calculations

### Week 3: Optimization
- [ ] Add caching layer
- [ ] Pre-compute aggregations
- [ ] Implement parallel processing
- [ ] Create performance benchmarks

### Week 4: Production
- [ ] Deploy Parquet-based system
- [ ] Monitor performance
- [ ] Document API changes
- [ ] Train users on new system

## Risk Mitigation

### Data Quality
- Maintain CSV fallback for validation
- Implement data quality checks
- Log schema mismatches
- Handle missing columns gracefully

### Performance
- Monitor query performance
- Implement query timeout
- Add resource limits
- Cache expensive computations

### Compatibility
- Support both CSV and Parquet initially
- Gradual migration path
- Backward compatibility layer
- Clear deprecation timeline

## Success Metrics

1. **Performance**: 20x+ speedup for leaderboard generation
2. **Storage**: 90%+ reduction in disk usage
3. **Reliability**: <0.1% data discrepancies vs CSV
4. **Scalability**: Support for 5+ years of data in memory
5. **User Experience**: <1s response time for queries

## Conclusion

The migration to Parquet-based processing represents a fundamental improvement in the BESS revenue analysis system. By leveraging columnar storage, efficient compression, and optimized query patterns, we can provide near real-time leaderboard updates while reducing infrastructure costs and improving analytical capabilities.

The phased approach ensures minimal disruption while maximizing the benefits of the new architecture. The comprehensive schema documentation and column evolution tracking ensure long-term maintainability as ERCOT continues to evolve its data formats.