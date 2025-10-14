# Data Integrity Documentation - BESS Location Mapping

## The CROSSETT Problem: A Case Study in Data Corruption

### What Happened
CROSSETT BESS (a 200MW facility operated by Jupiter Power in Crane County, West Texas) was incorrectly placed in Harris County (Houston area) - over 400 miles away from its actual location.

### Root Cause Analysis

1. **Hardcoded Fake Data** (`match_bess_improved.py` line 80):
   ```python
   'CROSSETT': 'HARRIS',  # Likely Houston area  <-- WRONG!
   ```
   Someone assumed "Crossett" sounded like it was in Houston and hardcoded this mapping.

2. **Name Confusion**: 
   - CROSSETT was matched to CROCKETT BESS (different project in Harris County)
   - 77% name similarity allowed the match despite being completely different projects

3. **Algorithm Flaw**: 
   - Matching algorithm allowed high name similarity (>70%) to override county verification
   - Should have REQUIRED county match, not treated it as optional

4. **Cascading Errors**:
   - Wrong county → Wrong EIA match → Wrong coordinates → Wrong physical zone
   - Houston coordinates (29.77°, -95.38°) instead of West Texas (31.19°, -102.32°)

## The Correct Approach: Data-Driven County Mapping

### NEVER Use Hardcoded Mappings
**DO NOT** create dictionaries like:
```python
county_mappings = {
    'CROSSETT': 'HARRIS',  # NO! This is fake data!
    'ANCHOR': 'EASTLAND',  # NO! Verify from real sources!
    # etc...
}
```

### ALWAYS Use Real Data Sources

#### Method 1: Google Places API with LLM Validation
Located in: `create_bess_mapping_with_coordinates_v2.py`

```python
def geocode_with_google_places(location_name: str, county: str):
    """Geocode using Google Places API with LLM validation"""
    # 1. Search Google Places for "{substation} substation {county} County Texas"
    # 2. Get coordinates and address
    # 3. Validate with LLM that it's actually a substation
    # 4. Extract county from validated address
```

**LLM Validation Process**:
- Checks if result contains power/energy/substation keywords
- Verifies it's in the correct Texas county
- Rejects false matches (restaurants, businesses with similar names)

#### Method 2: EIA Generator Database
- Contains verified coordinates for operational facilities
- Includes county information
- Source: `/home/enrico/projects/battalion-platform/data/EIA/generators/EIA_generators_latest.xlsx`

#### Method 3: ERCOT Interconnection Queue
- Contains county data for projects
- Located in: `/home/enrico/projects/power_market_pipeline/interconnection_queue_clean/`

### Implementation Files

1. **`create_substation_county_mapping.py`** - Creates real mapping from Google Places
2. **`substation_county_mapping_from_google.csv`** - Output mapping file
3. **`match_bess_eia_generators_FIXED.py`** - Matching that REQUIRES county verification

### Critical Rules

1. **County Match is MANDATORY**
   - Never allow name similarity to override county mismatch
   - If counties don't match, it's NOT the same facility

2. **Unknown is Better than Wrong**
   - Return `None` if county is unknown
   - Never guess or assume based on name

3. **Document Data Sources**
   - Every mapping must trace back to a real source
   - Google Places, EIA, ERCOT IQ, etc.

4. **Validate with Multiple Sources**
   - Cross-check coordinates against zone boundaries
   - Verify county matches between sources
   - Use LLM to validate ambiguous results

### Testing Checklist

- [ ] CROSSETT maps to Crane County (West Texas)
- [ ] CROCKETT maps to Harris County (Houston)  
- [ ] No cross-county matches without explicit verification
- [ ] All mappings trace to real data sources
- [ ] No hardcoded county assignments

### Example: Correct CROSSETT Mapping

**Source**: EIA Generators Database
```
Plant Name: Crossett Power Management LLC
County: Crane
State: TX
Latitude: 31.191687
Longitude: -102.3172
Technology: Batteries
Capacity: 200 MW
```

**Settlement Zone**: LZ_WEST (correct for West Texas location)
**Physical Zone**: West (verified by GeoJSON boundaries)

## Lessons Learned

1. **Never Make Assumptions** - "Crossett" doesn't mean Houston
2. **Verify Everything** - Check multiple data sources
3. **County First** - Geographic hierarchy matters
4. **Document Sources** - Every data point needs provenance
5. **Test Edge Cases** - Similar names can cause confusion

## Contact

If you find hardcoded data or mapping errors:
1. Remove the hardcoded data immediately
2. Replace with data-driven lookup
3. Document the real data source
4. Test the correction

Remember: **REAL DATA ONLY - NO EXCEPTIONS**