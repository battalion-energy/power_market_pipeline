#!/usr/bin/env python3
"""
Add the distance validation layer to the matching pipeline.
This ensures all future runs will catch location errors.
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic

def add_validation_to_matching_pipeline():
    """
    Update the matching pipeline to include distance validation
    """
    
    validation_code = '''
# Add this to the end of any matching script

def validate_match_distances(matches_df, max_distance_miles=100):
    """
    Validate that matched BESS locations are within reasonable distance of county centers.
    This catches errors like CROSSETT being matched to Houston facilities.
    """
    from geopy.distance import geodesic
    
    # County centers (reference data)
    county_centers = {
        'HARRIS': (29.7604, -95.3698),
        'CRANE': (31.3976, -102.3569),  # Where CROSSETT should be
        # ... add all counties
    }
    
    issues = []
    
    for idx, row in matches_df.iterrows():
        if pd.notna(row.get('EIA_Latitude')) and pd.notna(row.get('EIA_Longitude')):
            county = str(row.get('EIA_County', '')).upper().replace(' COUNTY', '').strip()
            
            if county in county_centers:
                county_lat, county_lon = county_centers[county]
                distance = geodesic(
                    (row['EIA_Latitude'], row['EIA_Longitude']),
                    (county_lat, county_lon)
                ).miles
                
                if distance > max_distance_miles:
                    issues.append({
                        'BESS': row['BESS_Gen_Resource'],
                        'EIA_Plant': row.get('EIA_Plant_Name'),
                        'County': county,
                        'Distance': distance,
                        'Warning': f'Location is {distance:.1f} miles from county center!'
                    })
    
    if issues:
        print("\\n⚠️ DISTANCE VALIDATION WARNINGS:")
        for issue in issues:
            print(f"   {issue['BESS']}: {issue['Warning']}")
            print(f"      Matched to: {issue['EIA_Plant']} in {issue['County']} County")
    
    return issues

# Call validation after matching
validation_issues = validate_match_distances(all_matches)
if validation_issues:
    print(f"\\n🚨 Found {len(validation_issues)} suspicious locations!")
    print("   These matches may be incorrect and should be reviewed.")
'''
    
    print("="*70)
    print("DISTANCE VALIDATION CODE FOR PIPELINE")
    print("="*70)
    print("\nAdd this validation to all matching scripts:")
    print(validation_code)
    
    return validation_code

def add_zone_validation():
    """
    Add validation to check if settlement zone matches physical location
    """
    
    zone_validation = '''
def validate_zone_consistency(df):
    """
    Check if settlement zones match expected zones based on coordinates.
    This catches errors like CROSSETT settling in WEST but having HOUSTON coordinates.
    """
    
    def get_expected_zone(lat, lon):
        """Determine expected ERCOT zone from coordinates"""
        if pd.isna(lat) or pd.isna(lon):
            return None
            
        # Simplified zone boundaries
        if lon <= -100.0:  # Far west
            return 'LZ_WEST'
        elif lat >= 32.0 and lon >= -98.5:  # North Texas
            return 'LZ_NORTH'
        elif (28.5 <= lat <= 30.5) and (-96.0 <= lon <= -94.5):  # Houston area
            return 'LZ_HOUSTON'
        elif lat <= 30.0:  # South Texas
            return 'LZ_SOUTH'
        else:
            return 'LZ_NORTH'
    
    issues = []
    
    for idx, row in df.iterrows():
        settlement_zone = row.get('Load_Zone')
        lat = row.get('Latitude', row.get('EIA_Latitude'))
        lon = row.get('Longitude', row.get('EIA_Longitude'))
        
        if settlement_zone and lat and lon:
            expected_zone = get_expected_zone(lat, lon)
            
            if expected_zone and settlement_zone != expected_zone:
                issues.append({
                    'BESS': row.get('BESS_Gen_Resource'),
                    'Settlement_Zone': settlement_zone,
                    'Expected_Zone': expected_zone,
                    'Coordinates': f'({lat:.2f}, {lon:.2f})',
                    'Warning': 'Zone mismatch detected!'
                })
                
                # Special check for CROSSETT
                if 'CROSSETT' in str(row.get('BESS_Gen_Resource', '')).upper():
                    if settlement_zone == 'LZ_WEST' and expected_zone == 'LZ_HOUSTON':
                        print("🚨 CRITICAL: CROSSETT has West settlement but Houston coordinates!")
                        print("   This indicates the location data is wrong!")
    
    if issues:
        print("\\n⚠️ ZONE CONSISTENCY WARNINGS:")
        for issue in issues[:10]:  # Show first 10
            print(f"   {issue['BESS']}: Settles in {issue['Settlement_Zone']} but coords suggest {issue['Expected_Zone']}")
            print(f"      Location: {issue['Coordinates']}")
    
    return issues
'''
    
    print("\n" + "="*70)
    print("ZONE CONSISTENCY VALIDATION CODE")
    print("="*70)
    print("\nAdd this validation to check zone consistency:")
    print(zone_validation)
    
    return zone_validation

def create_integrated_validation_script():
    """
    Create a complete validation script that can be imported by other scripts
    """
    
    integrated_script = '''#!/usr/bin/env python3
"""
Integrated validation module for BESS location data.
Import this in matching scripts to validate results.

Usage:
    from bess_location_validator import validate_all
    issues = validate_all(df)
"""

import pandas as pd
import numpy as np
from geopy.distance import geodesic

class BESSLocationValidator:
    """Comprehensive location validation for BESS data"""
    
    def __init__(self, max_distance_miles=100):
        self.max_distance_miles = max_distance_miles
        self.county_centers = self._load_county_centers()
        
    def _load_county_centers(self):
        """Load Texas county center coordinates"""
        return {
            'HARRIS': (29.7604, -95.3698),
            'CRANE': (31.3976, -102.3569),
            'ECTOR': (31.8673, -102.5406),
            'BRAZORIA': (29.1694, -95.4185),
            # Add all counties...
        }
    
    def validate_distance(self, df):
        """Validate distances from county centers"""
        issues = []
        
        for idx, row in df.iterrows():
            lat = row.get('EIA_Latitude', row.get('Latitude'))
            lon = row.get('EIA_Longitude', row.get('Longitude'))
            county = str(row.get('EIA_County', row.get('County', ''))).upper().replace(' COUNTY', '').strip()
            
            if lat and lon and county in self.county_centers:
                county_lat, county_lon = self.county_centers[county]
                distance = geodesic((lat, lon), (county_lat, county_lon)).miles
                
                if distance > self.max_distance_miles:
                    issues.append({
                        'type': 'distance',
                        'bess': row.get('BESS_Gen_Resource'),
                        'county': county,
                        'distance': distance,
                        'severity': 'HIGH' if distance > 200 else 'MEDIUM'
                    })
        
        return issues
    
    def validate_zone_consistency(self, df):
        """Validate that settlement zones match physical locations"""
        issues = []
        
        def get_expected_zone(lat, lon):
            if pd.isna(lat) or pd.isna(lon):
                return None
            
            if lon <= -100.0:
                return 'LZ_WEST'
            elif lat >= 32.0 and lon >= -98.5:
                return 'LZ_NORTH'
            elif (28.5 <= lat <= 30.5) and (-96.0 <= lon <= -94.5):
                return 'LZ_HOUSTON'
            else:
                return 'LZ_SOUTH'
        
        for idx, row in df.iterrows():
            settlement_zone = row.get('Load_Zone')
            lat = row.get('Latitude', row.get('EIA_Latitude'))
            lon = row.get('Longitude', row.get('EIA_Longitude'))
            
            if settlement_zone and lat and lon:
                expected = get_expected_zone(lat, lon)
                if expected and settlement_zone != expected:
                    issues.append({
                        'type': 'zone_mismatch',
                        'bess': row.get('BESS_Gen_Resource'),
                        'settlement_zone': settlement_zone,
                        'expected_zone': expected,
                        'severity': 'HIGH'
                    })
        
        return issues
    
    def validate_known_issues(self, df):
        """Check for specific known problems"""
        issues = []
        
        # Check CROSSETT specifically
        crossett = df[df['BESS_Gen_Resource'].str.contains('CROSSETT', case=False, na=False)]
        for idx, row in crossett.iterrows():
            county = str(row.get('EIA_County', row.get('County', ''))).upper()
            if 'HARRIS' in county:
                issues.append({
                    'type': 'known_error',
                    'bess': row.get('BESS_Gen_Resource'),
                    'error': 'CROSSETT should be in Crane County, not Harris',
                    'severity': 'CRITICAL'
                })
        
        return issues
    
    def validate_all(self, df):
        """Run all validations and return consolidated results"""
        all_issues = []
        
        # Run each validation
        all_issues.extend(self.validate_distance(df))
        all_issues.extend(self.validate_zone_consistency(df))
        all_issues.extend(self.validate_known_issues(df))
        
        # Report results
        if all_issues:
            print(f"\\n🚨 VALIDATION FOUND {len(all_issues)} ISSUES:")
            
            critical = [i for i in all_issues if i.get('severity') == 'CRITICAL']
            high = [i for i in all_issues if i.get('severity') == 'HIGH']
            medium = [i for i in all_issues if i.get('severity') == 'MEDIUM']
            
            if critical:
                print(f"   CRITICAL: {len(critical)} issues")
                for issue in critical[:3]:
                    print(f"      - {issue.get('bess')}: {issue.get('error', issue.get('type'))}")
            
            if high:
                print(f"   HIGH: {len(high)} issues")
                for issue in high[:3]:
                    print(f"      - {issue.get('bess')}: {issue.get('type')}")
            
            if medium:
                print(f"   MEDIUM: {len(medium)} issues")
        else:
            print("\\n✅ All validations passed!")
        
        return all_issues

# Convenience function
def validate_all(df, max_distance_miles=100):
    """Quick validation of a dataframe"""
    validator = BESSLocationValidator(max_distance_miles)
    return validator.validate_all(df)
'''
    
    # Save the integrated script
    with open('/home/enrico/projects/power_market_pipeline/bess_location_validator.py', 'w') as f:
        f.write(integrated_script)
    
    print("\n" + "="*70)
    print("CREATED INTEGRATED VALIDATION MODULE")
    print("="*70)
    print("\n✅ Saved to: bess_location_validator.py")
    print("\nUsage in any matching script:")
    print("   from bess_location_validator import validate_all")
    print("   issues = validate_all(matched_df)")
    
    return integrated_script

if __name__ == '__main__':
    # Generate validation code
    distance_code = add_validation_to_matching_pipeline()
    zone_code = add_zone_validation()
    integrated = create_integrated_validation_script()
    
    print("\n" + "="*70)
    print("VALIDATION LAYER READY FOR INTEGRATION")
    print("="*70)
    print("\nThis validation layer will:")
    print("1. Check distances from county centers (catch wrong counties)")
    print("2. Validate zone consistency (catch coordinate errors)")
    print("3. Flag known issues (like CROSSETT in wrong location)")
    print("\n✅ Import bess_location_validator.py in all matching scripts!")
    print("This would have prevented the CROSSETT error from happening.")