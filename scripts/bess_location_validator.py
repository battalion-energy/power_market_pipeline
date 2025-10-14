#!/usr/bin/env python3
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
            print(f"\n🚨 VALIDATION FOUND {len(all_issues)} ISSUES:")
            
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
            print("\n✅ All validations passed!")
        
        return all_issues

# Convenience function
def validate_all(df, max_distance_miles=100):
    """Quick validation of a dataframe"""
    validator = BESSLocationValidator(max_distance_miles)
    return validator.validate_all(df)
