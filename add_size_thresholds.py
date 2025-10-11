#!/usr/bin/env python3
"""
Add size thresholds to customer class designations
"""

import json

# Size threshold data based on verified research
SIZE_THRESHOLDS = {
    "MA-CONNECTED-SOLUTIONS": {
        "residential_max_kw": 50,
        "commercial_min_kw": 50,
        "size_threshold_notes": "Systems < 50kW qualify as residential. Systems >= 50kW should enroll through Commercial ConnectedSolutions Program."
    },
    "MA-CLEAN-PEAK": {
        "residential_max_kw": 50,
        "commercial_min_kw": 50,
        "size_threshold_notes": "Residential customers: inverter size < 50kW. Commercial/Industrial: >= 50kW."
    },
    "CAISO-ELRP": {
        "residential_max_kw": None,
        "commercial_min_kw": 1,
        "size_threshold_notes": "Residential: No minimum. Non-residential: Must be able to reduce load by minimum of 1 kW. Residential compensated at $1/kWh, non-residential at $2/kWh."
    },
    "RI-CONNECTED-SOLUTIONS": {
        "residential_max_kw": None,
        "commercial_min_kw": None,
        "size_threshold_notes": "Residential battery storage program. No specific size threshold documented."
    },
    "NH-CONNECTED-SOLUTIONS": {
        "residential_max_kw": None,
        "commercial_min_kw": None,
        "size_threshold_notes": "Residential battery storage program. No specific size threshold documented."
    },
    "CA-DSGS": {
        "residential_max_kw": None,
        "commercial_min_kw": None,
        "size_threshold_notes": "Residential battery storage program (Powerwall and similar home batteries). No specific size threshold documented."
    },
    "ERCOT-ERS": {
        "residential_max_kw": None,
        "commercial_min_kw": None,
        "size_threshold_notes": "Wholesale market program requiring QSE status - commercial/industrial only. No specific size thresholds."
    },
    "ISONE-FCM-DR": {
        "residential_max_kw": None,
        "commercial_min_kw": None,
        "size_threshold_notes": "Wholesale capacity market - commercial/industrial only. Minimum capacity varies by resource type."
    },
    "PJM-ECONOMIC-DR": {
        "residential_max_kw": None,
        "commercial_min_kw": None,
        "size_threshold_notes": "Wholesale energy market - commercial/industrial only. No specific size thresholds."
    },
    "CONED-CSRP-DLRP": {
        "residential_max_kw": None,
        "commercial_min_kw": None,
        "size_threshold_notes": "Commercial and industrial customers only. No specific size thresholds documented."
    }
}

def main():
    # Read catalog
    with open('demand_response_programs_catalog.json', 'r') as f:
        catalog = json.load(f)

    # Update each program
    for program in catalog['programs']:
        program_id = program['program_id']
        if program_id in SIZE_THRESHOLDS:
            threshold_info = SIZE_THRESHOLDS[program_id]

            # Add size thresholds to customer_classes
            if 'eligibility' in program and 'customer_classes' in program['eligibility']:
                customer_classes = program['eligibility']['customer_classes']

                if threshold_info['residential_max_kw'] is not None:
                    customer_classes['residential_max_kw'] = threshold_info['residential_max_kw']

                if threshold_info['commercial_min_kw'] is not None:
                    customer_classes['commercial_min_kw'] = threshold_info['commercial_min_kw']

                customer_classes['size_threshold_notes'] = threshold_info['size_threshold_notes']

    # Update catalog metadata version
    catalog['catalog_metadata']['version'] = '1.2'
    catalog['catalog_metadata']['data_quality_note'] = "This catalog contains ONLY verified data from official sources. Any unavailable data is explicitly marked as 'not available' with source URLs provided for manual verification. v1.1 adds customer class designations (residential, commercial, industrial). v1.2 adds size thresholds for customer class eligibility."

    # Write updated catalog
    with open('demand_response_programs_catalog.json', 'w') as f:
        json.dump(catalog, f, indent=2)

    print("✓ Updated catalog with size thresholds")
    print(f"✓ Updated {len(catalog['programs'])} programs")

    # Print summary
    print("\nSize Threshold Summary:")
    for program in catalog['programs']:
        program_id = program['program_id']
        classes = program['eligibility'].get('customer_classes', {})
        res_max = classes.get('residential_max_kw', 'N/A')
        com_min = classes.get('commercial_min_kw', 'N/A')
        print(f"{program_id:30} Res Max: {str(res_max):>10} kW  |  Com Min: {str(com_min):>10} kW")

if __name__ == '__main__':
    main()
