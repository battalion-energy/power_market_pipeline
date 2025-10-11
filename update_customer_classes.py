#!/usr/bin/env python3
"""
Update demand response catalog with customer class designations
"""

import json

# Customer class mapping based on verified research
CUSTOMER_CLASSES = {
    "ERCOT-ERS": {
        "residential": False,
        "commercial": True,
        "industrial": True,
        "notes": "Wholesale market program requiring QSE status - commercial/industrial only"
    },
    "CAISO-ELRP": {
        "residential": True,
        "commercial": True,
        "industrial": True,
        "notes": "Both residential (Power Saver Rewards at $1/kWh) and non-residential ($2/kWh). Expanded to residential in 2022."
    },
    "MA-CONNECTED-SOLUTIONS": {
        "residential": True,
        "commercial": False,
        "industrial": False,
        "notes": "Residential battery storage and smart thermostat program. Systems < 50kW."
    },
    "RI-CONNECTED-SOLUTIONS": {
        "residential": True,
        "commercial": False,
        "industrial": False,
        "notes": "Residential battery storage program"
    },
    "NH-CONNECTED-SOLUTIONS": {
        "residential": True,
        "commercial": False,
        "industrial": False,
        "notes": "Residential battery storage program"
    },
    "CA-DSGS": {
        "residential": True,
        "commercial": False,
        "industrial": False,
        "notes": "Residential battery storage program (Powerwall and similar home batteries)"
    },
    "MA-CLEAN-PEAK": {
        "residential": True,
        "commercial": True,
        "industrial": True,
        "notes": "Both residential (< 50kW) and commercial/industrial (>= 50kW) eligible"
    },
    "ISONE-FCM-DR": {
        "residential": False,
        "commercial": True,
        "industrial": True,
        "notes": "Wholesale capacity market - commercial/industrial only"
    },
    "PJM-ECONOMIC-DR": {
        "residential": False,
        "commercial": True,
        "industrial": True,
        "notes": "Wholesale energy market - commercial/industrial only"
    },
    "CONED-CSRP-DLRP": {
        "residential": False,
        "commercial": True,
        "industrial": True,
        "notes": "Commercial and industrial customers only"
    }
}

def main():
    # Read catalog
    with open('demand_response_programs_catalog.json', 'r') as f:
        catalog = json.load(f)

    # Update each program
    for program in catalog['programs']:
        program_id = program['program_id']
        if program_id in CUSTOMER_CLASSES:
            class_info = CUSTOMER_CLASSES[program_id]

            # Add customer_classes to eligibility section
            if 'eligibility' in program:
                program['eligibility']['customer_classes'] = {
                    "residential": class_info['residential'],
                    "commercial": class_info['commercial'],
                    "industrial": class_info['industrial']
                }

                # Update notes if they exist
                if 'notes' in program['eligibility']:
                    program['eligibility']['notes'] = f"{program['eligibility']['notes']} Customer classes: {class_info['notes']}"
                else:
                    program['eligibility']['notes'] = f"Customer classes: {class_info['notes']}"

    # Update catalog metadata version
    catalog['catalog_metadata']['version'] = '1.1'
    catalog['catalog_metadata']['last_updated'] = '2025-10-10'
    catalog['catalog_metadata']['data_quality_note'] = "This catalog contains ONLY verified data from official sources. Any unavailable data is explicitly marked as 'not available' with source URLs provided for manual verification. v1.1 adds customer class designations (residential, commercial, industrial)."

    # Write updated catalog
    with open('demand_response_programs_catalog.json', 'w') as f:
        json.dump(catalog, f, indent=2)

    print("✓ Updated catalog with customer class designations")
    print(f"✓ Updated {len(catalog['programs'])} programs")

    # Print summary
    print("\nCustomer Class Summary:")
    for program in catalog['programs']:
        program_id = program['program_id']
        classes = program['eligibility'].get('customer_classes', {})
        res = '✓' if classes.get('residential') else '✗'
        com = '✓' if classes.get('commercial') else '✗'
        ind = '✓' if classes.get('industrial') else '✗'
        print(f"{program_id:30} Res:{res} Com:{com} Ind:{ind}")

if __name__ == '__main__':
    main()
