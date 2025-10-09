import json
from datetime import datetime

# Read current state
with open('ercot_download_state.json', 'r') as f:
    state = json.load(f)

# Update SCED Gen start date to June 18, 2025 (we have data through June 17, 2025)
state['datasets']['60d_SCED_Gen_Resources'] = {
    'last_timestamp': '2025-06-17T23:55:00',  # Last date we have in parquet
    'last_download': datetime.now().isoformat(),
    'last_records_count': 0,
    'download_history': []
}

state['last_updated'] = datetime.now().isoformat()

# Write updated state
with open('ercot_download_state.json', 'w') as f:
    json.dump(state, f, indent=2)

print("Updated SCED Gen start date to: 2025-06-17T23:55:00")
print("Will download from: June 18, 2025 → Aug 9, 2025 (~52 days)")
