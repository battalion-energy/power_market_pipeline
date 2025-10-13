# Demand Response Event Generator

## Overview

The DR Event Generator creates realistic event schedules for demand response programs based on program parameters from the catalog. These events are used as inputs to battery energy storage optimization algorithms.

## Features

- **Realistic Event Generation**: Creates events that respect program constraints (call windows, max events, duration limits)
- **Multiple Event Types**: Handles emergency-based, market-based, and scheduled dispatch programs
- **Reproducible**: Uses random seed for consistent output
- **Timezone Aware**: Generates events in program-specific timezones
- **Payment Information**: Includes payment rates for optimization calculations

## Usage

### Basic Usage

```bash
# Generate events for CAISO ELRP in 2024
python3 generate_dr_events.py --program "CAISO-ELRP" --year 2024 --output events.json

# Generate with specific seed for reproducibility
python3 generate_dr_events.py --program "MA-CONNECTED-SOLUTIONS" --year 2025 --seed 42

# Output to stdout (for piping)
python3 generate_dr_events.py --program "ERCOT-ERS" --year 2024
```

### Command-Line Options

- `--program` (required): Program ID from catalog (e.g., "CAISO-ELRP")
- `--year` (required): Year to generate events for
- `--output`: Output JSON file path (default: stdout)
- `--seed`: Random seed for reproducible generation
- `--catalog`: Path to DR programs catalog (default: demand_response_programs_catalog.json)

## Event Generation Strategies

### Emergency-Based Programs
**Programs**: CAISO-ELRP, ERCOT-ERS, Con Ed CSRP/DLRP

- Random events within defined call windows
- Respects max events per season
- Variable duration within min/max bounds
- Weekday restrictions if applicable
- Examples: 10-15 events in summer, 4-9 PM window

### Market-Based Programs
**Programs**: ISO-NE FCM, PJM Economic DR

- Events generated based on economic dispatch patterns
- Higher frequency during peak seasons
- ~20% of days have events
- Variable start times and durations
- Reflects market volatility

### Scheduled Dispatch Programs
**Programs**: MA Clean Peak Standard

- Events on regular schedule during peak periods
- Weekday-only events
- Fixed time windows (e.g., 2-6 PM)
- Predictable pattern for long-term optimization

## Output Format

```json
{
  "program_id": "CAISO-ELRP",
  "year": 2024,
  "total_events": 36,
  "total_hours": 95.8,
  "events": [
    {
      "event_id": "CAISO-ELRP-20240503T1600",
      "start_time": "2024-05-03T16:00:00-07:00",
      "end_time": "2024-05-03T19:23:00-07:00",
      "duration_hours": 3.38,
      "season": "Summer Peak",
      "event_type": "emergency",
      "payment_rate": {
        "has_capacity": false,
        "has_performance": true,
        "performance_value": 2,
        "performance_unit": "$/kWh"
      },
      "program_id": "CAISO-ELRP",
      "advance_notice_hours": 12.5
    }
  ],
  "metadata": {
    "generated_at": "2025-10-10T...",
    "generator_version": "1.0",
    "seed": 42
  }
}
```

## Integration with Optimization Algorithms

### Expected Use Case

The event generator output provides inputs for battery optimization:

1. **Event Timing**: When to expect DR calls (start_time, end_time)
2. **Payment Rates**: Revenue opportunity per kWh discharged
3. **Advance Notice**: How much time to prepare battery state
4. **Duration**: How long battery needs to perform
5. **Seasonal Patterns**: Year-round optimization with seasonal variations

### Example Integration

```python
import json
from datetime import datetime

# Load generated events
with open('caiso_elrp_2024_events.json', 'r') as f:
    dr_events = json.load(f)

# Use in optimization
for event in dr_events['events']:
    event_start = datetime.fromisoformat(event['start_time'])
    duration = event['duration_hours']
    payment = event['payment_rate']['performance_value']  # $/kWh

    # Optimize battery dispatch for this event
    optimal_discharge = optimize_battery(
        event_time=event_start,
        duration_hours=duration,
        payment_rate_per_kwh=payment,
        advance_notice_hours=event['advance_notice_hours']
    )
```

## Program-Specific Examples

### CAISO ELRP (Emergency)
- **Season**: May-October
- **Call Window**: 4-9 PM any day
- **Typical Events**: 8-15 per season
- **Duration**: 1-5 hours
- **Payment**: $2/kWh (non-res), $1/kWh (res)
- **Max Hours/Year**: 60

```bash
python3 generate_dr_events.py --program "CAISO-ELRP" --year 2024 --seed 42
# Output: ~36 events, ~96 total hours
```

### MA Connected Solutions (Emergency)
- **Season**: June-September
- **Call Window**: 3-8 PM weekdays
- **Typical Events**: 10-20 per summer
- **Duration**: 2-3 hours
- **Payment**: $275/kW-year (capacity-based)

```bash
python3 generate_dr_events.py --program "MA-CONNECTED-SOLUTIONS" --year 2025 --seed 123
# Output: ~13 events, ~32 total hours
```

### ERCOT ERS (Emergency - Rare)
- **Season**: Year-round
- **Call Window**: Any time
- **Typical Events**: Very rare (2 events in 2020-2024)
- **Duration**: Variable (1-10 hours)
- **Payment**: Auction-based capacity

```bash
python3 generate_dr_events.py --program "ERCOT-ERS" --year 2024 --seed 789
# Output: ~5-10 events (simulated, actual is much rarer)
```

### MA Clean Peak (Scheduled)
- **Season**: Year-round
- **Call Window**: Scheduled peak periods
- **Events**: Daily during peak hours
- **Duration**: 4 hours
- **Payment**: Market-based CPEC pricing

```bash
python3 generate_dr_events.py --program "MA-CLEAN-PEAK" --year 2024
# Output: ~260 events (weekdays only)
```

## Validation and Constraints

The generator respects all program constraints:

✓ **Max Events**: Never exceeds max_events_per_season/year
✓ **Call Windows**: Only generates events within allowed hours
✓ **Days of Week**: Respects weekday/weekend restrictions
✓ **Duration Limits**: Stays within min/max duration bounds
✓ **Max Hours**: Total hours won't exceed annual/seasonal limits
✓ **Seasons**: Events only in active program seasons
✓ **Timezone**: All events in program-specific timezone

## Limitations and Assumptions

### Current Limitations

1. **Historical Accuracy**: Generated events are statistically realistic but don't match actual historical events
2. **Weather Correlation**: Doesn't model weather patterns (e.g., heat waves triggering multiple consecutive events)
3. **Market Prices**: Market-based programs don't model actual LMP patterns
4. **Grid Conditions**: Emergency programs don't model actual grid stress conditions

### Assumptions

1. **Event Distribution**: Emergency events are uniformly random within constraints
2. **Duration**: Uniformly distributed within min/max bounds
3. **Market Events**: ~20% of days have market events
4. **Notice Time**: Uniformly distributed between min/max notice hours

## Future Enhancements

Potential improvements for more realistic event generation:

1. **Historical Patterns**: Train on actual historical events
2. **Weather Data**: Correlate events with temperature forecasts
3. **Price Signals**: Use historical LMP data for market programs
4. **Multi-Year Patterns**: Model year-to-year variations
5. **Stacking**: Generate events for multiple programs simultaneously
6. **Constraints**: Add minimum rest period between events
7. **Clustering**: Model heat wave clustering (consecutive events)

## Testing

Test all program types:

```bash
# Emergency programs
python3 generate_dr_events.py --program "CAISO-ELRP" --year 2024 --seed 1
python3 generate_dr_events.py --program "MA-CONNECTED-SOLUTIONS" --year 2024 --seed 2
python3 generate_dr_events.py --program "ERCOT-ERS" --year 2024 --seed 3

# Market programs
python3 generate_dr_events.py --program "PJM-ECONOMIC-DR" --year 2024 --seed 4
python3 generate_dr_events.py --program "ISONE-FCM-DR" --year 2024 --seed 5

# Scheduled programs
python3 generate_dr_events.py --program "MA-CLEAN-PEAK" --year 2024 --seed 6
```

## Version History

- **v1.0** (2025-10-10): Initial release with emergency, market, and scheduled event generation

## Related Files

- `demand_response_programs_catalog.json` - Source program parameters
- `demand_response_schema.json` - Schema definition
- `DEMAND_RESPONSE_CATALOG_README.md` - Program catalog documentation
