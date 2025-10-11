#!/usr/bin/env python3
"""
Demand Response Event Generator

Generates randomized or scheduled events for DR programs based on program parameters.
This output feeds into battery energy storage optimization algorithms.

Usage:
    python3 generate_dr_events.py --program "CAISO-ELRP" --year 2024
    python3 generate_dr_events.py --program "MA-CONNECTED-SOLUTIONS" --year 2025 --output events.json
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pytz


class DREventGenerator:
    """Generate DR events based on program parameters"""

    def __init__(self, catalog_path: str = "demand_response_programs_catalog.json"):
        """Load the DR programs catalog"""
        with open(catalog_path, 'r') as f:
            self.catalog = json.load(f)
        self.programs = {p['program_id']: p for p in self.catalog['programs']}

    def get_program(self, program_id: str) -> Dict[str, Any]:
        """Get program by ID"""
        if program_id not in self.programs:
            available = ", ".join(self.programs.keys())
            raise ValueError(f"Program '{program_id}' not found. Available: {available}")
        return self.programs[program_id]

    def parse_season_dates(self, season: Dict, year: int) -> tuple:
        """Parse season start/end dates for given year"""
        start_str = season['start_date']  # Format: "MM-DD"
        end_str = season['end_date']

        if start_str == "varies" or end_str == "varies":
            return None, None

        start_month, start_day = map(int, start_str.split('-'))
        end_month, end_day = map(int, end_str.split('-'))

        start_date = datetime(year, start_month, start_day)
        end_date = datetime(year, end_month, end_day)

        return start_date, end_date

    def generate_events(self, program_id: str, year: int, seed: int = None) -> List[Dict]:
        """
        Generate events for a program in a given year

        Args:
            program_id: Program identifier (e.g., "CAISO-ELRP")
            year: Year to generate events for
            seed: Random seed for reproducibility

        Returns:
            List of event dictionaries with start_time, end_time, duration_hours, payment info
        """
        if seed is not None:
            random.seed(seed)

        program = self.get_program(program_id)
        events = []

        # Get timezone
        call_windows = program.get('call_windows', [])
        if not call_windows:
            print(f"Warning: No call windows defined for {program_id}")
            return events

        timezone_str = call_windows[0].get('timezone', 'America/New_York')
        tz = pytz.timezone(timezone_str)

        # Determine event generation strategy
        special_programs = program.get('special_programs', {})
        if special_programs.get('scheduled_dispatch', False):
            events = self._generate_scheduled_events(program, year, tz)
        elif program.get('program_type') in ['iso', 'wholesale_market']:
            events = self._generate_market_events(program, year, tz)
        else:
            events = self._generate_emergency_events(program, year, tz)

        return sorted(events, key=lambda x: x['start_time'])

    def _generate_scheduled_events(self, program: Dict, year: int, tz) -> List[Dict]:
        """Generate events for scheduled dispatch programs (e.g., MA Clean Peak)"""
        events = []
        # For scheduled programs, generate events during defined peak periods
        # This is simplified - in reality, you'd need the actual seasonal peak schedules
        seasons = program.get('seasons', [])

        for season in seasons:
            start_date, end_date = self.parse_season_dates(season, year)
            if not start_date or not end_date:
                continue

            # Generate events on weekdays during typical peak hours (e.g., 2-7 PM)
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Weekday
                    # Assume 2-6 PM peak period
                    event_start = tz.localize(datetime.combine(
                        current_date.date(),
                        datetime.min.time().replace(hour=14)
                    ))
                    event_end = tz.localize(datetime.combine(
                        current_date.date(),
                        datetime.min.time().replace(hour=18)
                    ))

                    events.append({
                        'event_id': f"{program['program_id']}-{current_date.strftime('%Y%m%d')}",
                        'start_time': event_start.isoformat(),
                        'end_time': event_end.isoformat(),
                        'duration_hours': 4.0,
                        'season': season['season_name'],
                        'event_type': 'scheduled',
                        'payment_rate': self._get_payment_rate(program),
                        'program_id': program['program_id']
                    })

                current_date += timedelta(days=1)

        return events

    def _generate_market_events(self, program: Dict, year: int, tz) -> List[Dict]:
        """Generate events for market-based programs (ISO/RTO programs)"""
        events = []
        # Market programs dispatch based on economics, so we generate variable events
        # throughout the year, with higher frequency during summer

        seasons = program.get('seasons', [])
        if not seasons:
            # Year-round market, generate across whole year
            seasons = [{'season_name': 'Year-Round', 'start_date': '01-01', 'end_date': '12-31'}]

        for season in seasons:
            start_date, end_date = self.parse_season_dates(season, year)
            if not start_date or not end_date:
                continue

            # Market events: more frequent in summer, during peak hours
            num_days = (end_date - start_date).days
            # Assume ~20% of days have market events
            num_events = int(num_days * 0.2)

            for _ in range(num_events):
                # Random day in season
                random_days = random.randint(0, num_days)
                event_date = start_date + timedelta(days=random_days)

                # Call windows
                call_windows = program.get('call_windows', [])
                if call_windows:
                    window = random.choice(call_windows)
                    start_hour = window.get('start_hour', 14)
                    end_hour = window.get('end_hour', 20)

                    # Random start within window
                    event_hour = random.randint(start_hour, end_hour - 2)
                    duration = random.uniform(1, 4)  # 1-4 hours

                    event_start = tz.localize(datetime.combine(
                        event_date.date(),
                        datetime.min.time().replace(hour=event_hour)
                    ))
                    event_end = event_start + timedelta(hours=duration)

                    events.append({
                        'event_id': f"{program['program_id']}-{event_start.strftime('%Y%m%d%H%M')}",
                        'start_time': event_start.isoformat(),
                        'end_time': event_end.isoformat(),
                        'duration_hours': round(duration, 2),
                        'season': season['season_name'],
                        'event_type': 'market',
                        'payment_rate': self._get_payment_rate(program),
                        'program_id': program['program_id']
                    })

        return events

    def _generate_emergency_events(self, program: Dict, year: int, tz) -> List[Dict]:
        """Generate events for emergency-based DR programs"""
        events = []
        event_params = program.get('event_parameters', {})
        seasons = program.get('seasons', [])

        for season in seasons:
            start_date, end_date = self.parse_season_dates(season, year)
            if not start_date or not end_date:
                continue

            # Get max events for this season
            max_events = season.get('max_events')
            if max_events == "not specified" or max_events is None:
                max_events = event_params.get('max_events_per_season', 15)
            if isinstance(max_events, str):
                max_events = 15  # default

            # Generate random number of events (between 50-100% of max)
            num_events = random.randint(int(max_events * 0.5), int(max_events))

            # Get call windows
            call_windows = [w for w in program.get('call_windows', [])
                          if season['season_name'] in w.get('seasons', [])]

            if not call_windows:
                call_windows = program.get('call_windows', [])

            if not call_windows:
                continue

            # Generate events
            for i in range(num_events):
                # Random date in season
                num_days = (end_date - start_date).days
                random_days = random.randint(0, num_days)
                event_date = start_date + timedelta(days=random_days)

                # Select random call window
                window = random.choice(call_windows)
                start_hour = window.get('start_hour', 16)
                end_hour = window.get('end_hour', 21)

                # Check days of week
                days_of_week = window.get('days_of_week', [])
                if days_of_week:
                    day_name = event_date.strftime('%A')
                    if day_name not in days_of_week:
                        continue

                # Random start time within window
                if isinstance(start_hour, int) and isinstance(end_hour, int):
                    event_hour = random.randint(start_hour, min(end_hour - 1, 23))
                else:
                    event_hour = 16  # default

                # Duration
                typical_duration = event_params.get('typical_duration_hours', 2)
                min_duration = event_params.get('minimum_duration_hours', 1)
                max_duration = event_params.get('maximum_duration_hours', 5)

                if isinstance(typical_duration, str) or typical_duration == "not specified":
                    typical_duration = 2
                if isinstance(min_duration, str) or min_duration == "not specified":
                    min_duration = 1
                if isinstance(max_duration, str) or max_duration == "not specified":
                    max_duration = 5

                duration = random.uniform(float(min_duration), float(max_duration))

                event_start = tz.localize(datetime.combine(
                    event_date.date(),
                    datetime.min.time().replace(hour=event_hour)
                ))
                event_end = event_start + timedelta(hours=duration)

                # Get payment rate
                payment_rate = self._get_payment_rate(program, season['season_name'])

                events.append({
                    'event_id': f"{program['program_id']}-{event_start.strftime('%Y%m%d%H%M')}",
                    'start_time': event_start.isoformat(),
                    'end_time': event_end.isoformat(),
                    'duration_hours': round(duration, 2),
                    'season': season['season_name'],
                    'event_type': 'emergency',
                    'payment_rate': payment_rate,
                    'program_id': program['program_id'],
                    'advance_notice_hours': self._get_notice_hours(program)
                })

        return events

    def _get_payment_rate(self, program: Dict, season: str = None) -> Dict[str, Any]:
        """Extract payment rate information"""
        payment_structure = program.get('payment_structure', {})

        rate_info = {
            'has_capacity': payment_structure.get('has_capacity_payment', False),
            'has_performance': payment_structure.get('has_performance_payment', False)
        }

        # Performance payment
        if rate_info['has_performance']:
            perf_rate = payment_structure.get('performance_rate', {})
            rate_info['performance_value'] = perf_rate.get('value')
            rate_info['performance_unit'] = perf_rate.get('unit')

        # Capacity payment
        if rate_info['has_capacity']:
            cap_rate = payment_structure.get('capacity_rate', {})
            rate_info['capacity_value'] = cap_rate.get('value')
            rate_info['capacity_unit'] = cap_rate.get('unit')

            # Check for seasonal rates
            if cap_rate.get('varies_by_season') and season:
                seasonal_rates = cap_rate.get('seasonal_rates', [])
                for sr in seasonal_rates:
                    if sr.get('season') == season:
                        rate_info['capacity_value'] = sr.get('rate')
                        rate_info['capacity_unit'] = sr.get('unit')

        return rate_info

    def _get_notice_hours(self, program: Dict) -> float:
        """Get typical advance notice hours"""
        notification = program.get('notification', {})
        min_notice = notification.get('minimum_notice_hours')
        max_notice = notification.get('maximum_notice_hours')

        if isinstance(min_notice, (int, float)) and isinstance(max_notice, (int, float)):
            return random.uniform(min_notice, max_notice)
        elif isinstance(max_notice, (int, float)):
            return max_notice
        elif notification.get('day_ahead_notice'):
            return 24.0
        else:
            return 2.0  # default


def main():
    parser = argparse.ArgumentParser(description='Generate DR events for optimization')
    parser.add_argument('--program', required=True, help='Program ID (e.g., CAISO-ELRP)')
    parser.add_argument('--year', type=int, required=True, help='Year to generate events for')
    parser.add_argument('--output', default=None, help='Output JSON file (default: stdout)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--catalog', default='demand_response_programs_catalog.json',
                       help='Path to DR programs catalog')

    args = parser.parse_args()

    # Generate events
    generator = DREventGenerator(args.catalog)
    events = generator.generate_events(args.program, args.year, args.seed)

    # Create output
    output = {
        'program_id': args.program,
        'year': args.year,
        'total_events': len(events),
        'total_hours': sum(e['duration_hours'] for e in events),
        'events': events,
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0',
            'seed': args.seed
        }
    }

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Generated {len(events)} events for {args.program} in {args.year}")
        print(f"✓ Total event hours: {output['total_hours']:.1f}")
        print(f"✓ Output saved to: {args.output}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
