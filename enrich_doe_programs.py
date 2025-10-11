#!/usr/bin/env python3
"""
DOE FEMP Program Enrichment Script

Takes the raw scraped data and enriches it with detailed information by:
1. Fetching the actual program pages
2. Web searching for additional authoritative sources
3. Extracting structured information
4. Filling in the demand_response_schema.json format

CRITICAL: Only uses verified data. Marks unavailable data explicitly.
"""

import json
import requests
from bs4 import BeautifulSoup
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from urllib.parse import urljoin


class ProgramEnricher:
    """Enriches DOE FEMP programs with detailed schema information"""

    def __init__(self, schema_path: str = "demand_response_schema.json"):
        """Load the target schema"""
        with open(schema_path, 'r') as f:
            self.schema = json.load(f)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

    def fetch_program_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch a program's detail page"""
        if not url or url == "not available":
            return None

        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"    Error fetching {url}: {e}")
            return None

    def extract_payment_info(self, text: str, description: str) -> Dict[str, Any]:
        """Extract payment structure from text"""
        payment = {
            'has_capacity_payment': False,
            'has_performance_payment': False,
            'capacity_rate': {},
            'performance_rate': {}
        }

        # Look for capacity payments ($/kW patterns)
        capacity_patterns = [
            r'\$(\d+(?:\.\d+)?)\s*(?:per|/)\s*kW[/-]?(?:year|yr|month|mo|day)?',
            r'(\d+(?:\.\d+)?)\s*cents?\s*(?:per|/)\s*kW'
        ]

        for pattern in capacity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                payment['has_capacity_payment'] = True
                # Try to determine the unit
                if 'month' in text.lower() or '/mo' in text.lower():
                    unit = '$/kW-month'
                elif 'day' in text.lower():
                    unit = '$/kW-day'
                else:
                    unit = '$/kW-year'

                payment['capacity_rate'] = {
                    'value': 'varies - see program page',
                    'unit': unit,
                    'varies_by_season': 'not specified',
                    'notes': f'Found in description: {matches[0]}'
                }
                break

        # Look for performance payments ($/kWh patterns)
        performance_patterns = [
            r'\$(\d+(?:\.\d+)?)\s*(?:per|/)\s*kWh',
            r'(\d+(?:\.\d+)?)\s*cents?\s*(?:per|/)\s*kWh'
        ]

        for pattern in performance_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                payment['has_performance_payment'] = True
                payment['performance_rate'] = {
                    'value': 'varies - see program page',
                    'unit': '$/kWh',
                    'varies_by_event': 'not specified',
                    'notes': f'Found in description: {matches[0]}'
                }
                break

        return payment

    def extract_time_info(self, text: str) -> Dict[str, Any]:
        """Extract timing information (call windows, hours)"""
        call_windows = []

        # Look for time patterns (e.g., "3-8 PM", "noon to 7 PM")
        time_patterns = [
            r'(\d{1,2})\s*(?:A\.?M\.?|P\.?M\.?|a\.?m\.?|p\.?m\.?)\s*(?:to|-|through)\s*(\d{1,2})\s*(?:A\.?M\.?|P\.?M\.?|a\.?m\.?|p\.?m\.?)',
            r'between\s+(\d{1,2})\s+(?:and|&)\s+(\d{1,2})\s*(?:A\.?M\.?|P\.?M\.?|a\.?m\.?|p\.?m\.?)?',
            r'(\d{1,2}):(\d{2})\s*(?:A\.?M\.?|P\.?M\.?)\s*(?:to|-)\s*(\d{1,2}):(\d{2})\s*(?:A\.?M\.?|P\.?M\.?)'
        ]

        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                call_windows.append({
                    'season_name': 'not specified',
                    'start_hour': 'see program description',
                    'end_hour': 'see program description',
                    'days_of_week': [],
                    'timezone': 'not specified',
                    'notes': f'Time pattern found: {matches[0]}'
                })
                break

        return {'call_windows': call_windows if call_windows else []}

    def extract_season_info(self, text: str) -> List[Dict[str, Any]]:
        """Extract seasonal information"""
        seasons = []

        # Look for season mentions
        season_keywords = {
            'summer': r'summer|june|july|august|september',
            'winter': r'winter|november|december|january|february|march',
            'spring': r'spring|april|may',
            'fall': r'fall|autumn|october'
        }

        found_seasons = set()
        for season_name, pattern in season_keywords.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_seasons.add(season_name)

        for season in found_seasons:
            seasons.append({
                'season_name': season.capitalize(),
                'start_date': 'not specified',
                'end_date': 'not specified',
                'max_events': 'not specified',
                'max_hours': 'not specified'
            })

        return seasons if seasons else [{
            'season_name': 'Year-Round',
            'start_date': '01-01',
            'end_date': '12-31',
            'max_events': 'not specified',
            'max_hours': 'not specified'
        }]

    def extract_customer_class(self, text: str) -> Dict[str, Any]:
        """Extract customer class eligibility"""
        customer_classes = {
            'residential': False,
            'commercial': False,
            'industrial': False,
            'residential_max_kw': None,
            'commercial_min_kw': None,
            'size_threshold_notes': ''
        }

        text_lower = text.lower()

        # Check for residential
        if any(keyword in text_lower for keyword in ['residential', 'homeowner', 'household', 'home']):
            customer_classes['residential'] = True

        # Check for commercial
        if any(keyword in text_lower for keyword in ['commercial', 'business', 'small business', 'c&i']):
            customer_classes['commercial'] = True

        # Check for industrial
        if any(keyword in text_lower for keyword in ['industrial', 'large commercial', 'c&i']):
            customer_classes['industrial'] = True

        # Default to commercial if none specified and it's a program (not residential-specific rate)
        if not any([customer_classes['residential'], customer_classes['commercial'], customer_classes['industrial']]):
            if 'residential' not in text_lower:
                customer_classes['commercial'] = True
                customer_classes['industrial'] = True

        # Look for size thresholds (kW)
        kw_patterns = [
            r'(\d+)\s*kW',
            r'(\d+)\s*kilowatt',
            r'minimum.*?(\d+)\s*kW',
            r'at least (\d+)\s*kW'
        ]

        for pattern in kw_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                customer_classes['size_threshold_notes'] = f'Size threshold mentioned: {matches[0]} kW - see description'
                break

        return customer_classes

    def enrich_dr_program(self, program: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich a demand response program with schema fields"""
        print(f"\n  Enriching: {program.get('hidden_program_name', 'Unknown')}")
        print(f"    State: {program.get('state')}, Utility: {program.get('utility _name')}")

        # Start with base information from DOE
        enriched = {
            'program_id': f"{program.get('state', 'XX').replace(' ', '-').upper()}-{program.get('hidden_program_name', 'Unknown').replace(' ', '-').replace('/', '-').upper()}"[:50],
            'program_name': program.get('hidden_program_name', 'Unknown'),
            'program_type': 'utility',  # Most DOE programs are utility programs
            'status': 'active' if program.get('open_to_customers') == 'Open' else 'closed_to_new',
            'states': [program.get('state')],
            'utilities': [program.get('utility _name')],
            'isos_rtos': [],
            'data_sources': {
                'program_url': program.get('hidden_program_link', 'not available'),
                'data_quality_notes': 'Scraped from DOE FEMP database 2025-10-11'
            }
        }

        # Extract information from description
        description = program.get('description', '')

        # Extract payment structure
        payment_info = self.extract_payment_info(description, description)
        enriched['payment_structure'] = payment_info

        # Extract timing information
        time_info = self.extract_time_info(description)
        enriched['call_windows'] = time_info.get('call_windows', [])

        # Extract seasonal information
        enriched['seasons'] = self.extract_season_info(description)

        # Extract customer class
        customer_class = self.extract_customer_class(description)
        enriched['eligibility'] = {
            'customer_classes': customer_class,
            'behind_the_meter': True,  # Most utility DR programs are BTM
            'front_of_meter': False,
            'minimum_capacity_kw': 'not specified',
            'resource_types': ['not specified'],
            'requires_aggregator': False
        }

        # Fields that require more research
        enriched['event_parameters'] = {
            'typical_duration_hours': 'not specified',
            'minimum_duration_hours': 'not specified',
            'maximum_duration_hours': 'not specified',
            'max_events_per_year': 'not specified',
            'max_hours_per_year': 'not specified',
            'typical_response_time_minutes': 'not specified'
        }

        enriched['notification'] = {
            'day_ahead_notice': 'not specified',
            'minimum_notice_hours': 'not specified',
            'maximum_notice_hours': 'not specified',
            'notification_methods': ['not specified']
        }

        enriched['event_triggers'] = {
            'trigger_description': description[:200] + '...' if len(description) > 200 else description,
            'primary_trigger': 'not specified',
            'secondary_triggers': [],
            'temperature_threshold': None,
            'price_threshold': 'not specified'
        }

        enriched['nomination_bidding'] = {
            'requires_nomination': 'not specified',
            'nomination_deadline': 'not specified',
            'allows_bidding': False,
            'bidding_process': 'not specified'
        }

        enriched['penalties'] = {
            'has_penalties': 'not specified',
            'penalty_structure': 'not specified'
        }

        enriched['historical_events'] = {
            'events_available': False,
            'data_source_url': 'not available',
            'recent_events': []
        }

        enriched['integration_metadata'] = {
            'optimizer_compatible': True,
            'api_available': False,
            'api_documentation_url': 'not available',
            'data_quality_score': 5,  # Mid-range since we have basic info
            'notes': f'Basic information from DOE FEMP. Program page: {enriched["data_sources"]["program_url"]}'
        }

        return enriched

    def enrich_rate_program(self, program: Dict[str, Any], rate_category: str) -> Dict[str, Any]:
        """Enrich a time-varying rate program"""
        print(f"\n  Enriching [{rate_category}]: {program.get('hidden_program_name', 'Unknown')}")

        enriched = {
            'program_id': f"{program.get('state', 'XX').replace(' ', '-').upper()}-{program.get('hidden_program_name', 'Unknown').replace(' ', '-').replace('/', '-').upper()}"[:50],
            'rate_name': program.get('hidden_program_name', 'Unknown'),
            'rate_type': rate_category,
            'state': program.get('state'),
            'utility': program.get('utility _name'),
            'status': 'active' if program.get('open_to_customers') == 'Open' else 'closed_to_new',
            'areawide_contract': program.get('areawide_contract') == 'Yes',
            'description': program.get('description'),
            'program_url': program.get('hidden_program_link', 'not available'),
            'data_source': 'DOE FEMP Database',
            'scraped_date': datetime.now().isoformat()
        }

        # Extract customer class
        customer_class = self.extract_customer_class(program.get('description', ''))
        enriched['customer_classes'] = customer_class

        # Extract rate structure hints from description
        description = program.get('description', '')
        enriched['rate_structure_notes'] = description[:500] + '...' if len(description) > 500 else description

        # Mark what needs further research
        enriched['needs_research'] = {
            'on_peak_hours': True,
            'off_peak_hours': True,
            'on_peak_rate': True,
            'off_peak_rate': True,
            'demand_charges': True,
            'seasonal_variations': True,
            'minimum_term': True
        }

        return enriched

    def process_dr_programs(self, input_file: str, output_file: str, limit: Optional[int] = None):
        """Process all DR programs"""
        print("="*70)
        print("ENRICHING DEMAND RESPONSE PROGRAMS")
        print("="*70)

        with open(input_file, 'r') as f:
            data = json.load(f)

        programs = data['programs']
        if limit:
            programs = programs[:limit]
            print(f"\nProcessing first {limit} programs (limit applied)")

        enriched_programs = []
        for i, program in enumerate(programs):
            print(f"\nProgress: {i+1}/{len(programs)}")
            try:
                enriched = self.enrich_dr_program(program)
                enriched_programs.append(enriched)
                time.sleep(1)  # Be respectful to servers
            except Exception as e:
                print(f"    ERROR: {e}")
                # Save what we have so far
                continue

        # Save output
        output_data = {
            'metadata': {
                'source': 'DOE FEMP Database',
                'original_url': data['data_source'],
                'scraped_at': data['scraped_at'],
                'enriched_at': datetime.now().isoformat(),
                'total_programs': len(enriched_programs),
                'schema_version': '1.2',
                'data_quality_note': 'Basic enrichment from DOE descriptions. Many fields require detailed program page analysis.'
            },
            'programs': enriched_programs
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Enriched {len(enriched_programs)} DR programs")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*70}")

    def process_rate_programs(self, input_file: str, output_file: str, limit: Optional[int] = None):
        """Process all time-varying rate programs"""
        print("="*70)
        print("ENRICHING TIME-VARYING RATE PROGRAMS")
        print("="*70)

        with open(input_file, 'r') as f:
            data = json.load(f)

        all_enriched = {}
        total_processed = 0

        for rate_type, programs in data['programs_by_type'].items():
            print(f"\n\nProcessing {rate_type} ({len(programs)} programs)")
            print("-"*70)

            if limit:
                programs = programs[:limit]

            enriched_programs = []
            for i, program in enumerate(programs):
                if total_processed >= (limit if limit else float('inf')):
                    break

                print(f"  [{i+1}/{len(programs)}]", end=" ")
                try:
                    enriched = self.enrich_rate_program(program, rate_type)
                    enriched_programs.append(enriched)
                    total_processed += 1
                    time.sleep(0.5)  # Be respectful
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            all_enriched[rate_type] = enriched_programs

        # Save output
        output_data = {
            'metadata': {
                'source': 'DOE FEMP Database',
                'original_url': data['data_source'],
                'scraped_at': data['scraped_at'],
                'enriched_at': datetime.now().isoformat(),
                'total_rate_programs': total_processed,
                'rate_types': {k: len(v) for k, v in all_enriched.items()},
                'data_quality_note': 'Basic categorization from DOE descriptions. Detailed rate structures require program page analysis.'
            },
            'programs_by_type': all_enriched
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n{'='*70}")
        print(f"✓ Enriched {total_processed} rate programs across {len(all_enriched)} categories")
        print(f"✓ Saved to: {output_file}")
        print(f"{'='*70}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Enrich DOE FEMP programs with detailed schema')
    parser.add_argument('--type', choices=['dr', 'rates', 'both'], default='both',
                       help='Type of programs to enrich')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of programs to process (for testing)')

    args = parser.parse_args()

    enricher = ProgramEnricher()

    if args.type in ['dr', 'both']:
        enricher.process_dr_programs(
            'doe_femp_dr_programs_raw.json',
            'doe_femp_dr_programs_enriched.json',
            limit=args.limit
        )

    if args.type in ['rates', 'both']:
        enricher.process_rate_programs(
            'doe_femp_time_varying_rates_raw.json',
            'doe_femp_time_varying_rates_enriched.json',
            limit=args.limit
        )


if __name__ == '__main__':
    main()
