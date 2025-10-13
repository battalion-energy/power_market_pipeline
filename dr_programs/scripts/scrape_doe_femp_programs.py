#!/usr/bin/env python3
"""
DOE FEMP Demand Response and Time-Variable Pricing Programs Scraper

Scrapes all program data from:
https://www.energy.gov/femp/demand-response-and-time-variable-pricing-programs-search

Separates DR programs from time-varying rate programs.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
from typing import List, Dict, Any
from datetime import datetime
import re


class DOEFEMPScraper:
    """Scraper for DOE FEMP DR and pricing programs"""

    BASE_URL = "https://www.energy.gov"
    SEARCH_URL = "https://www.energy.gov/femp/demand-response-and-time-variable-pricing-programs-search"

    # Time-varying rate types
    RATE_TYPES = {
        'TOU': ['TOU', 'Time-of-Use', 'Time of Use'],
        'Real-Time Energy': ['Real-Time', 'Real Time Energy', 'RTP'],
        'Real-Time Indexed': ['Real-Time Indexed', 'RT Indexed'],
        'Day-Ahead Indexed': ['Day-Ahead Indexed', 'DA Indexed'],
        'Interruptible/Curtailable': ['Interruptible', 'Curtailable', 'Interruptible/Curtailable'],
        'Load Reduction Rate/Rider': ['Load Reduction', 'Rider'],
        'Critical Peak Pricing': ['CPP', 'Critical Peak'],
        'Variable Peak Pricing': ['VPP', 'Variable Peak'],
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

    def fetch_page(self, url: str, retries: int = 3) -> BeautifulSoup:
        """Fetch and parse a page"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                print(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def scrape_main_table(self) -> List[Dict[str, Any]]:
        """Scrape the main programs table from JavaScript DataTables data"""
        print(f"Fetching main page: {self.SEARCH_URL}")
        soup = self.fetch_page(self.SEARCH_URL)

        programs = []

        # The data is embedded in JavaScript using DataTables
        # Look for script tags containing datatableRows
        scripts = soup.find_all('script')

        columns = []
        rows_data = []

        for script in scripts:
            if not script.string:
                continue

            script_text = script.string

            # Extract column definitions
            if 'datatableColumns' in script_text:
                # Find the datatableColumns array
                match = re.search(r'datatableColumns\s*:\s*(\[.*?\])', script_text, re.DOTALL)
                if match:
                    try:
                        # Extract column data - look for data/name fields
                        column_matches = re.findall(r'\{\s*data\s*:\s*[\'"](\w+)[\'"]', match.group(1))
                        if column_matches:
                            columns = column_matches
                            print(f"Found columns: {columns}")
                    except Exception as e:
                        print(f"Error parsing columns: {e}")

            # Extract row data
            if 'datatableRows' in script_text:
                # Find start of datatableRows array
                start_match = re.search(r'"datatableRows"\s*:\s*\[', script_text)
                if start_match:
                    start_pos = start_match.end() - 1  # Position of opening bracket
                    # Use bracket counter to find matching closing bracket
                    bracket_count = 0
                    in_string = False
                    escape_next = False
                    end_pos = start_pos

                    for i in range(start_pos, len(script_text)):
                        char = script_text[i]

                        if escape_next:
                            escape_next = False
                            continue

                        if char == '\\':
                            escape_next = True
                            continue

                        if char == '"' and not escape_next:
                            in_string = not in_string
                            continue

                        if not in_string:
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end_pos = i + 1
                                    break

                    if end_pos > start_pos:
                        try:
                            rows_json = script_text[start_pos:end_pos]
                            rows_data = json.loads(rows_json)
                            print(f"Found {len(rows_data)} programs in DataTables data")
                        except json.JSONDecodeError as e:
                            print(f"JSON parsing error: {e}")
                            print(f"Extracted {len(rows_json)} characters")
                        except Exception as e:
                            print(f"Error parsing rows: {e}")

        # If we have both columns and rows, combine them
        if rows_data:
            for row in rows_data:
                if isinstance(row, dict):
                    # Data is already in dictionary format
                    programs.append(row)
                elif isinstance(row, list) and columns:
                    # Data is in array format, map to columns
                    program = {}
                    for i, col_name in enumerate(columns):
                        if i < len(row):
                            program[col_name] = row[i]
                    programs.append(program)

            print(f"Successfully extracted {len(programs)} programs")

        if not programs:
            print("\n⚠ Could not extract DataTables data")
            print("Attempting to save raw script for manual inspection...")

            # Save script content for debugging
            for i, script in enumerate(scripts):
                if script.string and ('datatable' in script.string.lower() or 'programs' in script.string.lower()):
                    with open(f'debug_script_{i}.txt', 'w') as f:
                        f.write(script.string)
                    print(f"  Saved script to debug_script_{i}.txt")

        return programs

    def categorize_program(self, program: Dict[str, Any]) -> str:
        """Categorize program as DR or specific rate type"""
        # Use actual field names from DataTables
        program_type = program.get('program_or_rate', '').lower()
        program_name = program.get('hidden_program_name', program.get('program_name', '')).lower()
        description = program.get('description', '').lower()

        combined_text = f"{program_type} {program_name} {description}"

        # Check for time-varying rate types
        for rate_type, keywords in self.RATE_TYPES.items():
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    return rate_type

        # Check if it's explicitly a rate vs program
        if 'rate' in program_type or 'pricing' in program_type or 'tariff' in combined_text:
            return 'Other Rate Type'

        # Default to DR program
        return 'Demand Response Program'

    def scrape_program_details(self, program: Dict[str, Any]) -> Dict[str, Any]:
        """Scrape detailed information from program page if available"""
        link = program.get('Program Name_link')
        if not link:
            return program

        try:
            print(f"  Fetching details for: {program.get('Program Name', 'Unknown')}")
            soup = self.fetch_page(link)

            # Extract main content
            content_div = soup.find('div', class_=re.compile(r'content|main|article'))
            if content_div:
                # Get all text
                full_text = content_div.get_text(separator='\n', strip=True)
                program['detailed_description'] = full_text[:2000]  # Limit length

            # Look for specific fields
            # This will vary by page structure

            time.sleep(1)  # Be respectful to the server

        except Exception as e:
            print(f"  Error fetching details: {e}")

        return program

    def save_programs(self, programs: List[Dict[str, Any]]):
        """Separate and save programs into appropriate files"""

        dr_programs = []
        rate_programs = {
            'TOU': [],
            'Real-Time Energy': [],
            'Real-Time Indexed': [],
            'Day-Ahead Indexed': [],
            'Interruptible/Curtailable': [],
            'Load Reduction Rate/Rider': [],
            'Critical Peak Pricing': [],
            'Variable Peak Pricing': [],
            'Other Rate Type': []
        }

        print("\nCategorizing programs...")
        for program in programs:
            category = self.categorize_program(program)
            print(f"  {program.get('Program Name', 'Unknown')}: {category}")

            if category == 'Demand Response Program':
                dr_programs.append(program)
            elif category in rate_programs:
                rate_programs[category].append(program)
            else:
                # New category discovered
                if category not in rate_programs:
                    rate_programs[category] = []
                rate_programs[category].append(program)

        # Save DR programs
        dr_output = {
            'data_source': self.SEARCH_URL,
            'scraped_at': datetime.now().isoformat(),
            'total_programs': len(dr_programs),
            'programs': dr_programs
        }

        with open('doe_femp_dr_programs_raw.json', 'w') as f:
            json.dump(dr_output, f, indent=2)
        print(f"\n✓ Saved {len(dr_programs)} DR programs to: doe_femp_dr_programs_raw.json")

        # Save time-varying rates
        total_rates = sum(len(programs) for programs in rate_programs.values())
        rates_output = {
            'data_source': self.SEARCH_URL,
            'scraped_at': datetime.now().isoformat(),
            'total_rate_programs': total_rates,
            'rate_types': {k: len(v) for k, v in rate_programs.items() if v},
            'programs_by_type': {k: v for k, v in rate_programs.items() if v}
        }

        with open('doe_femp_time_varying_rates_raw.json', 'w') as f:
            json.dump(rates_output, f, indent=2)
        print(f"✓ Saved {total_rates} time-varying rate programs to: doe_femp_time_varying_rates_raw.json")

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total programs scraped: {len(programs)}")
        print(f"  - Demand Response Programs: {len(dr_programs)}")
        print(f"  - Time-Varying Rate Programs: {total_rates}")
        if rate_programs:
            print("\nRate Program Breakdown:")
            for rate_type, progs in sorted(rate_programs.items()):
                if progs:
                    print(f"    - {rate_type}: {len(progs)}")

        return dr_programs, rate_programs

    def run(self, fetch_details: bool = False):
        """Run the full scraping process"""
        print("="*60)
        print("DOE FEMP Programs Scraper")
        print("="*60)

        # Scrape main table
        programs = self.scrape_main_table()

        if not programs:
            print("\n⚠ No programs found. The page structure may have changed.")
            print("  Please manually inspect the page and update the scraper.")
            return

        # Optionally fetch detailed information
        if fetch_details:
            print("\nFetching detailed information for each program...")
            for i, program in enumerate(programs):
                print(f"Progress: {i+1}/{len(programs)}")
                programs[i] = self.scrape_program_details(program)

        # Save programs
        self.save_programs(programs)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Scrape DOE FEMP DR and pricing programs')
    parser.add_argument('--fetch-details', action='store_true',
                       help='Fetch detailed information from each program page (slower)')

    args = parser.parse_args()

    scraper = DOEFEMPScraper()
    scraper.run(fetch_details=args.fetch_details)


if __name__ == '__main__':
    main()
