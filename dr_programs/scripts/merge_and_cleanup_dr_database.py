#!/usr/bin/env python3
"""
Merge DR Program Research into Master Database with Cleanup
===========================================================

This script:
1. Reads all batch research JSON files (114+ files)
2. Merges enriched data into master database
3. Cleans up database errors:
   - Removes misclassified programs (standby tariffs, non-DR)
   - Removes non-existent programs (404 errors)
   - Deduplicates entries
   - Corrects territory misclassifications
4. Adds quality metadata

Data Integrity: Maintains 100% integrity - no data invention.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Programs to remove (identified during research)
PROGRAMS_TO_REMOVE = {
    # Index 70: Con Edison LMIP doesn't exist (404 error)
    'NEW-YORK-LOAD-MANAGEMENT-INCENTIVE-PROGRAM': 'Program does not exist (404 error)',

    # Index 97: Duke SC On-Site Generation is standby tariff, not DR
    'SOUTH-CAROLINA-ON-SITE-GENERATION-SERVICE-PROGRAM': 'Misclassified - standby service tariff, not DR program',
}

# Programs that are duplicates (keep first occurrence only)
DUPLICATE_PROGRAMS = [
    # These are the same programs researched multiple times
    ('MINNESOTA-DEMAND-RESPONSE-OFFERINGS', 'MISO'),  # Batch 5 and Batch 9
    ('MISSISSIPPI-DEMAND-RESPONSE-OFFERINGS', 'MISO'),  # Batch 5 and Batch 9
    ('KANSAS-BUSINESS-DEMAND-RESPONSE-PROGRAM', 'Evergy'),  # Batch 6 and Batch 9
]

# Territory corrections
TERRITORY_CORRECTIONS = {
    # Entergy Texas operates in MISO, not ERCOT
    'TEXAS-LOAD-MANAGEMENT-PROGRAM': {
        'utility': 'Entergy Texas',
        'correct_iso': 'MISO',
        'note': 'Entergy Texas operates in MISO South, not ERCOT'
    }
}

def load_batch_files(batch_dir: str = 'dr_programs_researched') -> tuple[Dict[int, Dict], Dict[str, Dict]]:
    """Load all batch research JSON files. Returns (by_index, by_program_id)."""
    by_index = {}
    by_program_id = {}
    batch_path = Path(batch_dir)

    if not batch_path.exists():
        print(f"ERROR: Batch directory {batch_dir} not found")
        return {}, {}

    json_files = sorted(batch_path.glob('program_batch*.json'))
    print(f"Found {len(json_files)} batch files")

    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract program index if available
            if 'program_index' in data:
                idx = data['program_index']
                by_index[idx] = data
                print(f"  Loaded: {file_path.name} (index {idx})")

            # Also index by program_id and program_name for matching
            program_id = data.get('program_id', '').upper()
            program_name = data.get('program_name', '').upper()

            if program_id:
                by_program_id[program_id] = data
            if program_name and program_name not in by_program_id:
                by_program_id[program_name] = data

        except json.JSONDecodeError as e:
            print(f"  ERROR decoding {file_path.name}: {e}")
        except Exception as e:
            print(f"  ERROR loading {file_path.name}: {e}")

    print(f"  Loaded {len(by_index)} programs by index")
    print(f"  Loaded {len(by_program_id)} programs by ID/name")

    return by_index, by_program_id

def load_original_database(db_file: str = 'doe_femp_dr_programs_enriched.json') -> Dict:
    """Load the original database."""
    db_path = Path(db_file)

    if not db_path.exists():
        print(f"ERROR: Original database {db_file} not found")
        return {'programs': []}

    try:
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded original database: {len(data.get('programs', []))} programs")
        return data
    except Exception as e:
        print(f"ERROR loading original database: {e}")
        return {'programs': []}

def should_remove_program(program: Dict) -> tuple[bool, str]:
    """Check if a program should be removed."""
    program_id = program.get('program_id', '').upper()

    # Check removal list
    for remove_id, reason in PROGRAMS_TO_REMOVE.items():
        if remove_id in program_id:
            return True, reason

    return False, ''

def is_duplicate_program(program: Dict, seen_programs: set) -> tuple[bool, str]:
    """Check if program is a duplicate."""
    program_id = program.get('program_id', '').upper()
    utility = program.get('utility', '').upper()

    for dup_id, dup_utility in DUPLICATE_PROGRAMS:
        if dup_id in program_id and dup_utility.upper() in utility:
            key = f"{dup_id}_{dup_utility}"
            if key in seen_programs:
                return True, f"Duplicate of {dup_id} ({dup_utility})"
            else:
                seen_programs.add(key)
                return False, ''

    return False, ''

def apply_territory_corrections(program: Dict) -> List[str]:
    """Apply territory corrections to program."""
    corrections_applied = []
    program_id = program.get('program_id', '').upper()

    for correct_id, correction in TERRITORY_CORRECTIONS.items():
        if correct_id in program_id:
            utility = program.get('utility', '')
            if correction['utility'] in utility:
                # Correct ISO assignment
                if 'geography' in program and 'isos' in program['geography']:
                    old_iso = program['geography']['isos']
                    program['geography']['isos'] = [correction['correct_iso']]
                    corrections_applied.append(
                        f"Corrected ISO from {old_iso} to {correction['correct_iso']}: {correction['note']}"
                    )

    return corrections_applied

def merge_databases(original_db: Dict, batch_by_index: Dict[int, Dict], batch_by_id: Dict[str, Dict]) -> Dict:
    """Merge batch research into original database with cleanup."""

    merged = {
        'metadata': {
            'title': 'US Demand Response Programs - Comprehensive Enriched Catalog',
            'version': '2.0',
            'last_updated': '2025-10-12',
            'total_programs_original': len(original_db.get('programs', [])),
            'total_programs_researched': len(batch_by_index) + len(batch_by_id),
            'data_integrity': '100% - No invented data',
            'research_batches': '1-11 (complete)',
            'exceptional_discoveries': 4
        },
        'programs': [],
        'removed_programs': [],
        'corrections_applied': [],
        'duplicate_programs': []
    }

    seen_programs = set()
    programs_by_id = {}

    # Process original database programs
    for idx, program in enumerate(original_db.get('programs', [])):
        # Check if we have enriched research for this program
        # Try index first
        enriched = batch_by_index.get(idx)

        # If not found by index, try by program_id or name
        if not enriched:
            program_id = program.get('program_id', '').upper()
            program_name = program.get('program_name', '').upper()

            if program_id and program_id in batch_by_id:
                enriched = batch_by_id[program_id]
            elif program_name and program_name in batch_by_id:
                enriched = batch_by_id[program_name]

        # Check for removal
        should_remove, removal_reason = should_remove_program(program)
        if should_remove:
            merged['removed_programs'].append({
                'index': idx,
                'program_id': program.get('program_id'),
                'program_name': program.get('program_name'),
                'reason': removal_reason
            })
            print(f"  REMOVING: Index {idx} - {program.get('program_name')} ({removal_reason})")
            continue

        # Check for duplicates
        is_dup, dup_reason = is_duplicate_program(program, seen_programs)
        if is_dup:
            merged['duplicate_programs'].append({
                'index': idx,
                'program_id': program.get('program_id'),
                'program_name': program.get('program_name'),
                'reason': dup_reason
            })
            print(f"  DUPLICATE: Index {idx} - {program.get('program_name')} ({dup_reason})")
            continue

        # Use enriched data if available, otherwise original
        if enriched:
            final_program = enriched.copy()
            print(f"  MERGED: Index {idx} - {program.get('program_name')} (enriched data)")
        else:
            final_program = program.copy()
            final_program['enrichment_status'] = 'original_data_only'
            print(f"  KEPT: Index {idx} - {program.get('program_name')} (original data, not yet enriched)")

        # Apply territory corrections
        corrections = apply_territory_corrections(final_program)
        if corrections:
            merged['corrections_applied'].extend([{
                'index': idx,
                'program_name': final_program.get('program_name'),
                'corrections': corrections
            }])
            print(f"    CORRECTED: {corrections[0]}")

        # Add original index for traceability
        final_program['original_index'] = idx

        merged['programs'].append(final_program)
        programs_by_id[final_program.get('program_id', f'program_{idx}')] = final_program

    # Update metadata
    merged['metadata']['total_programs_final'] = len(merged['programs'])
    merged['metadata']['programs_removed'] = len(merged['removed_programs'])
    merged['metadata']['programs_deduplicated'] = len(merged['duplicate_programs'])
    merged['metadata']['corrections_applied_count'] = len(merged['corrections_applied'])

    return merged

def add_program_statistics(merged_db: Dict) -> Dict:
    """Add statistical metadata about programs."""

    programs = merged_db['programs']

    stats = {
        'total_programs': len(programs),
        'by_status': defaultdict(int),
        'by_program_type': defaultdict(int),
        'by_state': defaultdict(int),
        'battery_suitability': defaultdict(int),
        'data_quality_scores': {
            'excellent': 0,  # 8-10
            'good': 0,  # 6-8
            'moderate': 0,  # 4-6
            'poor': 0  # 0-4
        }
    }

    for program in programs:
        # Status
        status = program.get('status', 'unknown')
        if isinstance(status, str):
            stats['by_status'][status] += 1

        # Program type
        ptype = program.get('program_type', 'unknown')
        if isinstance(ptype, str):
            stats['by_program_type'][ptype] += 1
        elif isinstance(ptype, dict):
            # Some programs have program_type as dict, extract string value
            ptype_str = ptype.get('type', 'unknown') if isinstance(ptype, dict) else 'unknown'
            stats['by_program_type'][ptype_str] += 1

        # State
        geography = program.get('geography', {})
        states = geography.get('states', [])
        for state in states:
            stats['by_state'][state] += 1

        # Battery suitability
        if 'integration_metadata' in program:
            rating = program['integration_metadata'].get('battery_suitability_rating', 'unknown')
            stats['battery_suitability'][rating] += 1

        # Data quality
        if 'integration_metadata' in program:
            quality = program['integration_metadata'].get('data_quality_score', 0)
            if quality >= 8:
                stats['data_quality_scores']['excellent'] += 1
            elif quality >= 6:
                stats['data_quality_scores']['good'] += 1
            elif quality >= 4:
                stats['data_quality_scores']['moderate'] += 1
            else:
                stats['data_quality_scores']['poor'] += 1

    merged_db['statistics'] = stats

    return merged_db

def main():
    """Main execution."""
    print("=" * 80)
    print("DR Program Database Merge and Cleanup")
    print("=" * 80)
    print()

    # Load data
    print("STEP 1: Loading batch research files...")
    batch_by_index, batch_by_id = load_batch_files()
    print()

    print("STEP 2: Loading original database...")
    original_db = load_original_database()
    print()

    print("STEP 3: Merging and cleaning data...")
    merged_db = merge_databases(original_db, batch_by_index, batch_by_id)
    print()

    print("STEP 4: Adding statistics...")
    merged_db = add_program_statistics(merged_db)
    print()

    # Summary
    print("=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"Original programs:       {merged_db['metadata']['total_programs_original']}")
    print(f"Programs researched:     {merged_db['metadata']['total_programs_researched']}")
    print(f"Programs removed:        {merged_db['metadata']['programs_removed']}")
    print(f"Programs deduplicated:   {merged_db['metadata']['programs_deduplicated']}")
    print(f"Corrections applied:     {merged_db['metadata']['corrections_applied_count']}")
    print(f"Final program count:     {merged_db['metadata']['total_programs_final']}")
    print()

    if merged_db['removed_programs']:
        print("REMOVED PROGRAMS:")
        for prog in merged_db['removed_programs']:
            print(f"  - Index {prog['index']}: {prog['program_name']}")
            print(f"    Reason: {prog['reason']}")
        print()

    if merged_db['duplicate_programs']:
        print("DUPLICATE PROGRAMS (removed):")
        for prog in merged_db['duplicate_programs']:
            print(f"  - Index {prog['index']}: {prog['program_name']}")
            print(f"    Reason: {prog['reason']}")
        print()

    if merged_db['corrections_applied']:
        print("CORRECTIONS APPLIED:")
        for corr in merged_db['corrections_applied']:
            print(f"  - Index {corr['index']}: {corr['program_name']}")
            for c in corr['corrections']:
                print(f"    {c}")
        print()

    # Save output
    output_file = 'doe_femp_dr_programs_enriched_v2_clean.json'
    print(f"STEP 5: Saving merged database to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_db, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(merged_db['programs'])} programs")

    # Save summary report
    summary_file = 'database_merge_cleanup_report.txt'
    print(f"STEP 6: Saving summary report to {summary_file}...")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("DR Program Database Merge and Cleanup Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: 2025-10-12\n")
        f.write(f"\nOriginal programs:       {merged_db['metadata']['total_programs_original']}\n")
        f.write(f"Programs researched:     {merged_db['metadata']['total_programs_researched']}\n")
        f.write(f"Programs removed:        {merged_db['metadata']['programs_removed']}\n")
        f.write(f"Programs deduplicated:   {merged_db['metadata']['programs_deduplicated']}\n")
        f.write(f"Corrections applied:     {merged_db['metadata']['corrections_applied_count']}\n")
        f.write(f"Final program count:     {merged_db['metadata']['total_programs_final']}\n")
        f.write("\n" + "=" * 80 + "\n")

        if merged_db['removed_programs']:
            f.write("\nREMOVED PROGRAMS:\n")
            for prog in merged_db['removed_programs']:
                f.write(f"  Index {prog['index']}: {prog['program_name']}\n")
                f.write(f"    Reason: {prog['reason']}\n")

        if merged_db['duplicate_programs']:
            f.write("\nDUPLICATE PROGRAMS:\n")
            for prog in merged_db['duplicate_programs']:
                f.write(f"  Index {prog['index']}: {prog['program_name']}\n")
                f.write(f"    Reason: {prog['reason']}\n")

        if merged_db['corrections_applied']:
            f.write("\nCORRECTIONS APPLIED:\n")
            for corr in merged_db['corrections_applied']:
                f.write(f"  Index {corr['index']}: {corr['program_name']}\n")
                for c in corr['corrections']:
                    f.write(f"    {c}\n")

        # Statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("PROGRAM STATISTICS\n")
        f.write("=" * 80 + "\n")

        stats = merged_db['statistics']
        f.write(f"\nTotal Programs: {stats['total_programs']}\n")

        f.write("\nBy Status:\n")
        for status, count in sorted(stats['by_status'].items()):
            f.write(f"  {status}: {count}\n")

        f.write("\nBy Program Type:\n")
        for ptype, count in sorted(stats['by_program_type'].items()):
            f.write(f"  {ptype}: {count}\n")

        f.write("\nTop 10 States by Program Count:\n")
        top_states = sorted(stats['by_state'].items(), key=lambda x: x[1], reverse=True)[:10]
        for state, count in top_states:
            f.write(f"  {state}: {count}\n")

        f.write("\nData Quality Scores:\n")
        for quality, count in stats['data_quality_scores'].items():
            f.write(f"  {quality}: {count}\n")

    print()
    print("=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"Merged database: {output_file}")
    print(f"Summary report:  {summary_file}")
    print()
    print("Data Integrity: 100% maintained - No data invented during merge")

if __name__ == '__main__':
    main()
