#!/usr/bin/env python3
"""
Evaluate the four-pass matching results and propose improvements
"""

import pandas as pd
import numpy as np
import json

def analyze_four_pass_results():
    """Comprehensive analysis of four-pass matching results"""
    
    # Load the results
    df = pd.read_csv('/home/enrico/projects/power_market_pipeline/BESS_FOUR_PASS_MATCHED.csv')
    
    print('='*70)
    print('FOUR-PASS MATCHING EVALUATION REPORT')
    print('='*70)
    
    # Overall statistics
    total = len(df)
    
    # Categorize by match quality
    excellent = df[df['match_score'] >= 90]
    good = df[(df['match_score'] >= 70) & (df['match_score'] < 90)]
    fair = df[(df['match_score'] >= 50) & (df['match_score'] < 70)]
    poor = df[(df['match_score'] > 0) & (df['match_score'] < 50)]
    no_match = df[df['match_score'] == 0]
    
    print('\n📊 OVERALL MATCH QUALITY DISTRIBUTION')
    print('-'*40)
    print(f'Total BESS Resources: {total}')
    print(f'  Excellent (≥90%): {len(excellent):3} ({100*len(excellent)/total:5.1f}%)')
    print(f'  Good (70-89%):    {len(good):3} ({100*len(good)/total:5.1f}%)')
    print(f'  Fair (50-69%):    {len(fair):3} ({100*len(fair)/total:5.1f}%)')
    print(f'  Poor (<50%):      {len(poor):3} ({100*len(poor)/total:5.1f}%)')
    print(f'  No match:         {len(no_match):3} ({100*len(no_match)/total:5.1f}%)')
    print(f'\nTotal Matched: {total - len(no_match)} ({100*(total - len(no_match))/total:.1f}%)')
    
    # Analyze by source
    print('\n📈 MATCHES BY DATA SOURCE')
    print('-'*40)
    matched = df[df['match_score'] > 0]
    if 'Source' in matched.columns:
        source_counts = matched['Source'].value_counts()
        for source, count in source_counts.items():
            print(f'{source:25} {count:3} ({100*count/len(matched):5.1f}%)')
    
    # Pass-by-pass contribution analysis
    print('\n🔄 PASS-BY-PASS CONTRIBUTION')
    print('-'*40)
    
    # Estimate contributions based on source
    pass1_operational = len(df[(df['Source'] == 'Operational') | (df['Source'] == 'Standalone')])
    pass2_llm_operational = len(df[df['Source'] == 'LLM Match'])
    pass3_planned_heuristic = len(df[df['Source'].str.contains('Planned', na=False) & 
                                     ~df['Source'].str.contains('LLM', na=False)])
    pass4_llm_planned = len(df[df['Source'] == 'LLM Match (Planned)'])
    
    print(f'Pass 1 (Heuristic/Operational): {pass1_operational:3} matches')
    print(f'Pass 2 (LLM/Operational):       {pass2_llm_operational:3} matches (+{pass2_llm_operational})')
    print(f'Pass 3 (Heuristic/Planned):     {pass3_planned_heuristic:3} matches (+{pass3_planned_heuristic})')
    print(f'Pass 4 (LLM/Planned):           {pass4_llm_planned:3} matches (+{pass4_llm_planned})')
    
    cumulative = pass1_operational + pass2_llm_operational + pass3_planned_heuristic + pass4_llm_planned
    print(f'\nCumulative improvement: {cumulative}/{total} ({100*cumulative/total:.1f}%)')
    
    # Analyze LLM effectiveness
    print('\n🤖 LLM MATCHING EFFECTIVENESS')
    print('-'*40)
    
    llm_matches = df[df['Source'].str.contains('LLM', na=False)]
    if len(llm_matches) > 0:
        print(f'Total LLM matches: {len(llm_matches)}')
        print(f'Average confidence: {llm_matches["match_score"].mean():.1f}%')
        
        # Show sample LLM matches
        print('\nSample LLM Matches:')
        sample_cols = ['BESS_Gen_Resource', 'match_score', 'Source']
        if 'IQ_Project_Name' in llm_matches.columns:
            sample_cols.append('IQ_Project_Name')
        elif 'Project Name' in llm_matches.columns:
            sample_cols.append('Project Name')
        
        available_cols = [col for col in sample_cols if col in llm_matches.columns]
        print(llm_matches[available_cols].head(5).to_string(index=False))
    
    # Analyze unmatched patterns
    print('\n❌ UNMATCHED RESOURCES ANALYSIS')
    print('-'*40)
    print(f'Total unmatched: {len(no_match)}')
    
    if len(no_match) > 0:
        # Common prefixes
        prefixes = {}
        for name in no_match['BESS_Gen_Resource']:
            if pd.notna(name) and '_' in name:
                prefix = name.split('_')[0]
                prefixes[prefix] = prefixes.get(prefix, 0) + 1
        
        print('\nTop unmatched prefixes:')
        for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f'  {prefix:15} {count:2} resources')
        
        # Missing location data
        no_zone = no_match[no_match['Load_Zone'].isna()]
        no_sub = no_match[no_match['Substation'].isna()]
        print(f'\nMissing location data:')
        print(f'  Without Load Zone:  {len(no_zone)}')
        print(f'  Without Substation: {len(no_sub)}')
    
    # Success patterns
    print('\n✅ SUCCESSFUL MATCHING PATTERNS')
    print('-'*40)
    
    if 'match_reason' in df.columns:
        exact_matches = df[df['match_reason'].str.contains('exact', case=False, na=False)]
        county_matches = df[df['match_reason'].str.contains('county', case=False, na=False)]
        similarity_matches = df[df['match_reason'].str.contains('similarity', case=False, na=False)]
        
        print(f'Exact matches:      {len(exact_matches)}')
        print(f'County verified:    {len(county_matches)}')
        print(f'Name similarity:    {len(similarity_matches)}')
    
    return df

def generate_improvement_recommendations():
    """Generate specific recommendations for improving matching"""
    
    print('\n' + '='*70)
    print('RECOMMENDATIONS FOR IMPROVEMENT')
    print('='*70)
    
    recommendations = {
        "1. Data Enrichment": [
            "• Add more county mappings for unmatched substations",
            "• Include Texas railroad district data for geographic matching",
            "• Map developer subsidiaries and joint ventures",
            "• Add historical project name changes/variations"
        ],
        
        "2. Heuristic Improvements": [
            "• Lower similarity threshold for Planned projects (0.5 instead of 0.6)",
            "• Add fuzzy matching for county names (adjacent counties)",
            "• Include capacity-based filtering (±20% range)",
            "• Match on partial POI names (e.g., '138kV' suffix patterns)"
        ],
        
        "3. LLM Prompt Enhancements": [
            "• Add specific Texas county adjacency information",
            "• Include more developer name variations and abbreviations",
            "• Provide COD timeline context (2025-2028 typical for new projects)",
            "• Add examples of successful matches for few-shot learning"
        ],
        
        "4. Multi-Pass Strategy": [
            "• Add Pass 5: Relaxed matching (lower thresholds, broader search)",
            "• Consider geographic clustering for unmatched resources",
            "• Use embedding-based similarity for semantic matching",
            "• Implement confidence-weighted ensemble of multiple methods"
        ],
        
        "5. Specific Pattern Recognition": [
            "• SWOOSE variants (military/aviation themed names)",
            "• Numeric suffixes (_UNIT1, _BES1) as version indicators",
            "• Developer-specific patterns (BRP uses coded names)",
            "• Geographic abbreviations (HEIGHTTN, LOPENO, MADERO)"
        ],
        
        "6. Process Optimization": [
            "• Cache LLM responses for similar queries",
            "• Batch process by developer/region for context",
            "• Use cheaper models for initial filtering",
            "• Implement active learning from manual corrections"
        ]
    }
    
    for category, items in recommendations.items():
        print(f'\n{category}')
        print('-'*40)
        for item in items:
            print(item)
    
    # Specific unmatched resource strategies
    print('\n7. Strategies for Common Unmatched Patterns')
    print('-'*40)
    
    unmatched_strategies = {
        "SWOOSE projects": "Military/aviation theme - check Ward/Reeves counties",
        "BRP_ projects": "Broad Reach Power - check their project pipeline",
        "MADERO/LOPENO": "Rio Grande Valley projects - check Hidalgo/Starr counties",
        "HEIGHTTN": "Houston area - check Harris County projects",
        "ESS/BESS doubles": "Check for registration errors or test projects"
    }
    
    for pattern, strategy in unmatched_strategies.items():
        print(f'  {pattern:15} → {strategy}')
    
    # Implementation priority
    print('\n📋 IMPLEMENTATION PRIORITY')
    print('-'*40)
    print('1. HIGH: Add Pass 5 with relaxed thresholds')
    print('2. HIGH: Enhance county mapping database')
    print('3. MEDIUM: Implement embedding-based matching')
    print('4. MEDIUM: Add few-shot examples to LLM prompts')
    print('5. LOW: Build active learning pipeline')

def calculate_improvement_potential():
    """Calculate potential improvement with recommendations"""
    
    print('\n' + '='*70)
    print('IMPROVEMENT POTENTIAL ANALYSIS')
    print('='*70)
    
    current_unmatched = 69  # From four-pass results
    total = 195
    
    improvements = {
        "Relaxed thresholds (Pass 5)": 10,
        "Enhanced county mapping": 8,
        "Embedding-based matching": 12,
        "Better LLM prompts": 5,
        "Developer pattern recognition": 7
    }
    
    print('\nEstimated additional matches with improvements:')
    print('-'*40)
    
    cumulative = 0
    for improvement, estimate in improvements.items():
        cumulative += estimate
        print(f'{improvement:30} +{estimate:2} matches')
    
    print(f'\nTotal potential improvement: +{cumulative} matches')
    
    current_matched = total - current_unmatched
    potential_matched = current_matched + cumulative
    
    print(f'\nCurrent match rate: {current_matched}/{total} ({100*current_matched/total:.1f}%)')
    print(f'Potential match rate: {potential_matched}/{total} ({100*potential_matched/total:.1f}%)')
    print(f'Remaining unmatched: {total - potential_matched} (likely new/unnamed projects)')
    
    # ROI Analysis
    print('\n💰 RETURN ON INVESTMENT')
    print('-'*40)
    
    implementation_hours = {
        "Relaxed thresholds": 2,
        "County mapping": 4,
        "Embedding matching": 8,
        "LLM improvements": 3,
        "Pattern recognition": 5
    }
    
    total_hours = sum(implementation_hours.values())
    matches_per_hour = cumulative / total_hours
    
    print(f'Estimated implementation: {total_hours} hours')
    print(f'Expected improvement: {cumulative} matches')
    print(f'Efficiency: {matches_per_hour:.1f} matches/hour of work')
    
    if matches_per_hour > 1.5:
        print('\n✅ Recommendation: HIGH VALUE - Proceed with improvements')
    else:
        print('\n⚠️  Recommendation: Evaluate if manual matching is more efficient')

if __name__ == '__main__':
    # Analyze results
    df = analyze_four_pass_results()
    
    # Generate recommendations
    generate_improvement_recommendations()
    
    # Calculate potential
    calculate_improvement_potential()
    
    print('\n' + '='*70)
    print('EVALUATION COMPLETE')
    print('='*70)