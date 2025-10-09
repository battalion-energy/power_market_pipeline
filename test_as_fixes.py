#!/usr/bin/env python3
"""
Test the fixed AS revenue calculations across different years
"""

import polars as pl
from pathlib import Path
from bess_revenue_calculator import BESSRevenueCalculator

def test_as_calculations():
    base_dir = Path("/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data")

    # Test batteries
    test_batteries = {
        'BATCAVE_BES1': 'BATCAVE_LD1',  # 100 MW
        'TINPD_BES1': 'TINPD_LD3',  # Should have good ECRS revenue
    }

    print("=" * 80)
    print("Testing AS Revenue Calculations - Old vs New Columns")
    print("=" * 80)

    # Test 2020 (old column structure)
    print("\n### Testing 2020 (Old Column Structure) ###")
    try:
        calc_2020 = BESSRevenueCalculator(base_dir, year=2020)
        for gen_name, load_name in test_batteries.items():
            print(f"\n{gen_name}:")

            # Test Gen AS
            gen_as = calc_2020.calculate_dam_as_revenue(gen_name, is_gen=True)
            print(f"  Gen RegUp:    ${gen_as.get('RegUp', 0):>12,.0f}")
            print(f"  Gen RegDown:  ${gen_as.get('RegDown', 0):>12,.0f}")
            print(f"  Gen RRS:      ${gen_as.get('RRS', 0):>12,.0f}")
            print(f"  Gen ECRS:     ${gen_as.get('ECRS', 0):>12,.0f}")
            print(f"  Gen NonSpin:  ${gen_as.get('NonSpin', 0):>12,.0f}")

            # Test Load AS
            load_as = calc_2020.calculate_dam_as_revenue(load_name, is_gen=False)
            print(f"  Load RegUp:   ${load_as.get('RegUp', 0):>12,.0f}")
            print(f"  Load RegDown: ${load_as.get('RegDown', 0):>12,.0f}")
            print(f"  Load RRS:     ${load_as.get('RRS', 0):>12,.0f}")
            print(f"  Load ECRS:    ${load_as.get('ECRS', 0):>12,.0f}")
            print(f"  Load NonSpin: ${load_as.get('NonSpin', 0):>12,.0f}")
    except Exception as e:
        print(f"ERROR in 2020: {e}")
        import traceback
        traceback.print_exc()

    # Test 2024 (new column structure)
    print("\n### Testing 2024 (New Column Structure) ###")
    try:
        calc_2024 = BESSRevenueCalculator(base_dir, year=2024)
        for gen_name, load_name in test_batteries.items():
            print(f"\n{gen_name}:")

            # Test Gen AS
            gen_as = calc_2024.calculate_dam_as_revenue(gen_name, is_gen=True)
            print(f"  Gen RegUp:    ${gen_as.get('RegUp', 0):>12,.0f}")
            print(f"  Gen RegDown:  ${gen_as.get('RegDown', 0):>12,.0f}")
            print(f"  Gen RRS:      ${gen_as.get('RRS', 0):>12,.0f}  <- Should be BIG!")
            print(f"  Gen ECRS:     ${gen_as.get('ECRS', 0):>12,.0f}  <- Should be BIG!")
            print(f"  Gen NonSpin:  ${gen_as.get('NonSpin', 0):>12,.0f}")

            # Test Load AS
            load_as = calc_2024.calculate_dam_as_revenue(load_name, is_gen=False)
            print(f"  Load RegUp:   ${load_as.get('RegUp', 0):>12,.0f}")
            print(f"  Load RegDown: ${load_as.get('RegDown', 0):>12,.0f}")
            print(f"  Load RRS:     ${load_as.get('RRS', 0):>12,.0f}  <- Should be BIG!")
            print(f"  Load ECRS:    ${load_as.get('ECRS', 0):>12,.0f}  <- Should be visible!")
            print(f"  Load NonSpin: ${load_as.get('NonSpin', 0):>12,.0f}")
    except Exception as e:
        print(f"ERROR in 2024: {e}")
        import traceback
        traceback.print_exc()

    # Verify TINPD Q3 2024 ECRS specifically (we know it should be ~$35k)
    print("\n### Specific Q3 2024 ECRS Verification (TINPD_LD3) ###")
    try:
        dam_load_file = base_dir / "rollup_files/DAM_Load_Resources/2024.parquet"
        df = pl.read_parquet(dam_load_file)

        # Filter to TINPD_LD3, Q3 2024
        df_q3 = df.filter(
            (pl.col("Load Resource Name") == "TINPD_LD3") &
            (pl.col("DeliveryDate") >= pl.date(2024, 7, 1)) &
            (pl.col("DeliveryDate") <= pl.date(2024, 9, 30))
        )

        # Calculate ECRS revenue directly from raw data
        ecrs_sd = (df_q3["ECRSSD Awarded"] * df_q3["ECRS MCPC"]).sum()
        ecrs_md = (df_q3["ECRSMD Awarded"] * df_q3["ECRS MCPC"]).sum()
        total_ecrs_q3 = ecrs_sd + ecrs_md

        print(f"  Raw data ECRS SD: ${ecrs_sd:>12,.2f}")
        print(f"  Raw data ECRS MD: ${ecrs_md:>12,.2f}")
        print(f"  Raw data total:   ${total_ecrs_q3:>12,.2f}  <- Should be ~$35k")

        # Now test what our calculator gives for full year
        load_as = calc_2024.calculate_dam_as_revenue("TINPD_LD3", is_gen=False)
        print(f"  Calculator (full year): ${load_as.get('ECRS', 0):>12,.2f}")
        print(f"  Calculator should be 3-4x Q3 for full year")

    except Exception as e:
        print(f"ERROR in Q3 verification: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_as_calculations()
