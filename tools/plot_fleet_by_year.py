#!/usr/bin/env python3
"""
Fleet-by-year stacked bars for DA net, RT net, and AS revenues.
Reads bess_revenue_<year>*.csv files (prefers *_TELEMETERED when present).
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def load_year(year: int) -> pd.DataFrame | None:
    telem = Path(f"bess_revenue_{year}_TELEMETERED.csv")
    reg = Path(f"bess_revenue_{year}.csv")
    f = telem if telem.exists() and telem.stat().st_size > 0 else (reg if reg.exists() and reg.stat().st_size > 0 else None)
    if not f:
        return None
    return pd.read_csv(f)


def main():
    years = [2020, 2021, 2022, 2023, 2024]
    rows = []
    for y in years:
        df = load_year(y)
        if df is None or df.empty:
            continue
        da = df['da_net_energy'].sum() if 'da_net_energy' in df.columns else df['dam_discharge_revenue'].sum()
        rt = df['rt_net_revenue'].sum() if 'rt_net_revenue' in df.columns else 0.0
        as_rev = 0.0
        for c in df.columns:
            if c.startswith('dam_as_gen_') or c.startswith('dam_as_load_'):
                as_rev += df[c].sum()
        rows.append({'year': y, 'DA': da, 'RT': rt, 'AS': as_rev})

    if not rows:
        print('No data found.')
        return
    agg = pd.DataFrame(rows).set_index('year').sort_index()

    # Plot
    ax = agg[['DA','RT','AS']].plot(kind='bar', stacked=True, figsize=(10,6), color=['#8B7BC8','#5FD4AF','#5DADE2'])
    ax.set_ylabel('Revenue ($)')
    ax.set_title('ERCOT BESS Fleet Revenue by Year (DA net, RT net, AS)')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    out = Path('bess_fleet_revenue_by_year.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
