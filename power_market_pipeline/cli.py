"""Command-line interface for power market pipeline."""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_db
from downloaders.base_v2 import DownloadConfig
from services.data_fetcher import DataFetcher
from services.dataset_registry import DatasetRegistry

load_dotenv()


@click.group()
def cli():
    """Power Market Pipeline - ISO data collection and processing."""
    pass


@cli.command()
def init():
    """Initialize the database schema."""
    click.echo("Initializing database...")
    init_db()
    
    # Register datasets
    registry = DatasetRegistry()
    for iso in ["ERCOT", "CAISO", "ISONE", "NYISO"]:
        for dataset_key in ["lmp_dam", "lmp_rtm", "as_dam", "load_forecast"]:
            try:
                dataset = registry.register_dataset(iso, f"{iso.lower()}_{dataset_key}")
                click.echo(f"✓ Registered {dataset.dataset_id}")
            except Exception as e:
                click.echo(f"✗ Failed to register {iso}_{dataset_key}: {str(e)}")
    
    click.echo("Database initialization complete!")


@cli.command()
@click.option("--iso", multiple=True, default=["ERCOT"], help="ISO(s) to download data for")
@click.option("--start-date", default=None, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD), defaults to today")
@click.option("--days", type=int, default=None, help="Number of days to download (from today backwards)")
@click.option("--data-types", multiple=True, default=["lmp"], help="Data types to download")
@click.option("--output-dir", default="./data", help="Output directory for raw files")
def download(iso, start_date, end_date, days, data_types, output_dir):
    """Download data from ISOs."""
    # Parse dates
    if days:
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now() - timedelta(days=7)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
    
    # Create config
    config = DownloadConfig(
        start_date=start_dt,
        end_date=end_dt,
        data_types=list(data_types),
        output_dir=output_dir
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Downloading data for {', '.join(iso)}")
    click.echo(f"Date range: {start_dt.date()} to {end_dt.date()}")
    click.echo(f"Data types: {', '.join(data_types)}")
    
    # Run data fetcher
    fetcher = DataFetcher(config)
    results = asyncio.run(fetcher.fetch_all_data(
        isos=list(iso),
        start_date=start_dt,
        end_date=end_dt,
        data_types=list(data_types)
    ))
    
    # Display results
    for iso_code, iso_results in results.items():
        click.echo(f"\n{iso_code}:")
        for data_type, count in iso_results.items():
            click.echo(f"  {data_type}: {count:,} records")


@cli.command()
@click.option("--iso", multiple=True, default=["ERCOT", "CAISO", "ISONE", "NYISO"])
@click.option("--start-date", default="2019-01-01", help="Start date for backfill")
@click.option("--output-dir", default="./data", help="Output directory")
def backfill(iso, start_date, output_dir):
    """Run historical data backfill."""
    click.echo("Starting historical backfill...")
    click.echo(f"ISOs: {', '.join(iso)}")
    click.echo(f"Start date: {start_date}")
    
    # Create config
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    config = DownloadConfig(
        start_date=start_dt,
        end_date=datetime.now(),
        data_types=["lmp", "ancillary", "load"],
        output_dir=output_dir
    )
    
    # Run backfill
    fetcher = DataFetcher(config)
    
    async def run():
        await fetcher.run_historical_backfill(list(iso), start_dt)
    
    asyncio.run(run())


@cli.command()
@click.option("--iso", multiple=True, default=["ERCOT"], help="ISOs to update")
@click.option("--data-types", default="lmp,load", help="Comma-separated data types")
@click.option("--ercot-only", is_flag=True, help="Use specialized ERCOT updater")
def realtime(iso, data_types, ercot_only):
    """Start real-time data updates with 5-minute intervals."""
    from services.realtime_updater import RealtimeUpdater, ERCOTRealtimeUpdater
    
    click.echo("Starting real-time updates...")
    click.echo(f"ISOs: {', '.join(iso)}")
    
    # Parse data types
    data_type_list = [dt.strip() for dt in data_types.split(",")]
    click.echo(f"Data types: {', '.join(data_type_list)}")
    
    async def run():
        # Initialize database
        init_db()
        
        # Create appropriate updater
        if ercot_only or (len(iso) == 1 and iso[0] == "ERCOT"):
            click.echo("Using specialized ERCOT real-time updater")
            updater = ERCOTRealtimeUpdater()
        else:
            updater = RealtimeUpdater(isos=list(iso), data_types=data_type_list)
        
        # Run forever
        await updater.run_forever()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nStopping real-time updates...")


@cli.command()
@click.option("--iso", default=None, help="Filter by ISO")
def catalog(iso):
    """Display data catalog."""
    registry = DatasetRegistry()
    datasets = registry.get_dataset_catalog(iso)
    
    if not datasets:
        click.echo("No datasets found")
        return
    
    # Group by ISO
    by_iso = {}
    for dataset in datasets:
        iso_code = dataset.get("iso", "Unknown")
        if iso_code not in by_iso:
            by_iso[iso_code] = []
        by_iso[iso_code].append(dataset)
    
    # Display
    for iso_code, iso_datasets in sorted(by_iso.items()):
        click.echo(f"\n{iso_code}")
        click.echo("=" * 80)
        
        for ds in iso_datasets:
            click.echo(f"\n  {ds['dataset_id']}")
            click.echo(f"  {ds['description']}")
            click.echo(f"  Update: {ds['update_frequency']} | "
                      f"Spatial: {ds['spatial_resolution']} | "
                      f"Temporal: {ds['temporal_resolution']}")
            
            if ds['latest_data']:
                click.echo(f"  Latest: {ds['latest_data']} | "
                          f"Records: {ds.get('total_rows', 0):,}")


@cli.command()
@click.option("--dataset", required=True, help="Dataset name")
def update_stats(dataset):
    """Update dataset statistics."""
    registry = DatasetRegistry()
    registry.update_dataset_statistics(dataset)
    click.echo(f"Updated statistics for {dataset}")


if __name__ == "__main__":
    cli()