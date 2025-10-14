#!/usr/bin/env python3
"""
Fast Training Script for RT Price Spike Model

Uses parquet files for 10-100x faster loading.
Optimized for RTX 4070 (12GB VRAM).
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

# Add ml_models to path
sys.path.insert(0, str(Path(__file__).parent / "ml_models"))

from feature_engineering_parquet import ERCOTFeatureEngineerParquet
from price_spike_model import PriceSpikeModelTrainer


def check_gpu():
    """Check GPU availability and specs."""
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")
        print()
    else:
        print("\n⚠️  No GPU found - training will be slow on CPU\n")


def prepare_data(data_dir: Path,
                 years: list = [2024, 2025],
                 hub_col: str = 'HB_HOUSTON',
                 test_split: float = 0.15,
                 val_split: float = 0.15):
    """
    Prepare data for training using parquet files.

    Returns:
        train_df, val_df, test_df
    """
    print(f"\n{'='*80}")
    print("DATA PREPARATION (PARQUET-BASED - FAST)")
    print(f"{'='*80}\n")

    # Initialize feature engineer
    fe = ERCOTFeatureEngineerParquet(data_dir)

    # Build master feature set
    master_df = fe.build_master_feature_set(years=years, hub_col=hub_col)

    if master_df.empty:
        print("❌ No data available. Check parquet files.")
        return None, None, None

    # Sort by timestamp
    master_df = master_df.sort_values('timestamp').reset_index(drop=True)

    # Train/Val/Test split (chronological)
    n = len(master_df)
    train_end = int(n * (1 - test_split - val_split))
    val_end = int(n * (1 - test_split))

    train_df = master_df.iloc[:train_end].copy()
    val_df = master_df.iloc[train_end:val_end].copy()
    test_df = master_df.iloc[val_end:].copy()

    print(f"\n✅ Data split:")
    print(f"   Train: {len(train_df):,} samples ({train_df['timestamp'].min()} to {train_df['timestamp'].max()})")
    print(f"   Val:   {len(val_df):,} samples ({val_df['timestamp'].min()} to {val_df['timestamp'].max()})")
    print(f"   Test:  {len(test_df):,} samples ({test_df['timestamp'].min()} to {test_df['timestamp'].max()})")

    # Check spike distribution
    print(f"\n   Spike rate - Train: {train_df['price_spike'].mean()*100:.2f}%")
    print(f"   Spike rate - Val:   {val_df['price_spike'].mean()*100:.2f}%")
    print(f"   Spike rate - Test:  {test_df['price_spike'].mean()*100:.2f}%")

    return train_df, val_df, test_df


def train_price_spike_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                            epochs: int = 100, batch_size: int = 256):
    """Train Model 3: RT Price Spike Prediction."""

    print(f"\n{'='*80}")
    print("TRAINING MODEL 3: RT PRICE SPIKE PROBABILITY")
    print(f"{'='*80}\n")

    trainer = PriceSpikeModelTrainer(device='cuda')
    model, history = trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4
    )

    # Plot training history
    trainer.plot_training_history(history)

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train ERCOT RT Price Spike Model (Fast Parquet Version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings
    python train_spike_model_fast.py

    # Train with custom settings
    python train_spike_model_fast.py --epochs 100 --batch-size 256

    # Train on specific hub
    python train_spike_model_fast.py --hub HB_NORTH

    # Use 2023-2025 data
    python train_spike_model_fast.py --years 2023 2024 2025
        """
    )

    parser.add_argument('--data-dir', type=str,
                       default='/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data',
                       help='Data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--years', type=int, nargs='+', default=[2024, 2025],
                       help='Years to include in training')
    parser.add_argument('--hub', type=str, default='HB_HOUSTON',
                       help='Hub to predict (HB_HOUSTON, HB_NORTH, HB_SOUTH, HB_WEST)')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split fraction')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split fraction')

    args = parser.parse_args()

    # Check GPU
    check_gpu()

    # Prepare data
    train_df, val_df, test_df = prepare_data(
        data_dir=Path(args.data_dir),
        years=args.years,
        hub_col=args.hub,
        test_split=args.test_split,
        val_split=args.val_split
    )

    if train_df is None:
        print("❌ Data not ready. Exiting.")
        return 1

    # Save prepared data for future use
    print("\n💾 Saving prepared datasets...")
    output_dir = Path(args.data_dir)
    train_df.to_parquet(output_dir / 'train_data_spike.parquet', index=False)
    val_df.to_parquet(output_dir / 'val_data_spike.parquet', index=False)
    test_df.to_parquet(output_dir / 'test_data_spike.parquet', index=False)
    print("   ✅ Saved train/val/test datasets")

    # Train model
    model, history = train_price_spike_model(
        train_df, val_df,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"\nModel trained on hub: {args.hub}")
    print(f"Training data: {args.years}")
    print(f"Best validation AUC: {max(history['val_auc']):.4f}")
    print(f"\nNext steps:")
    print("  1. Evaluate model on test set")
    print("  2. Backtest on Winter Storm Uri (Feb 2021)")
    print("  3. Backtest on summer heat waves")
    print("  4. Deploy to production")
    print(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
