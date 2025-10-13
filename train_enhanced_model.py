#!/usr/bin/env python3
"""
Enhanced Training Script with Wind/Solar Forecast Errors

This adds the CRITICAL features missing from baseline model:
- Wind forecast errors (actual - STWPF)
- Solar forecast errors (actual - STPPF)

Expected improvement: AUC 0.51 → 0.75+
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

from feature_engineering_enhanced import ERCOTFeatureEngineerEnhanced
from price_spike_model import PriceSpikeModelTrainer


def check_gpu():
    """Check GPU availability."""
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   VRAM: {props.total_memory / 1024**3:.1f} GB\n")
    else:
        print("\n⚠️  No GPU - training on CPU\n")


def prepare_enhanced_data(data_dir: Path,
                          years: list = [2024, 2025],
                          hub_col: str = 'HB_HOUSTON',
                          test_split: float = 0.15,
                          val_split: float = 0.15):
    """Prepare enhanced data with forecast errors."""

    print(f"\n{'='*80}")
    print("ENHANCED DATA PREPARATION (WITH FORECAST ERRORS)")
    print(f"{'='*80}\n")

    fe = ERCOTFeatureEngineerEnhanced(data_dir)
    master_df = fe.build_master_feature_set_enhanced(years=years, hub_col=hub_col)

    if master_df.empty:
        print("❌ No data available")
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

    print(f"\n   Spike rate - Train: {train_df['price_spike'].mean()*100:.2f}%")
    print(f"   Spike rate - Val:   {val_df['price_spike'].mean()*100:.2f}%")
    print(f"   Spike rate - Test:  {test_df['price_spike'].mean()*100:.2f}%")

    # Show feature statistics
    forecast_error_cols = [c for c in train_df.columns if 'wind_error' in c or 'solar_error' in c]
    if forecast_error_cols:
        print(f"\n📊 Forecast Error Statistics:")
        for col in forecast_error_cols:
            if train_df[col].notna().sum() > 0:
                print(f"   {col}: mean={train_df[col].mean():.1f}, std={train_df[col].std():.1f}")

    return train_df, val_df, test_df


def train_enhanced_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        epochs: int = 50, batch_size: int = 256, sequence_length: int = 24):
    """Train enhanced model with forecast errors.

    Args:
        sequence_length: Number of hours to use as input (default: 24 = 1 day)
    """

    print(f"\n{'='*80}")
    print("TRAINING ENHANCED MODEL (WITH FORECAST ERRORS)")
    print(f"{'='*80}\n")

    trainer = PriceSpikeModelTrainer(device='cuda')
    model, history = trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=epochs,
        batch_size=batch_size,
        lr=1e-4,
        sequence_length=sequence_length
    )

    # Plot training history
    trainer.plot_training_history(history)

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train Enhanced ERCOT Price Spike Model (with forecast errors)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--data-dir', type=str,
                       default='/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data',
                       help='Data directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--years', type=int, nargs='+', default=[2024, 2025],
                       help='Years to include')
    parser.add_argument('--hub', type=str, default='HB_HOUSTON',
                       help='Hub to predict')

    args = parser.parse_args()

    check_gpu()

    # Prepare enhanced data
    train_df, val_df, test_df = prepare_enhanced_data(
        data_dir=Path(args.data_dir),
        years=args.years,
        hub_col=args.hub
    )

    if train_df is None:
        print("❌ Data not ready. Exiting.")
        return 1

    # Save prepared data
    print("\n💾 Saving enhanced datasets...")
    output_dir = Path(args.data_dir)
    train_df.to_parquet(output_dir / 'train_data_enhanced.parquet', index=False)
    val_df.to_parquet(output_dir / 'val_data_enhanced.parquet', index=False)
    test_df.to_parquet(output_dir / 'test_data_enhanced.parquet', index=False)
    print("   ✅ Saved train/val/test datasets")

    # Train enhanced model
    model, history = train_enhanced_model(
        train_df, val_df,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Summary
    print(f"\n{'='*80}")
    print("ENHANCED MODEL TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"\nFeatures added: Wind/Solar forecast errors")
    print(f"Model trained on hub: {args.hub}")
    print(f"Training data: {args.years}")
    print(f"Best validation AUC: {max(history['val_auc']):.4f}")
    print(f"\n🎯 Target AUC: > 0.88 (Industry Benchmark)")
    print(f"📈 Expected improvement: 0.51 (baseline) → 0.75+ (with forecast errors)")
    print(f"\nNext steps:")
    print("  1. Evaluate on test set")
    print("  2. Compare baseline vs enhanced")
    print("  3. Production deployment")
    print(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
