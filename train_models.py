#!/usr/bin/env python3
"""
Main Training Script for ERCOT Price Forecasting Models

Trains all 3 models optimized for RTX 4070 (12GB VRAM):
1. Model 1: DA Price Forecasting (LSTM-Attention)
2. Model 2: RT Price Forecasting (TCN-LSTM)
3. Model 3: RT Price Spike Probability (Transformer) - MOST CRITICAL

Usage:
    # Train all models
    python train_models.py --all

    # Train specific model
    python train_models.py --model spike

    # With custom settings
    python train_models.py --model spike --epochs 100 --batch-size 256
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

from feature_engineering import ERCOTFeatureEngineer
from price_spike_model import PriceSpikeModelTrainer, PriceSpikeTransformer
# from da_price_model import DAPriceModelTrainer  # To be implemented
# from rt_price_model import RTPriceModelTrainer  # To be implemented


def check_gpu():
    """Check GPU availability and specs."""
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"   VRAM: {props.total_memory / 1024**3:.1f} GB")
        print(f"   Compute Capability: {props.major}.{props.minor}")

        # Check if RTX 4070 (or similar)
        if props.total_memory / 1024**3 >= 11:
            print(f"   ✅ Sufficient VRAM for all models")
        else:
            print(f"   ⚠️  Limited VRAM - may need smaller batch sizes")
    else:
        print("\n⚠️  No GPU found - training will be slow on CPU")
    print()


def prepare_data(data_dir: Path, weather_dir: Path,
                 test_split: float = 0.15,
                 val_split: float = 0.15):
    """
    Prepare data for training.

    Returns:
        train_df, val_df, test_df
    """
    print(f"\n{'='*80}")
    print("DATA PREPARATION")
    print(f"{'='*80}\n")

    # Initialize feature engineer
    fe = ERCOTFeatureEngineer(data_dir)

    # Build master feature set
    master_df = fe.build_master_feature_set(weather_dir=weather_dir)

    if master_df.empty:
        print("❌ No data available yet. Wait for downloads to complete.")
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

    print(f"✅ Data split:")
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


def train_da_price_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                         epochs: int = 100, batch_size: int = 512):
    """Train Model 1: DA Price Forecasting."""

    print(f"\n{'='*80}")
    print("TRAINING MODEL 1: DA PRICE FORECASTING")
    print(f"{'='*80}\n")

    print("⚠️  DA Price model not yet implemented")
    print("   Will use LSTM-Attention architecture")
    print("   Target: MAE < $5/MWh, R² > 0.85")
    return None, None


def train_rt_price_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                        epochs: int = 100, batch_size: int = 256):
    """Train Model 2: RT Price Forecasting."""

    print(f"\n{'='*80}")
    print("TRAINING MODEL 2: RT PRICE FORECASTING")
    print(f"{'='*80}\n")

    print("⚠️  RT Price model not yet implemented")
    print("   Will use TCN-LSTM architecture")
    print("   Target: MAE < $15/MWh")
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Train ERCOT Price Forecasting Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all models
    python train_models.py --all

    # Train only price spike model (Model 3)
    python train_models.py --model spike

    # Custom settings
    python train_models.py --model spike --epochs 100 --batch-size 256

    # Use specific data directory
    python train_models.py --all --data-dir /path/to/data
        """
    )

    parser.add_argument('--model', choices=['spike', 'da', 'rt', 'all'],
                       default='all', help='Which model to train')
    parser.add_argument('--data-dir', type=str,
                       default='/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data',
                       help='Data directory')
    parser.add_argument('--weather-dir', type=str,
                       default='/home/enrico/data/weather_data',
                       help='Weather data directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
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
        weather_dir=Path(args.weather_dir),
        test_split=args.test_split,
        val_split=args.val_split
    )

    if train_df is None:
        print("❌ Data not ready. Exiting.")
        return 1

    # Save prepared data for future use
    print("\n💾 Saving prepared datasets...")
    train_df.to_parquet(Path(args.data_dir) / 'train_data.parquet', index=False)
    val_df.to_parquet(Path(args.data_dir) / 'val_data.parquet', index=False)
    test_df.to_parquet(Path(args.data_dir) / 'test_data.parquet', index=False)
    print("   ✅ Saved train/val/test datasets")

    # Train models
    models_trained = []

    if args.model in ['spike', 'all']:
        model, history = train_price_spike_model(
            train_df, val_df,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        if model:
            models_trained.append('spike')

    if args.model in ['da', 'all']:
        model, history = train_da_price_model(
            train_df, val_df,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        if model:
            models_trained.append('da')

    if args.model in ['rt', 'all']:
        model, history = train_rt_price_model(
            train_df, val_df,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        if model:
            models_trained.append('rt')

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"\nModels trained: {', '.join(models_trained) if models_trained else 'None'}")
    print(f"\nNext steps:")
    print("  1. Evaluate models on test set")
    print("  2. Backtest on Winter Storm Uri (Feb 2021)")
    print("  3. Backtest on summer heat waves")
    print("  4. Deploy to production")
    print(f"\n{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
