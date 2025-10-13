"""
Train Multi-Horizon Price Spike Model on 2019-2025 Data

Trains on ALL available historical data (55,658 samples with 540 spike events)
vs previous 2024-2025 only (103 spike events).

21x more spike training data!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from price_spike_multihorizon_model import MultiHorizonModelTrainer


def load_and_split_data(data_file: Path, val_split: float = 0.2, test_split: float = 0.1):
    """
    Load data and split into train/val/test.

    Uses time-based split to avoid data leakage.
    """
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}\n")

    df = pd.read_parquet(data_file)
    print(f"Loaded: {len(df):,} samples")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Features: {len(df.columns)}")

    # Time-based split (train on early years, validate on recent)
    n = len(df)
    test_size = int(n * test_split)
    val_size = int(n * val_split)
    train_size = n - val_size - test_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]

    print(f"\n📊 Data Split (time-based):")
    print(f"  Train: {len(train_df):,} samples ({train_df.index.min()} to {train_df.index.max()})")
    print(f"  Val:   {len(val_df):,} samples ({val_df.index.min()} to {val_df.index.max()})")
    print(f"  Test:  {len(test_df):,} samples ({test_df.index.min()} to {test_df.index.max()})")

    # Check spike distribution
    print(f"\n📈 High Price Spike Distribution (>$400, 1h ahead):")
    print(f"  Train: {train_df['spike_high_1h'].sum():,} ({train_df['spike_high_1h'].mean()*100:.2f}%)")
    print(f"  Val:   {val_df['spike_high_1h'].sum():,} ({val_df['spike_high_1h'].mean()*100:.2f}%)")
    print(f"  Test:  {test_df['spike_high_1h'].sum():,} ({test_df['spike_high_1h'].mean()*100:.2f}%)")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description='Train multi-horizon spike model')
    parser.add_argument('--data-file', type=str,
                       default='/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_multihorizon_2019_2025.parquet',
                       help='Path to master features file')
    parser.add_argument('--no-uri', action='store_true',
                       help='Use dataset without Winter Storm Uri')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (256-512 optimal for RTX 4070)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=12,
                       help='Hours of history to use (12 = 12 hours)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split (default: 20%)')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Test split (default: 10%)')

    args = parser.parse_args()

    # Load data (use no-uri version if requested)
    if args.no_uri:
        args.data_file = args.data_file.replace('.parquet', '_no_uri.parquet')
        print("\n⚠️  Using dataset WITHOUT Winter Storm Uri")

    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("\nRun feature engineering first:")
        print("  uv run python ml_models/feature_engineering_multihorizon.py")
        return

    # Load and split data
    train_df, val_df, test_df = load_and_split_data(
        data_file,
        val_split=args.val_split,
        test_split=args.test_split
    )

    # Train model
    trainer = MultiHorizonModelTrainer()
    model, history = trainer.train(
        train_df=train_df,
        val_df=val_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sequence_length=args.sequence_length
    )

    # Save final model
    output_dir = Path('ml_models')
    output_dir.mkdir(exist_ok=True)

    model_name = 'multihorizon_model_2019_2025'
    if args.no_uri:
        model_name += '_no_uri'

    import torch
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'sequence_length': args.sequence_length,
            'data_file': str(data_file),
        }
    }, output_dir / f'{model_name}_final.pth')

    print(f"\n💾 Saved final model: {model_name}_final.pth")

    # Save training history
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Epoch')

    axes[0, 1].plot(history['val_auc_high_1h'], label='1h ahead')
    axes[0, 1].plot(history['val_auc_high_24h'], label='24h ahead')
    axes[0, 1].plot(history['val_auc_high_48h'], label='48h ahead')
    axes[0, 1].axhline(y=0.88, color='r', linestyle='--', label='Target (0.88)')
    axes[0, 1].set_title('Validation AUC (High Prices)')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')

    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_training_history.png', dpi=150)
    print(f"💾 Saved training history: {model_name}_training_history.png")

    # Test set evaluation
    print(f"\n{'='*80}")
    print("TEST SET EVALUATION")
    print(f"{'='*80}\n")

    from price_spike_multihorizon_model import MultiHorizonDataset
    from torch.utils.data import DataLoader
    import torch
    from sklearn.metrics import roc_auc_score

    test_dataset = MultiHorizonDataset(test_df, sequence_length=args.sequence_length)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_features, batch_targets in test_loader:
            batch_features = batch_features.to(trainer.device)
            output = model(batch_features)
            all_preds.append(output.cpu().numpy())
            all_targets.append(batch_targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_probs = 1 / (1 + np.exp(-all_preds))

    # Calculate AUC for each horizon (high prices)
    print("High Price Prediction AUC by Horizon:")
    print(f"{'Horizon':<10} {'AUC':<8} {'Spike %':<10}")
    print("-" * 30)
    for h in [1, 6, 12, 24, 36, 48]:
        idx = h - 1
        if all_targets[:, idx].sum() > 10:  # Need enough positive examples
            auc = roc_auc_score(all_targets[:, idx], all_probs[:, idx])
            spike_pct = all_targets[:, idx].mean() * 100
            print(f"{h:2d}h ahead  {auc:.4f}   {spike_pct:.2f}%")

    print(f"\n{'='*80}")
    print("✅ Training Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
