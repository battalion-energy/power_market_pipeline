#!/usr/bin/env python3
"""
Test the trained unified DA+RT model on test set
"""

import torch
import pandas as pd
import polars as pl
import numpy as np
from pathlib import Path
from train_unified_da_rt_quantile import prepare_data, UnifiedDART_Forecaster, UnifiedDataset, QuantileLoss, validate

MODEL_DIR = Path('models')
DATA_FILE = '/pool/ssd8tb/data/iso/ERCOT/ercot_market_data/ERCOT_data/master_features_enhanced_with_ordc_load_2019_2025.parquet'
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

def main():
    # Load data
    print('Loading data...')
    df_pl = pl.read_parquet(DATA_FILE)
    df, feature_cols = prepare_data(df_pl)
    print(f'Dataset: {len(df):,} samples, {len(feature_cols)} features')

    # Split
    n = len(df)
    val_end = int(n * 0.85)

    df_test = df.iloc[val_end:].copy()
    print(f'Test set: {len(df_test):,} samples')

    # Create test dataset
    test_dataset = UnifiedDataset(df_test, feature_cols)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UnifiedDART_Forecaster(input_dim=len(feature_cols), n_quantiles=len(QUANTILES)).to(device)
    checkpoint = torch.load(MODEL_DIR / 'unified_da_rt_best.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test evaluation
    print('\nEvaluating on test set...')
    criterion = QuantileLoss(QUANTILES)
    test_loss_da, test_loss_rt, test_mae_da, test_mae_rt = validate(model, test_loader, criterion, device)

    print('\n' + '='*60)
    print('TEST SET RESULTS')
    print('='*60)
    print(f'DA Price - Test MAE: ${test_mae_da:.2f}/MWh')
    print(f'RT Price - Test MAE: ${test_mae_rt:.2f}/MWh')
    print(f'DA Price - Test Loss: {test_loss_da:.4f}')
    print(f'RT Price - Test Loss: {test_loss_rt:.4f}')
    print('='*60)

if __name__ == "__main__":
    main()
