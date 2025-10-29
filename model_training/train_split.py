"""
Train/Validation/Test Split
Creates chronological splits for time-series modeling
CRITICAL: No random shuffling - respects temporal ordering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "feature_tables"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_master_features():
    """
    Load the master feature matrix

    Returns:
        DataFrame with all features
    """
    filepath = FEATURE_DIR / "master_features.parquet"

    if not filepath.exists():
        raise FileNotFoundError(
            f"Master features not found at {filepath}\n"
            "Please run: python feature_engineering/build_features.py"
        )

    df = pd.read_parquet(filepath)
    print(f"Loaded master features: {df.shape}")

    return df


def create_chronological_splits(df, train_end='2022-10-01', val_end='2023-10-01'):
    """
    Create train/validation/test splits based on dates
    CRITICAL: Chronological split to prevent data leakage

    Args:
        df: Master features DataFrame
        train_end: End date for training set (exclusive)
        val_end: End date for validation set (exclusive)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure game_date is datetime
    df['game_date'] = pd.to_datetime(df['game_date'])

    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    # Split chronologically
    train = df[df['game_date'] < train_end_dt].copy()
    val = df[(df['game_date'] >= train_end_dt) & (df['game_date'] < val_end_dt)].copy()
    test = df[df['game_date'] >= val_end_dt].copy()

    print("\n" + "="*60)
    print("CHRONOLOGICAL SPLIT SUMMARY")
    print("="*60)

    print(f"\nTrain set:")
    print(f"  Date range: {train['game_date'].min()} to {train['game_date'].max()}")
    print(f"  Shape: {train.shape}")
    print(f"  Unique players: {train['player_id'].nunique()}")

    print(f"\nValidation set:")
    print(f"  Date range: {val['game_date'].min()} to {val['game_date'].max()}")
    print(f"  Shape: {val.shape}")
    print(f"  Unique players: {val['player_id'].nunique()}")

    print(f"\nTest set:")
    print(f"  Date range: {test['game_date'].min()} to {test['game_date'].max()}")
    print(f"  Shape: {test.shape}")
    print(f"  Unique players: {test['player_id'].nunique()}")

    # Validate no temporal leakage
    assert train['game_date'].max() < val['game_date'].min(), "Train dates overlap with validation!"
    assert val['game_date'].max() < test['game_date'].min(), "Validation dates overlap with test!"

    print("\n✓ Temporal split validation passed: No date overlap between sets")

    return train, val, test


def check_target_distribution(train, val, test):
    """
    Check target variable distribution across splits

    Args:
        train, val, test: DataFrames for each split
    """
    print("\n" + "="*60)
    print("TARGET DISTRIBUTION")
    print("="*60)

    for name, df in [('Train', train), ('Validation', val), ('Test', test)]:
        if 'target_pra' in df.columns:
            print(f"\n{name} set PRA:")
            print(f"  Mean: {df['target_pra'].mean():.2f}")
            print(f"  Median: {df['target_pra'].median():.2f}")
            print(f"  Std: {df['target_pra'].std():.2f}")
            print(f"  Min: {df['target_pra'].min():.2f}")
            print(f"  Max: {df['target_pra'].max():.2f}")

    print("\n" + "="*60)


def prepare_for_modeling(df, drop_cols=None):
    """
    Prepare DataFrame for modeling
    Separates features from target and identifiers

    Args:
        df: DataFrame to prepare
        drop_cols: Additional columns to drop

    Returns:
        Tuple of (X, y, metadata)
    """
    if drop_cols is None:
        drop_cols = []

    # Columns to exclude from features
    exclude_cols = [
        'target_pra',  # Target variable
        'player_id',   # Identifier
        'game_id',     # Identifier
        'game_date',   # Identifier
        'player_name', # Identifier
        'season'       # Identifier (can be feature if you want)
    ] + drop_cols

    # Metadata columns
    metadata_cols = ['player_id', 'game_id', 'game_date', 'player_name', 'season']
    metadata = df[metadata_cols].copy()

    # Target
    if 'target_pra' in df.columns:
        y = df['target_pra'].copy()
    else:
        raise ValueError("Target variable 'target_pra' not found!")

    # Features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    print(f"\nPrepared for modeling:")
    print(f"  Features (X): {X.shape}")
    print(f"  Target (y): {y.shape}")
    print(f"  Metadata: {metadata.shape}")

    return X, y, metadata


def handle_missing_values(X_train, X_val, X_test, strategy='median'):
    """
    Handle missing values in feature matrices
    IMPORTANT: Fit on train, transform on val/test

    Args:
        X_train, X_val, X_test: Feature matrices
        strategy: 'median', 'mean', or 'zero'

    Returns:
        Tuple of (X_train, X_val, X_test) with imputed values
    """
    print("\n" + "="*60)
    print("HANDLING MISSING VALUES")
    print("="*60)

    # Calculate fill values from training set only
    if strategy == 'median':
        fill_values = X_train.median()
    elif strategy == 'mean':
        fill_values = X_train.mean()
    elif strategy == 'zero':
        fill_values = pd.Series(0, index=X_train.columns)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Count missing before
    train_missing_before = X_train.isnull().sum().sum()
    val_missing_before = X_val.isnull().sum().sum()
    test_missing_before = X_test.isnull().sum().sum()

    # Fill missing values
    X_train_filled = X_train.fillna(fill_values)
    X_val_filled = X_val.fillna(fill_values)
    X_test_filled = X_test.fillna(fill_values)

    # Count missing after
    train_missing_after = X_train_filled.isnull().sum().sum()
    val_missing_after = X_val_filled.isnull().sum().sum()
    test_missing_after = X_test_filled.isnull().sum().sum()

    print(f"\nImputation strategy: {strategy}")
    print(f"\nTrain set: {train_missing_before} → {train_missing_after} missing values")
    print(f"Validation set: {val_missing_before} → {val_missing_after} missing values")
    print(f"Test set: {test_missing_before} → {test_missing_after} missing values")

    if train_missing_after > 0 or val_missing_after > 0 or test_missing_after > 0:
        print("\n⚠️  WARNING: Some missing values remain after imputation!")
        # Fill remaining with 0
        X_train_filled = X_train_filled.fillna(0)
        X_val_filled = X_val_filled.fillna(0)
        X_test_filled = X_test_filled.fillna(0)
        print("Filled remaining missing values with 0")

    return X_train_filled, X_val_filled, X_test_filled


def save_splits(train_data, val_data, test_data):
    """
    Save train/val/test splits to parquet

    Args:
        train_data, val_data, test_data: Tuples of (X, y, metadata)
    """
    X_train, y_train, meta_train = train_data
    X_val, y_val, meta_val = val_data
    X_test, y_test, meta_test = test_data

    # Combine X, y, metadata for each split
    train_df = pd.concat([meta_train, X_train, y_train.rename('target_pra')], axis=1)
    val_df = pd.concat([meta_val, X_val, y_val.rename('target_pra')], axis=1)
    test_df = pd.concat([meta_test, X_test, y_test.rename('target_pra')], axis=1)

    # Save
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    print("\n" + "="*60)
    print("SAVED SPLITS")
    print("="*60)
    print(f"Train: {PROCESSED_DIR / 'train.parquet'} ({train_df.shape})")
    print(f"Validation: {PROCESSED_DIR / 'val.parquet'} ({val_df.shape})")
    print(f"Test: {PROCESSED_DIR / 'test.parquet'} ({test_df.shape})")


def main():
    """
    Main function to create train/val/test splits
    """
    print("="*60)
    print("CREATING TRAIN/VALIDATION/TEST SPLITS")
    print("="*60)

    # Load master features
    df = load_master_features()

    # Create chronological splits
    # Adjust dates based on your data range
    train, val, test = create_chronological_splits(
        df,
        train_end='2022-10-01',   # Everything before this is training
        val_end='2023-10-01'      # Between train_end and val_end is validation
    )

    # Check target distribution
    check_target_distribution(train, val, test)

    # Prepare for modeling
    print("\n" + "="*60)
    print("PREPARING FOR MODELING")
    print("="*60)

    X_train, y_train, meta_train = prepare_for_modeling(train)
    X_val, y_val, meta_val = prepare_for_modeling(val)
    X_test, y_test, meta_test = prepare_for_modeling(test)

    # Handle missing values
    X_train, X_val, X_test = handle_missing_values(X_train, X_val, X_test, strategy='median')

    # Save splits
    save_splits(
        (X_train, y_train, meta_train),
        (X_val, y_val, meta_val),
        (X_test, y_test, meta_test)
    )

    print("\n" + "="*60)
    print("✓ SPLIT CREATION COMPLETE")
    print("="*60)
    print("Ready for model training!")
    print("\nNext step: python model_training/train_model.py")


if __name__ == "__main__":
    main()
