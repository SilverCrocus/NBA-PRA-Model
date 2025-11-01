"""
Train/Validation/Test Split
Creates chronological splits for time-series modeling
CRITICAL: No random shuffling - respects temporal ordering
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import argparse
from typing import List, Dict, Tuple

from model_training.utils import prepare_features_target, setup_logger
from model_training.config import (
    FEATURE_DIR,
    PROCESSED_DIR,
    CV_TRAINING_WINDOW_YEARS,
    CV_GAP_GAMES,
    CV_MIN_TEST_GAMES,
    CV_VAL_SPLIT,
    CV_FOLDS_DIR
)

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
if CV_FOLDS_DIR:
    CV_FOLDS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logger = logging.getLogger(__name__)


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
    logger.info(f"Loaded master features: {df.shape}")

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

    logger.info("\n" + "="*60)
    logger.info("CHRONOLOGICAL SPLIT SUMMARY")
    logger.info("="*60)

    logger.info(f"\nTrain set:")
    logger.info(f"  Date range: {train['game_date'].min()} to {train['game_date'].max()}")
    logger.info(f"  Shape: {train.shape}")
    logger.info(f"  Unique players: {train['player_id'].nunique()}")

    logger.info(f"\nValidation set:")
    logger.info(f"  Date range: {val['game_date'].min()} to {val['game_date'].max()}")
    logger.info(f"  Shape: {val.shape}")
    logger.info(f"  Unique players: {val['player_id'].nunique()}")

    logger.info(f"\nTest set:")
    logger.info(f"  Date range: {test['game_date'].min()} to {test['game_date'].max()}")
    logger.info(f"  Shape: {test.shape}")
    logger.info(f"  Unique players: {test['player_id'].nunique()}")

    # Validate no temporal leakage
    assert train['game_date'].max() < val['game_date'].min(), "Train dates overlap with validation!"
    assert val['game_date'].max() < test['game_date'].min(), "Validation dates overlap with test!"

    logger.info("\n✓ Temporal split validation passed: No date overlap between sets")

    return train, val, test


def check_target_distribution(train, val, test):
    """
    Check target variable distribution across splits

    Args:
        train, val, test: DataFrames for each split
    """
    logger.info("\n" + "="*60)
    logger.info("TARGET DISTRIBUTION")
    logger.info("="*60)

    for name, df in [('Train', train), ('Validation', val), ('Test', test)]:
        if 'target_pra' in df.columns:
            logger.info(f"\n{name} set PRA:")
            logger.info(f"  Mean: {df['target_pra'].mean():.2f}")
            logger.info(f"  Median: {df['target_pra'].median():.2f}")
            logger.info(f"  Std: {df['target_pra'].std():.2f}")
            logger.info(f"  Min: {df['target_pra'].min():.2f}")
            logger.info(f"  Max: {df['target_pra'].max():.2f}")

    logger.info("\n" + "="*60)


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
    logger.info("\n" + "="*60)
    logger.info("HANDLING MISSING VALUES")
    logger.info("="*60)

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

    logger.info(f"\nImputation strategy: {strategy}")
    logger.info(f"\nTrain set: {train_missing_before} → {train_missing_after} missing values")
    logger.info(f"Validation set: {val_missing_before} → {val_missing_after} missing values")
    logger.info(f"Test set: {test_missing_before} → {test_missing_after} missing values")

    if train_missing_after > 0 or val_missing_after > 0 or test_missing_after > 0:
        logger.info("\n⚠️  WARNING: Some missing values remain after imputation!")
        # Fill remaining with 0
        X_train_filled = X_train_filled.fillna(0)
        X_val_filled = X_val_filled.fillna(0)
        X_test_filled = X_test_filled.fillna(0)
        logger.info("Filled remaining missing values with 0")

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

    logger.info("\n" + "="*60)
    logger.info("SAVED SPLITS")
    logger.info("="*60)
    logger.info(f"Train: {PROCESSED_DIR / 'train.parquet'} ({train_df.shape})")
    logger.info(f"Validation: {PROCESSED_DIR / 'val.parquet'} ({val_df.shape})")
    logger.info(f"Test: {PROCESSED_DIR / 'test.parquet'} ({test_df.shape})")


# ============================================================================
# TIME-SERIES CROSS-VALIDATION FUNCTIONS
# ============================================================================

def detect_season(game_date: pd.Timestamp) -> str:
    """
    Map game_date to NBA season string
    NBA season: Oct Year1 - Jun Year2 → "Year1-Year2"

    Args:
        game_date: Game date timestamp

    Returns:
        Season string (e.g., "2023-24")

    Example:
        >>> detect_season(pd.Timestamp('2023-10-15'))
        '2023-24'
        >>> detect_season(pd.Timestamp('2024-01-15'))
        '2023-24'
    """
    year = game_date.year
    month = game_date.month

    if month >= 10:  # Oct-Dec: start of season
        return f"{year}-{str(year+1)[-2:]}"
    else:  # Jan-Sep: end of season
        return f"{year-1}-{str(year)[-2:]}"


def get_available_seasons(df: pd.DataFrame) -> List[str]:
    """
    Extract unique NBA seasons from data

    Args:
        df: DataFrame with game_date column

    Returns:
        Sorted list of season strings

    Example:
        Returns: ['2015-16', '2016-17', '2017-18', ...]
    """
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['season_detected'] = df['game_date'].apply(detect_season)
    seasons = sorted(df['season_detected'].unique())

    logger.info(f"\nDetected {len(seasons)} seasons in data:")
    for season in seasons:
        season_games = df[df['season_detected'] == season]
        logger.info(f"  {season}: {len(season_games):,} games "
              f"({season_games['game_date'].min().strftime('%Y-%m-%d')} to "
              f"{season_games['game_date'].max().strftime('%Y-%m-%d')})")

    return seasons


def create_timeseries_cv_splits(
    df: pd.DataFrame,
    training_window_years: int = CV_TRAINING_WINDOW_YEARS,
    gap_games: int = CV_GAP_GAMES,
    min_test_games: int = CV_MIN_TEST_GAMES,
    val_split: float = CV_VAL_SPLIT
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create rolling training windows for time-series CV

    Default values are imported from model_training.config

    Args:
        df: Master features DataFrame
        training_window_years: Number of years for training window (default: from config.CV_TRAINING_WINDOW_YEARS)
        gap_games: Gap between train end and test start per player (default: from config.CV_GAP_GAMES)
        min_test_games: Minimum games required in test set (default: from config.CV_MIN_TEST_GAMES)
        val_split: Fraction of training data for validation

    Returns:
        List of fold dictionaries, each containing:
        {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'fold_id': int,
            'train_seasons': list,
            'test_season': str,
            'train_start': date,
            'train_end': date,
            'test_start': date,
            'test_end': date
        }

    Example:
        For 2015-24 data (9 seasons) with 3-year windows:
        Fold 0: Train[2015-16, 2016-17, 2017-18] → Test[2018-19]
        Fold 1: Train[2016-17, 2017-18, 2018-19] → Test[2019-20]
        ...
        Fold 5: Train[2020-21, 2021-22, 2022-23] → Test[2023-24]
    """
    logger.info("\n" + "="*60)
    logger.info("CREATING TIME-SERIES CV FOLDS")
    logger.info("="*60)
    logger.info(f"Training window: {training_window_years} years")
    logger.info(f"Gap: {gap_games} games per player")
    logger.info(f"Validation split: {val_split:.0%} of training data")

    # Ensure game_date is datetime and add season column
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['season_detected'] = df['game_date'].apply(detect_season)

    # Get available seasons
    seasons = get_available_seasons(df)

    # Generate folds
    folds = []
    fold_id = 0

    # Rolling window: train on N years, test on year N+1
    for i in range(len(seasons) - training_window_years):
        train_seasons = seasons[i:i+training_window_years]
        test_season = seasons[i+training_window_years]

        # Extract data for this fold
        train_val_data = df[df['season_detected'].isin(train_seasons)].copy()
        test_data = df[df['season_detected'] == test_season].copy()

        # Check minimum test size
        if len(test_data) < min_test_games:
            logger.info(f"\n⚠️  Skipping fold {fold_id}: Test set too small "
                  f"({len(test_data)} < {min_test_games} games)")
            continue

        # Enforce gap: For each player, remove first gap_games from test set
        players_with_gap_violation = []
        for player_id in test_data['player_id'].unique():
            player_train_val = train_val_data[train_val_data['player_id'] == player_id]
            player_test = test_data[test_data['player_id'] == player_id]

            if len(player_train_val) > 0 and len(player_test) > 0:
                last_train_date = player_train_val['game_date'].max()

                # Remove test games within gap
                player_test_sorted = player_test.sort_values('game_date')
                player_test_games_after_gap = player_test_sorted.iloc[min(gap_games, len(player_test_sorted)):]

                # Update test_data to exclude gap games for this player
                test_data = test_data[
                    ~((test_data['player_id'] == player_id) &
                      (test_data['game_date'] < player_test_games_after_gap['game_date'].min()))
                ]

        # Re-check test size after gap enforcement
        if len(test_data) < min_test_games:
            logger.info(f"\n⚠️  Skipping fold {fold_id}: Test set too small after gap enforcement "
                  f"({len(test_data)} < {min_test_games} games)")
            continue

        # Split train_val into train and val (chronological)
        train_val_sorted = train_val_data.sort_values('game_date')
        split_idx = int(len(train_val_sorted) * (1 - val_split))

        train_data = train_val_sorted.iloc[:split_idx].copy()
        val_data = train_val_sorted.iloc[split_idx:].copy()

        # Create fold dictionary
        fold = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'fold_id': fold_id,
            'train_seasons': train_seasons,
            'test_season': test_season,
            'train_start': train_data['game_date'].min(),
            'train_end': train_data['game_date'].max(),
            'val_start': val_data['game_date'].min(),
            'val_end': val_data['game_date'].max(),
            'test_start': test_data['game_date'].min(),
            'test_end': test_data['game_date'].max()
        }

        folds.append(fold)

        # Print fold summary
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Train seasons: {', '.join(train_seasons)}")
        logger.info(f"Test season: {test_season}")
        logger.info(f"\nTrain set:")
        logger.info(f"  Date range: {fold['train_start'].strftime('%Y-%m-%d')} to {fold['train_end'].strftime('%Y-%m-%d')}")
        logger.info(f"  Games: {len(train_data):,}")
        logger.info(f"  Players: {train_data['player_id'].nunique()}")
        logger.info(f"\nValidation set:")
        logger.info(f"  Date range: {fold['val_start'].strftime('%Y-%m-%d')} to {fold['val_end'].strftime('%Y-%m-%d')}")
        logger.info(f"  Games: {len(val_data):,}")
        logger.info(f"  Players: {val_data['player_id'].nunique()}")
        logger.info(f"\nTest set:")
        logger.info(f"  Date range: {fold['test_start'].strftime('%Y-%m-%d')} to {fold['test_end'].strftime('%Y-%m-%d')}")
        logger.info(f"  Games: {len(test_data):,}")
        logger.info(f"  Players: {test_data['player_id'].nunique()}")

        # Validate no temporal leakage
        assert fold['train_end'] <= fold['val_start'], f"Fold {fold_id}: Train dates overlap with validation!"
        assert fold['val_end'] <= fold['test_start'], f"Fold {fold_id}: Validation dates overlap with test!"

        logger.info(f"\n✓ Temporal validation passed for fold {fold_id}")

        fold_id += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"CREATED {len(folds)} CV FOLDS")
    logger.info(f"{'='*60}")

    return folds


def save_cv_splits(folds: List[Dict], output_dir: Path = None):
    """
    Save all CV folds to disk

    Args:
        folds: List of fold dictionaries from create_timeseries_cv_splits()
        output_dir: Base directory for CV folds (default: from config.CV_FOLDS_DIR)

    Output structure:
        data/processed/cv_folds/
            fold_0/
                train.parquet
                val.parquet
                test.parquet
                fold_info.txt
            fold_1/
                ...
    """
    if output_dir is None:
        output_dir = CV_FOLDS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "="*60)
    logger.info("SAVING CV FOLDS")
    logger.info("="*60)
    logger.info(f"Output directory: {output_dir}")

    for fold in folds:
        fold_id = fold['fold_id']
        fold_dir = output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save train/val/test
        fold['train'].to_parquet(fold_dir / "train.parquet", index=False)
        fold['val'].to_parquet(fold_dir / "val.parquet", index=False)
        fold['test'].to_parquet(fold_dir / "test.parquet", index=False)

        # Save fold metadata
        with open(fold_dir / "fold_info.txt", 'w') as f:
            f.write(f"Fold {fold_id} Information\n")
            f.write("="*60 + "\n\n")
            f.write(f"Train seasons: {', '.join(fold['train_seasons'])}\n")
            f.write(f"Test season: {fold['test_season']}\n\n")
            f.write(f"Train: {len(fold['train']):,} games, "
                   f"{fold['train_start'].strftime('%Y-%m-%d')} to {fold['train_end'].strftime('%Y-%m-%d')}\n")
            f.write(f"Val: {len(fold['val']):,} games, "
                   f"{fold['val_start'].strftime('%Y-%m-%d')} to {fold['val_end'].strftime('%Y-%m-%d')}\n")
            f.write(f"Test: {len(fold['test']):,} games, "
                   f"{fold['test_start'].strftime('%Y-%m-%d')} to {fold['test_end'].strftime('%Y-%m-%d')}\n")

        logger.info(f"\n✓ Saved fold {fold_id} to {fold_dir}")
        logger.info(f"  train.parquet: {fold['train'].shape}")
        logger.info(f"  val.parquet: {fold['val'].shape}")
        logger.info(f"  test.parquet: {fold['test'].shape}")

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ ALL {len(folds)} FOLDS SAVED")
    logger.info(f"{'='*60}")


def main():
    """
    Main function to create train/val/test splits
    Supports both single-split and CV modes
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Create train/val/test splits')
    parser.add_argument('--cv-mode', action='store_true',
                       help='Generate time-series CV folds instead of single split')
    parser.add_argument('--training-window', type=int, default=3,
                       help='Training window in years for CV (default: 3)')
    parser.add_argument('--gap-games', type=int, default=15,
                       help='Gap between train and test in games (default: 15)')
    parser.add_argument('--min-test-games', type=int, default=1000,
                       help='Minimum games per test fold (default: 1000)')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split fraction (default: 0.2)')
    parser.add_argument('--train-end', type=str, default='2022-10-01',
                       help='Training end date for single split (default: 2022-10-01)')
    parser.add_argument('--val-end', type=str, default='2023-10-01',
                       help='Validation end date for single split (default: 2023-10-01)')

    args = parser.parse_args()

    # Load master features
    df = load_master_features()

    if args.cv_mode:
        # CV MODE: Create time-series cross-validation folds
        logger.info("="*60)
        logger.info("MODE: TIME-SERIES CROSS-VALIDATION")
        logger.info("="*60)

        folds = create_timeseries_cv_splits(
            df,
            training_window_years=args.training_window,
            gap_games=args.gap_games,
            min_test_games=args.min_test_games,
            val_split=args.val_split
        )

        save_cv_splits(folds)

        logger.info("\n" + "="*60)
        logger.info("✓ CV FOLD CREATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Created {len(folds)} CV folds")
        logger.info("Ready for CV training!")
        logger.info("\nNext step: uv run model_training/training.py --cv --model-type xgboost")

    else:
        # SINGLE SPLIT MODE: Create single train/val/test split
        logger.info("="*60)
        logger.info("MODE: SINGLE TRAIN/VALIDATION/TEST SPLIT")
        logger.info("="*60)

        # Create chronological splits
        train, val, test = create_chronological_splits(
            df,
            train_end=args.train_end,
            val_end=args.val_end
        )

        # Check target distribution
        check_target_distribution(train, val, test)

        # Prepare for modeling
        logger.info("\n" + "="*60)
        logger.info("PREPARING FOR MODELING")
        logger.info("="*60)

        X_train, y_train, meta_train = prepare_features_target(train, return_metadata=True)
        X_val, y_val, meta_val = prepare_features_target(val, return_metadata=True)
        X_test, y_test, meta_test = prepare_features_target(test, return_metadata=True)

        # Handle missing values
        X_train, X_val, X_test = handle_missing_values(X_train, X_val, X_test, strategy='median')

        # Save splits
        save_splits(
            (X_train, y_train, meta_train),
            (X_val, y_val, meta_val),
            (X_test, y_test, meta_test)
        )

        logger.info("\n" + "="*60)
        logger.info("✓ SPLIT CREATION COMPLETE")
        logger.info("="*60)
        logger.info("Ready for model training!")
        logger.info("\nNext step: uv run model_training/training.py --model-type xgboost")


if __name__ == "__main__":
    main()
