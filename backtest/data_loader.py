"""
Data Loading and Preparation for Walk-Forward Backtest

Handles loading master features, historical odds, and applying CTG imputation.

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import Tuple, Optional, List
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.config import (
    MASTER_FEATURES_PATH, HISTORICAL_ODDS_PATH, PLAYER_GAMES_PATH,
    TARGET_SEASON, TARGET_START_DATE, TARGET_END_DATE,
    TRAINING_WINDOW_YEARS, EXCLUDE_COLUMNS
)
from feature_engineering.features.ctg_imputation import (
    calculate_position_baselines,
    impute_missing_ctg,
    create_imputation_flags,
    create_position_relative_features
)

logger = logging.getLogger(__name__)


def load_master_features(season_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Load master features dataset

    Args:
        season_filter: Optional season to filter (e.g., '2024-25'). If None, loads all seasons.

    Returns:
        DataFrame with all features
    """
    logger.info(f"Loading master features from {MASTER_FEATURES_PATH}...")

    if not MASTER_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Master features not found at {MASTER_FEATURES_PATH}")

    df = pd.read_parquet(MASTER_FEATURES_PATH)

    # Convert game_date to datetime
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Filter by season if specified
    if season_filter:
        df = df[df['season'] == season_filter].copy()
        logger.info(f"  Filtered to season {season_filter}: {len(df)} games")

    logger.info(f"  Loaded {len(df)} games for {df['player_id'].nunique()} players")
    logger.info(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    logger.info(f"  Features: {len(df.columns)} columns")

    return df


def load_historical_odds() -> pd.DataFrame:
    """
    Load historical betting lines for 2024-25 season

    Returns:
        DataFrame with betting lines (player_name, event_date, line, bookmaker, etc.)
    """
    logger.info(f"Loading historical odds from {HISTORICAL_ODDS_PATH}...")

    if not HISTORICAL_ODDS_PATH.exists():
        logger.warning(f"  Historical odds not found at {HISTORICAL_ODDS_PATH}")
        logger.warning(f"  Continuing without betting lines...")
        return pd.DataFrame()

    odds = pd.read_csv(HISTORICAL_ODDS_PATH)

    # Convert event_date to datetime
    odds['event_date'] = pd.to_datetime(odds['event_date'])

    logger.info(f"  Loaded {len(odds)} betting lines")
    logger.info(f"  Date range: {odds['event_date'].min()} to {odds['event_date'].max()}")
    logger.info(f"  Players: {odds['player_name'].nunique()}")

    return odds


def get_training_window(all_data: pd.DataFrame,
                        prediction_date: pd.Timestamp,
                        window_years: int = TRAINING_WINDOW_YEARS) -> pd.DataFrame:
    """
    Get training data within N-year rolling window before prediction date

    Args:
        all_data: Full dataset
        prediction_date: Date we're predicting for
        window_years: Number of years to include in training window

    Returns:
        DataFrame with training data (strictly before prediction_date)
    """
    # Training data must be BEFORE prediction date (strict inequality)
    training_cutoff = prediction_date - timedelta(days=1)

    # Rolling window: N years before prediction date
    window_start = prediction_date - timedelta(days=365 * window_years)

    # Filter to window
    training_data = all_data[
        (all_data['game_date'] >= window_start) &
        (all_data['game_date'] <= training_cutoff)
    ].copy()

    logger.debug(f"  Training window: {window_start.date()} to {training_cutoff.date()}")
    logger.debug(f"  Training games: {len(training_data)}")

    return training_data


def match_predictions_to_betting_lines(predictions_df: pd.DataFrame,
                                        odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Match predictions to betting lines by player and date

    Args:
        predictions_df: DataFrame with columns [player_name, game_date, prediction, actual_pra]
        odds_df: DataFrame with betting lines [player_name, event_date, line, bookmaker]

    Returns:
        DataFrame with predictions + betting lines merged
    """
    if odds_df.empty:
        logger.warning("No betting lines available, skipping line matching")
        predictions_df['betting_line'] = np.nan
        return predictions_df

    # For simplicity, use the first available line per player/date
    # In production, you might want to use a specific bookmaker or average across books
    odds_agg = odds_df.groupby(['player_name', 'event_date']).agg({
        'line': 'first',  # Take first line (could also use mean or specific bookmaker)
        'over_price': 'first',
        'under_price': 'first'
    }).reset_index()

    # Merge predictions with lines
    merged = predictions_df.merge(
        odds_agg,
        left_on=['player_name', 'game_date'],
        right_on=['player_name', 'event_date'],
        how='left'
    )

    # Clean up
    merged = merged.drop(columns=['event_date'], errors='ignore')
    merged = merged.rename(columns={'line': 'betting_line'})

    matched_count = (~merged['betting_line'].isna()).sum()
    logger.info(f"  Matched {matched_count}/{len(predictions_df)} predictions to betting lines")

    return merged


def apply_imputation_with_baselines(data: pd.DataFrame,
                                     position_baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Apply CTG imputation using pre-calculated position baselines

    This ensures we don't leak test data into training by using test-derived baselines.

    Args:
        data: Data to impute (can be train, val, or test)
        position_baselines: Pre-calculated position statistics from training data

    Returns:
        DataFrame with imputed CTG features and imputation flags
    """
    # Impute missing CTG features
    data_imputed = impute_missing_ctg(data, position_baselines, impute_method='position_mean')

    # Note: Imputation flags should already be in the data from advanced_metrics.py
    # But if not, we can create them here
    if 'has_ctg_data' not in data.columns:
        logger.warning("Imputation flags not found in data, creating now...")
        flags = create_imputation_flags(data)
        data_imputed = data_imputed.merge(
            flags,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    return data_imputed


def prepare_features_for_training(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for model training by separating X and y

    Args:
        df: DataFrame with features and target

    Returns:
        Tuple of (X_features, y_target)
    """
    # Identify target column (try both 'target_pra' and 'pra')
    if 'target_pra' in df.columns:
        target_col = 'target_pra'
    elif 'pra' in df.columns:
        target_col = 'pra'
    else:
        raise ValueError(f"Target column ('target_pra' or 'pra') not found in dataframe. Available columns: {df.columns.tolist()[:10]}...")

    # Identify columns to exclude
    exclude_cols = [col for col in EXCLUDE_COLUMNS if col in df.columns]

    # Features = all columns except target and excluded columns
    feature_cols = [col for col in df.columns if col not in exclude_cols + [target_col]]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    logger.debug(f"  Prepared {len(feature_cols)} features, {len(y)} samples")

    return X, y


def get_unique_game_days(df: pd.DataFrame,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> List[pd.Timestamp]:
    """
    Get list of unique game days in dataset (days with at least one game)

    Args:
        df: DataFrame with game_date column
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        Sorted list of unique game dates
    """
    df_filtered = df.copy()

    if start_date:
        df_filtered = df_filtered[df_filtered['game_date'] >= pd.Timestamp(start_date)]

    if end_date:
        df_filtered = df_filtered[df_filtered['game_date'] <= pd.Timestamp(end_date)]

    game_days = sorted(df_filtered['game_date'].unique())

    logger.info(f"Found {len(game_days)} unique game days")
    if game_days:
        logger.info(f"  First game day: {game_days[0].date()}")
        logger.info(f"  Last game day: {game_days[-1].date()}")

    return game_days


def load_backtest_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all data required for backtesting

    Returns:
        Tuple of (all_data, test_data, odds_data)
        - all_data: Full historical dataset (for training windows)
        - test_data: 2024-25 season data (target for predictions)
        - odds_data: Historical betting lines
    """
    logger.info("Loading backtest datasets...")

    # Load full historical data (for training)
    all_data = load_master_features(season_filter=None)

    # Load target season data (for testing)
    test_data = load_master_features(season_filter=TARGET_SEASON)

    if test_data.empty:
        raise ValueError(f"No data found for target season {TARGET_SEASON}")

    # Load betting lines
    odds_data = load_historical_odds()

    logger.info(f"\nData loading complete:")
    logger.info(f"  All data: {len(all_data)} games ({all_data['season'].nunique()} seasons)")
    logger.info(f"  Test data (2024-25): {len(test_data)} games")
    logger.info(f"  Betting lines: {len(odds_data)} lines")

    return all_data, test_data, odds_data


def validate_data_quality(df: pd.DataFrame) -> None:
    """
    Validate data quality and log warnings

    Args:
        df: DataFrame to validate
    """
    # Check for missing values
    missing_pct = df.isnull().sum() / len(df)
    high_missing = missing_pct[missing_pct > 0.3]

    if len(high_missing) > 0:
        logger.warning(f"Found {len(high_missing)} features with >30% missing values:")
        for col, pct in high_missing.items():
            logger.warning(f"  - {col}: {pct:.1%} missing")

    # Check for duplicates
    if 'player_id' in df.columns and 'game_id' in df.columns:
        duplicates = df.duplicated(subset=['player_id', 'game_id']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate player-game records")

    # Check date range
    if 'game_date' in df.columns:
        date_range = (df['game_date'].max() - df['game_date'].min()).days
        logger.info(f"  Date range: {date_range} days")

    logger.info("✓ Data quality validation complete")
