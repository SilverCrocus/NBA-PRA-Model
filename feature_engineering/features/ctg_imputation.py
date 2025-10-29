"""
CTG Missing Data Imputation Module

Handles missing CleaningTheGlass (CTG) data using positional averages and imputation flags.
This prevents bias from forward-filling and allows the model to learn when CTG data is unreliable.

Key principles:
- Use positional averages for rookies/missing players (NOT forward-fill)
- Create binary flags so XGBoost can learn interaction effects
- Calculate baselines from training data only (prevent temporal leakage)
- Separate prior-season features from current season

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from feature_engineering.utils import (
    validate_not_empty,
    validate_required_columns,
    validate_grain_uniqueness,
    create_feature_base
)
from feature_engineering.config import GRAIN_COLUMNS, CTG_SEASON_MAPPING

logger = logging.getLogger(__name__)

# CTG features that need imputation
CTG_FEATURES = [
    'usage_rate', 'assist_rate', 'turnover_rate',
    'points_per_shot_attempt', 'true_shooting_pct', 'efg_pct',
    'offensive_rebound_pct', 'defensive_rebound_pct',
    'usage_rate_rank', 'assist_rate_rank', 'turnover_rate_rank',
    'psa_rank', 'efficiency_rank'
]


def calculate_position_baselines(train_df: pd.DataFrame,
                                  features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate position-level statistics from training data ONLY.

    CRITICAL: Must be called separately for each CV fold to prevent temporal leakage.
    Do NOT calculate from entire dataset.

    Args:
        train_df: Training data for current fold
        features: List of CTG features to calculate baselines for (defaults to CTG_FEATURES)

    Returns:
        DataFrame with columns: position, feature_mean, feature_std, feature_median

    Example:
        >>> train_fold = pd.read_parquet('data/processed/cv_folds/fold_0/train.parquet')
        >>> baselines = calculate_position_baselines(train_fold)
        >>> # Apply to val/test using these baselines
    """
    validate_not_empty(train_df, 'calculate_position_baselines')

    if features is None:
        features = CTG_FEATURES

    # Check if position column exists
    if 'position' not in train_df.columns:
        logger.warning("No 'position' column found. Will use inferred position or default.")
        # If position doesn't exist, create a default 'Unknown' position
        train_df = train_df.copy()
        train_df['position'] = 'Unknown'

    logger.info(f"Calculating position baselines for {len(features)} features from {len(train_df)} training games")

    # Calculate statistics by position
    agg_dict = {feat: ['mean', 'std', 'median', 'count'] for feat in features if feat in train_df.columns}

    if not agg_dict:
        logger.warning("No CTG features found in training data")
        return pd.DataFrame()

    position_stats = train_df.groupby('position').agg(agg_dict).reset_index()

    # Flatten multi-level columns
    position_stats.columns = ['position'] + [
        f'{col[0]}_{col[1]}' for col in position_stats.columns[1:]
    ]

    logger.info(f"Calculated baselines for {len(position_stats)} positions")

    return position_stats


def create_imputation_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary flags indicating CTG data availability and quality.

    Flags enable XGBoost to learn interaction effects like:
    - is_rookie=1 → discount long-term averages
    - has_ctg_data=0 → rely more on minutes/playing time

    Args:
        df: Player games with CTG features

    Returns:
        DataFrame with imputation flag features on standard grain
    """
    validate_not_empty(df, 'create_imputation_flags')
    validate_required_columns(df, GRAIN_COLUMNS, 'create_imputation_flags')

    features = create_feature_base(df)

    # Flag 1: Has CTG data for current game (usage_rate is primary CTG feature)
    features['has_ctg_data'] = df['usage_rate'].notna().astype(int)

    # Flag 2: Is rookie (no prior season CTG data available)
    # Detect by checking if current season maps to a previous season
    if 'season' in df.columns:
        df_temp = df.copy()
        df_temp['prior_season'] = df_temp['season'].map(CTG_SEASON_MAPPING)
        features['is_rookie'] = df_temp['prior_season'].isna().astype(int)
    else:
        logger.warning("No 'season' column found, setting is_rookie=0 for all")
        features['is_rookie'] = 0

    # Flag 3: Number of seasons with CTG history (cumulative count per player)
    if 'player_id' in df.columns:
        features['ctg_seasons_available'] = (
            df.groupby('player_id')['usage_rate']
            .transform(lambda x: x.notna().expanding().sum())
            .fillna(0)
            .astype(int)
        )
    else:
        features['ctg_seasons_available'] = 0

    # Flag 4: CTG data quality score (0-1, higher = more reliable)
    # Based on: has data AND multiple seasons of history
    features['ctg_data_quality'] = np.clip(
        (features['has_ctg_data'] * features['ctg_seasons_available']) / 5.0,
        0, 1
    )

    logger.info(f"Created imputation flags for {len(features)} games")
    logger.info(f"  - {features['has_ctg_data'].sum()} games with CTG data")
    logger.info(f"  - {features['is_rookie'].sum()} rookie games")
    logger.info(f"  - Avg CTG quality score: {features['ctg_data_quality'].mean():.3f}")

    return features


def create_position_relative_features(df: pd.DataFrame,
                                       position_baselines: pd.DataFrame) -> pd.DataFrame:
    """
    Create features relative to position averages (e.g., usage_vs_position).

    Provides positional context when CTG data is missing or for interpreting player roles.
    A guard with 0.18 usage might be below league average but normal for guards.

    Args:
        df: Player games with CTG features and position
        position_baselines: Position statistics from calculate_position_baselines()

    Returns:
        DataFrame with position-relative features
    """
    validate_not_empty(df, 'create_position_relative_features')
    validate_required_columns(df, GRAIN_COLUMNS, 'create_position_relative_features')

    if position_baselines.empty:
        logger.warning("Empty position baselines, skipping position-relative features")
        return create_feature_base(df)

    features = create_feature_base(df)

    # Merge position baselines
    if 'position' not in df.columns:
        logger.warning("No 'position' column in data, cannot create position-relative features")
        return features

    df_merged = df.merge(position_baselines, on='position', how='left')

    # Create delta features for key CTG metrics
    relative_features = [
        'usage_rate', 'assist_rate', 'turnover_rate',
        'points_per_shot_attempt', 'true_shooting_pct', 'efg_pct'
    ]

    for feat in relative_features:
        if feat in df.columns and f'{feat}_mean' in df_merged.columns:
            # Delta from position average
            features[f'{feat}_vs_position'] = (
                df_merged[feat] - df_merged[f'{feat}_mean']
            )

            # Z-score relative to position (standardized)
            std_col = f'{feat}_std'
            if std_col in df_merged.columns:
                features[f'{feat}_position_zscore'] = (
                    (df_merged[feat] - df_merged[f'{feat}_mean']) /
                    df_merged[std_col].replace(0, np.nan)
                )

    created_features = [c for c in features.columns if c not in GRAIN_COLUMNS]
    logger.info(f"Created {len(created_features)} position-relative features")

    return features


def create_prior_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create separate features for prior-season CTG stats (not imputed into current season).

    This allows model to learn: "Player has no current CTG, but last year they had X usage."
    Better than forward-filling because model can learn to discount stale data.

    Args:
        df: Player games with CTG data

    Returns:
        DataFrame with prior-season CTG features
    """
    validate_not_empty(df, 'create_prior_season_features')
    validate_required_columns(df, GRAIN_COLUMNS, 'create_prior_season_features')

    features = create_feature_base(df)

    # For now, just create placeholders for the prior-season features
    # These would be filled in by merging prior-season CTG data explicitly
    prior_features = ['usage_rate', 'assist_rate', 'points_per_shot_attempt']

    for feat in prior_features:
        features[f'prev_season_{feat}'] = np.nan  # Will be filled by merge logic

    logger.info(f"Created prior-season feature placeholders for {len(features)} games")

    return features


def impute_missing_ctg(df: pd.DataFrame,
                       position_baselines: pd.DataFrame,
                       impute_method: str = 'position_mean') -> pd.DataFrame:
    """
    Impute missing CTG features using positional averages.

    Strategy:
    - For players with prior-season CTG: Use that (via CTG_SEASON_MAPPING)
    - For rookies/missing players: Use position mean from training data
    - NEVER forward-fill current season data

    Args:
        df: Player games potentially with missing CTG features
        position_baselines: Position statistics from calculate_position_baselines()
        impute_method: 'position_mean', 'position_median', or 'league_mean'

    Returns:
        DataFrame with imputed CTG features
    """
    validate_not_empty(df, 'impute_missing_ctg')

    if position_baselines.empty:
        logger.warning("Empty position baselines, cannot impute")
        return df

    if 'position' not in df.columns:
        logger.warning("No 'position' column, cannot impute using position averages")
        return df

    df_imputed = df.copy()

    # Merge position baselines
    df_imputed = df_imputed.merge(position_baselines, on='position', how='left')

    stat_suffix = '_mean' if impute_method == 'position_mean' else '_median'

    # Impute each CTG feature
    imputed_counts = {}
    for feat in CTG_FEATURES:
        if feat not in df_imputed.columns:
            continue

        baseline_col = f'{feat}{stat_suffix}'
        if baseline_col not in df_imputed.columns:
            continue

        # Count missing before imputation
        missing_mask = df_imputed[feat].isna()
        missing_count = missing_mask.sum()

        if missing_count > 0:
            # Impute with position baseline
            df_imputed.loc[missing_mask, feat] = df_imputed.loc[missing_mask, baseline_col]
            imputed_counts[feat] = missing_count

    # Clean up merged baseline columns
    baseline_cols = [c for c in df_imputed.columns if any(c.endswith(s) for s in ['_mean', '_std', '_median', '_count'])]
    df_imputed = df_imputed.drop(columns=baseline_cols, errors='ignore')

    logger.info(f"Imputed missing CTG features using {impute_method}:")
    for feat, count in imputed_counts.items():
        logger.info(f"  - {feat}: {count} values imputed")

    return df_imputed


def apply_ctg_imputation_pipeline(df: pd.DataFrame,
                                   position_baselines: Optional[pd.DataFrame] = None,
                                   calculate_baselines: bool = False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Full CTG imputation pipeline: flags + imputation + position-relative features.

    Use this for consistent imputation across train/val/test splits.

    Args:
        df: Player games with CTG data
        position_baselines: Pre-calculated position baselines (pass None if calculate_baselines=True)
        calculate_baselines: If True, calculate baselines from df (use for training data only)

    Returns:
        Tuple of (imputed_data, feature_tables_dict)
        - imputed_data: Original df with imputations applied
        - feature_tables_dict: Dict with keys 'imputation_flags', 'position_relative', 'prior_season'

    Example:
        >>> # For training fold:
        >>> train_imputed, train_features = apply_ctg_imputation_pipeline(train_df, calculate_baselines=True)
        >>>
        >>> # For val/test (use training baselines):
        >>> baselines = train_features['position_baselines']
        >>> val_imputed, val_features = apply_ctg_imputation_pipeline(val_df, position_baselines=baselines)
    """
    validate_not_empty(df, 'apply_ctg_imputation_pipeline')

    feature_tables = {}

    # Step 1: Calculate or validate position baselines
    if calculate_baselines:
        logger.info("Calculating position baselines from input data")
        position_baselines = calculate_position_baselines(df)
        feature_tables['position_baselines'] = position_baselines
    elif position_baselines is None:
        raise ValueError("Must provide position_baselines or set calculate_baselines=True")
    else:
        feature_tables['position_baselines'] = position_baselines

    # Step 2: Create imputation flags BEFORE imputing (to track what was missing)
    logger.info("Creating imputation flags")
    imputation_flags = create_imputation_flags(df)
    feature_tables['imputation_flags'] = imputation_flags

    # Step 3: Impute missing CTG features
    logger.info("Imputing missing CTG features")
    df_imputed = impute_missing_ctg(df, position_baselines, impute_method='position_mean')

    # Step 4: Create position-relative features
    logger.info("Creating position-relative features")
    position_relative = create_position_relative_features(df_imputed, position_baselines)
    feature_tables['position_relative'] = position_relative

    # Step 5: Create prior-season features
    logger.info("Creating prior-season features")
    prior_season = create_prior_season_features(df_imputed)
    feature_tables['prior_season'] = prior_season

    logger.info(f"CTG imputation pipeline complete. Created {len(feature_tables)} feature tables.")

    return df_imputed, feature_tables
