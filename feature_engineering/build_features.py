"""
Build Features - Master Feature Pipeline
Joins all feature tables and creates final feature matrix for modeling
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)
import numpy as np
from pathlib import Path
from typing import List, Optional
from data_loader import load_player_gamelogs

# Import shared utilities and configuration
from utils import validate_grain_uniqueness
from config import PROJECT_ROOT, FEATURE_DIR, GRAIN_COLUMNS


def load_feature_table(filename: str) -> Optional[pd.DataFrame]:
    """
    Load a single feature table from the feature_tables directory

    Args:
        filename: Name of parquet file (e.g., "rolling_features.parquet")

    Returns:
        pd.DataFrame: Feature table with grain [player_id, game_id, game_date]
        None: If file doesn't exist (with warning printed)

    Examples:
        >>> features = load_feature_table("rolling_features.parquet")
        Loaded rolling_features.parquet: (587034, 53)

        >>> features = load_feature_table("nonexistent.parquet")
        Warning: nonexistent.parquet not found. Skipping.
        >>> features is None
        True

    Notes:
        - Expects file in FEATURE_DIR (data/feature_tables/)
        - Returns None gracefully if file missing (allows partial pipeline runs)
        - Prints shape for verification
    """
    filepath = FEATURE_DIR / filename

    if not filepath.exists():
        logger.warning(f"Warning: {filename} not found. Skipping.")
        return None

    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {filename}: {df.shape}")
    return df


def validate_grain(
    df: pd.DataFrame,
    grain_cols: List[str] = ['player_id', 'game_id', 'game_date']
) -> None:
    """
    Validate that DataFrame has exactly one row per grain combination

    CRITICAL: This ensures all feature tables can be joined 1:1 without duplicates.
    Grain violations indicate bugs in feature calculations (e.g., cartesian joins).

    Args:
        df: DataFrame to validate
        grain_cols: Columns that define the grain (default: [player_id, game_id, game_date])

    Raises:
        AssertionError: If duplicate rows found on grain columns

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'player_id': [1, 1, 2],
        ...     'game_id': [101, 102, 101],
        ...     'game_date': ['2023-10-01', '2023-10-02', '2023-10-01'],
        ...     'pra': [25, 30, 15]
        ... })
        >>> validate_grain(df)
        ✓ Grain validation passed: 3 unique rows on ['player_id', 'game_id', 'game_date']

        # Duplicate grain will fail
        >>> df_duplicate = pd.DataFrame({
        ...     'player_id': [1, 1],
        ...     'game_id': [101, 101],
        ...     'game_date': ['2023-10-01', '2023-10-01'],
        ...     'pra': [25, 27]
        ... })
        >>> validate_grain(df_duplicate)
        AssertionError: Grain violation: 1 duplicates found

    Notes:
        - Used before every feature table merge in build_master_features()
        - Prevents many-to-many joins that create duplicate rows
        - Prints first 5 duplicate rows for debugging
    """
    duplicate_count = df.duplicated(subset=grain_cols).sum()

    if duplicate_count > 0:
        logger.error(f"ERROR: Found {duplicate_count} duplicate rows on grain {grain_cols}")
        duplicates = df[df.duplicated(subset=grain_cols, keep=False)]
        logger.info(duplicates.head())
        raise AssertionError(f"Grain violation: {duplicate_count} duplicates found")

    logger.info(f"✓ Grain validation passed: {len(df)} unique rows on {grain_cols}")


def merge_features(
    base_df: pd.DataFrame,
    feature_tables: List[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge all feature tables onto base DataFrame

    Args:
        base_df: Base DataFrame with target variable
        feature_tables: List of feature DataFrames to merge

    Returns:
        Merged DataFrame with all features
    """
    grain_cols = ['player_id', 'game_id', 'game_date']

    # Start with base
    merged = base_df.copy()
    logger.info(f"\nStarting with base: {merged.shape}")

    # Merge each feature table
    for i, feature_df in enumerate(feature_tables):
        if feature_df is None:
            continue

        # Validate grain of feature table
        validate_grain(feature_df, grain_cols)

        # Merge
        before_shape = merged.shape
        merged = merged.merge(
            feature_df,
            on=grain_cols,
            how='left',
            validate='1:1'  # Ensures one-to-one relationship
        )

        logger.info(f"After merging table {i+1}: {merged.shape} (added {merged.shape[1] - before_shape[1]} columns)")

    return merged


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable (PRA) from box score stats

    Args:
        df: Player game logs

    Returns:
        DataFrame with target variable
    """
    df = df.copy()

    # Target: Points + Rebounds + Assists
    if 'pra' in df.columns:
        df['target_pra'] = df['pra']
    else:
        df['target_pra'] = df['points'] + df['rebounds'] + df['assists']

    return df


def create_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple base features from raw box score

    Args:
        df: Player game logs

    Returns:
        DataFrame with base features
    """
    grain_cols = ['player_id', 'game_id', 'game_date']

    base = df[grain_cols].copy()

    # Add target
    base['target_pra'] = df['target_pra']

    # Add basic identifiers
    base['season'] = df['season']
    base['player_name'] = df['player_name']

    return base


def check_feature_quality(df: pd.DataFrame) -> None:
    """
    Check quality of feature matrix

    Args:
        df: Feature matrix

    Prints:
        Summary statistics and warnings
    """
    logger.info("\n" + "="*60)
    logger.info("FEATURE QUALITY REPORT")
    logger.info("="*60)

    # Basic info
    logger.info(f"\nDataset shape: {df.shape}")
    logger.info(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    logger.info(f"Unique players: {df['player_id'].nunique()}")
    logger.info(f"Unique games: {df['game_id'].nunique()}")

    # Missing values
    logger.info("\n" + "-"*60)
    logger.info("MISSING VALUES")
    logger.info("-"*60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })

    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if len(missing_df) > 0:
        logger.info("\nColumns with missing values:")
        logger.info(missing_df.head(20))

        high_missing = missing_df[missing_df['Missing %'] > 50]
        if len(high_missing) > 0:
            logger.warning(f"\n⚠️  WARNING: {len(high_missing)} columns have >50% missing values")
    else:
        logger.info("✓ No missing values found")

    # Target variable check
    logger.info("\n" + "-"*60)
    logger.info("TARGET VARIABLE")
    logger.info("-"*60)

    if 'target_pra' in df.columns:
        logger.info(f"Mean PRA: {df['target_pra'].mean():.2f}")
        logger.info(f"Median PRA: {df['target_pra'].median():.2f}")
        logger.info(f"Std PRA: {df['target_pra'].std():.2f}")
        logger.info(f"Min PRA: {df['target_pra'].min():.2f}")
        logger.info(f"Max PRA: {df['target_pra'].max():.2f}")
        logger.info(f"Missing PRA: {df['target_pra'].isnull().sum()}")

        if df['target_pra'].isnull().sum() > 0:
            logger.warning("⚠️  WARNING: Target variable has missing values!")
    else:
        logger.warning("⚠️  ERROR: Target variable 'target_pra' not found!")

    # Feature counts
    logger.info("\n" + "-"*60)
    logger.info("FEATURE SUMMARY")
    logger.info("-"*60)

    feature_cols = [col for col in df.columns if col not in ['player_id', 'game_id', 'game_date', 'season', 'player_name', 'target_pra']]

    logger.info(f"Total features: {len(feature_cols)}")

    # Feature categories
    rolling_features = [col for col in feature_cols if 'avg' in col or 'std' in col or 'ewma' in col or 'trend' in col]
    matchup_features = [col for col in feature_cols if 'opp' in col or 'opponent' in col or 'rest' in col]
    contextual_features = [col for col in feature_cols if 'is_' in col or 'day' in col or 'game_' in col]
    advanced_features = [col for col in feature_cols if 'usage' in col or 'assist' in col or 'efficiency' in col or 'pct' in col]

    logger.info(f"  Rolling features: {len(rolling_features)}")
    logger.info(f"  Matchup features: {len(matchup_features)}")
    logger.info(f"  Contextual features: {len(contextual_features)}")
    logger.info(f"  Advanced features: {len(advanced_features)}")

    logger.info("\n" + "="*60)


def build_master_features() -> Optional[pd.DataFrame]:
    """
    Main entry point to build master feature matrix for modeling

    Orchestrates the complete feature building pipeline:
    1. Load player game logs with target variable (PRA)
    2. Create base features (grain, target, identifiers)
    3. Load all 6 feature tables from parquet files
    4. Validate grain uniqueness for each table
    5. Merge all features with 1:1 validation
    6. Quality check (missing values, distributions)
    7. Save master feature matrix to parquet

    Returns:
        pd.DataFrame: Master feature matrix with grain [player_id, game_id, game_date],
            target variable (target_pra), and 165+ features
        None: If no feature tables found

        Saved to: data/feature_tables/master_features.parquet

    Raises:
        AssertionError: If grain validation fails during merge

    Examples:
        >>> from feature_engineering.build_features import build_master_features
        >>> master = build_master_features()
        ============================================================
        BUILDING MASTER FEATURE MATRIX
        ============================================================
        Loading base player game logs...
        Loaded 587034 games
        ...
        ============================================================
        ✓ MASTER FEATURES SAVED
        ============================================================
        Location: .../master_features.parquet
        Shape: (587034, 171)
        Ready for model training!

        >>> master.shape
        (587034, 171)  # 165+ features + 6 meta columns

        >>> master.columns[:10]
        Index(['player_id', 'game_id', 'game_date', 'target_pra', 'season',
               'player_name', 'pra_avg_last5', 'pra_avg_last10', ...])

    Feature Tables Loaded (6):
        1. rolling_features.parquet: ~50 features (averages, EWMA, trends)
        2. matchup_features.parquet: ~9 features (opponent defense, pace)
        3. contextual_features.parquet: ~23 features (home/away, rest, timing)
        4. advanced_metrics.parquet: ~22 features (CTG usage, efficiency)
        5. position_features.parquet: ~14 features (z-scores, percentiles)
        6. injury_features.parquet: ~16 features (DNP, availability)

    Quality Checks Performed:
        - Grain uniqueness on [player_id, game_id, game_date]
        - Missing value analysis with warnings for >50% missing
        - Target variable statistics (mean, median, std)
        - Feature count by category (rolling, matchup, contextual, advanced)
        - Date range and player count verification

    Notes:
        - Expected runtime: 1-2 minutes for full dataset
        - Requires all 6 feature tables to exist (run pipeline first)
        - Uses validate='1:1' on all merges to prevent duplication
        - Missing feature tables will skip with warning
        - High-impact features: position_features, injury_features
        - Ready for model_training/train_split.py after this step
    """
    logger.info("="*60)
    logger.info("BUILDING MASTER FEATURE MATRIX")
    logger.info("="*60)

    # Load base data with target
    logger.info("\nLoading base player game logs...")
    df = load_player_gamelogs()
    df = create_target_variable(df)

    logger.info(f"Loaded {len(df)} games")

    # Create base features
    logger.info("\nCreating base features...")
    base = create_base_features(df)

    # Load all feature tables
    logger.info("\nLoading feature tables...")
    logger.info("-"*60)

    feature_tables = [
        load_feature_table("rolling_features.parquet"),
        load_feature_table("matchup_features.parquet"),
        load_feature_table("contextual_features.parquet"),
        load_feature_table("advanced_metrics.parquet"),
        load_feature_table("position_features.parquet"),  # NEW: High-impact
        load_feature_table("injury_features.parquet"),     # NEW: High-impact
    ]

    # Filter out None values
    feature_tables = [ft for ft in feature_tables if ft is not None]

    if len(feature_tables) == 0:
        logger.warning("\n⚠️  ERROR: No feature tables found!")
        logger.info("Please run the individual feature scripts first:")
        logger.info("  - python feature_engineering/rolling_features.py")
        logger.info("  - python feature_engineering/matchup_features.py")
        logger.info("  - python feature_engineering/contextual_features.py")
        logger.info("  - python feature_engineering/advanced_metrics.py")
        logger.info("  - python feature_engineering/position_features.py  [HIGH IMPACT]")
        logger.info("  - python feature_engineering/injury_features.py    [HIGH IMPACT]")
        return None

    # Merge all features
    logger.info("\n" + "="*60)
    logger.info("MERGING FEATURE TABLES")
    logger.info("="*60)

    master_features = merge_features(base, feature_tables)

    # Quality check
    check_feature_quality(master_features)

    # Save master feature matrix
    output_path = FEATURE_DIR / "master_features.parquet"
    master_features.to_parquet(output_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ MASTER FEATURES SAVED")
    logger.info(f"{'='*60}")
    logger.info(f"Location: {output_path}")
    logger.info(f"Shape: {master_features.shape}")
    logger.info(f"Ready for model training!")

    return master_features


if __name__ == "__main__":
    build_master_features()
