"""
Build Features - Master Feature Pipeline
Joins all feature tables and creates final feature matrix for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_player_gamelogs

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "feature_tables"


def load_feature_table(filename):
    """
    Load a feature table from parquet

    Args:
        filename: Name of parquet file

    Returns:
        DataFrame or None if file doesn't exist
    """
    filepath = FEATURE_DIR / filename

    if not filepath.exists():
        print(f"Warning: {filename} not found. Skipping.")
        return None

    df = pd.read_parquet(filepath)
    print(f"Loaded {filename}: {df.shape}")
    return df


def validate_grain(df, grain_cols=['player_id', 'game_id', 'game_date']):
    """
    Validate that DataFrame has unique grain

    Args:
        df: DataFrame to validate
        grain_cols: Columns that define the grain

    Raises:
        AssertionError if grain is not unique
    """
    duplicate_count = df.duplicated(subset=grain_cols).sum()

    if duplicate_count > 0:
        print(f"ERROR: Found {duplicate_count} duplicate rows on grain {grain_cols}")
        duplicates = df[df.duplicated(subset=grain_cols, keep=False)]
        print(duplicates.head())
        raise AssertionError(f"Grain violation: {duplicate_count} duplicates found")

    print(f"✓ Grain validation passed: {len(df)} unique rows on {grain_cols}")


def merge_features(base_df, feature_tables):
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
    print(f"\nStarting with base: {merged.shape}")

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

        print(f"After merging table {i+1}: {merged.shape} (added {merged.shape[1] - before_shape[1]} columns)")

    return merged


def create_target_variable(df):
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


def create_base_features(df):
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


def check_feature_quality(df):
    """
    Check quality of feature matrix

    Args:
        df: Feature matrix

    Prints:
        Summary statistics and warnings
    """
    print("\n" + "="*60)
    print("FEATURE QUALITY REPORT")
    print("="*60)

    # Basic info
    print(f"\nDataset shape: {df.shape}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique players: {df['player_id'].nunique()}")
    print(f"Unique games: {df['game_id'].nunique()}")

    # Missing values
    print("\n" + "-"*60)
    print("MISSING VALUES")
    print("-"*60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct
    })

    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if len(missing_df) > 0:
        print("\nColumns with missing values:")
        print(missing_df.head(20))

        high_missing = missing_df[missing_df['Missing %'] > 50]
        if len(high_missing) > 0:
            print(f"\n⚠️  WARNING: {len(high_missing)} columns have >50% missing values")
    else:
        print("✓ No missing values found")

    # Target variable check
    print("\n" + "-"*60)
    print("TARGET VARIABLE")
    print("-"*60)

    if 'target_pra' in df.columns:
        print(f"Mean PRA: {df['target_pra'].mean():.2f}")
        print(f"Median PRA: {df['target_pra'].median():.2f}")
        print(f"Std PRA: {df['target_pra'].std():.2f}")
        print(f"Min PRA: {df['target_pra'].min():.2f}")
        print(f"Max PRA: {df['target_pra'].max():.2f}")
        print(f"Missing PRA: {df['target_pra'].isnull().sum()}")

        if df['target_pra'].isnull().sum() > 0:
            print("⚠️  WARNING: Target variable has missing values!")
    else:
        print("⚠️  ERROR: Target variable 'target_pra' not found!")

    # Feature counts
    print("\n" + "-"*60)
    print("FEATURE SUMMARY")
    print("-"*60)

    feature_cols = [col for col in df.columns if col not in ['player_id', 'game_id', 'game_date', 'season', 'player_name', 'target_pra']]

    print(f"Total features: {len(feature_cols)}")

    # Feature categories
    rolling_features = [col for col in feature_cols if 'avg' in col or 'std' in col or 'ewma' in col or 'trend' in col]
    matchup_features = [col for col in feature_cols if 'opp' in col or 'opponent' in col or 'rest' in col]
    contextual_features = [col for col in feature_cols if 'is_' in col or 'day' in col or 'game_' in col]
    advanced_features = [col for col in feature_cols if 'usage' in col or 'assist' in col or 'efficiency' in col or 'pct' in col]

    print(f"  Rolling features: {len(rolling_features)}")
    print(f"  Matchup features: {len(matchup_features)}")
    print(f"  Contextual features: {len(contextual_features)}")
    print(f"  Advanced features: {len(advanced_features)}")

    print("\n" + "="*60)


def build_master_features():
    """
    Main function to build master feature matrix
    Loads all feature tables and combines them
    """
    print("="*60)
    print("BUILDING MASTER FEATURE MATRIX")
    print("="*60)

    # Load base data with target
    print("\nLoading base player game logs...")
    df = load_player_gamelogs()
    df = create_target_variable(df)

    print(f"Loaded {len(df)} games")

    # Create base features
    print("\nCreating base features...")
    base = create_base_features(df)

    # Load all feature tables
    print("\nLoading feature tables...")
    print("-"*60)

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
        print("\n⚠️  ERROR: No feature tables found!")
        print("Please run the individual feature scripts first:")
        print("  - python feature_engineering/rolling_features.py")
        print("  - python feature_engineering/matchup_features.py")
        print("  - python feature_engineering/contextual_features.py")
        print("  - python feature_engineering/advanced_metrics.py")
        print("  - python feature_engineering/position_features.py  [HIGH IMPACT]")
        print("  - python feature_engineering/injury_features.py    [HIGH IMPACT]")
        return None

    # Merge all features
    print("\n" + "="*60)
    print("MERGING FEATURE TABLES")
    print("="*60)

    master_features = merge_features(base, feature_tables)

    # Quality check
    check_feature_quality(master_features)

    # Save master feature matrix
    output_path = FEATURE_DIR / "master_features.parquet"
    master_features.to_parquet(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"✓ MASTER FEATURES SAVED")
    print(f"{'='*60}")
    print(f"Location: {output_path}")
    print(f"Shape: {master_features.shape}")
    print(f"Ready for model training!")

    return master_features


if __name__ == "__main__":
    build_master_features()
