"""
Advanced Metrics for NBA PRA Model
Integrates CleaningTheGlass (CTG) advanced statistics

These metrics provide deeper insights into player efficiency and role
"""

# Add project root to path when running as script
if __name__ == "__main__":
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

import pandas as pd
import logging

logger = logging.getLogger(__name__)
import numpy as np
from pathlib import Path
from feature_engineering.data_loader import load_player_gamelogs, consolidate_ctg_data_all_seasons, load_ctg_player_data

# Import shared utilities and configuration
from feature_engineering.utils import (convert_minutes_to_float, create_feature_base,
                   validate_not_empty, validate_required_columns)
from feature_engineering.config import (FEATURE_DIR, CTG_SEASON_MAPPING, HIGH_USAGE_THRESHOLD,
                    LOW_USAGE_THRESHOLD, PRIMARY_PLAYMAKER_THRESHOLD,
                    THREE_POINT_SPECIALIST_THRESHOLD)
from feature_engineering.features.ctg_imputation import (
    create_imputation_flags,
    create_position_relative_features,
    calculate_position_baselines
)


def load_and_merge_ctg_data(player_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load CTG data and merge with player games using PREVIOUS season's data
    to prevent temporal leakage.

    CTG data contains season-level aggregates. Using current season's data would
    include information from future games. Instead, we use the previous season's
    stats as a proxy for player ability/role.

    Args:
        player_games_df: Player game logs from NBA API

    Returns:
        DataFrame with CTG metrics merged
    """
    # Input validation
    validate_not_empty(player_games_df, 'load_and_merge_ctg_data')
    validate_required_columns(
        player_games_df,
        ['player_id', 'game_id', 'game_date', 'season', 'player_name'],
        'load_and_merge_ctg_data'
    )

    logger.info("Loading CTG offensive overview data...")
    ctg_data = consolidate_ctg_data_all_seasons()

    if ctg_data.empty:
        logger.warning("Warning: No CTG data found. Creating placeholder features.")
        return player_games_df

    # Clean CTG player names for matching
    ctg_data['player'] = ctg_data['player'].str.strip()

    # Extract position from CTG data (Pos column)
    # Position is available in CTG offensive overview
    if 'Pos' in ctg_data.columns:
        logger.info(f"  Found position data in CTG: {ctg_data['Pos'].nunique()} unique positions")
    else:
        logger.warning("  No position column found in CTG data")

    # Map current season to previous season for CTG lookup
    # Using imported CTG_SEASON_MAPPING from config
    player_games_df['previous_season'] = player_games_df['season'].map(CTG_SEASON_MAPPING)

    # Merge using PREVIOUS season's CTG data
    merged = player_games_df.merge(
        ctg_data,
        left_on=['player_name', 'previous_season'],
        right_on=['player', 'season'],
        how='left',
        suffixes=('', '_ctg')
    )

    # Rename CTG position column if it exists
    if 'Pos' in merged.columns:
        merged = merged.rename(columns={'Pos': 'position'})
        logger.info(f"  Extracted position for {(~merged['position'].isna()).sum()} games")

    # Drop temporary column
    merged = merged.drop(columns=['previous_season'], errors='ignore')

    logger.info(f"  Merged CTG data: {(~merged['usage'].isna()).sum() if 'usage' in merged.columns else 0} games with CTG stats")

    return merged


def create_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create usage rate and related features from CTG data

    Args:
        df: Player games with CTG data merged

    Returns:
        DataFrame with usage features
    """
    # Input validation
    validate_not_empty(df, 'create_usage_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date'],
        'create_usage_features'
    )

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Usage Rate (percentage of team possessions used)
    if 'usage' in df.columns:
        features['usage_rate'] = df['usage']
        features['usage_rate_rank'] = df.get('usage_rank', np.nan)
    else:
        features['usage_rate'] = np.nan
        features['usage_rate_rank'] = np.nan

    # Points per shot attempt (efficiency)
    if 'psa' in df.columns:
        features['points_per_shot_attempt'] = df['psa']
        features['psa_rank'] = df.get('psa_rank', np.nan)
    else:
        features['points_per_shot_attempt'] = np.nan
        features['psa_rank'] = np.nan

    return features


def create_playmaking_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create assist and playmaking features from CTG data

    Args:
        df: Player games with CTG data

    Returns:
        DataFrame with playmaking features
    """
    # Input validation
    validate_not_empty(df, 'create_playmaking_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date'],
        'create_playmaking_features'
    )

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Assist percentage
    if 'ast%' in df.columns:
        features['assist_rate'] = df['ast%']
        features['assist_rate_rank'] = df.get('ast%_rank', np.nan)
    else:
        features['assist_rate'] = np.nan
        features['assist_rate_rank'] = np.nan

    # Assist to usage ratio (playmaking efficiency)
    if 'ast:usg' in df.columns:
        features['ast_to_usage_ratio'] = df['ast:usg']
    else:
        features['ast_to_usage_ratio'] = np.nan

    # Turnover percentage
    if 'tov%' in df.columns:
        features['turnover_rate'] = df['tov%']
        features['turnover_rate_rank'] = df.get('tov%_rank', np.nan)
    else:
        features['turnover_rate'] = np.nan
        features['turnover_rate_rank'] = np.nan

    return features


def create_efficiency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create shooting efficiency features

    Args:
        df: Player games with CTG data

    Returns:
        DataFrame with efficiency features
    """
    # Input validation
    validate_not_empty(df, 'create_efficiency_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date'],
        'create_efficiency_features'
    )

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Effective field goal percentage
    # Calculate from box score if not in CTG
    if 'fg_made' in df.columns and 'fg_attempted' in df.columns and 'fg3_made' in df.columns:
        # eFG% = (FGM + 0.5 * 3PM) / FGA
        features['efg_pct'] = (
            (df['fg_made'] + 0.5 * df['fg3_made']) / df['fg_attempted'].replace(0, np.nan)
        )
    else:
        features['efg_pct'] = np.nan

    # True shooting percentage
    if 'fg_attempted' in df.columns and 'ft_attempted' in df.columns and 'points' in df.columns:
        # TS% = PTS / (2 * (FGA + 0.44 * FTA))
        features['true_shooting_pct'] = (
            df['points'] / (2 * (df['fg_attempted'] + 0.44 * df['ft_attempted'])).replace(0, np.nan)
        )
    else:
        features['true_shooting_pct'] = np.nan

    return features


def create_rebounding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rebounding features from CTG data

    Args:
        df: Player games with CTG data

    Returns:
        DataFrame with rebounding features
    """
    # Input validation
    validate_not_empty(df, 'create_rebounding_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date'],
        'create_rebounding_features'
    )

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Raw rebounds from box score
    if 'offensive_rebounds' in df.columns:
        features['offensive_rebound_pct'] = df['offensive_rebounds'] / df.get('rebounds', 1).replace(0, np.nan)
    else:
        features['offensive_rebound_pct'] = np.nan

    if 'defensive_rebounds' in df.columns:
        features['defensive_rebound_pct'] = df['defensive_rebounds'] / df.get('rebounds', 1).replace(0, np.nan)
    else:
        features['defensive_rebound_pct'] = np.nan

    return features


def create_defense_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create defensive features from box score

    Args:
        df: Player games DataFrame

    Returns:
        DataFrame with defensive features
    """
    # Input validation
    validate_not_empty(df, 'create_defense_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date'],
        'create_defense_features'
    )

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Defensive stats from box score
    features['steals_per_game'] = df.get('steals', np.nan)
    features['blocks_per_game'] = df.get('blocks', np.nan)
    features['defensive_stocks'] = df.get('steals', 0) + df.get('blocks', 0)

    # Per-minute defensive stats (if minutes available)
    if 'minutes' in df.columns:
        df['minutes_float'] = df['minutes'].apply(convert_minutes_to_float)

        features['steals_per_36'] = (
            df.get('steals', 0) / df['minutes_float'].replace(0, np.nan) * 36
        )

        features['blocks_per_36'] = (
            df.get('blocks', 0) / df['minutes_float'].replace(0, np.nan) * 36
        )

    return features


def create_role_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create indicators for player role based on stats

    Args:
        df: Player games with metrics

    Returns:
        DataFrame with role indicators
    """
    # Input validation
    validate_not_empty(df, 'create_role_indicators')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date'],
        'create_role_indicators'
    )

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Calculate role indicators from recent games
    # Using imported thresholds from config

    # High usage player (star/primary option)
    if 'usage' in df.columns:
        features['is_high_usage'] = (df['usage'] >= HIGH_USAGE_THRESHOLD).astype(int)
        features['is_low_usage'] = (df['usage'] < LOW_USAGE_THRESHOLD).astype(int)
    else:
        features['is_high_usage'] = 0
        features['is_low_usage'] = 0

    # Primary playmaker
    if 'ast%' in df.columns:
        features['is_primary_playmaker'] = (df['ast%'] >= PRIMARY_PLAYMAKER_THRESHOLD).astype(int)
    else:
        features['is_primary_playmaker'] = 0

    # Three-point specialist
    if 'fg3_attempted' in df.columns and 'fg_attempted' in df.columns:
        three_pt_rate = df['fg3_attempted'] / df['fg_attempted'].replace(0, np.nan)
        features['is_three_point_specialist'] = (three_pt_rate >= THREE_POINT_SPECIALIST_THRESHOLD).astype(int)
    else:
        features['is_three_point_specialist'] = 0

    return features


def validate_advanced_metrics(features_df: pd.DataFrame) -> None:
    """
    Validate advanced metrics features

    Args:
        features_df: Computed features
    """
    # Check usage rate is in reasonable range (if present)
    if 'usage_rate' in features_df.columns:
        usage_mean = features_df['usage_rate'].dropna().mean()
        if not np.isnan(usage_mean):
            assert 0.10 < usage_mean < 0.35, f"Usage rate mean unusual: {usage_mean}"

    # Check true shooting is in reasonable range
    if 'true_shooting_pct' in features_df.columns:
        ts_mean = features_df['true_shooting_pct'].dropna().mean()
        if not np.isnan(ts_mean):
            assert 0.3 < ts_mean < 0.8, f"True shooting % mean unusual: {ts_mean}"

    logger.info("✓ Advanced metrics validation passed")


def build_advanced_metrics() -> pd.DataFrame:
    """
    Main function to build all advanced metrics features
    """
    logger.info("Loading player game logs...")
    df = load_player_gamelogs()

    # Input validation
    validate_not_empty(df, 'build_advanced_metrics')

    logger.info(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    logger.info("\nMerging CTG data...")
    df_with_ctg = load_and_merge_ctg_data(df)

    logger.info("Creating usage features...")
    usage = create_usage_features(df_with_ctg)

    logger.info("Creating playmaking features...")
    playmaking = create_playmaking_features(df_with_ctg)

    logger.info("Creating efficiency features...")
    efficiency = create_efficiency_features(df_with_ctg)

    logger.info("Creating rebounding features...")
    rebounding = create_rebounding_features(df_with_ctg)

    logger.info("Creating defense features...")
    defense = create_defense_features(df_with_ctg)

    logger.info("Creating role indicators...")
    roles = create_role_indicators(df_with_ctg)

    logger.info("\nMerging all advanced metrics features...")
    features = usage.copy()

    for feature_df in [playmaking, efficiency, rebounding, defense, roles]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    logger.info(f"Created {len(features.columns) - 3} advanced metrics features")

    # NEW: Add imputation features
    logger.info("\nCreating imputation features...")

    # Step 1: Create imputation flags (track what was missing)
    logger.info("  Creating imputation flags...")
    imputation_flags = create_imputation_flags(df_with_ctg)

    # Step 2: Calculate position baselines from ALL data
    # Note: In walk-forward backtest, this will be calculated per-fold from training data only
    logger.info("  Calculating position baselines...")
    if 'position' in df_with_ctg.columns:
        position_baselines = calculate_position_baselines(df_with_ctg)

        # Step 3: Create position-relative features
        logger.info("  Creating position-relative features...")
        position_relative = create_position_relative_features(df_with_ctg, position_baselines)

        # Merge new features
        features = features.merge(
            imputation_flags,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

        features = features.merge(
            position_relative,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

        logger.info(f"  Added {len(imputation_flags.columns) - 3} imputation flags")
        logger.info(f"  Added {len(position_relative.columns) - 3} position-relative features")
    else:
        logger.warning("  No position data available, skipping position-relative features")
        # Still add imputation flags
        features = features.merge(
            imputation_flags,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )
        logger.info(f"  Added {len(imputation_flags.columns) - 3} imputation flags")

    logger.info(f"\nTotal features created: {len(features.columns) - 3}")

    # Validate
    logger.info("\nValidating advanced metrics...")
    validate_advanced_metrics(features)

    # Save
    output_path = FEATURE_DIR / "advanced_metrics.parquet"
    features.to_parquet(output_path, index=False)

    logger.info(f"\n✓ Saved advanced metrics to {output_path}")
    logger.info(f"  Shape: {features.shape}")
    logger.info(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_advanced_metrics()
