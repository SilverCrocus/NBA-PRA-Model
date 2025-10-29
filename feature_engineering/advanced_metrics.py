"""
Advanced Metrics for NBA PRA Model
Integrates CleaningTheGlass (CTG) advanced statistics

These metrics provide deeper insights into player efficiency and role
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_player_gamelogs, consolidate_ctg_data_all_seasons, load_ctg_player_data

# Output directory
FEATURE_DIR = Path(__file__).parent.parent / "data" / "feature_tables"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def load_and_merge_ctg_data(player_games_df):
    """
    Load CTG data and merge with player games

    Args:
        player_games_df: Player game logs from NBA API

    Returns:
        DataFrame with CTG metrics merged
    """
    print("Loading CTG offensive overview data...")
    ctg_data = consolidate_ctg_data_all_seasons()

    if ctg_data.empty:
        print("Warning: No CTG data found. Creating placeholder features.")
        return player_games_df

    # Clean CTG player names for matching
    ctg_data['player'] = ctg_data['player'].str.strip()

    # Merge on player name and season
    # Note: This is season-level data, so same values for all games in a season
    merged = player_games_df.merge(
        ctg_data,
        left_on=['player_name', 'season'],
        right_on=['player', 'season'],
        how='left'
    )

    return merged


def create_usage_features(df):
    """
    Create usage rate and related features from CTG data

    Args:
        df: Player games with CTG data merged

    Returns:
        DataFrame with usage features
    """
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


def create_playmaking_features(df):
    """
    Create assist and playmaking features from CTG data

    Args:
        df: Player games with CTG data

    Returns:
        DataFrame with playmaking features
    """
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


def create_efficiency_features(df):
    """
    Create shooting efficiency features

    Args:
        df: Player games with CTG data

    Returns:
        DataFrame with efficiency features
    """
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


def create_rebounding_features(df):
    """
    Create rebounding features from CTG data

    Args:
        df: Player games with CTG data

    Returns:
        DataFrame with rebounding features
    """
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


def create_defense_features(df):
    """
    Create defensive features from box score

    Args:
        df: Player games DataFrame

    Returns:
        DataFrame with defensive features
    """
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


def create_role_indicators(df):
    """
    Create indicators for player role based on stats

    Args:
        df: Player games with metrics

    Returns:
        DataFrame with role indicators
    """
    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Calculate role indicators from recent games
    # Group by player and calculate rolling averages (season-based)

    # High usage player (star/primary option)
    if 'usage' in df.columns:
        features['is_high_usage'] = (df['usage'] >= 0.25).astype(int)
        features['is_low_usage'] = (df['usage'] < 0.18).astype(int)
    else:
        features['is_high_usage'] = 0
        features['is_low_usage'] = 0

    # Primary playmaker
    if 'ast%' in df.columns:
        features['is_primary_playmaker'] = (df['ast%'] >= 0.25).astype(int)
    else:
        features['is_primary_playmaker'] = 0

    # Three-point specialist
    if 'fg3_attempted' in df.columns and 'fg_attempted' in df.columns:
        three_pt_rate = df['fg3_attempted'] / df['fg_attempted'].replace(0, np.nan)
        features['is_three_point_specialist'] = (three_pt_rate >= 0.5).astype(int)
    else:
        features['is_three_point_specialist'] = 0

    return features


def convert_minutes_to_float(minutes):
    """Convert minutes to float"""
    if pd.isna(minutes):
        return 0.0
    if isinstance(minutes, (int, float)):
        return float(minutes)
    if isinstance(minutes, str) and ':' in minutes:
        try:
            parts = minutes.split(':')
            return float(parts[0]) + float(parts[1]) / 60
        except:
            return 0.0
    try:
        return float(minutes)
    except:
        return 0.0


def validate_advanced_metrics(features_df):
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

    print("✓ Advanced metrics validation passed")


def build_advanced_metrics():
    """
    Main function to build all advanced metrics features
    """
    print("Loading player game logs...")
    df = load_player_gamelogs()

    print(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    print("\nMerging CTG data...")
    df_with_ctg = load_and_merge_ctg_data(df)

    print("Creating usage features...")
    usage = create_usage_features(df_with_ctg)

    print("Creating playmaking features...")
    playmaking = create_playmaking_features(df_with_ctg)

    print("Creating efficiency features...")
    efficiency = create_efficiency_features(df_with_ctg)

    print("Creating rebounding features...")
    rebounding = create_rebounding_features(df_with_ctg)

    print("Creating defense features...")
    defense = create_defense_features(df_with_ctg)

    print("Creating role indicators...")
    roles = create_role_indicators(df_with_ctg)

    print("\nMerging all advanced metrics features...")
    features = usage.copy()

    for feature_df in [playmaking, efficiency, rebounding, defense, roles]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    print(f"Created {len(features.columns) - 3} advanced metrics features")

    # Validate
    print("\nValidating advanced metrics...")
    validate_advanced_metrics(features)

    # Save
    output_path = FEATURE_DIR / "advanced_metrics.parquet"
    features.to_parquet(output_path, index=False)

    print(f"\n✓ Saved advanced metrics to {output_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_advanced_metrics()
