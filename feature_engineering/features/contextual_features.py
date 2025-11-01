"""
Contextual Features for NBA PRA Model
Creates game context features (home/away, time factors, etc.)
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
from feature_engineering.data_loader import load_player_gamelogs

# Import shared utilities and configuration
from feature_engineering.utils import (convert_minutes_to_float, create_feature_base,
                   validate_not_empty, validate_required_columns)
from feature_engineering.config import FEATURE_DIR, DEFAULT_REST_DAYS, RECENT_PERFORMANCE_DAYS


def create_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create home/away related features

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with home/away features
    """
    # Input validation
    validate_not_empty(df, 'create_home_away_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date', 'is_home'],
        'create_home_away_features'
    )

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Home game indicator (already in data)
    features['is_home'] = df['is_home']

    return features


def create_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rest and fatigue related features

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with rest features
    """
    # Input validation
    validate_not_empty(df, 'create_rest_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date'],
        'create_rest_features'
    )

    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Days since last game
    features['days_since_last_game'] = (
        df.groupby('player_id')['game_date']
        .diff()
        .dt.days
    )

    # Fill first game (start of season) with 3 days
    features['days_since_last_game'] = features['days_since_last_game'].fillna(3)

    # Back-to-back games (most fatiguing)
    features['is_back_to_back'] = (features['days_since_last_game'] == 1).astype(int)

    # One day rest
    features['is_one_day_rest'] = (features['days_since_last_game'] == 2).astype(int)

    # Well rested (3+ days)
    features['is_well_rested'] = (features['days_since_last_game'] >= 3).astype(int)

    # Games in last 7 days (intensity indicator)
    # CRITICAL: Use custom calculation to exclude current game and avoid leakage
    def count_games_in_window(dates, days):
        """Count games in last N days excluding current game"""
        result = []
        for idx in range(len(dates)):
            current_date = dates.iloc[idx]
            # Get games in the window BEFORE current game
            window_start = current_date - pd.Timedelta(days=days)
            mask = (dates >= window_start) & (dates < current_date)
            count = mask.sum()
            result.append(count)
        return result

    features['games_last_7_days'] = (
        df.groupby('player_id')['game_date']
        .transform(lambda x: count_games_in_window(x, days=7))
    )

    # Games in last 14 days
    features['games_last_14_days'] = (
        df.groupby('player_id')['game_date']
        .transform(lambda x: count_games_in_window(x, days=14))
    )

    return features


def create_season_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to timing in the season

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with season timing features
    """
    # Input validation
    validate_not_empty(df, 'create_season_timing_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date', 'season'],
        'create_season_timing_features'
    )

    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Number of prior games this season for this player (prevents leakage)
    # cumcount() is 0-indexed: game 1 shows 0, game 2 shows 1, etc.
    # This is correct - we want count of PRIOR games, not current game number
    features['game_number_season'] = (
        df.groupby(['player_id', 'season']).cumcount()
    )

    # Extract month from game date
    features['game_month'] = df['game_date'].dt.month

    # Season stage indicators
    # October-November: Early season (10, 11)
    features['is_early_season'] = features['game_month'].isin([10, 11]).astype(int)

    # December-February: Mid season (12, 1, 2)
    features['is_mid_season'] = features['game_month'].isin([12, 1, 2]).astype(int)

    # March-April: Late season (3, 4)
    features['is_late_season'] = features['game_month'].isin([3, 4]).astype(int)

    # Playoff months (May-June) - though we're focusing on regular season
    features['is_playoff_month'] = features['game_month'].isin([5, 6]).astype(int)

    # Day of week (some players perform better on certain days)
    features['day_of_week'] = df['game_date'].dt.dayofweek

    # Weekend game indicator
    features['is_weekend'] = features['day_of_week'].isin([4, 5, 6]).astype(int)

    return features


def create_recent_performance_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary indicators for recent performance patterns

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with performance indicators
    """
    # Input validation
    validate_not_empty(df, 'create_recent_performance_indicators')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date', 'pra', 'season'],
        'create_recent_performance_indicators'
    )

    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Recent PRA performance relative to season average
    season_avg = (
        df.groupby(['player_id', 'season'])['pra']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    last_game_pra = df.groupby('player_id')['pra'].shift(1)

    # Hot streak: last game above season average
    features['last_game_above_avg'] = (last_game_pra > season_avg).astype(int)

    # Last 3 games average vs season average
    last3_avg = (
        df.groupby('player_id')['pra']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    features['hot_last_3_games'] = (last3_avg > season_avg * 1.1).astype(int)
    features['cold_last_3_games'] = (last3_avg < season_avg * 0.9).astype(int)

    return features


def create_minutes_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features related to playing time context

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with minutes context features
    """
    # Input validation
    validate_not_empty(df, 'create_minutes_context')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date', 'minutes'],
        'create_minutes_context'
    )

    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Convert minutes to float if needed
    if 'minutes' in df.columns:
        df['minutes_float'] = df['minutes'].apply(convert_minutes_to_float)

        # Recent minutes trend
        features['minutes_last_game'] = df.groupby('player_id')['minutes_float'].shift(1)

        features['minutes_avg_last5'] = (
            df.groupby('player_id')['minutes_float']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

        # Minutes stability (low std = consistent role)
        features['minutes_std_last5'] = (
            df.groupby('player_id')['minutes_float']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
        )

        # Heavy minutes indicator (likely starter)
        features['is_high_minutes'] = (features['minutes_avg_last5'] >= 28).astype(int)

        # Limited minutes indicator (bench role)
        features['is_low_minutes'] = (features['minutes_avg_last5'] < 20).astype(int)

    return features


def validate_contextual_features(features_df: pd.DataFrame) -> None:
    """
    Validate contextual features

    Args:
        features_df: Computed features
    """
    # Check binary features are 0 or 1
    binary_cols = [col for col in features_df.columns if col.startswith('is_')]

    for col in binary_cols:
        unique_vals = features_df[col].dropna().unique()
        assert set(unique_vals).issubset({0, 1}), f"{col} contains non-binary values"

    # Check days since last game is reasonable
    # NOTE: NBA teams typically play every 2-3 days, but with All-Star breaks,
    # schedule gaps, and small sample sizes, the average can be higher
    if 'days_since_last_game' in features_df.columns:
        avg_days = features_df['days_since_last_game'].mean()
        # Allow wider range: back-to-backs (1 day) to extended breaks (10+ days)
        # With SAMPLE_SIZE=10, tracked players have gaps between games
        assert 1 < avg_days < 12, (
            f"Average days between games out of reasonable range: {avg_days:.2f}. "
            f"Expected 1-12 days (accounts for back-to-backs, normal schedule, breaks, sample size effects)."
        )

    logger.info("✓ Contextual features validation passed")


def build_contextual_features() -> pd.DataFrame:
    """
    Main function to build all contextual features
    """
    logger.info("Loading player game logs...")
    df = load_player_gamelogs()

    # Input validation
    validate_not_empty(df, 'build_contextual_features')

    logger.info(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    logger.info("\nCreating home/away features...")
    home_away = create_home_away_features(df)

    logger.info("Creating rest features...")
    rest = create_rest_features(df)

    logger.info("Creating season timing features...")
    timing = create_season_timing_features(df)

    logger.info("Creating recent performance indicators...")
    performance = create_recent_performance_indicators(df)

    logger.info("Creating minutes context features...")
    minutes = create_minutes_context(df)

    logger.info("\nMerging all contextual features...")
    features = home_away.copy()

    for feature_df in [rest, timing, performance, minutes]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    logger.info(f"Created {len(features.columns) - 3} contextual features")

    # Validate
    logger.info("\nValidating contextual features...")
    validate_contextual_features(features)

    # Sort by player and game_date for temporal validation
    features = features.sort_values(['player_id', 'game_date'])

    # Save
    output_path = FEATURE_DIR / "contextual_features.parquet"
    features.to_parquet(output_path, index=False)

    logger.info(f"\n✓ Saved contextual features to {output_path}")
    logger.info(f"  Shape: {features.shape}")
    logger.info(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_contextual_features()
