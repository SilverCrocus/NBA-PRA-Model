"""
Matchup Features for NBA PRA Model
Creates opponent-based features for player performance prediction

CRITICAL: Only uses data available BEFORE game date
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
from feature_engineering.data_loader import load_player_gamelogs, NBA_API_DIR

# Import shared utilities and configuration
from feature_engineering.utils import (create_feature_base,
                   validate_not_empty, validate_required_columns)
from feature_engineering.config import FEATURE_DIR, OPPONENT_DEFENSE_WINDOW


def calculate_opponent_defensive_stats(
    df: pd.DataFrame,
    window: int = 10
) -> pd.DataFrame:
    """
    Calculate opponent's defensive performance (points/PRA allowed)

    IMPORTANT: This function calculates TEAM-LEVEL defensive stats, not player-level.
    - Input: Player-level game logs (one row per player per game)
    - Process: Aggregates ALL players to TEAM totals per game
    - Output: Team defensive performance (e.g., total PRA allowed per game)
    - Expected values: ~169 PRA per game (NBA team average), range 100-250

    The resulting features represent how many points/PRA the opponent's
    ENTIRE TEAM allowed in recent games, not individual player stats.

    Example:
        If Lakers played Warriors and the Warriors lineup scored:
        - Player A: 25 PRA, Player B: 22 PRA, ..., Player J: 15 PRA
        - Total: 169 PRA scored against Lakers
        - Feature: opp_pra_allowed_avg = 169 (Lakers' team defense quality)

    Args:
        df: Player game logs DataFrame
        window: Rolling window for opponent stats (default 10 games)

    Returns:
        DataFrame with opponent defensive features (team-level aggregated)
    """
    # Input validation
    validate_not_empty(df, 'calculate_opponent_defensive_stats')
    validate_required_columns(
        df,
        ['opponent_team', 'game_id', 'game_date', 'points', 'pra'],
        'calculate_opponent_defensive_stats'
    )

    # Step 1: Aggregate to game level - total stats scored AGAINST each opponent per game
    # This gives us the opponent's defensive performance (how many points they allowed)
    game_level = df.groupby(['opponent_team', 'game_id', 'game_date']).agg({
        'points': 'sum',      # Total points scored against this opponent in this game
        'pra': 'sum',         # Total PRA scored against this opponent
        'rebounds': 'sum',
        'assists': 'sum'
    }).reset_index()

    # Step 2: Sort by opponent and date for rolling calculations
    game_level = game_level.sort_values(['opponent_team', 'game_date']).reset_index(drop=True)

    # Step 3: Calculate rolling averages of opponent's defensive performance
    # Use shift(1) to avoid leakage
    game_level['opp_points_allowed_avg'] = (
        game_level.groupby('opponent_team')['points']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    game_level['opp_pra_allowed_avg'] = (
        game_level.groupby('opponent_team')['pra']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    game_level['opp_rebounds_allowed_avg'] = (
        game_level.groupby('opponent_team')['rebounds']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    game_level['opp_assists_allowed_avg'] = (
        game_level.groupby('opponent_team')['assists']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Step 4: Merge back to player-game level
    features = df[['player_id', 'game_id', 'game_date', 'opponent_team']].copy()

    features = features.merge(
        game_level[['opponent_team', 'game_id', 'opp_points_allowed_avg',
                   'opp_pra_allowed_avg', 'opp_rebounds_allowed_avg', 'opp_assists_allowed_avg']],
        on=['opponent_team', 'game_id'],
        how='left'
    )

    return features.drop(columns=['opponent_team'])


def calculate_opponent_pace(
    df: pd.DataFrame,
    window: int = 10
) -> pd.DataFrame:
    """
    Calculate opponent's pace (estimated from total points)

    Args:
        df: Player game logs DataFrame
        window: Rolling window

    Returns:
        DataFrame with pace features
    """
    # Input validation
    validate_not_empty(df, 'calculate_opponent_pace')
    validate_required_columns(
        df,
        ['opponent_team', 'game_date', 'points'],
        'calculate_opponent_pace'
    )

    df = df.sort_values('game_date').reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date', 'opponent_team']].copy()

    # Estimate team pace from total points scored by team in games
    # Group by opponent and calculate rolling average of points (proxy for pace)
    team_scoring = df.groupby(['opponent_team', 'game_date']).agg({
        'points': 'sum'  # Total points by all players vs this opponent
    }).reset_index()

    team_scoring = team_scoring.sort_values(['opponent_team', 'game_date'])

    team_scoring['opp_pace_proxy'] = (
        team_scoring.groupby('opponent_team')['points']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Merge back
    features = features.merge(
        team_scoring[['opponent_team', 'game_date', 'opp_pace_proxy']],
        on=['opponent_team', 'game_date'],
        how='left'
    )

    return features.drop(columns=['opponent_team'])


def calculate_player_vs_opponent_history(
    df: pd.DataFrame,
    window: int = 5
) -> pd.DataFrame:
    """
    Calculate player's historical performance vs specific opponent

    Args:
        df: Player game logs DataFrame
        window: Number of recent matchups to average

    Returns:
        DataFrame with head-to-head features
    """
    # Input validation
    validate_not_empty(df, 'calculate_player_vs_opponent_history')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date', 'opponent_team', 'pra', 'points'],
        'calculate_player_vs_opponent_history'
    )

    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date', 'opponent_team']].copy()

    # Calculate player's average vs this specific opponent
    features['pra_vs_opponent_avg'] = (
        df.groupby(['player_id', 'opponent_team'])['pra']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    features['points_vs_opponent_avg'] = (
        df.groupby(['player_id', 'opponent_team'])['points']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    features['games_vs_opponent'] = (
        df.groupby(['player_id', 'opponent_team']).cumcount()
    )

    return features.drop(columns=['opponent_team'])


def calculate_team_strength_features(
    df: pd.DataFrame,
    window: int = 10
) -> pd.DataFrame:
    """
    Calculate rolling team strength indicators

    Args:
        df: Player game logs DataFrame (needs team info)
        window: Rolling window

    Returns:
        DataFrame with team strength features
    """
    # Extract player's team from matchup
    # Input validation
    validate_not_empty(df, 'calculate_team_strength_features')
    validate_required_columns(
        df,
        ['matchup', 'game_date', 'win_loss'],
        'calculate_team_strength_features'
    )

    df = df.copy()
    df['player_team'] = df['matchup'].str.extract(r'([A-Z]{3})')[0]

    df = df.sort_values('game_date').reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Calculate team win rate (rolling)
    df['is_win'] = (df['win_loss'] == 'W').astype(int)

    team_wins = df.groupby(['player_team', 'game_date']).agg({
        'is_win': 'first'  # One win/loss per game
    }).reset_index()

    team_wins = team_wins.sort_values(['player_team', 'game_date'])

    team_wins['team_win_pct_last10'] = (
        team_wins.groupby('player_team')['is_win']
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    # Merge back
    df_with_team = df[['player_id', 'game_id', 'game_date', 'player_team']].merge(
        team_wins[['player_team', 'game_date', 'team_win_pct_last10']],
        on=['player_team', 'game_date'],
        how='left'
    )

    features['team_win_pct_last10'] = df_with_team['team_win_pct_last10']

    return features


def validate_matchup_features(features_df: pd.DataFrame) -> None:
    """
    Validate matchup features for data quality

    Args:
        features_df: Computed features DataFrame

    Note:
        opp_pra_allowed_avg is TEAM-LEVEL aggregated PRA (not player-level).
        It represents the total PRA scored against an opponent team per game.
        Expected range: 120-220 for full NBA teams, smaller for test samples.
    """
    # Check for reasonable ranges
    # NOTE: opp_pra_allowed_avg is TEAM-LEVEL (sum of all players vs opponent per game)
    if 'opp_pra_allowed_avg' in features_df.columns:
        avg_opp_pra = features_df['opp_pra_allowed_avg'].mean()
        non_null_count = features_df['opp_pra_allowed_avg'].notna().sum()

        # For test datasets (SAMPLE_SIZE < 50), team aggregations are incomplete
        # Only validate if we have enough data for meaningful team-level stats
        if non_null_count > 100:
            # Full dataset: expect NBA team-level PRA (typical range 100-250)
            # NBA team averages: ~110 points + ~43 rebounds + ~23 assists = ~176 PRA
            assert 100 < avg_opp_pra < 250, (
                f"Team-level opponent PRA allowed avg out of expected range: {avg_opp_pra:.1f}. "
                f"Expected 100-250 for full team aggregation. "
                f"This represents TOTAL PRA scored against opponent per game (team-level, not player-level)."
            )
        else:
            # Test dataset: just check for reasonable non-negative values
            assert avg_opp_pra > 0, (
                f"Opponent PRA allowed avg must be positive: {avg_opp_pra:.1f}. "
                f"Note: Small sample size ({non_null_count} observations), team aggregations may be incomplete."
            )

    if 'days_rest' in features_df.columns:
        avg_rest = features_df['days_rest'].mean()
        assert 0 < avg_rest < 10, f"Average rest days out of range: {avg_rest}"

    logger.info("✓ Matchup features validation passed")


def build_matchup_features() -> pd.DataFrame:
    """
    Main function to build all matchup features
    """
    logger.info("Loading player game logs...")
    df = load_player_gamelogs()

    # Input validation
    validate_not_empty(df, 'build_matchup_features')

    logger.info(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    logger.info("\nCalculating opponent defensive stats...")
    opp_defense = calculate_opponent_defensive_stats(df, window=10)

    logger.info("Calculating opponent pace...")
    opp_pace = calculate_opponent_pace(df, window=10)

    logger.info("Calculating player vs opponent history...")
    vs_opponent = calculate_player_vs_opponent_history(df, window=5)

    logger.info("Calculating team strength...")
    team_strength = calculate_team_strength_features(df, window=10)

    logger.info("\nMerging all matchup features...")
    # Note: Rest features moved to contextual_features.py to avoid duplication
    features = opp_defense.copy()

    for feature_df in [opp_pace, vs_opponent, team_strength]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    logger.info(f"Created {len(features.columns) - 3} matchup features")

    # Validate
    logger.info("\nValidating matchup features...")
    validate_matchup_features(features)

    # Save
    output_path = FEATURE_DIR / "matchup_features.parquet"
    # Sort by player and game_date for temporal validation
    features = features.sort_values(['player_id', 'game_date'])
    features.to_parquet(output_path, index=False)

    logger.info(f"\n✓ Saved matchup features to {output_path}")
    logger.info(f"  Shape: {features.shape}")
    logger.info(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_matchup_features()
