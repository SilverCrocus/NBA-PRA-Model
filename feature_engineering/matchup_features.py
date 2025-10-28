"""
Matchup Features for NBA PRA Model
Creates opponent-based features for player performance prediction

CRITICAL: Only uses data available BEFORE game date
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_player_gamelogs, NBA_API_DIR

# Output directory
FEATURE_DIR = Path(__file__).parent.parent / "data" / "feature_tables"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def calculate_opponent_defensive_stats(df, window=10):
    """
    Calculate opponent's defensive performance (points/PRA allowed)

    FIXED: Now correctly calculates opponent's TEAM defensive stats
    by aggregating all points scored against them per game

    Args:
        df: Player game logs DataFrame
        window: Rolling window for opponent stats

    Returns:
        DataFrame with opponent defensive features
    """
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


def calculate_opponent_pace(df, window=10):
    """
    Calculate opponent's pace (estimated from total points)

    Args:
        df: Player game logs DataFrame
        window: Rolling window

    Returns:
        DataFrame with pace features
    """
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


def calculate_player_vs_opponent_history(df, window=5):
    """
    Calculate player's historical performance vs specific opponent

    Args:
        df: Player game logs DataFrame
        window: Number of recent matchups to average

    Returns:
        DataFrame with head-to-head features
    """
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


def calculate_team_strength_features(df, window=10):
    """
    Calculate rolling team strength indicators

    Args:
        df: Player game logs DataFrame (needs team info)
        window: Rolling window

    Returns:
        DataFrame with team strength features
    """
    # Extract player's team from matchup
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


def validate_matchup_features(features_df):
    """
    Validate matchup features for data quality

    Args:
        features_df: Computed features DataFrame
    """
    # Check for reasonable ranges (relaxed for small test samples)
    if 'opp_pra_allowed_avg' in features_df.columns:
        avg_opp_pra = features_df['opp_pra_allowed_avg'].mean()
        assert 5 < avg_opp_pra < 60, f"Opponent PRA allowed avg out of range: {avg_opp_pra}"

    if 'days_rest' in features_df.columns:
        avg_rest = features_df['days_rest'].mean()
        assert 0 < avg_rest < 10, f"Average rest days out of range: {avg_rest}"

    print("✓ Matchup features validation passed")


def build_matchup_features():
    """
    Main function to build all matchup features
    """
    print("Loading player game logs...")
    df = load_player_gamelogs()

    print(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    print("\nCalculating opponent defensive stats...")
    opp_defense = calculate_opponent_defensive_stats(df, window=10)

    print("Calculating opponent pace...")
    opp_pace = calculate_opponent_pace(df, window=10)

    print("Calculating player vs opponent history...")
    vs_opponent = calculate_player_vs_opponent_history(df, window=5)

    print("Calculating team strength...")
    team_strength = calculate_team_strength_features(df, window=10)

    print("\nMerging all matchup features...")
    # Note: Rest features moved to contextual_features.py to avoid duplication
    features = opp_defense.copy()

    for feature_df in [opp_pace, vs_opponent, team_strength]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    print(f"Created {len(features.columns) - 3} matchup features")

    # Validate
    print("\nValidating matchup features...")
    validate_matchup_features(features)

    # Save
    output_path = FEATURE_DIR / "matchup_features.parquet"
    features.to_parquet(output_path, index=False)

    print(f"\n✓ Saved matchup features to {output_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_matchup_features()
