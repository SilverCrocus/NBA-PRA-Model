"""
Rolling Features for NBA PRA Model
Calculates rolling averages and trends for player performance

CRITICAL: All features use .shift(1) to avoid data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_player_gamelogs, NBA_API_DIR

# Output directory for feature tables
FEATURE_DIR = Path(__file__).parent.parent / "data" / "feature_tables"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rolling_features(df, windows=[5, 10, 20]):
    """
    Calculate rolling average features for player performance

    Args:
        df: Player game logs DataFrame (must be sorted by player_id, game_date)
        windows: List of rolling window sizes

    Returns:
        DataFrame with rolling features
    """
    # Ensure data is sorted
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Key stats to create rolling features for
    stats_to_roll = ['pra', 'points', 'rebounds', 'assists', 'minutes']

    for stat in stats_to_roll:
        if stat not in df.columns:
            continue

        # Convert minutes to numeric if it's a string (e.g., "25:30")
        if stat == 'minutes':
            df[stat] = df[stat].apply(convert_minutes_to_float)

        for window in windows:
            # CRITICAL: shift(1) excludes current game to prevent data leakage
            features[f'{stat}_avg_last{window}'] = (
                df.groupby('player_id')[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            )

            # Rolling standard deviation (volatility)
            features[f'{stat}_std_last{window}'] = (
                df.groupby('player_id')[stat]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
            )

    return features


def calculate_ewma_features(df, halflives=[3, 7, 14]):
    """
    Calculate exponentially weighted moving averages
    Recent games weighted more heavily than older games

    Args:
        df: Player game logs DataFrame
        halflives: List of halflife values for EWMA

    Returns:
        DataFrame with EWMA features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    stats_to_ewma = ['pra', 'points', 'rebounds', 'assists']

    for stat in stats_to_ewma:
        if stat not in df.columns:
            continue

        for halflife in halflives:
            # CRITICAL: shift(1) to avoid leakage
            features[f'{stat}_ewma_hl{halflife}'] = (
                df.groupby('player_id')[stat]
                .transform(lambda x: x.shift(1).ewm(halflife=halflife, min_periods=1).mean())
            )

    return features


def calculate_trend_features(df, window=10):
    """
    Calculate linear trend over recent games
    Positive = improving, Negative = declining

    Args:
        df: Player game logs DataFrame
        window: Number of games to calculate trend over

    Returns:
        DataFrame with trend features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    stats_for_trend = ['pra', 'points', 'minutes']

    for stat in stats_for_trend:
        if stat not in df.columns:
            continue

        if stat == 'minutes':
            df[stat] = df[stat].apply(convert_minutes_to_float)

        # Calculate slope of linear regression over window
        features[f'{stat}_trend_last{window}'] = (
            df.groupby('player_id')[stat]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).apply(calculate_slope))
        )

    return features


def calculate_slope(series):
    """
    Calculate slope of linear regression for a series

    Args:
        series: Pandas Series

    Returns:
        Slope coefficient
    """
    if len(series) < 3:
        return 0

    x = np.arange(len(series))
    y = series.values

    # Handle NaN values
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 3:
        return 0

    x = x[valid_mask]
    y = y[valid_mask]

    # Linear regression: y = mx + b
    slope = np.polyfit(x, y, 1)[0]
    return slope


def calculate_home_away_splits(df, window=10):
    """
    Calculate rolling averages split by home/away

    Args:
        df: Player game logs DataFrame
        window: Rolling window size

    Returns:
        DataFrame with home/away split features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Create separate rolling averages for home and away games
    for stat in ['pra', 'points']:
        if stat not in df.columns:
            continue

        # Calculate home/away averages using a different approach
        # Create masked versions for home and away
        df_temp = df.copy()
        df_temp[f'{stat}_home'] = df_temp[stat].where(df_temp['is_home'] == 1)
        df_temp[f'{stat}_away'] = df_temp[stat].where(df_temp['is_home'] == 0)

        # Home games rolling average
        features[f'{stat}_avg_home_last{window}'] = (
            df_temp.groupby('player_id')[f'{stat}_home']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

        # Away games rolling average
        features[f'{stat}_avg_away_last{window}'] = (
            df_temp.groupby('player_id')[f'{stat}_away']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    return features


def calculate_games_played_features(df):
    """
    Calculate cumulative games played this season

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with games played features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Cumulative games this season
    features['games_played_season'] = (
        df.groupby(['player_id', 'season']).cumcount()
    )

    # Note: games_last_N_days is calculated in contextual_features.py
    # Removed from here to avoid duplication

    return features


def convert_minutes_to_float(minutes):
    """
    Convert minutes from string format (MM:SS) to float

    Args:
        minutes: Minutes as string (e.g., "25:30") or float

    Returns:
        Minutes as float
    """
    if pd.isna(minutes):
        return 0.0

    if isinstance(minutes, (int, float)):
        return float(minutes)

    if isinstance(minutes, str):
        if ':' in minutes:
            try:
                parts = minutes.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            except:
                return 0.0
        else:
            try:
                return float(minutes)
            except:
                return 0.0

    return 0.0


def validate_no_leakage(df, features_df):
    """
    Validate that features don't contain future information

    Args:
        df: Original game logs
        features_df: Computed features

    Raises:
        AssertionError if validation fails
    """
    # Check 1: For first game of each player, rolling features should be NaN or minimal
    first_games = df.groupby('player_id').head(1)

    for game_id in first_games['game_id'].values:
        feature_row = features_df[features_df['game_id'] == game_id].iloc[0]

        # First game should not have full 10-game average
        if 'pra_avg_last10' in feature_row:
            assert pd.isna(feature_row['pra_avg_last10']) or feature_row['pra_avg_last10'] == 0, \
                f"Leakage detected: First game has rolling avg"

    print("✓ Leakage validation passed: No future information in features")


def build_rolling_features():
    """
    Main function to build all rolling features
    Loads data, computes features, validates, and saves
    """
    print("Loading player game logs...")
    df = load_player_gamelogs()

    print(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    print("\nCalculating rolling features...")
    rolling = calculate_rolling_features(df, windows=[5, 10, 20])

    print("Calculating EWMA features...")
    ewma = calculate_ewma_features(df, halflives=[3, 7, 14])

    print("Calculating trend features...")
    trends = calculate_trend_features(df, window=10)

    print("Calculating home/away splits...")
    splits = calculate_home_away_splits(df, window=10)

    print("Calculating games played features...")
    games_played = calculate_games_played_features(df)

    print("\nMerging all rolling features...")
    # Merge all feature sets
    features = rolling.copy()

    for feature_df in [ewma, trends, splits, games_played]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    print(f"Created {len(features.columns) - 3} rolling features")

    # Validate no data leakage
    print("\nValidating data leakage prevention...")
    validate_no_leakage(df, features)

    # Save to parquet
    output_path = FEATURE_DIR / "rolling_features.parquet"
    features.to_parquet(output_path, index=False)

    print(f"\n✓ Saved rolling features to {output_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_rolling_features()
