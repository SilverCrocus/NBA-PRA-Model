"""
Contextual Features for NBA PRA Model
Creates game context features (home/away, time factors, etc.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_player_gamelogs

# Output directory
FEATURE_DIR = Path(__file__).parent.parent / "data" / "feature_tables"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def create_home_away_features(df):
    """
    Create home/away related features

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with home/away features
    """
    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Home game indicator (already in data)
    features['is_home'] = df['is_home']

    return features


def create_rest_features(df):
    """
    Create rest and fatigue related features

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with rest features
    """
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


def create_season_timing_features(df):
    """
    Create features related to timing in the season

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with season timing features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Game number in season for this player
    features['game_number_season'] = (
        df.groupby(['player_id', 'season']).cumcount() + 1
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


def create_recent_performance_indicators(df):
    """
    Create binary indicators for recent performance patterns

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with performance indicators
    """
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


def create_minutes_context(df):
    """
    Create features related to playing time context

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with minutes context features
    """
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


def convert_minutes_to_float(minutes):
    """
    Convert minutes from string format to float

    Args:
        minutes: Minutes as string or float

    Returns:
        Float minutes
    """
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


def validate_contextual_features(features_df):
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
    if 'days_since_last_game' in features_df.columns:
        avg_days = features_df['days_since_last_game'].mean()
        assert 1 < avg_days < 5, f"Average days between games unusual: {avg_days}"

    print("✓ Contextual features validation passed")


def build_contextual_features():
    """
    Main function to build all contextual features
    """
    print("Loading player game logs...")
    df = load_player_gamelogs()

    print(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    print("\nCreating home/away features...")
    home_away = create_home_away_features(df)

    print("Creating rest features...")
    rest = create_rest_features(df)

    print("Creating season timing features...")
    timing = create_season_timing_features(df)

    print("Creating recent performance indicators...")
    performance = create_recent_performance_indicators(df)

    print("Creating minutes context features...")
    minutes = create_minutes_context(df)

    print("\nMerging all contextual features...")
    features = home_away.copy()

    for feature_df in [rest, timing, performance, minutes]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    print(f"Created {len(features.columns) - 3} contextual features")

    # Validate
    print("\nValidating contextual features...")
    validate_contextual_features(features)

    # Save
    output_path = FEATURE_DIR / "contextual_features.parquet"
    features.to_parquet(output_path, index=False)

    print(f"\n✓ Saved contextual features to {output_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_contextual_features()
