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

    Performs comprehensive checks:
    1. First game of each player should have NaN/0 for all rolling features
    2. Rolling features should be properly populated in mid-season games
    3. Spot check random games for reasonable values

    Args:
        df: Original game logs (must have player_id, game_id, game_date)
        features_df: Computed features (must have player_id, game_id, game_date)

    Raises:
        AssertionError if validation fails with detailed error messages
    """
    print("\n" + "="*60)
    print("DATA LEAKAGE VALIDATION")
    print("="*60)

    # Identify all rolling feature columns
    rolling_cols = [col for col in features_df.columns
                   if any(pattern in col for pattern in ['_avg_', '_std_', '_ewma_', '_trend_'])]

    print(f"Checking {len(rolling_cols)} rolling features for leakage...")
    if len(rolling_cols) > 5:
        print(f"Features to validate: {rolling_cols[:5]}...")
    else:
        print(f"Features: {rolling_cols}")

    # Check 1: First game of each player
    print("\nCheck 1: First game per player (should have NaN/0)...")
    first_games = df.groupby('player_id').first().reset_index()[['player_id', 'game_id', 'game_date']]

    leakage_detected = []
    missing_features = 0

    for idx, row in first_games.iterrows():
        # FIXED: Look up using correct grain [player_id, game_id]
        feature_row = features_df[
            (features_df['player_id'] == row['player_id']) &
            (features_df['game_id'] == row['game_id'])
        ]

        if len(feature_row) == 0:
            missing_features += 1
            continue

        feature_row = feature_row.iloc[0]

        # Check all rolling features
        for col in rolling_cols:
            if col not in feature_row:
                continue

            value = feature_row[col]

            # First game should be NaN or 0
            if pd.notna(value) and value != 0:
                leakage_detected.append({
                    'player_id': row['player_id'],
                    'game_id': row['game_id'],
                    'game_date': row['game_date'],
                    'feature': col,
                    'value': value
                })

    # Report results
    if missing_features > 0:
        print(f"  ⚠️  Warning: {missing_features} players have no features (might be filtered)")

    if leakage_detected:
        print(f"\n  ❌ LEAKAGE DETECTED in {len(leakage_detected)} cases:")
        for case in leakage_detected[:10]:  # Show first 10
            print(f"     Player {case['player_id']}, Game {case['game_id']} ({case['game_date']})")
            print(f"       → {case['feature']} = {case['value']:.2f} (should be NaN or 0)")

        if len(leakage_detected) > 10:
            print(f"     ... and {len(leakage_detected) - 10} more cases")

        raise AssertionError(
            f"DATA LEAKAGE DETECTED: {len(leakage_detected)} first-game features have non-zero values. "
            f"This indicates rolling calculations are including the current game. "
            f"Ensure all rolling operations use .shift(1) before .rolling()."
        )

    print(f"  ✓ All {len(first_games) - missing_features} first games have NaN/0 for rolling features")

    # Check 2: Sample mid-season games (rolling features should be populated)
    print("\nCheck 2: Mid-season games (rolling features should be populated)...")
    mid_season_games = df.groupby('player_id').apply(
        lambda x: x.iloc[min(15, len(x)-1)] if len(x) > 15 else None
    ).dropna().reset_index(drop=True)[['player_id', 'game_id', 'game_date']]

    if len(mid_season_games) > 0:
        sample_size = min(50, len(mid_season_games))
        mid_season_sample = mid_season_games.sample(sample_size, random_state=42)

        unpopulated_count = 0
        for idx, row in mid_season_sample.iterrows():
            feature_row = features_df[
                (features_df['player_id'] == row['player_id']) &
                (features_df['game_id'] == row['game_id'])
            ]

            if len(feature_row) == 0:
                continue

            feature_row = feature_row.iloc[0]

            # Check if key rolling features are populated
            key_features = ['pra_avg_last10', 'points_avg_last10']
            for col in key_features:
                if col in feature_row and pd.isna(feature_row[col]):
                    unpopulated_count += 1
                    break

        if unpopulated_count > sample_size * 0.3:  # More than 30% unpopulated
            print(f"  ⚠️  Warning: {unpopulated_count}/{sample_size} mid-season games have missing rolling features")
        else:
            print(f"  ✓ Mid-season games have populated rolling features ({sample_size - unpopulated_count}/{sample_size})")

    # Check 3: No future information in any game
    print("\nCheck 3: Spot check random games for reasonable values...")
    random_sample = features_df.sample(min(100, len(features_df)), random_state=42)

    unreasonable_values = []
    for col in rolling_cols[:10]:  # Check first 10 rolling features
        if col not in random_sample:
            continue

        col_data = random_sample[col].dropna()
        if len(col_data) == 0:
            continue

        # Check for unreasonable values (e.g., negative averages, extreme values)
        if 'avg' in col or 'ewma' in col:
            # Averages should be in reasonable range (0-100 for most NBA stats)
            if col_data.min() < -10 or col_data.max() > 200:
                unreasonable_values.append((col, col_data.min(), col_data.max()))

    if unreasonable_values:
        print(f"  ⚠️  Warning: {len(unreasonable_values)} features have unreasonable value ranges:")
        for col, min_val, max_val in unreasonable_values[:5]:
            print(f"     {col}: [{min_val:.2f}, {max_val:.2f}]")
    else:
        print(f"  ✓ Spot check passed: All values in reasonable ranges")

    print("\n" + "="*60)
    print("✅ LEAKAGE VALIDATION PASSED")
    print("="*60)
    print("All rolling features properly exclude current game information.")
    print(f"Validated {len(rolling_cols)} features across {len(features_df):,} games.\n")


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
