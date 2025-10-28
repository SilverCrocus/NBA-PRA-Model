"""
Injury and DNP Tracking Features for NBA PRA Model
Tracks missed games, injury returns, and load management patterns

HIGH IMPACT: Expected 4-6% RMSE improvement
Rationale: Players returning from injury follow predictable ramp-up patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_player_gamelogs

# Output directory
FEATURE_DIR = Path(__file__).parent.parent / "data" / "feature_tables"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def detect_missed_games(df, expected_games_per_month=8):
    """
    Detect when players miss games (DNP, injury, rest)

    Args:
        df: Player game logs
        expected_games_per_month: Typical games per month in NBA (8-10)

    Returns:
        DataFrame with missed game indicators
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Calculate days since last game
    features['days_since_last_game'] = (
        df.groupby('player_id')['game_date']
        .diff()
        .dt.days
    )

    # Detect long gaps (likely missed games)
    # NBA teams play ~4 games per week, so >7 days suggests missed games
    features['likely_missed_games'] = (
        (features['days_since_last_game'] > 7) &
        (features['days_since_last_game'] <= 30)  # Within same season
    ).astype(int)

    # Estimate number of games missed
    # Assume 0.5 games per day when missing
    features['estimated_games_missed'] = (
        (features['days_since_last_game'] - 3) * 0.5
    ).clip(lower=0)

    # Filter to only actual missed games (not season start)
    features['estimated_games_missed'] = features['estimated_games_missed'].where(
        features['likely_missed_games'] == 1, 0
    )

    return features


def calculate_injury_return_features(df):
    """
    Track games since return from injury/absence

    Args:
        df: Player game logs

    Returns:
        DataFrame with injury return features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Detect multi-game absences (3+ games missed)
    days_gap = df.groupby('player_id')['game_date'].diff().dt.days
    is_return_from_absence = (days_gap > 10).astype(int)

    # Games since last multi-game absence
    # Reset counter when player returns from absence
    features['games_since_absence'] = 0

    for player_id in df['player_id'].unique():
        player_mask = df['player_id'] == player_id
        player_returns = is_return_from_absence[player_mask]

        # Calculate games since last return
        games_counter = []
        counter = 0

        for is_return in player_returns.values:
            if is_return:
                counter = 0  # Reset on return
            games_counter.append(counter)
            counter += 1

        features.loc[player_mask, 'games_since_absence'] = games_counter

    # Returning from injury indicator (first 5 games back)
    features['is_returning_from_injury'] = (
        (features['games_since_absence'] > 0) &
        (features['games_since_absence'] <= 5)
    ).astype(int)

    # Early return (first 2 games)
    features['is_early_return'] = (
        (features['games_since_absence'] > 0) &
        (features['games_since_absence'] <= 2)
    ).astype(int)

    return features


def calculate_dnp_patterns(df):
    """
    Calculate DNP (Did Not Play) patterns and load management indicators

    Args:
        df: Player game logs

    Returns:
        DataFrame with DNP pattern features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Detect DNP games (minutes = 0 or very low)
    if 'minutes' in df.columns:
        # Convert minutes to numeric
        df['minutes_float'] = df['minutes'].apply(convert_minutes_to_float)

        df['is_dnp'] = (df['minutes_float'] < 3).astype(int)
        features['is_dnp'] = df['is_dnp']

        # Rolling DNP rate (last 30 days)
        features['dnp_rate_last30d'] = (
            df.groupby('player_id')['is_dnp']
            .transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
        )

        # Load management indicator (strategic rest)
        # High DNP rate with normal minutes when playing
        features['likely_load_managed'] = (
            (features['dnp_rate_last30d'] > 0.15) &
            (df['minutes_float'] > 0)
        ).astype(int)

        # Consecutive games played (durability)
        features['consecutive_games_played'] = 0

        for player_id in df['player_id'].unique():
            player_mask = df['player_id'] == player_id
            player_dnps = features.loc[player_mask, 'is_dnp'].values

            # Count consecutive non-DNP games
            consecutive = []
            count = 0

            for dnp in player_dnps:
                if dnp == 0:  # Playing
                    count += 1
                else:  # DNP
                    count = 0
                consecutive.append(count)

            features.loc[player_mask, 'consecutive_games_played'] = consecutive

    return features


def calculate_minutes_recovery_features(df):
    """
    Calculate minutes recovery patterns after absences

    Args:
        df: Player game logs

    Returns:
        DataFrame with recovery pattern features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    if 'minutes' not in df.columns:
        return features

    df['minutes_float'] = df['minutes'].apply(convert_minutes_to_float)

    # Expected minutes (rolling 10-game average)
    df['expected_minutes'] = (
        df.groupby('player_id')['minutes_float']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    # Minutes deficit (playing fewer minutes than expected)
    features['minutes_deficit'] = (
        df['expected_minutes'] - df['minutes_float']
    ).clip(lower=0)

    # Ramping up indicator (gradually increasing minutes)
    df['minutes_trend_last3'] = (
        df.groupby('player_id')['minutes_float']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=2).apply(
            lambda y: (y.iloc[-1] - y.iloc[0]) / len(y) if len(y) > 1 else 0
        ))
    )

    features['is_ramping_up'] = (df['minutes_trend_last3'] > 2).astype(int)

    # Minutes restriction (consistently below expected)
    df['minutes_restriction_games'] = 0

    for player_id in df['player_id'].unique():
        player_mask = df['player_id'] == player_id

        actual = df.loc[player_mask, 'minutes_float'].values
        expected = df.loc[player_mask, 'expected_minutes'].values

        # Count consecutive games with significantly fewer minutes
        consecutive_restricted = []
        count = 0

        for act, exp in zip(actual, expected):
            if not np.isnan(exp) and act < (exp * 0.8):  # <80% of expected
                count += 1
            else:
                count = 0
            consecutive_restricted.append(count)

        features.loc[player_mask, 'minutes_restriction_games'] = consecutive_restricted

    features['is_minutes_restricted'] = (
        features['minutes_restriction_games'] >= 2
    ).astype(int)

    return features


def calculate_availability_score(df):
    """
    Calculate overall player availability/health score

    Args:
        df: Player game logs

    Returns:
        DataFrame with availability score
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Calculate games played in last 30 days
    features['games_played_last30d'] = (
        df.groupby('player_id')
        .rolling(window='30D', on='game_date')
        ['game_id']
        .count()
        .reset_index(drop=True)
    )

    # Availability score (0-100)
    # 100 = fully available, 0 = frequently missing games
    # Expect ~12 games per 30 days for healthy player
    features['availability_score'] = (
        (features['games_played_last30d'] / 12) * 100
    ).clip(upper=100)

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


def validate_injury_features(features_df):
    """
    Validate injury features

    Args:
        features_df: Computed features
    """
    # Check availability score is 0-100
    if 'availability_score' in features_df.columns:
        assert features_df['availability_score'].min() >= 0
        assert features_df['availability_score'].max() <= 100

    # Check games_since_absence is non-negative
    if 'games_since_absence' in features_df.columns:
        assert features_df['games_since_absence'].min() >= 0

    print("✓ Injury features validation passed")


def build_injury_features():
    """
    Main function to build all injury/DNP tracking features
    """
    print("Loading player game logs...")
    df = load_player_gamelogs()

    print(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    print("\nDetecting missed games...")
    missed = detect_missed_games(df)

    print("Calculating injury return features...")
    injury_return = calculate_injury_return_features(df)

    print("Calculating DNP patterns...")
    dnp = calculate_dnp_patterns(df)

    print("Calculating minutes recovery features...")
    minutes_recovery = calculate_minutes_recovery_features(df)

    print("Calculating availability score...")
    availability = calculate_availability_score(df)

    print("\nMerging all injury features...")
    features = missed.copy()

    for feature_df in [injury_return, dnp, minutes_recovery, availability]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    print(f"Created {len(features.columns) - 3} injury/DNP features")

    # Validate
    print("\nValidating injury features...")
    validate_injury_features(features)

    # Save
    output_path = FEATURE_DIR / "injury_features.parquet"
    features.to_parquet(output_path, index=False)

    print(f"\n✓ Saved injury features to {output_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")

    # Print summary statistics
    if 'is_returning_from_injury' in features.columns:
        return_pct = features['is_returning_from_injury'].mean() * 100
        print(f"\nSummary: {return_pct:.1f}% of games are injury returns")

    if 'availability_score' in features.columns:
        avg_avail = features['availability_score'].mean()
        print(f"Average availability score: {avg_avail:.1f}/100")

    return features


if __name__ == "__main__":
    build_injury_features()
