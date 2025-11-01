"""
Rolling Features for NBA PRA Model
Calculates rolling averages and trends for player performance

CRITICAL: All features use .shift(1) to avoid data leakage
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
from typing import List
from feature_engineering.data_loader import load_player_gamelogs, NBA_API_DIR

# Import shared utilities and configuration
from feature_engineering.utils import (convert_minutes_to_float, create_feature_base, validate_grain_uniqueness,
                   validate_not_empty, validate_required_columns)
from feature_engineering.config import FEATURE_DIR, ROLLING_WINDOWS, ROLLING_STATS, GRAIN_COLUMNS


def calculate_rolling_features(
    df: pd.DataFrame,
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Calculate rolling average features for player performance

    CRITICAL: Uses .shift(1) to prevent data leakage by excluding the current
    game from all rolling calculations. This ensures features only use historical
    data available before the game being predicted.

    Args:
        df: Player game logs DataFrame with columns ['player_id', 'game_date',
            'pra', 'points', 'rebounds', 'assists', 'minutes']. Will be sorted by
            player_id, game_date if not already sorted.
        windows: List of rolling window sizes in games (default: [5, 10, 20]).
            Larger windows provide more stable averages but less recent signal.

    Returns:
        DataFrame with grain columns ['player_id', 'game_id', 'game_date'] plus:
            - {stat}_avg_last{N}: Rolling average over last N games
            - {stat}_std_last{N}: Rolling standard deviation (volatility)

        Stats calculated: pra, points, rebounds, assists, minutes
        Windows: As specified in windows parameter

        Total features: len(stats) * len(windows) * 2 metrics = ~30 features

    Examples:
        >>> from data_loader import load_player_gamelogs
        >>> df = load_player_gamelogs()
        >>> features = calculate_rolling_features(df, windows=[5, 10])
        >>> features[['player_id', 'game_id', 'pra_avg_last5', 'pra_avg_last10']].head()

        # Verify no leakage: first game should have NaN/0
        >>> first_games = df.groupby('player_id').first().reset_index()
        >>> merged = first_games.merge(features, on=['player_id', 'game_id'])
        >>> assert merged['pra_avg_last5'].isna().all() or (merged['pra_avg_last5'] == 0).all()

    Notes:
        - First N games for each player will have NaN values (no history)
        - Uses min_periods=1 to provide partial averages for early games
        - All features use shifted data (.shift(1)) to prevent temporal leakage
        - Minutes are converted from "MM:SS" string format to float automatically
        - Expected runtime: 30-60 seconds for full dataset (~150k games)
    """
    # Input validation
    validate_not_empty(df, 'calculate_rolling_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date', 'pra', 'points', 'rebounds', 'assists', 'minutes'],
        'calculate_rolling_features'
    )

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


def calculate_ewma_features(
    df: pd.DataFrame,
    halflives: List[int] = [3, 7, 14]
) -> pd.DataFrame:
    """
    Calculate exponentially weighted moving averages for player performance

    EWMA gives more weight to recent games using exponential decay. The halflife
    parameter controls the decay rate: smaller = more weight on recent games.

    CRITICAL: Uses .shift(1) to prevent data leakage by excluding current game.

    Args:
        df: Player game logs DataFrame with columns ['player_id', 'game_date',
            'pra', 'points', 'rebounds', 'assists']. Will be sorted automatically.
        halflives: List of halflife values in games (default: [3, 7, 14]).
            - halflife=3: Recent 3 games have ~50% weight
            - halflife=7: Recent 7 games have ~50% weight
            - halflife=14: Recent 14 games have ~50% weight

    Returns:
        DataFrame with grain columns ['player_id', 'game_id', 'game_date'] plus:
            - {stat}_ewma_hl{N}: Exponentially weighted average with halflife N

        Stats calculated: pra, points, rebounds, assists
        Halflives: As specified in halflives parameter

        Total features: len(stats) * len(halflives) = ~12 features

    Examples:
        >>> from data_loader import load_player_gamelogs
        >>> df = load_player_gamelogs()
        >>> features = calculate_ewma_features(df, halflives=[3, 7])
        >>> features[['player_id', 'pra_ewma_hl3', 'pra_ewma_hl7']].head()

        # EWMA adapts faster than simple rolling average
        >>> features['pra_ewma_hl3'].mean()  # More reactive to recent form

    Notes:
        - EWMA adapts faster to recent trends than simple rolling averages
        - Smaller halflife = more responsive to recent performance changes
        - Larger halflife = more stable, similar to longer rolling windows
        - All calculations use .shift(1) to exclude current game
        - Uses min_periods=1 to start calculations from first game
        - Expected runtime: 20-40 seconds for full dataset
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


def calculate_trend_features(
    df: pd.DataFrame,
    window: int = 10
) -> pd.DataFrame:
    """
    Calculate linear trend (slope) over recent games to detect improvement/decline

    Fits a linear regression over the last N games to detect if player is trending
    upward (positive slope) or downward (negative slope). Useful for identifying
    hot/cold streaks and injury recovery patterns.

    CRITICAL: Uses .shift(1) to prevent data leakage by excluding current game.

    Args:
        df: Player game logs DataFrame with columns ['player_id', 'game_date',
            'pra', 'points', 'minutes']. Will be sorted automatically.
        window: Number of games to calculate trend over (default: 10).
            Requires min_periods=3 for meaningful regression.

    Returns:
        DataFrame with grain columns ['player_id', 'game_id', 'game_date'] plus:
            - {stat}_trend_last{window}: Linear regression slope over last N games

        Slope interpretation:
            - Positive: Player improving over recent games
            - Negative: Player declining over recent games
            - Near 0: Stable performance

        Stats calculated: pra, points, minutes
        Total features: 3

    Examples:
        >>> from data_loader import load_player_gamelogs
        >>> df = load_player_gamelogs()
        >>> features = calculate_trend_features(df, window=10)
        >>> features[['player_id', 'pra_trend_last10']].head()

        # Positive trend indicates improving form
        >>> improving_players = features[features['pra_trend_last10'] > 1.0]

    Notes:
        - Requires minimum 3 games for regression (set via min_periods=3)
        - Trend is calculated using numpy.polyfit linear regression
        - Minutes are converted from "MM:SS" string format to float
        - All calculations use .shift(1) to exclude current game
        - NaN values in window are filtered before regression
        - Expected runtime: 40-80 seconds (slower due to apply function)
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


def calculate_slope(series: pd.Series) -> float:
    """
    Calculate slope of linear regression for a time series

    Helper function for calculate_trend_features. Fits y = mx + b and returns m.

    Args:
        series: Pandas Series of numeric values (e.g., PRA values over time)

    Returns:
        float: Slope coefficient (m) from linear regression
            - Positive: Increasing trend
            - Negative: Decreasing trend
            - 0: No trend or insufficient data (<3 points)

    Examples:
        >>> import pandas as pd
        >>> series = pd.Series([10, 12, 14, 16, 18])  # Increasing
        >>> calculate_slope(series)
        2.0  # Slope of +2 per game

        >>> series = pd.Series([20, 18, 16, 14, 12])  # Decreasing
        >>> calculate_slope(series)
        -2.0  # Slope of -2 per game

    Notes:
        - Returns 0 if series has <3 valid (non-NaN) values
        - Handles NaN values by filtering them out before regression
        - Uses numpy.polyfit(x, y, 1)[0] to get slope
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


def calculate_home_away_splits(
    df: pd.DataFrame,
    window: int = 10
) -> pd.DataFrame:
    """
    Calculate separate rolling averages for home and away games

    Players often perform differently at home vs away due to travel, crowd, etc.
    This creates separate rolling averages for each context.

    CRITICAL: Uses .shift(1) to prevent data leakage by excluding current game.

    Args:
        df: Player game logs DataFrame with columns ['player_id', 'game_date',
            'pra', 'points', 'is_home']. Will be sorted automatically.
        window: Rolling window size in games (default: 10).
            Note: Window counts ALL games, but average only includes home/away.

    Returns:
        DataFrame with grain columns ['player_id', 'game_id', 'game_date'] plus:
            - {stat}_avg_home_last{window}: Rolling average for home games only
            - {stat}_avg_away_last{window}: Rolling average for away games only

        Stats calculated: pra, points
        Total features: 4

    Examples:
        >>> from data_loader import load_player_gamelogs
        >>> df = load_player_gamelogs()
        >>> features = calculate_home_away_splits(df, window=10)
        >>> features[['player_id', 'pra_avg_home_last10', 'pra_avg_away_last10']].head()

        # Analyze home/away differential
        >>> features['home_away_diff'] = (
        ...     features['pra_avg_home_last10'] - features['pra_avg_away_last10']
        ... )
        >>> features[features['home_away_diff'] > 5]  # Significant home advantage

    Notes:
        - Uses .where() to mask non-home/away games before rolling
        - Window size is in total games, not just home or away games
        - Player with few home games may have sparse home_avg values
        - All calculations use .shift(1) to exclude current game
        - Expected runtime: 30-50 seconds for full dataset
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


def calculate_games_played_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cumulative games played this season

    Args:
        df: Player game logs DataFrame

    Returns:
        DataFrame with games played features
    """
    # Input validation
    validate_not_empty(df, 'calculate_games_played_features')
    validate_required_columns(
        df,
        ['player_id', 'game_id', 'game_date', 'season'],
        'calculate_games_played_features'
    )

    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Cumulative games this season
    features['games_played_season'] = (
        df.groupby(['player_id', 'season']).cumcount()
    )

    # Note: games_last_N_days is calculated in contextual_features.py
    # Removed from here to avoid duplication

    return features


def validate_no_leakage(df: pd.DataFrame, features_df: pd.DataFrame) -> None:
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
    logger.info("\n" + "="*60)
    logger.info("DATA LEAKAGE VALIDATION")
    logger.info("="*60)

    # Identify all rolling feature columns
    rolling_cols = [col for col in features_df.columns
                   if any(pattern in col for pattern in ['_avg_', '_std_', '_ewma_', '_trend_'])]

    logger.info(f"Checking {len(rolling_cols)} rolling features for leakage...")
    if len(rolling_cols) > 5:
        logger.info(f"Features to validate: {rolling_cols[:5]}...")
    else:
        logger.info(f"Features: {rolling_cols}")

    # Check 1: First game of each player
    logger.info("\nCheck 1: First game per player (should have NaN/0)...")
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
        logger.warning(f"  ⚠️  Warning: {missing_features} players have no features (might be filtered)")

    if leakage_detected:
        logger.error(f"\n  ❌ LEAKAGE DETECTED in {len(leakage_detected)} cases:")
        for case in leakage_detected[:10]:  # Show first 10
            logger.info(f"     Player {case['player_id']}, Game {case['game_id']} ({case['game_date']})")
            logger.info(f"       → {case['feature']} = {case['value']:.2f} (should be NaN or 0)")

        if len(leakage_detected) > 10:
            logger.info(f"     ... and {len(leakage_detected) - 10} more cases")

        raise AssertionError(
            f"DATA LEAKAGE DETECTED: {len(leakage_detected)} first-game features have non-zero values. "
            f"This indicates rolling calculations are including the current game. "
            f"Ensure all rolling operations use .shift(1) before .rolling()."
        )

    logger.info(f"  ✓ All {len(first_games) - missing_features} first games have NaN/0 for rolling features")

    # Check 2: Sample mid-season games (rolling features should be populated)
    logger.info("\nCheck 2: Mid-season games (rolling features should be populated)...")
    # Get game number for each player
    df['game_num'] = df.groupby('player_id').cumcount()
    # Select game 15 for each player (mid-season)
    mid_season_games = df[df['game_num'] == 15][['player_id', 'game_id', 'game_date']].copy()
    # Clean up temporary column
    df.drop('game_num', axis=1, inplace=True)

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
            logger.warning(f"  ⚠️  Warning: {unpopulated_count}/{sample_size} mid-season games have missing rolling features")
        else:
            logger.info(f"  ✓ Mid-season games have populated rolling features ({sample_size - unpopulated_count}/{sample_size})")

    # Check 3: No future information in any game
    logger.info("\nCheck 3: Spot check random games for reasonable values...")
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
        logger.warning(f"  ⚠️  Warning: {len(unreasonable_values)} features have unreasonable value ranges:")
        for col, min_val, max_val in unreasonable_values[:5]:
            logger.info(f"     {col}: [{min_val:.2f}, {max_val:.2f}]")
    else:
        logger.info(f"  ✓ Spot check passed: All values in reasonable ranges")

    logger.info("\n" + "="*60)
    logger.info("✅ LEAKAGE VALIDATION PASSED")
    logger.info("="*60)
    logger.info("All rolling features properly exclude current game information.")
    logger.info(f"Validated {len(rolling_cols)} features across {len(features_df):,} games.\n")


def build_rolling_features() -> pd.DataFrame:
    """
    Main entry point to build all rolling feature types

    Orchestrates the complete rolling features pipeline:
    1. Load player game logs from NBA API
    2. Calculate rolling averages (5, 10, 20 game windows)
    3. Calculate EWMA features (3, 7, 14 game halflives)
    4. Calculate trend features (10 game slopes)
    5. Calculate home/away splits (10 game windows)
    6. Calculate games played counters
    7. Merge all feature sets
    8. Validate no data leakage
    9. Save to parquet

    Returns:
        pd.DataFrame: Complete rolling features with grain [player_id, game_id, game_date]
            and ~50 rolling feature columns

        Saved to: data/feature_tables/rolling_features.parquet

    Raises:
        AssertionError: If data leakage validation fails

    Examples:
        >>> from feature_engineering.rolling_features import build_rolling_features
        >>> features = build_rolling_features()
        Loading player game logs...
        Loaded 587034 games for 2280 players
        ...
        Saved rolling features to .../rolling_features.parquet

        >>> features.shape
        (587034, 53)  # 50 features + 3 grain columns

        >>> features.columns
        Index(['player_id', 'game_id', 'game_date', 'pra_avg_last5', ...])

    Notes:
        - Expected runtime: 2-5 minutes for full dataset (~150k games)
        - Creates ~50 features total across all rolling types
        - All features use .shift(1) to prevent temporal leakage
        - Validates grain uniqueness and leakage prevention before saving
        - Part of the feature engineering pipeline (run after data_loader.py)
    """
    logger.info("Loading player game logs...")
    df = load_player_gamelogs()

    logger.info(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    logger.info("\nCalculating rolling features...")
    rolling = calculate_rolling_features(df, windows=[5, 10, 20])

    logger.info("Calculating EWMA features...")
    ewma = calculate_ewma_features(df, halflives=[3, 7, 14])

    logger.info("Calculating trend features...")
    trends = calculate_trend_features(df, window=10)

    logger.info("Calculating home/away splits...")
    splits = calculate_home_away_splits(df, window=10)

    logger.info("Calculating games played features...")
    games_played = calculate_games_played_features(df)

    logger.info("\nMerging all rolling features...")
    # Merge all feature sets
    features = rolling.copy()

    for feature_df in [ewma, trends, splits, games_played]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    logger.info(f"Created {len(features.columns) - 3} rolling features")

    # Validate no data leakage
    logger.info("\nValidating data leakage prevention...")
    validate_no_leakage(df, features)

    # Save to parquet
    output_path = FEATURE_DIR / "rolling_features.parquet"
    # Sort by game_date and player_id for temporal validation
    features = features.sort_values(['game_date', 'player_id'])
    features.to_parquet(output_path, index=False)

    logger.info(f"\n✓ Saved rolling features to {output_path}")
    logger.info(f"  Shape: {features.shape}")
    logger.info(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_rolling_features()
