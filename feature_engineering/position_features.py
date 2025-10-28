"""
Position Normalization Features for NBA PRA Model
Normalizes player stats relative to their position

HIGH IMPACT: Expected 3-5% RMSE improvement
Rationale: 30 PRA for a center is very different from 30 PRA for a guard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_player_gamelogs

# Output directory
FEATURE_DIR = Path(__file__).parent.parent / "data" / "feature_tables"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def infer_player_position(df):
    """
    Infer player position from their stat profile

    Uses minutes-weighted averages to determine position based on:
    - Rebounds (high for bigs)
    - Assists (high for guards)
    - Three-point attempt rate (high for guards/wings)

    Args:
        df: Player game logs

    Returns:
        DataFrame with position column added
    """
    df = df.copy()

    # Calculate per-game averages for each player across all seasons
    player_profiles = df.groupby('player_id').agg({
        'rebounds': 'mean',
        'assists': 'mean',
        'fg3_attempted': 'mean',
        'fg_attempted': 'mean',
        'points': 'mean',
        'player_name': 'first'
    }).reset_index()

    # Calculate three-point attempt rate
    player_profiles['three_pt_rate'] = (
        player_profiles['fg3_attempted'] / player_profiles['fg_attempted'].replace(0, np.nan)
    )

    # Create composite scores for position classification
    # Guard score: high assists, high 3PT rate, low rebounds
    player_profiles['guard_score'] = (
        player_profiles['assists'] * 0.5 +
        player_profiles['three_pt_rate'].fillna(0) * 10 -
        player_profiles['rebounds'] * 0.3
    )

    # Big score: high rebounds, low assists, low 3PT rate
    player_profiles['big_score'] = (
        player_profiles['rebounds'] * 0.8 -
        player_profiles['assists'] * 0.3 -
        player_profiles['three_pt_rate'].fillna(0) * 5
    )

    # Classify positions
    def classify_position(row):
        if row['big_score'] > 3:
            return 'Center'
        elif row['big_score'] > 1:
            return 'Forward'
        elif row['guard_score'] > 2:
            return 'Guard'
        else:
            return 'Wing'  # Hybrid/versatile players

    player_profiles['position'] = player_profiles.apply(classify_position, axis=1)

    # Merge position back to game logs
    df = df.merge(
        player_profiles[['player_id', 'position']],
        on='player_id',
        how='left'
    )

    # Fill any missing positions with 'Wing' (neutral)
    df['position'] = df['position'].fillna('Wing')

    return df


def calculate_position_normalized_features(df):
    """
    Create position-normalized features using z-scores

    Args:
        df: Player game logs with position column

    Returns:
        DataFrame with position-normalized features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Ensure position exists
    if 'position' not in df.columns:
        df = infer_player_position(df)

    # For each stat, calculate position-specific mean and std
    # Then compute z-score for each player
    stats_to_normalize = ['pra', 'points', 'rebounds', 'assists']

    for stat in stats_to_normalize:
        if stat not in df.columns:
            continue

        # Calculate position-specific statistics (using shift to avoid leakage)
        # Group by position and game_date, calculate rolling stats
        df_sorted = df.sort_values(['position', 'game_date']).reset_index(drop=True)

        # Calculate expanding mean and std for each position (historical only)
        position_stats = df_sorted.groupby(['position']).apply(
            lambda g: pd.DataFrame({
                'game_date': g['game_date'],
                f'{stat}_position_mean': g[stat].shift(1).expanding().mean(),
                f'{stat}_position_std': g[stat].shift(1).expanding().std()
            })
        ).reset_index(drop=True)

        # Merge back to original dataframe
        df_sorted = pd.concat([df_sorted.reset_index(drop=True), position_stats], axis=1)

        # Calculate z-score: (value - position_mean) / position_std
        features[f'{stat}_position_zscore'] = (
            (df_sorted[stat] - df_sorted[f'{stat}_position_mean']) /
            df_sorted[f'{stat}_position_std'].replace(0, np.nan)
        )

        # Also include raw position mean for context
        features[f'{stat}_vs_position_avg'] = (
            df_sorted[stat] - df_sorted[f'{stat}_position_mean']
        )

    return features


def calculate_position_percentiles(df):
    """
    Calculate player's percentile rank within their position

    Args:
        df: Player game logs with position

    Returns:
        DataFrame with position percentile features
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    if 'position' not in df.columns:
        df = infer_player_position(df)

    # Calculate rolling 10-game average PRA for each player
    df['pra_avg_last10'] = (
        df.groupby('player_id')['pra']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # For each game, calculate percentile rank within position
    # Based on recent 10-game averages
    def calculate_position_percentile_row(row, df_with_pos):
        """Calculate percentile rank within position for a single game"""
        current_date = row['game_date']
        current_avg = row['pra_avg_last10']
        position = row['position']

        # Get all players at this position with data before this date
        historical = df_with_pos[
            (df_with_pos['position'] == position) &
            (df_with_pos['game_date'] < current_date)
        ]['pra_avg_last10'].dropna()

        if len(historical) > 0 and not pd.isna(current_avg):
            percentile = (historical < current_avg).sum() / len(historical) * 100
        else:
            percentile = 50  # Default to median for first games

        return percentile

    df_with_position = df[['game_date', 'pra_avg_last10', 'position']].copy()
    features['pra_position_percentile'] = df.apply(
        lambda row: calculate_position_percentile_row(row, df_with_position),
        axis=1
    )

    return features


def create_position_role_indicators(df):
    """
    Create indicators for position-specific roles

    Args:
        df: Player game logs with position

    Returns:
        DataFrame with role indicators
    """
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    features = df[['player_id', 'game_id', 'game_date']].copy()

    if 'position' not in df.columns:
        df = infer_player_position(df)

    # Position indicators (one-hot encoding)
    features['is_guard'] = (df['position'] == 'Guard').astype(int)
    features['is_wing'] = (df['position'] == 'Wing').astype(int)
    features['is_forward'] = (df['position'] == 'Forward').astype(int)
    features['is_center'] = (df['position'] == 'Center').astype(int)

    # Position-specific thresholds for "elite" performance
    position_thresholds = {
        'Guard': {'points': 20, 'assists': 7, 'rebounds': 5},
        'Wing': {'points': 18, 'assists': 5, 'rebounds': 6},
        'Forward': {'points': 18, 'assists': 4, 'rebounds': 8},
        'Center': {'points': 16, 'assists': 3, 'rebounds': 10}
    }

    # Calculate recent averages
    for stat in ['points', 'rebounds', 'assists']:
        df[f'{stat}_avg_last5'] = (
            df.groupby('player_id')[stat]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )

    # Elite performer for position indicator
    def is_elite_for_position(row):
        position = row['position']
        thresholds = position_thresholds.get(position, position_thresholds['Wing'])

        return int(
            row['points_avg_last5'] >= thresholds['points'] and
            (row['rebounds_avg_last5'] >= thresholds['rebounds'] or
             row['assists_avg_last5'] >= thresholds['assists'])
        )

    features['is_elite_for_position'] = df.apply(is_elite_for_position, axis=1)

    return features


def validate_position_features(features_df):
    """
    Validate position features

    Args:
        features_df: Computed features
    """
    # Check z-scores are in reasonable range
    zscore_cols = [col for col in features_df.columns if 'zscore' in col]

    for col in zscore_cols:
        mean_zscore = features_df[col].dropna().mean()
        std_zscore = features_df[col].dropna().std()

        # Z-scores should have mean ~0 and std ~1
        assert abs(mean_zscore) < 0.5, f"{col} mean z-score too far from 0: {mean_zscore}"
        # Note: std might not be exactly 1 due to sample size differences

    # Check percentiles are 0-100
    if 'pra_position_percentile' in features_df.columns:
        assert features_df['pra_position_percentile'].min() >= 0
        assert features_df['pra_position_percentile'].max() <= 100

    print("✓ Position features validation passed")


def build_position_features():
    """
    Main function to build all position normalization features
    """
    print("Loading player game logs...")
    df = load_player_gamelogs()

    print(f"Loaded {len(df)} games for {df['player_id'].nunique()} players")

    print("\nInferring player positions...")
    df = infer_player_position(df)

    position_counts = df.groupby('position')['player_id'].nunique()
    print(f"Position distribution:\n{position_counts}")

    print("\nCalculating position-normalized features...")
    normalized = calculate_position_normalized_features(df)

    print("Calculating position percentiles...")
    percentiles = calculate_position_percentiles(df)

    print("Creating position role indicators...")
    roles = create_position_role_indicators(df)

    print("\nMerging all position features...")
    features = normalized.copy()

    for feature_df in [percentiles, roles]:
        features = features.merge(
            feature_df,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

    print(f"Created {len(features.columns) - 3} position features")

    # Validate
    print("\nValidating position features...")
    validate_position_features(features)

    # Save
    output_path = FEATURE_DIR / "position_features.parquet"
    features.to_parquet(output_path, index=False)

    print(f"\n✓ Saved position features to {output_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Columns: {list(features.columns)}")

    return features


if __name__ == "__main__":
    build_position_features()
