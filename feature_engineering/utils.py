"""
Shared Utility Functions for Feature Engineering

This module contains common functions used across multiple feature engineering modules
to reduce code duplication and improve maintainability.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Optional


# ==============================================================================
# Data Type Conversion Utilities
# ==============================================================================

def convert_minutes_to_float(minutes):
    """
    Convert minutes from various formats to float

    Handles multiple input formats:
    - Already float/int: returns as-is
    - String "MM:SS": converts to decimal minutes
    - NaN/None: returns 0.0
    - Invalid: returns 0.0

    Args:
        minutes: Minutes in various formats (str, int, float, or NaN)

    Returns:
        float: Minutes as decimal float

    Examples:
        >>> convert_minutes_to_float("32:30")
        32.5
        >>> convert_minutes_to_float(25.5)
        25.5
        >>> convert_minutes_to_float(np.nan)
        0.0
    """
    if pd.isna(minutes):
        return 0.0

    if isinstance(minutes, (int, float)):
        return float(minutes)

    if isinstance(minutes, str) and ':' in minutes:
        try:
            parts = minutes.split(':')
            return float(parts[0]) + float(parts[1]) / 60
        except (ValueError, IndexError):
            return 0.0

    # Try to convert to float directly
    try:
        return float(minutes)
    except (ValueError, TypeError):
        return 0.0


# ==============================================================================
# DataFrame Utilities
# ==============================================================================

def create_feature_base(df: pd.DataFrame, grain_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create base feature DataFrame with grain columns

    Args:
        df: Input DataFrame
        grain_cols: List of grain columns (default: ['player_id', 'game_id', 'game_date'])

    Returns:
        DataFrame with only grain columns

    Examples:
        >>> df = pd.DataFrame({'player_id': [1], 'game_id': [100], 'game_date': ['2024-01-01'], 'points': [20]})
        >>> features = create_feature_base(df)
        >>> list(features.columns)
        ['player_id', 'game_id', 'game_date']
    """
    if grain_cols is None:
        grain_cols = ['player_id', 'game_id', 'game_date']

    return df[grain_cols].copy()


# ==============================================================================
# Validation Utilities
# ==============================================================================

def validate_required_columns(df: pd.DataFrame, required_cols: List[str], function_name: str = "function") -> None:
    """
    Validate that DataFrame contains required columns

    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        function_name: Name of calling function (for error messages)

    Raises:
        ValueError: If required columns are missing

    Examples:
        >>> df = pd.DataFrame({'a': [1], 'b': [2]})
        >>> validate_required_columns(df, ['a', 'b'], 'test_func')  # passes
        >>> validate_required_columns(df, ['a', 'c'], 'test_func')  # raises ValueError
    """
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{function_name}: Missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )


def validate_not_empty(df: pd.DataFrame, function_name: str = "function") -> None:
    """
    Validate that DataFrame is not empty

    Args:
        df: DataFrame to validate
        function_name: Name of calling function (for error messages)

    Raises:
        ValueError: If DataFrame is empty

    Examples:
        >>> df = pd.DataFrame({'a': [1, 2]})
        >>> validate_not_empty(df, 'test_func')  # passes
        >>> df_empty = pd.DataFrame()
        >>> validate_not_empty(df_empty, 'test_func')  # raises ValueError
    """
    if df.empty:
        raise ValueError(f"{function_name}: DataFrame is empty")


def validate_grain_uniqueness(df: pd.DataFrame, grain_cols: Optional[List[str]] = None) -> None:
    """
    Validate that grain columns form a unique key

    Args:
        df: DataFrame to validate
        grain_cols: List of grain columns (default: ['player_id', 'game_id', 'game_date'])

    Raises:
        AssertionError: If duplicates exist

    Examples:
        >>> df = pd.DataFrame({
        ...     'player_id': [1, 1, 2],
        ...     'game_id': [100, 101, 100],
        ...     'game_date': ['2024-01-01', '2024-01-02', '2024-01-01']
        ... })
        >>> validate_grain_uniqueness(df)  # passes (no duplicates)
    """
    if grain_cols is None:
        grain_cols = ['player_id', 'game_id', 'game_date']

    duplicates = df.duplicated(subset=grain_cols).sum()

    if duplicates > 0:
        raise AssertionError(
            f"Grain violation: {duplicates} duplicate rows found for columns {grain_cols}"
        )

    print(f"✓ Grain validation passed: {len(df):,} unique rows")


def validate_value_range(
    df: pd.DataFrame,
    col: str,
    min_val: float,
    max_val: float,
    strict: bool = False
) -> None:
    """
    Validate that column values are within expected range

    Args:
        df: DataFrame containing column
        col: Column name to validate
        min_val: Minimum expected value
        max_val: Maximum expected value
        strict: If True, raises exception on failure; if False, prints warning

    Raises:
        AssertionError: If strict=True and values out of range

    Examples:
        >>> df = pd.DataFrame({'score': [10, 20, 30]})
        >>> validate_value_range(df, 'score', 0, 100)  # passes
        >>> validate_value_range(df, 'score', 0, 25)  # warns (30 > 25)
    """
    if col not in df.columns:
        return

    values = df[col].dropna()

    if len(values) == 0:
        return

    actual_min, actual_max = values.min(), values.max()
    in_range = (actual_min >= min_val and actual_max <= max_val)

    if not in_range:
        msg = (
            f"{col} out of range: [{actual_min:.2f}, {actual_max:.2f}] "
            f"expected [{min_val}, {max_val}]"
        )

        if strict:
            raise AssertionError(msg)
        else:
            print(f"⚠️  Warning: {msg}")
    else:
        print(f"✓ {col} range validation passed: [{actual_min:.2f}, {actual_max:.2f}]")


# ==============================================================================
# Data Leakage Prevention Utilities
# ==============================================================================

def check_leakage_first_games(
    df: pd.DataFrame,
    features_df: pd.DataFrame,
    feature_cols: List[str],
    player_col: str = 'player_id'
) -> None:
    """
    Check for data leakage by validating first games have NaN/0 for rolling features

    Rolling features should have NaN or 0 for the first game of each player
    since there's no historical data.

    Args:
        df: Original data DataFrame
        features_df: Features DataFrame to validate
        feature_cols: List of feature columns to check
        player_col: Column containing player identifier

    Raises:
        AssertionError: If first games have non-null rolling features

    Examples:
        >>> df = pd.DataFrame({'player_id': [1, 1, 2], 'game_date': ['2024-01-01', '2024-01-02', '2024-01-01']})
        >>> features = pd.DataFrame({'player_id': [1, 1, 2], 'rolling_avg': [np.nan, 20, np.nan]})
        >>> check_leakage_first_games(df, features, ['rolling_avg'])  # passes
    """
    # Get first game for each player
    first_games = df.groupby(player_col).head(1).index

    # Check that rolling features are NaN/0 for first games
    for col in feature_cols:
        if col not in features_df.columns:
            continue

        first_game_values = features_df.loc[first_games, col]
        non_null_first_games = first_game_values.notna() & (first_game_values != 0)

        if non_null_first_games.any():
            problem_count = non_null_first_games.sum()
            raise AssertionError(
                f"Data leakage detected in {col}: {problem_count} first games have non-null/non-zero values"
            )

    print(f"✓ No leakage detected: First games have expected NaN/0 values for {len(feature_cols)} rolling features")


# ==============================================================================
# Statistical Utilities
# ==============================================================================

def calculate_z_score(values: pd.Series, mean: pd.Series, std: pd.Series) -> pd.Series:
    """
    Calculate z-scores safely (handles zero std)

    Args:
        values: Raw values
        mean: Mean values
        std: Standard deviation values

    Returns:
        Series of z-scores

    Examples:
        >>> values = pd.Series([20, 25, 30])
        >>> mean = pd.Series([22, 22, 22])
        >>> std = pd.Series([5, 5, 5])
        >>> calculate_z_score(values, mean, std)
        0   -0.4
        1    0.6
        2    1.6
        dtype: float64
    """
    return (values - mean) / std.replace(0, np.nan)


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """
    Safely divide two series, handling division by zero

    Args:
        numerator: Numerator series
        denominator: Denominator series
        fill_value: Value to use when denominator is 0 (default: 0.0)

    Returns:
        Series of quotients

    Examples:
        >>> a = pd.Series([10, 20, 30])
        >>> b = pd.Series([2, 0, 5])
        >>> safe_divide(a, b, fill_value=0.0)
        0     5.0
        1     0.0
        2     6.0
        dtype: float64
    """
    return (numerator / denominator.replace(0, np.nan)).fillna(fill_value)


# ==============================================================================
# Logging Utilities
# ==============================================================================

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance

    Examples:
        >>> logger = setup_logging(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create console handler
    handler = logging.StreamHandler()
    handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger


# ==============================================================================
# File System Utilities
# ==============================================================================

def ensure_directory_exists(path: Path) -> None:
    """
    Create directory if it doesn't exist

    Args:
        path: Directory path to create

    Examples:
        >>> from pathlib import Path
        >>> ensure_directory_exists(Path("data/output"))
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
