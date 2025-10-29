"""
Model Training Utilities
Shared functions for logging, data loading, metrics calculation, and validation
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

from model_training.config import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    TARGET_COLUMN,
    EXCLUDE_COLUMNS,
    LOG_LEVEL,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    CV_FOLDS_DIR,
)


def setup_logger(name: str, log_file: Optional[Path] = None, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Setup logger with consistent formatting
    Follows feature_engineering/utils.py pattern

    Args:
        name: Logger name (e.g., "training", "validation")
        log_file: Optional file path for logging
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_split_data(split_name: str) -> pd.DataFrame:
    """
    Load train/val/test split with validation

    Args:
        split_name: One of ['train', 'val', 'test']

    Returns:
        DataFrame with features and target

    Raises:
        FileNotFoundError: If split file doesn't exist
        ValueError: If split_name is invalid or required columns missing
    """
    # Validate split_name
    valid_splits = {'train': TRAIN_PATH, 'val': VAL_PATH, 'test': TEST_PATH}

    if split_name not in valid_splits:
        raise ValueError(
            f"Invalid split_name: '{split_name}'. Must be one of {list(valid_splits.keys())}"
        )

    filepath = valid_splits[split_name]

    # Check file exists
    if not filepath.exists():
        raise FileNotFoundError(
            f"{split_name.capitalize()} split not found at {filepath}\n"
            f"Please run: uv run model_training/train_split.py"
        )

    # Load data
    df = pd.read_parquet(filepath)

    # Validate required columns exist
    required_cols = [TARGET_COLUMN, 'player_id', 'game_id', 'game_date']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"{split_name.capitalize()} split missing required columns: {missing_cols}"
        )

    # Check for empty DataFrame
    if len(df) == 0:
        raise ValueError(f"{split_name.capitalize()} split is empty!")

    return df


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    exclude_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features from target, exclude non-feature columns

    Args:
        df: Full dataset with features and target
        target_col: Target column name
        exclude_cols: Columns to exclude from features (default: config.EXCLUDE_COLUMNS)

    Returns:
        Tuple of (X_features, y_target)

    Raises:
        ValueError: If target column missing or no features remain
    """
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLUMNS

    # Validate target exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame!")

    # Extract target
    y = df[target_col].copy()

    # Extract features (all columns except excluded)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    if len(feature_cols) == 0:
        raise ValueError(
            f"No features remaining after excluding {len(exclude_cols)} columns!"
        )

    X = df[feature_cols].copy()

    # Check for any remaining NaN values in target
    if y.isnull().any():
        raise ValueError(
            f"Target variable contains {y.isnull().sum()} missing values! "
            "Please handle missing values in train_split.py"
        )

    return X, y


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics

    Args:
        y_true: Actual values
        y_pred: Predicted values
        prefix: Prefix for metric names (e.g., "train_", "val_", "test_")

    Returns:
        Dictionary with RMSE, MAE, R², MAPE, and residual statistics
    """
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate residuals
    residuals = y_true - y_pred

    # Core metrics
    metrics = {
        f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred),
    }

    # MAPE (handle division by zero)
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        metrics[f"{prefix}mape"] = mape
    except (ValueError, ZeroDivisionError):
        metrics[f"{prefix}mape"] = np.nan

    # Residual statistics
    metrics[f"{prefix}residual_mean"] = residuals.mean()
    metrics[f"{prefix}residual_std"] = residuals.std()
    metrics[f"{prefix}residual_max"] = np.abs(residuals).max()

    # Median absolute error (robust to outliers)
    metrics[f"{prefix}median_ae"] = np.median(np.abs(residuals))

    return metrics


def validate_temporal_order(df: pd.DataFrame, date_col: str = 'game_date') -> bool:
    """
    Validate data is in chronological order within each player
    CRITICAL for time-series modeling

    Args:
        df: DataFrame with player_id and date column
        date_col: Name of date column

    Returns:
        True if properly ordered

    Raises:
        ValueError: If data is not chronologically sorted
    """
    # Check required columns
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame!")

    if 'player_id' not in df.columns:
        raise ValueError("'player_id' column not found in DataFrame!")

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Check if dates are monotonically increasing within each player
    df_sorted = df.sort_values(['player_id', date_col])
    is_sorted = df.equals(df_sorted)

    if not is_sorted:
        raise ValueError(
            f"Data is not chronologically sorted by {date_col} within player_id! "
            "This violates temporal ordering for time-series modeling."
        )

    return True


def check_missing_values(X: pd.DataFrame) -> Dict[str, int]:
    """
    Check for missing values in feature matrix

    Args:
        X: Feature DataFrame

    Returns:
        Dictionary mapping column name -> count of missing values
    """
    missing = X.isnull().sum()
    missing_dict = missing[missing > 0].to_dict()

    return missing_dict


def get_feature_groups(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Group features by category based on naming convention

    Args:
        feature_names: List of feature column names

    Returns:
        Dictionary mapping category -> list of feature names
    """
    groups = {
        'rolling': [],
        'matchup': [],
        'contextual': [],
        'advanced': [],
        'position': [],
        'injury': [],
        'other': []
    }

    for feature in feature_names:
        feature_lower = feature.lower()

        # Categorize based on feature name patterns
        if any(x in feature_lower for x in ['avg', 'std', 'ewma', 'trend', 'last']):
            groups['rolling'].append(feature)
        elif any(x in feature_lower for x in ['opp_', 'opponent', 'matchup']):
            groups['matchup'].append(feature)
        elif any(x in feature_lower for x in ['home', 'rest', 'back_to_back', 'season']):
            groups['contextual'].append(feature)
        elif any(x in feature_lower for x in ['usage', 'true_shooting', 'assist_rate', 'turnover']):
            groups['advanced'].append(feature)
        elif any(x in feature_lower for x in ['position', 'zscore', 'percentile']):
            groups['position'].append(feature)
        elif any(x in feature_lower for x in ['dnp', 'availability', 'absence', 'injury']):
            groups['injury'].append(feature)
        else:
            groups['other'].append(feature)

    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}

    return groups


def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """
    Format metrics dictionary as a nice table string

    Args:
        metrics: Dictionary of metric_name -> value
        title: Title for the table

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 60)
    lines.append(title.center(60))
    lines.append("=" * 60)

    # Group metrics by prefix (train, val, test)
    prefixes = set()
    for key in metrics.keys():
        if '_' in key:
            prefix = key.split('_')[0]
            if prefix in ['train', 'val', 'test']:
                prefixes.add(prefix)

    if prefixes:
        # Group by prefix
        for prefix in sorted(prefixes):
            lines.append(f"\n{prefix.capitalize()} Metrics:")
            prefix_metrics = {k: v for k, v in metrics.items() if k.startswith(f"{prefix}_")}
            for key, value in sorted(prefix_metrics.items()):
                clean_key = key.replace(f"{prefix}_", "").replace("_", " ").title()
                if isinstance(value, float):
                    lines.append(f"  {clean_key:.<40} {value:>10.4f}")
                else:
                    lines.append(f"  {clean_key:.<40} {value:>10}")
    else:
        # No prefix grouping
        for key, value in sorted(metrics.items()):
            clean_key = key.replace("_", " ").title()
            if isinstance(value, float):
                lines.append(f"  {clean_key:.<50} {value:>8.4f}")
            else:
                lines.append(f"  {clean_key:.<50} {value:>8}")

    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================================
# TIME-SERIES CROSS-VALIDATION UTILITIES
# ============================================================================

def load_cv_fold(fold_id: int, fold_dir: Path = None) -> Dict[str, pd.DataFrame]:
    """
    Load a specific CV fold (train/val/test)

    Args:
        fold_id: Fold number (0-indexed)
        fold_dir: Base directory for CV folds (default: CV_FOLDS_DIR from config)

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing a DataFrame

    Raises:
        FileNotFoundError: If fold directory or required files don't exist
        ValueError: If fold files are invalid

    Example:
        >>> fold_data = load_cv_fold(0)
        >>> X_train, y_train = prepare_features_target(fold_data['train'])
    """
    if fold_dir is None:
        fold_dir = CV_FOLDS_DIR

    # Construct fold directory path
    fold_path = fold_dir / f"fold_{fold_id}"

    # Check fold directory exists
    if not fold_path.exists():
        raise FileNotFoundError(
            f"Fold {fold_id} directory not found at {fold_path}\n"
            f"Available folds: {get_available_cv_folds(fold_dir)}\n"
            f"Please run: uv run model_training/train_split.py --cv-mode"
        )

    # Load train/val/test splits
    fold_data = {}
    for split_name in ['train', 'val', 'test']:
        split_path = fold_path / f"{split_name}.parquet"

        if not split_path.exists():
            raise FileNotFoundError(
                f"Fold {fold_id} {split_name} split not found at {split_path}"
            )

        df = pd.read_parquet(split_path)

        # Validate DataFrame not empty
        if len(df) == 0:
            raise ValueError(f"Fold {fold_id} {split_name} split is empty!")

        # Validate required columns
        required_cols = [TARGET_COLUMN, 'player_id', 'game_id', 'game_date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Fold {fold_id} {split_name} split missing columns: {missing_cols}"
            )

        fold_data[split_name] = df

    return fold_data


def get_available_cv_folds(fold_dir: Path = None) -> List[int]:
    """
    Detect available CV folds from directory structure

    Args:
        fold_dir: Base directory for CV folds (default: CV_FOLDS_DIR from config)

    Returns:
        Sorted list of fold IDs (e.g., [0, 1, 2, 3, 4])

    Example:
        >>> folds = get_available_cv_folds()
        >>> print(f"Found {len(folds)} folds: {folds}")
        Found 5 folds: [0, 1, 2, 3, 4]
    """
    if fold_dir is None:
        fold_dir = CV_FOLDS_DIR

    # Check if base directory exists
    if not fold_dir.exists():
        return []

    # Find all fold_N directories
    fold_dirs = [d for d in fold_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')]

    # Extract fold IDs
    fold_ids = []
    for fold_path in fold_dirs:
        try:
            fold_id = int(fold_path.name.split('_')[1])
            fold_ids.append(fold_id)
        except (ValueError, IndexError):
            # Skip directories that don't match fold_N pattern
            continue

    return sorted(fold_ids)


def calculate_cv_summary_statistics(
    fold_metrics: List[Dict[str, float]]
) -> pd.DataFrame:
    """
    Aggregate CV fold metrics into summary table

    Args:
        fold_metrics: List of metric dictionaries, one per fold
                      Each dict has structure: {'metric_name': value, ...}

    Returns:
        DataFrame with columns:
        - metric_name: Name of the metric
        - mean: Mean value across folds
        - std: Standard deviation across folds
        - min: Minimum value across folds
        - max: Maximum value across folds
        - cv: Coefficient of variation (std/mean), useful for assessing stability

    Example:
        >>> fold_metrics = [
        ...     {'test_mae': 3.5, 'test_rmse': 4.2, 'test_r2': 0.85},
        ...     {'test_mae': 3.7, 'test_rmse': 4.4, 'test_r2': 0.83},
        ...     {'test_mae': 3.6, 'test_rmse': 4.3, 'test_r2': 0.84}
        ... ]
        >>> summary = calculate_cv_summary_statistics(fold_metrics)
        >>> print(summary)
               metric_name  mean   std   min   max     cv
        0        test_mae  3.60  0.10  3.50  3.70  0.028
        1       test_rmse  4.30  0.10  4.20  4.40  0.023
        2         test_r2  0.84  0.01  0.83  0.85  0.012
    """
    if not fold_metrics:
        raise ValueError("fold_metrics list is empty!")

    # Convert list of dicts to DataFrame
    metrics_df = pd.DataFrame(fold_metrics)

    # Calculate summary statistics
    summary_rows = []
    for metric_name in metrics_df.columns:
        values = metrics_df[metric_name].values

        # Filter out NaN values
        values_clean = values[~np.isnan(values)]

        if len(values_clean) == 0:
            # All values are NaN
            summary_rows.append({
                'metric_name': metric_name,
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'cv': np.nan
            })
        else:
            mean_val = values_clean.mean()
            std_val = values_clean.std()

            # Coefficient of variation (std/mean), handle division by zero
            cv_val = (std_val / mean_val) if mean_val != 0 else np.nan

            summary_rows.append({
                'metric_name': metric_name,
                'mean': mean_val,
                'std': std_val,
                'min': values_clean.min(),
                'max': values_clean.max(),
                'cv': cv_val
            })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df
