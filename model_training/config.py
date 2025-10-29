"""
Model Training Configuration
Centralized configuration for model training parameters and paths
"""

from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PATHS (Following feature_engineering pattern)
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "feature_tables"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================

# Input data
MASTER_FEATURES_PATH = FEATURE_DIR / "master_features.parquet"

# Split data (created by train_split.py)
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VAL_PATH = PROCESSED_DIR / "val.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

# ============================================================================
# TARGET AND FEATURE CONFIGURATION
# ============================================================================

# Target variable
TARGET_COLUMN = "target_pra"

# Columns to exclude from features (identifiers, not predictive)
EXCLUDE_COLUMNS = [
    'target_pra',   # Target variable
    'player_id',    # Identifier
    'game_id',      # Identifier
    'game_date',    # Identifier (temporal, not feature)
    'player_name',  # Identifier
    'season',       # Identifier (could be feature, but using contextual features instead)
    'season_detected'  # Metadata (duplicate of season)
]

# Feature columns will be auto-detected (all columns except excluded)
FEATURE_COLUMNS = None  # Auto-detect from data

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# XGBoost configuration (optimized for MAE minimization)
XGBOOST_PARAMS: Dict[str, Any] = {
    # Core parameters
    "objective": "reg:absoluteerror",   # Optimizes MAE (changed from squarederror)
    "eval_metric": ["rmse", "mae"],

    # Learning parameters
    "learning_rate": 0.05,           # Conservative for stability
    "max_depth": 6,                  # Moderate depth, prevents overfitting
    "min_child_weight": 0,           # CRITICAL for MAE optimization (was 3)

    # Sampling parameters
    "subsample": 0.8,                # Row sampling (80%)
    "colsample_bytree": 0.8,         # Column sampling (80%)

    # Regularization
    "reg_alpha": 0.1,                # L1 regularization
    "reg_lambda": 1.0,               # L2 regularization

    # Training parameters
    "n_estimators": 500,             # Maximum iterations
    "early_stopping_rounds": 50,     # Stop if no improvement for 50 rounds

    # System parameters
    "random_state": 42,              # Reproducibility
    "n_jobs": -1,                    # Use all CPU cores
    "verbosity": 1,                  # Moderate logging
}

# LightGBM configuration (similar to XGBoost for comparison)
LIGHTGBM_PARAMS: Dict[str, Any] = {
    # Core parameters
    "objective": "regression",
    "metric": ["rmse", "mae"],
    "boosting_type": "gbdt",

    # Learning parameters
    "learning_rate": 0.05,
    "num_leaves": 31,                # 2^max_depth - 1 (approx max_depth=5)
    "max_depth": 6,
    "min_data_in_leaf": 20,

    # Sampling parameters
    "feature_fraction": 0.8,         # Column sampling
    "bagging_fraction": 0.8,         # Row sampling
    "bagging_freq": 5,               # Bagging frequency

    # Regularization
    "lambda_l1": 0.1,                # L1 regularization
    "lambda_l2": 1.0,                # L2 regularization

    # Training parameters
    "n_estimators": 500,
    "early_stopping_rounds": 50,

    # System parameters
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,                   # Suppress LightGBM warnings
}

# ============================================================================
# EXPECTED PERFORMANCE THRESHOLDS
# ============================================================================

# Based on CLAUDE.md documentation expectations
EXPECTED_R2_MIN = 0.75              # Minimum acceptable R²
EXPECTED_R2_TARGET = 0.84           # Target R² (midpoint of 0.82-0.86)
EXPECTED_RMSE_MAX = 5.0             # Maximum acceptable RMSE (PRA points)
EXPECTED_RMSE_TARGET = 3.75         # Target RMSE (midpoint of 3.5-4.0)
EXPECTED_MAE_MAX = 4.0              # Maximum acceptable MAE

# ============================================================================
# MLFLOW CONFIGURATION
# ============================================================================

MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "nba_pra_prediction"

# MLflow tags
MLFLOW_TAGS = {
    "project": "NBA_PRA",
    "task": "regression",
    "target": "pra",
    "data_type": "time_series",
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# VALIDATION CONFIGURATION
# ============================================================================

# Number of top features to display in importance plots
TOP_N_FEATURES = 30

# Residual analysis
RESIDUAL_PLOT_BINS = 50
RESIDUAL_OUTLIER_THRESHOLD = 3.0  # Standard deviations

# Confidence intervals
CONFIDENCE_LEVEL = 0.95

# ============================================================================
# TIME-SERIES CROSS-VALIDATION CONFIGURATION
# ============================================================================

# Training window (years of historical data per fold)
# Example: CV_TRAINING_WINDOW_YEARS = 3 means train on 2015-18, test on 2018-19
CV_TRAINING_WINDOW_YEARS = 3

# Gap between training end and test start (games per player)
# Prevents leakage from rolling features that use recent games (up to 20-game windows)
# Conservative buffer ensures test data never overlaps with feature calculation
CV_GAP_GAMES = 15

# Minimum games per test fold (ensures statistical significance)
CV_MIN_TEST_GAMES = 1000

# CV fold storage directory
CV_FOLDS_DIR = PROCESSED_DIR / "cv_folds"

# Number of folds (auto-determined from data, but can be overridden)
# None = auto-detect from available date range
# Example: 2015-24 data (9 seasons) with 3-year windows = 6 folds
CV_N_FOLDS = None  # Auto-detect

# Validation split within each training fold (for early stopping)
CV_VAL_SPLIT = 0.2  # 20% of training data for validation

# ============================================================================
# HYPERPARAMETER TUNING CONFIGURATION (Future use)
# ============================================================================

# Optuna configuration (when implementing hyperparameter tuning)
TUNING_N_TRIALS = 50
TUNING_TIMEOUT = 3600  # 1 hour
TUNING_METRIC = "val_rmse"  # Metric to optimize

# XGBoost search space
XGBOOST_SEARCH_SPACE = {
    "learning_rate": (0.01, 0.1),
    "max_depth": (3, 10),
    "min_child_weight": (1, 10),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "reg_alpha": (0.0, 1.0),
    "reg_lambda": (0.0, 2.0),
}
