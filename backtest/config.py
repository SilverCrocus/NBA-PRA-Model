"""
Walk-Forward Backtest Configuration

Constants and settings for the NBA PRA prediction backtest.

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

from pathlib import Path

# ==================== PATHS ====================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BACKTEST_DIR = PROJECT_ROOT / "backtest"
RESULTS_DIR = BACKTEST_DIR / "results"

# Input data paths
MASTER_FEATURES_PATH = DATA_DIR / "feature_tables" / "master_features.parquet"
PLAYER_GAMES_PATH = DATA_DIR / "nba_api" / "player_games.parquet"
HISTORICAL_ODDS_PATH = DATA_DIR / "historical_odds" / "2024-25" / "pra_odds.csv"

# Output paths
DAILY_PREDICTIONS_PATH = RESULTS_DIR / "daily_predictions.csv"
DAILY_METRICS_PATH = RESULTS_DIR / "daily_metrics.csv"
BETTING_PERFORMANCE_PATH = RESULTS_DIR / "betting_performance.csv"
PLAYER_ANALYSIS_PATH = RESULTS_DIR / "player_analysis.csv"
BACKTEST_REPORT_PATH = RESULTS_DIR / "backtest_report.md"

# ==================== BACKTEST SETTINGS ====================

# Training window
TRAINING_WINDOW_YEARS = 3  # Use 3 years of historical data before prediction date

# Ensemble settings
CV_FOLDS = 19  # Number of CV folds for ensemble (realistic production setup)
CV_VALIDATION_SPLIT = 0.2  # 20% of training data for validation

# Target season
TARGET_SEASON = "2024-25"
TARGET_START_DATE = "2024-10-01"  # Start of 2024-25 season
TARGET_END_DATE = "2025-06-30"    # End of season (including playoffs)

# ==================== BETTING SETTINGS ====================

# Standard American odds
BETTING_ODDS = -110  # Standard American odds (-110)
WIN_PAYOUT = 0.909   # Win pays: (100 / 110) = 0.909 units profit
LOSS_COST = -1.0     # Loss costs: 1 unit

# Break-even analysis
BREAK_EVEN_WIN_RATE = 0.524  # Need 52.4% win rate to profit at -110 odds

# Edge detection thresholds
EDGE_THRESHOLD_SMALL = 1.0  # Small edge: model differs from line by 1+ points
EDGE_THRESHOLD_MEDIUM = 2.0  # Medium edge: model differs by 2+ points
EDGE_THRESHOLD_LARGE = 3.0   # Large edge: model differs by 3+ points

# ==================== MODEL SETTINGS ====================

# XGBoost hyperparameters (from model_training/config.py)
XGBOOST_PARAMS = {
    "objective": "reg:absoluteerror",    # MAE optimization
    "eval_metric": ["rmse", "mae"],
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0  # Suppress XGBoost output
}

# ==================== MONTE CARLO SETTINGS ====================

# Enable/disable Monte Carlo probabilistic predictions
ENABLE_MONTE_CARLO = True  # Set to True to use MC predictions (default: False for backwards compatibility)

# Variance model hyperparameters
VARIANCE_MODEL_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 5,              # Shallower than mean model (variance is smoother)
    'learning_rate': 0.05,
    'n_estimators': 200,         # Fewer trees than mean model
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 10,      # Higher for stability
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# Distribution fitting
MC_DISTRIBUTION = 'gamma'           # 'gamma' distribution for PRA (non-negative, right-skewed)
MC_MIN_VARIANCE = 1.0               # Minimum variance to prevent degenerate distributions
MC_MAX_VARIANCE = 400.0             # Maximum variance ceiling

# Kelly criterion betting
KELLY_FRACTION = 0.25               # Fractional Kelly (0.25 = quarter Kelly for risk management)
MIN_EDGE_KELLY = 0.03               # Minimum 3% edge over breakeven to bet
MAX_BET_SIZE = 0.25                 # Maximum 25% of bankroll per bet

# Confidence filtering
MIN_CONFIDENCE = 0.6                # Minimum confidence score (0-1)
MAX_CV = 0.35                       # Maximum coefficient of variation (std/mean)

# Calibration
ENABLE_CALIBRATION = True           # Apply conformal calibration
CONFORMAL_ALPHA = 0.05              # Target 95% coverage (1 - alpha)

# MC probability calculation method
MC_PROBABILITY_METHOD = 'analytical'  # 'analytical' (fast, exact) or 'monte_carlo' (slower)

# Feature columns to exclude from training
EXCLUDE_COLUMNS = [
    'player_id', 'game_id', 'game_date', 'player_name', 'position',
    'opponent_team', 'season', 'season_detected', 'pra', 'target_pra',  # Target (both names)
    'points', 'rebounds', 'assists',  # Components of target
    'player', 'player_ctg', 'season_ctg'  # CTG merge artifacts
]

# ==================== PERFORMANCE THRESHOLDS ====================

# Expected performance benchmarks (from CV results)
EXPECTED_MAE_MIN = 2.5
EXPECTED_MAE_TARGET = 2.99
EXPECTED_MAE_MAX = 3.5

EXPECTED_RMSE_MIN = 3.5
EXPECTED_RMSE_TARGET = 4.25
EXPECTED_RMSE_MAX = 5.0

EXPECTED_R2_MIN = 0.80
EXPECTED_R2_TARGET = 0.868
EXPECTED_R2_MAX = 0.90

# Betting performance targets
TARGET_WIN_RATE = 0.53  # 53% win rate target
MIN_PROFITABLE_WIN_RATE = 0.525  # Minimum to be profitable after accounting for variance

# ==================== LOGGING SETTINGS ====================

import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Progress reporting
PROGRESS_LOG_INTERVAL = 10  # Log progress every N days
SAVE_CHECKPOINT_INTERVAL = 20  # Save intermediate results every N days

# ==================== IMPUTATION SETTINGS ====================

# CTG feature imputation
CTG_FEATURES = [
    'usage_rate', 'assist_rate', 'turnover_rate',
    'points_per_shot_attempt', 'true_shooting_pct', 'efg_pct',
    'offensive_rebound_pct', 'defensive_rebound_pct'
]

IMPUTATION_METHOD = 'position_mean'  # 'position_mean', 'position_median', or 'league_mean'

# ==================== VALIDATION SETTINGS ====================

# Minimum games required for metrics
MIN_GAMES_FOR_PLAYER_ANALYSIS = 10  # Minimum games to include player in analysis
MIN_GAMES_PER_DAY = 5  # Minimum games on a day to calculate daily metrics

# Data quality checks
MAX_MISSING_FEATURES_PCT = 0.3  # Warn if >30% of features missing
MIN_TRAINING_GAMES = 10000  # Minimum training games required

# ==================== HELPER FUNCTIONS ====================

def setup_logging(name: str = 'backtest', log_to_file: bool = True) -> logging.Logger:
    """
    Set up logging for backtest

    Args:
        name: Logger name
        log_to_file: Whether to also log to file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        log_file = RESULTS_DIR / f"backtest_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(LOG_LEVEL)
        file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to: {log_file}")

    return logger


def ensure_results_dir():
    """Ensure results directory exists"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Import pandas for timestamp in setup_logging
import pandas as pd

# Create results directory on import
ensure_results_dir()
