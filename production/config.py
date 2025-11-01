"""
Production Configuration

Central configuration for NBA PRA production prediction system.

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PROJECT STRUCTURE
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
PRODUCTION_DIR = PROJECT_ROOT / "production"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PRODUCTION_DIR / "models"
OUTPUTS_DIR = PRODUCTION_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
BETS_DIR = OUTPUTS_DIR / "bets"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
BETS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA PATHS
# ============================================================================

# Feature data
MASTER_FEATURES_PATH = DATA_DIR / "feature_tables" / "master_features.parquet"
PLAYER_GAMES_PATH = DATA_DIR / "nba_api" / "player_games.parquet"

# Feature tables (for incremental updates)
FEATURE_TABLES_DIR = DATA_DIR / "feature_tables"

# ============================================================================
# API CONFIGURATION
# ============================================================================

# TheOddsAPI
ODDS_API_KEY = os.getenv('ODDS_API_FREE_KEY', '5100c18e74058e57c1d33a747e8c2be1')
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_SPORT = "basketball_nba"
ODDS_REGIONS = "us"  # US bookmakers
ODDS_MARKETS = "player_points,player_rebounds,player_assists"  # Individual markets
ODDS_FORMAT = "american"  # -110 style odds

# Rate limiting for TheOddsAPI (free tier: 500 requests/month)
ODDS_API_RATE_LIMIT_DELAY = 1.0  # seconds between requests

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training window
TRAINING_WINDOW_YEARS = 3  # Use last 3 seasons
MIN_TRAINING_GAMES = 50000  # Minimum games required for training

# Cross-validation
CV_FOLDS = 19  # Number of CV folds (same as backtest)
CV_VALIDATION_SPLIT = 0.2  # 20% of each fold for validation

# Feature engineering
EXCLUDE_COLUMNS = [
    'pra',  # Target variable (from player_games)
    'target_pra',  # Target variable (from master_features)
    'game_id',
    'game_date',
    'player_id',
    'player_name',
    'team_abbreviation',
    'matchup',
    'season',
    'season_detected',
    'position'  # Categorical metadata (Guard, Center, Forward, Wing)
]

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# XGBoost hyperparameters (optimized from backtesting)
XGBOOST_PARAMS = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50,
    'eval_metric': 'rmse',
    'verbosity': 0
}

# Variance model parameters (for Monte Carlo)
VARIANCE_MODEL_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'gamma': 0.2,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# ============================================================================
# MONTE CARLO CONFIGURATION
# ============================================================================

ENABLE_MONTE_CARLO = True  # Use probabilistic predictions
MC_PROBABILITY_METHOD = 'analytical'  # 'analytical' (fast) or 'monte_carlo' (slow)

# Calibration
ENABLE_CALIBRATION = False  # Conformal calibration (set to True if needed)
CONFORMAL_ALPHA = 0.05  # 95% coverage

# ============================================================================
# BETTING CONFIGURATION
# ============================================================================

# Bet sizing
KELLY_FRACTION = 0.25  # Quarter Kelly for risk management
MIN_EDGE_KELLY = 0.03  # 3% minimum edge to bet
MIN_EDGE_DISPLAY = 0.02  # 2% minimum edge to display

# Confidence filtering
MIN_CONFIDENCE = 0.6  # 60% minimum confidence (how close to 0 or 1)
MAX_CV = 0.35  # 35% maximum coefficient of variation (std/mean)

# Odds
BETTING_ODDS = -110  # Standard American odds

# ============================================================================
# PLAYER FILTERING
# ============================================================================

# Minimum games to make prediction
MIN_CAREER_GAMES = 5  # Need at least 5 games of history
MIN_RECENT_GAMES = 3  # Need at least 3 games in last 30 days

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(name: str, level: str = LOG_LEVEL):
    """
    Setup logging configuration

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    import logging

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file = LOGS_DIR / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Prediction output columns
PREDICTION_COLUMNS = [
    'player_id',
    'player_name',
    'game_date',
    'team_abbreviation',
    'opponent',
    'home_game',
    'mean_pred',
    'std_dev',
    'betting_line',
    'pra_odds',  # Actual American odds from sportsbook
    'breakeven_prob',  # Breakeven probability calculated from odds
    'prob_over',
    'prob_under',
    'edge_over',
    'edge_under',
    'confidence_score',
    'cv'
]

# Betting output columns
BET_COLUMNS = [
    'player_name',
    'game_date',
    'team_abbreviation',
    'opponent',
    'betting_line',
    'direction',  # 'over' or 'under'
    'mean_pred',
    'prob_win',
    'edge',
    'kelly_size',
    'confidence_score',
    'bookmaker'
]

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []

    # Check required files exist
    if not MASTER_FEATURES_PATH.exists():
        errors.append(f"Master features not found: {MASTER_FEATURES_PATH}")

    # Check API key
    if not ODDS_API_KEY or ODDS_API_KEY == 'your_api_key_here':
        errors.append("ODDS_API_KEY not set in environment")

    # Check parameters
    if TRAINING_WINDOW_YEARS < 1:
        errors.append("TRAINING_WINDOW_YEARS must be >= 1")

    if CV_FOLDS < 5:
        errors.append("CV_FOLDS must be >= 5")

    if not (0 < KELLY_FRACTION <= 1):
        errors.append("KELLY_FRACTION must be in (0, 1]")

    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))

    return True

# Validate on import
try:
    validate_config()
except ValueError as e:
    # Don't raise during import, let modules handle it
    import warnings
    warnings.warn(str(e))
