"""
Configuration Module for NBA PRA Feature Engineering

Centralized configuration for paths, constants, and parameters used across
feature engineering modules. Provides single source of truth for all configuration.
"""

from pathlib import Path
from typing import List


# ==============================================================================
# Project Paths
# ==============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Data subdirectories
NBA_API_DIR = DATA_DIR / "nba_api"
CTG_DATA_DIR = DATA_DIR / "ctg_data_organized"
FEATURE_DIR = DATA_DIR / "feature_tables"
HISTORICAL_ODDS_DIR = DATA_DIR / "historical_odds"

# Create directories if they don't exist
NBA_API_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Data Grain (Critical - DO NOT MODIFY)
# ==============================================================================

# All feature tables MUST maintain this grain
GRAIN_COLUMNS: List[str] = ['player_id', 'game_id', 'game_date']


# ==============================================================================
# Rolling Window Parameters
# ==============================================================================

# Window sizes for rolling averages (in games)
ROLLING_WINDOWS: List[int] = [5, 10, 20]

# Half-lives for exponentially weighted moving averages (in games)
EWMA_HALFLIVES: List[int] = [3, 7, 14]

# Stats to calculate rolling features for
ROLLING_STATS: List[str] = ['pra', 'points', 'rebounds', 'assists']


# ==============================================================================
# Injury & DNP Detection Parameters
# ==============================================================================

# Days between games threshold to detect missed games
INJURY_GAP_THRESHOLD_DAYS: int = 7

# Maximum days for injury window (beyond this is likely season end/start)
INJURY_MAX_DAYS: int = 30

# Minutes threshold to detect DNP (Did Not Play)
DNP_MINUTES_THRESHOLD: float = 3.0

# DNP rate threshold for load management detection
LOAD_MANAGEMENT_RATE_THRESHOLD: float = 0.15

# Expected games per 30 days for healthy player
EXPECTED_GAMES_PER_30_DAYS: int = 12


# ==============================================================================
# Position Classification Parameters
# ==============================================================================

# Position classification thresholds (based on composite scores)
POSITION_THRESHOLDS = {
    'center_score': 3.0,     # big_score > 3 → Center
    'forward_score': 1.0,    # big_score > 1 → Forward
    'guard_score': 2.0,      # guard_score > 2 → Guard
}

# Position-specific elite performance thresholds
POSITION_ELITE_THRESHOLDS = {
    'Guard': {'points': 20, 'assists': 7, 'rebounds': 5},
    'Wing': {'points': 18, 'assists': 5, 'rebounds': 6},
    'Forward': {'points': 18, 'assists': 4, 'rebounds': 8},
    'Center': {'points': 16, 'assists': 3, 'rebounds': 10}
}


# ==============================================================================
# Matchup Features Parameters
# ==============================================================================

# Window for opponent defensive stats rolling averages
OPPONENT_DEFENSE_WINDOW: int = 10

# Expected games per week (for opponent scheduling)
GAMES_PER_WEEK: int = 4


# ==============================================================================
# Contextual Features Parameters
# ==============================================================================

# Days for "recent" performance windows
RECENT_PERFORMANCE_DAYS: int = 7

# Default rest days at season start (no previous game)
DEFAULT_REST_DAYS: int = 3


# ==============================================================================
# Advanced Metrics Parameters
# ==============================================================================

# CTG Season Mapping (current season → previous season for CTG lookup)
# Prevents temporal leakage by using only historical CTG data
CTG_SEASON_MAPPING = {
    '2025-26': '2024-25',  # Added for new season data
    '2024-25': '2023-24',
    '2023-24': '2022-23',
    '2022-23': '2021-22',
    '2021-22': '2020-21',
    '2020-21': '2019-20',
    '2019-20': '2018-19',
    '2018-19': '2017-18',
    '2017-18': '2016-17',
    '2016-17': '2015-16',
    '2015-16': '2014-15',
    '2014-15': '2013-14',
    '2013-14': '2012-13',
    '2012-13': '2011-12',
    '2011-12': '2010-11',
    '2010-11': '2009-10',
}

# Usage rate thresholds for role classification
HIGH_USAGE_THRESHOLD: float = 0.25  # Star/primary option
LOW_USAGE_THRESHOLD: float = 0.18   # Role player

# Assist rate threshold for primary playmaker
PRIMARY_PLAYMAKER_THRESHOLD: float = 0.25

# Three-point attempt rate for specialist classification
THREE_POINT_SPECIALIST_THRESHOLD: float = 0.50


# ==============================================================================
# Validation Parameters
# ==============================================================================

# Expected ranges for validation (used in feature validation)
VALIDATION_RANGES = {
    'usage_rate': (0.10, 0.35),
    'true_shooting_pct': (0.30, 0.80),
    'availability_score': (0, 100),
    'pra_position_percentile': (0, 100),
    'minutes_float': (0, 48),  # Max NBA game is 48 minutes + OT
    'points': (0, 100),        # Reasonable max for single game
    'rebounds': (0, 30),
    'assists': (0, 30),
}


# ==============================================================================
# NBA API Parameters
# ==============================================================================

# Rate limiting for NBA API calls (seconds between requests)
NBA_API_RATE_LIMIT: float = 2.0

# Sample size for testing (set to None for production)
SAMPLE_SIZE = None  # Change to 10 for testing

# Seasons to fetch
SEASONS_TO_FETCH: List[str] = [
    '2024-25', '2023-24', '2022-23', '2021-22', '2020-21',
    '2019-20', '2018-19', '2017-18', '2016-17', '2015-16',
    '2014-15', '2013-14', '2012-13', '2011-12', '2010-11'
]


# ==============================================================================
# Feature Engineering Metadata
# ==============================================================================

FEATURE_VERSION = "2.0.0"  # Updated after refactoring
LAST_UPDATED = "2025-01-29"


# ==============================================================================
# Logging Configuration
# ==============================================================================

LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
