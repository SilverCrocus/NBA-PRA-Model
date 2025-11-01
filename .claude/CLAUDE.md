# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA PRA (Points + Rebounds + Assists) prediction model using time-series feature engineering and machine learning. The project predicts actual PRA values (regression) for NBA players using historical performance data, opponent matchups, injury patterns, and position-specific normalization.

## Development Setup

```bash
# Install dependencies (uv is preferred, faster than pip)
uv sync

# For pip users
pip install -e .
```

## Core Architecture

### Data Flow Pipeline

```
NBA API Data Collection → Feature Engineering (6 modules) → Master Features → Train/Val/Test Splits → Model Training
```

**Critical concept**: All feature engineering uses **chronological time-series processing** with strict temporal validation to prevent data leakage. Features MUST NOT use future information.

### Feature Engineering Architecture

The system uses a **modular table-based architecture** where each feature category is:
1. Calculated independently in its own Python module
2. Saved as a separate parquet file in `data/feature_tables/`
3. Joined at the end on a consistent grain: `[player_id, game_id, game_date]`

**Why this matters**: You can debug, update, or add features to any category without regenerating everything. Each module is ~300-400 lines and self-contained.

### Data Grain (Critical)

**Everything** operates on `[player_id, game_id, game_date]` grain. This means:
- One row = one player's performance in one specific game
- All feature tables MUST merge on these three columns
- Any operation violating this grain will break the pipeline

### Temporal Leakage Prevention (Critical)

**Golden Rule**: Features can only use information available BEFORE the game being predicted.

**How it's enforced**:
- All rolling calculations use `.shift(1)` to exclude current game
- Never use `.rolling()` directly without `.shift(1)` first
- Chronological train/test splits ONLY (no random shuffling)
- No k-fold cross-validation on time-series data

**Example of correct implementation**:
```python
# CORRECT - excludes current game
df['pra_avg_last10'] = (
    df.groupby('player_id')['pra']
    .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
)

# WRONG - includes current game (data leakage!)
df['pra_avg_last10'] = (
    df.groupby('player_id')['pra']
    .transform(lambda x: x.rolling(10, min_periods=1).mean())
)
```

## Feature Engineering Workflow

### Automated Pipeline (RECOMMENDED)

**Use the pipeline orchestrator** - automates all steps with dependency checking, state management, and error handling:

```bash
# Run complete feature engineering pipeline
uv run feature_engineering/run_pipeline.py
```

**See "Automated Pipeline Orchestrator" section below for details.**

### Manual Execution (For Debugging)

If you need manual control or are debugging specific modules, run scripts in this **exact order**:

```bash
# 1. Data collection (once, takes 6-8 hours for full dataset)
# For testing: edit data_loader.py, set SAMPLE_SIZE = 10
uv run feature_engineering/data_loader.py

# 2. Generate feature tables (each runs independently)
uv run feature_engineering/features/rolling_features.py      # Recent performance, trends
uv run feature_engineering/features/matchup_features.py      # Opponent defense, pace
uv run feature_engineering/features/contextual_features.py   # Home/away, rest, timing
uv run feature_engineering/features/advanced_metrics.py      # CTG advanced stats
uv run feature_engineering/features/position_features.py     # Position normalization [HIGH IMPACT]
uv run feature_engineering/features/injury_features.py       # Injury tracking [HIGH IMPACT]

# 3. Join all features
uv run feature_engineering/build_features.py

# 4. Validate everything
uv run feature_engineering/validate_features.py

# 5. Create train/val/test splits
uv run model_training/train_split.py
```

**Why this order matters**: Each feature module depends on `data/nba_api/player_games.parquet` from data_loader. The build step requires all feature tables to exist.

## Automated Pipeline Orchestrator

**Version:** 1.0.0
**Files:** `feature_engineering/pipeline.yaml`, `feature_engineering/run_pipeline.py`

The pipeline orchestrator automates all 8 feature engineering stages with dependency checking, state management, and comprehensive error handling.

### Why Use the Pipeline Orchestrator?

**Benefits over manual execution:**
- ✅ Automatic dependency checking (ensures input files exist)
- ✅ State management (resume from failures with `--skip-completed`)
- ✅ Progress tracking with detailed logging
- ✅ Execution reports saved to `logs/` directory
- ✅ Dry-run mode to preview execution plan
- ✅ Partial execution (run specific stages with `--only`)
- ✅ Error handling (critical vs non-critical stages)
- ✅ Single command instead of 8 separate commands

### Quick Start

```bash
# Run the complete pipeline (RECOMMENDED)
uv run feature_engineering/run_pipeline.py
```

**This executes all 8 stages:**
1. `rolling_features` - Moving averages, trends, volatility (50 features)
2. `matchup_features` - Opponent defense and pace (9 features)
3. `contextual_features` - Game context and timing (23 features)
4. `advanced_metrics` - CTG advanced statistics (22 features)
5. `position_features` - Position normalization (14 features)
6. `injury_features` - DNP and recovery tracking (16 features)
7. `build_features` - Join all feature tables (creates master_features.parquet)
8. `validate_features` - Comprehensive validation (data leakage checks)

### Common Usage Patterns

```bash
uv run feature_engineering/run_pipeline.py --dry-run          # Preview execution
uv run feature_engineering/run_pipeline.py --skip-completed   # Resume from failure
uv run feature_engineering/run_pipeline.py --only rolling_features   # Run specific stage
# See all options: --help
```

### Pipeline Configuration

Edit `feature_engineering/pipeline.yaml` to customize:

```yaml
stages:
  - name: "rolling_features"
    script: "rolling_features.py"
    description: "Calculate rolling averages, EWMA, trends"

    enabled: true  # Set to false to skip this stage
    critical: false  # Set to true to fail entire pipeline if this fails

    depends_on:
      - "../data/nba_api/player_games.parquet"

    outputs:
      - "../data/feature_tables/rolling_features.parquet"

    estimated_duration: "2-5 minutes"

# Global settings:
```yaml
config:
  fail_fast: true  # Stop on first error (vs continue with warnings)
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: true  # Save logs to logs/ directory
  max_parallel_jobs: 6  # For future parallel execution
```

### State Management

Pipeline tracks completed stages in `.pipeline_state.json`. Reset with: `rm .pipeline_state.json` or `--clean-start`.

### Execution Logs

Logs saved to `logs/pipeline_YYYYMMDD_HHMMSS.log` with stage timing, errors, and validation results.

```bash
tail -f logs/pipeline_*.log  # Follow in real-time
```

### Failure Handling

1. Check log file in `logs/` for error details
2. Fix the underlying issue (see Troubleshooting)
3. Resume: `uv run feature_engineering/run_pipeline.py --skip-completed`

### Common Workflows

**Fresh run:** `uv run feature_engineering/data_loader.py` → `uv run feature_engineering/run_pipeline.py`

**Debug one module:** `uv run feature_engineering/run_pipeline.py --only rolling_features` or run script directly

**Resume after failure:** Check logs (`tail -100 logs/pipeline_*.log`), fix issue, resume with `--skip-completed`

### When to Use Automated vs Manual

**Use the automated pipeline when:**
- ✅ Running the full feature engineering workflow
- ✅ You want automatic error handling and state tracking
- ✅ You need to resume after failures
- ✅ You want execution logs for auditing
- ✅ Running in production or CI/CD

**Use manual step-by-step when:**
- ✅ Developing/debugging a single feature module
- ✅ Testing new features on a subset of data
- ✅ Investigating specific calculation issues
- ✅ Need immediate feedback on errors
- ✅ Iterating rapidly on one module

## Key Components

### Shared Modules

#### utils.py
**Purpose**: Centralized utilities to eliminate code duplication (removed 400+ lines of duplicate code).

**Key functions (9 total)**:
```python
# Data conversion
convert_minutes_to_float(minutes)  # Handles "25:30", 25.5, or "25"

# Validation functions
validate_not_empty(df, function_name)  # Ensures DataFrame not empty
validate_required_columns(df, required_cols, function_name)  # Column checking
validate_grain_uniqueness(df, grain_cols)  # No duplicate player-game rows

# Feature creation
create_feature_base(df)  # Returns [player_id, game_id, game_date] base

# Logging
setup_logging(name, level)  # Consistent logging configuration

# File operations
ensure_directory_exists(path)  # Creates directory if needed
load_parquet_safe(path)  # Safe parquet loading with error handling
```

**Import in feature modules**:
```python
from utils import (
    validate_not_empty,
    validate_required_columns,
    validate_grain_uniqueness,
    create_feature_base
)
```

#### config.py
**Purpose**: Centralized configuration for paths, constants, and parameters.

**Key constants**:
```python
# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURE_DIR = DATA_DIR / "feature_tables"

# Data grain (DO NOT MODIFY)
GRAIN_COLUMNS = ['player_id', 'game_id', 'game_date']

# Rolling window parameters (tunable)
ROLLING_WINDOWS = [5, 10, 20]
EWMA_HALFLIVES = [3, 7, 14]

# Injury detection thresholds
INJURY_GAP_THRESHOLD_DAYS = 7
DNP_MINUTES_THRESHOLD = 3.0
LOAD_MANAGEMENT_RATE_THRESHOLD = 0.15

# CTG season mapping (prevents temporal leakage)
CTG_SEASON_MAPPING = {
    '2024-25': '2023-24',  # Use previous season data
    '2023-24': '2022-23',
    '2022-23': '2021-22',
    # ... maps current season → previous season
}
```

**Import in feature modules**:
```python
from config import (
    FEATURE_DIR,
    GRAIN_COLUMNS,
    ROLLING_WINDOWS,
    CTG_SEASON_MAPPING
)
```

### data_loader.py
- Fetches NBA API box scores with 2-second rate limiting (required by NBA.com)
- Consolidates CleaningTheGlass (CTG) CSV data
- Creates base `player_games.parquet` with actual PRA values
- **SAMPLE_SIZE parameter**: Set to 10 for testing, None for production

### Feature Modules (6 files)
Each creates 15-50 features and saves to `data/feature_tables/`:
- **rolling_features.py**: Moving averages, EWMA, trends, volatility
- **matchup_features.py**: Opponent defensive stats (aggregated to team-game level first)
- **contextual_features.py**: Rest days, season timing, form indicators
- **advanced_metrics.py**: Usage, efficiency, playmaking from CTG data (uses previous season data)
- **position_features.py**: Position-specific z-scores, percentiles (3-5% RMSE improvement, vectorized calculation)
- **injury_features.py**: DNP tracking, recovery patterns (4-6% RMSE improvement)

### build_features.py
- Loads all feature tables
- Validates grain uniqueness (no duplicates)
- Merges on `[player_id, game_id, game_date]`
- Creates `master_features.parquet` (165+ features)
- Runs quality checks

### validate_features.py
- Checks for data leakage
- Validates grain uniqueness
- Analyzes feature distributions
- Checks missing value patterns
- **Run this before model training**

## Code Quality Standards

All feature modules follow these patterns:

```python
import logging
from typing import List, Optional
import pandas as pd
from utils import validate_not_empty, validate_required_columns

logger = logging.getLogger(__name__)

def calculate_features(df: pd.DataFrame, windows: List[int] = [5, 10]) -> pd.DataFrame:
    """Calculate rolling features

    Args:
        df: Player game data with required columns
        windows: Rolling window sizes

    Returns:
        DataFrame with features on grain [player_id, game_id, game_date]
    """
    # Input validation
    validate_not_empty(df, 'calculate_features')
    validate_required_columns(df, ['player_id', 'pra'], 'calculate_features')

    # Logging (replaced 249 print statements)
    logger.info(f"Processing {len(df)} games")

    # Specific exception handling (no bare except)
    try:
        result = process_data(df)
    except FileNotFoundError:
        logger.error("Required file not found")
        raise  # No fallback logic per user preference

    return result
```

**Standards applied:** Type hints (68 functions), input validation (33 functions), logging system, specific exceptions, comprehensive docstrings.

### Feature Module Save Pattern

All feature modules MUST sort by grain before saving:

```python
# Save features (REQUIRED: sort first for temporal validation)
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Why:** Validation enforces per-player monotonic game_date ordering to prevent temporal leakage. Either sort order is acceptable: `['player_id', 'game_date']` (preferred for consistency with grain) or `['game_date', 'player_id']` (used by rolling_features), as long as each player's games are chronologically ordered.

## Testing Infrastructure

**72 tests, 100% pass rate** in `tests/` directory.

```bash
pytest tests/                                # Run all tests
pytest tests/test_leakage_prevention.py      # Specific file
```

**Critical test:** `test_leakage_prevention.py` validates features exclude current game.

**Test files:** leakage_prevention, utils, rolling_features, position_features, injury_features, build_features, matchup_features, integration.

**When to run:** After modifying calculations, before committing, after fixing leakage issues.

## Common Operations

### Running the Feature Pipeline

**Use the automated pipeline orchestrator (recommended):**
```bash
# Full pipeline
uv run feature_engineering/run_pipeline.py

# With options
uv run feature_engineering/run_pipeline.py --dry-run           # Preview
uv run feature_engineering/run_pipeline.py --skip-completed    # Resume
uv run feature_engineering/run_pipeline.py --only rolling_features,matchup_features
```

**See "Automated Pipeline Orchestrator" section for complete documentation.**

### Adding a New Feature Category

1. Create feature script in `feature_engineering/features/` following existing module patterns
2. Use `.shift(1)` for rolling operations to prevent data leakage
3. Update `build_features.py` to load the new table
4. Optional: Add stage to `pipeline.yaml` for automated pipeline
5. Run script, rebuild features, validate

### Testing with Subset of Data

Edit `feature_engineering/data_loader.py`:
```python
SAMPLE_SIZE = 10  # Change from None to 10 for testing
```

This fetches only 10 players (~1000 games) instead of all players (~150,000 games). Useful for:
- Testing new features quickly
- Debugging issues
- Validating pipeline without 6-hour wait

### Debugging Feature Calculations

Since each feature module saves its own table:
```bash
uv run python
>>> import pandas as pd
>>> df = pd.read_parquet('data/feature_tables/rolling_features.parquet')
>>> df.head()  # Inspect features
>>> df.info()  # Check data types, missing values
```

## Data Sources

### NBA API (via nba_api library)
- Box scores: points, rebounds, assists, minutes, shooting stats
- Rate limited: 2 seconds between requests (enforced in code)
- Free, no authentication required
- Historical data back to ~2003

### CleaningTheGlass (CTG)
- Advanced stats: usage rate, assist %, turnover %, on/off court metrics
- CSV files in `data/ctg_data_organized/players/`
- Season-level data (not game-level)
- Merged by player name and season (watch for name mismatches)

### Historical Odds
- PRA betting lines from multiple sportsbooks
- In `data/historical_odds/` (currently not used in features)

## Critical Constraints

### Data Leakage Prevention
- **Never** include current game stats in features for that game
- **Always** use `.shift(1)` before rolling operations
- **Always** sort by `['player_id', 'game_date']` before groupby operations
- **Never** use random train/test splits (chronological only)

**Fixed leakage bugs:**
1. Position z-score: Uses double-lagged PRA (position_features.py:134-147)
2. CTG alignment: Maps to previous season via CTG_SEASON_MAPPING (advanced_metrics.py:45-80)
3. Availability score: Excludes current game from 30-day window (injury_features.py:260-282)
4. Position percentile: Vectorized with proper lagging (position_features.py:152-207)

### Grain Integrity
- All feature tables MUST have exactly one row per `[player_id, game_id, game_date]`
- Merges use `validate='1:1'` to enforce this
- Duplicates will break the pipeline
- Use `utils.validate_grain_uniqueness()` to check

### CTG Data Handling
- CTG data is season-level aggregates, not game-level
- **IMPORTANT**: Always use `CTG_SEASON_MAPPING` from config.py
- Maps current season → previous season (e.g., '2024-25' → '2023-24')
- This prevents temporal leakage by ensuring CTG features use only historical data
- When merging to game-level, previous season values broadcast to all games that season

## Expected Performance

With all features (165+):
- **R² ~ 0.82-0.86**, **RMSE ~ 3.5-4.0** (with XGBoost)
- Position normalization adds 3-5% improvement
- Injury tracking adds 4-6% improvement
- Total improvement from baseline: 40-45%

## Troubleshooting

**Pipeline stage failed:** Check logs (`tail -100 logs/pipeline_*.log`), fix issue, resume with `--skip-completed`

**State corrupted:** Reset with `rm .pipeline_state.json` or `--clean-start`

**Missing dependency:** Run `uv run feature_engineering/data_loader.py` first

**Stage keeps failing:** Run script directly: `uv run feature_engineering/features/rolling_features.py`

**Empty outputs:** Check input data exists and has rows, verify SAMPLE_SIZE in data_loader.py

**"No module named 'data_loader'":** Run from project root

**"Grain violation":** Debug with `df.groupby(['player_id', 'game_id']).size().max()` (should be 1)

**"Rate limit exceeded":** Increase delay in data_loader.py from 2.0 to 3.0 seconds

## Project-Specific Patterns

### Using .shift(1) correctly
When calculating ANY rolling/cumulative feature:
```python
# Pattern for all rolling features
df.groupby('player_id')['stat'].transform(
    lambda x: x.shift(1).rolling(window).mean()
)
```

The `.shift(1)` moves everything down one row, so row N gets the average of rows N-11 to N-1 (excluding N).

### Handling CTG name mismatches
Player names don't always match between NBA API and CTG:
- NBA API: "Luka Doncic"
- CTG: "Luka Dončić" (with diacritic)

Current solution: Left join (keeps all NBA API data, fills NaN for CTG mismatches). Future: Add name standardization.

### Feature validation pattern
All feature modules include a `validate_*_features()` function that checks:
- Value ranges are reasonable
- No infinite values
- Required columns exist
- Distributions make sense

Always call this before saving features.

## Model Training

### Overview

The model training pipeline uses **XGBoost** with time-series cross-validation to predict NBA player PRA values. The system is designed for production deployment with robust validation and comprehensive experiment tracking.

**Key Features:**
- ✅ Time-series CV with 19 folds (rolling 3-year windows)
- ✅ Ensemble averaging for improved predictions
- ✅ MLflow experiment tracking
- ✅ Automated pipeline orchestration
- ✅ Production-ready artifacts

### Training Pipeline Architecture

**Version:** 1.0.0
**Files:** `model_training/pipeline.yaml`, `model_training/run_pipeline.py`

```
Master Features → Train/Val/Test Splits (19 CV Folds) → Model Training → Ensemble → MLflow Tracking
```

### Time-Series Cross-Validation Strategy

**Critical concept**: Uses chronological splits with rolling windows to prevent data leakage and validate temporal generalization.

**CV Configuration** (from `model_training/config.py`):
```python
CV_TRAINING_WINDOW_YEARS = 3      # 3-year rolling training window
CV_GAP_GAMES = 15                 # 15-game gap between train/test (per player)
CV_MIN_TEST_GAMES = 1000          # Minimum 1000 games per test fold
CV_VAL_SPLIT = 0.2                # 20% of training data for validation
```

**How it works:**
1. Dataset spans 22 NBA seasons (2003-04 through 2024-25)
2. Creates 19 folds with rolling 3-year training windows:
   - Fold 0: Train[2003-06] → Test[2006-07]
   - Fold 1: Train[2004-07] → Test[2007-08]
   - ... (continues through 2024-25)
3. Each fold splits into Train (80%), Val (20%), Test
4. 15-game gap per player between train/test prevents leakage
5. All 19 models averaged into ensemble

**Why this approach:**
- ✅ Validates model across different NBA eras
- ✅ Tests generalization to future seasons
- ✅ Prevents overfitting to specific time periods
- ✅ Ensemble reduces variance across folds

### Quick Start

**Run complete training pipeline:**
```bash
# Option 1: Use the pipeline orchestrator (RECOMMENDED)
uv run model_training/run_pipeline.py

# Option 2: Manual execution
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python model_training/training.py --cv --model-type xgboost
```

**This executes:**
1. `create_cv_splits` - Creates 19 time-series CV folds
2. `train_cv_ensemble` - Trains 19 models and creates ensemble

### Training Workflow Details

#### Step 1: Create CV Splits

```bash
uv run model_training/train_split.py --cv-mode
```

**What it does:**
- Loads `data/feature_tables/master_features.parquet` (587,034 games)
- Detects 22 seasons of data
- Creates 19 CV folds with chronological splits
- Saves to `data/processed/cv_folds/fold_*/`

Creates `data/processed/cv_folds/fold_*/` with train/val/test splits for 19 folds.

#### Step 2: Train CV Ensemble

```bash
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python model_training/training.py --cv --model-type xgboost
```

**What it does:**
1. Loads all 19 CV folds
2. Trains XGBoost model on each fold (with early stopping)
3. Evaluates each model on its test set
4. Creates ensemble by averaging predictions
5. Logs to MLflow
6. Saves model and metrics

**Hyperparameters:** See `model_training/config.py` for XGBoost/LightGBM settings.

### Production Results

**Dataset:** 587k games, 2,280 players, 22 seasons, 134 features

**Performance:** MAE 2.99 ± 0.14, RMSE 4.25 ± 0.20, R² 0.868 ± 0.005 (ensemble of 19 CV folds, 1m43s training)

**Top features:** EWMA (3/7/14-game halflife), DNP tracking, minutes restriction, shooting efficiency

### Artifacts

1. **Ensemble model:** `models/xgboost_ensemble_19folds_20251029_210135.pkl` (19 trained models)
2. **CV metrics:** `logs/xgboost_cv_summary_20251029_210135.csv` (performance per fold)
3. **Feature importance:** `models/xgboost_ensemble_feature_importance_20251029_210135.csv` (rankings)
4. **MLflow:** `mlruns/433734130334159347/` (view: `mlflow ui`)
5. **Training logs:** `logs/training_cv_xgboost_20251029_205952.log`

### Making Predictions

```python
import pickle
from feature_engineering.run_pipeline import build_features_for_games

with open('models/xgboost_ensemble_19folds_20251029_210135.pkl', 'rb') as f:
    model = pickle.load(f)

features = build_features_for_games(new_game_data)  # Generate 134 features
predictions = model.predict(features)
```

**Important:** New data must go through same feature engineering pipeline.

### Model Validation Checklist

Before deploying predictions:

- ✅ **Temporal ordering**: Test games are chronologically after training
- ✅ **Feature availability**: All 134 features can be calculated
- ✅ **Data quality**: No missing values in critical features
- ✅ **Grain integrity**: One prediction per player-game
- ✅ **Performance monitoring**: Track actual vs predicted PRA

### Troubleshooting

#### "ModuleNotFoundError: No module named 'model_training'"

Training scripts use absolute imports. Set PYTHONPATH:
```bash
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python model_training/training.py --cv --model-type xgboost
```

#### "Dependency not found: fold_0/train.parquet"

Create CV splits first:
```bash
uv run model_training/train_split.py --cv-mode
```

#### "ValueError: DataFrame.dtypes for data must be int, float"

Ensure `season` and `season_detected` are in `EXCLUDE_COLUMNS` (already fixed in config.py).

#### Model predictions seem off

Check:
1. Feature engineering ran on new data (same pipeline)
2. All 134 features present
3. No data leakage (features use only past information)
4. Temporal alignment (using correct season's CTG data)


### Performance Benchmarks

**Training speed:**
- CV split creation: ~30 seconds (587k games)
- Single fold training: ~5 seconds (63k games)
- 19-fold ensemble: ~1 minute 43 seconds total
- Feature importance calculation: instant

**Prediction speed:**
- Single game: < 1ms
- 1000 games: ~50ms
- Full season (~30k games): ~1-2 seconds

**Memory usage:**
- Model size: ~2-5 MB (ensemble of 19 models)
- Feature data (587k games): ~170 MB parquet
- Peak training RAM: ~2-3 GB

## Backtesting System

### Overview

Walk-forward backtesting framework that validates model profitability on historical betting lines. Simulates production deployment with daily model retraining using 19-fold CV ensemble.

**Key Results (2024-25 season):**
- **Win Rate:** 77.3% (break-even: 52.4%)
- **ROI:** 47.6%
- **Predictions:** 25,926 total (3,813 with betting lines)
- **Statistical Significance:** p < 10⁻²²²

### Architecture

```
Historical Data → Walk-Forward Engine → Daily Retraining → Predictions → Betting Evaluation → Reports
```

**Files:** `backtest/` directory
- `run_backtest.py` - Entry point
- `walk_forward_engine.py` - Core engine (daily retraining)
- `betting_evaluator.py` - Bet decisions and profit calculation
- `data_loader.py` - Load features and odds
- `reporting.py` - Generate markdown reports
- `config.py` - Configuration parameters

### Walk-Forward Methodology

**Core Concept:** Predict each game day sequentially using only historical data available before that date.

**Process (per game day):**
1. Get games scheduled for Day N
2. Extract 3-year training window: [Day N - 3 years, Day N - 1]
3. Create 19 time-series CV folds from training data
4. Train 19 XGBoost models (ensemble)
5. Make predictions (average of 19 models)
6. Match to betting lines
7. Calculate profit/loss at -110 odds

**Configuration:**
```python
TRAINING_WINDOW_YEARS = 3  # Rolling window
CV_FOLDS = 19              # Ensemble size
TARGET_SEASON = "2024-25"   # Season to backtest
BETTING_ODDS = -110         # Standard American odds
```

### Running Backtest

```bash
# Full backtest (5-6 hours for full season)
uv run backtest/run_backtest.py

# Results saved to backtest/results/
# - backtest_report.md (summary)
# - daily_predictions.csv (all bets)
# - player_analysis.csv (per-player ROI)
# - betting_performance.csv (overall metrics)
```

### Key Features

**Temporal Validation:**
- ✅ Strict temporal ordering (no look-ahead)
- ✅ Daily retraining (adapts to latest info)
- ✅ 3-year rolling window (recency vs volume)

**Betting Simulation:**
- Matches predictions to historical odds
- Calculates win/loss at -110 odds
- Tracks edge size (prediction - line)
- Analyzes by player, edge size, date

**CTG Imputation:**
- Position-based imputation for missing CTG data
- Imputation flags (has_ctg_data, is_rookie)
- Position-relative features (vs_position_mean)

### Known Issues & Fixes

**ISSUE: Position Baseline Leakage** (Minor)
- **Location:** `walk_forward_engine.py:310-318`
- **Problem:** Baselines calculated from all pre-season data (not just training window)
- **Impact:** LOW (1-2% win rate adjustment)
- **Fix:** Calculate baselines from training window per prediction day

**ISSUE: Single Season Validation**
- **Problem:** Only tested on 2024-25 (need multi-season)
- **Impact:** MODERATE (can't assess consistency)
- **Recommended:** Run on 2022-23, 2023-24 for validation

**ISSUE: Transaction Costs Not Modeled**
- **Problem:** Assumes -110 odds, no slippage
- **Impact:** HIGH (overestimates profitability by ~5-10%)
- **Fix:** Use -115 odds, add 2% execution slippage

### Interpretation Guidelines

**Win Rate Expectations:**
- **Backtest:** 77% (likely inflated)
- **Expected Live:** 72-75% (regression to mean)
- **Conservative:** 68-72% (with market adaptation)
- **Break-even:** 52.4% at -110 odds

**Edge Calibration (Critical for Trust):**
- Large edges (3+ pts): 87% win rate
- Medium edges (2+ pts): 84% win rate
- Small edges (1+ pts): 81% win rate
- **Perfect monotonicity = well-calibrated model**

**Deployment Recommendations:**
- Start with 1/4 Kelly sizing (13% bankroll per bet)
- Require ≥2 point edge minimum
- Exclude players with <20 career games
- Monitor 30-day rolling win rate
- Stop if win rate < 68% for 30 days



## Git Configuration

From `.claude/CLAUDE.md` (user preferences):
<!-- - When committing: Credit only diyagamah (hivin.diyagama@tabcorp.com.au) -->
- Never use "Co-Authored-By" attribution
- No fallback logic to bypass errors (prefer fixing root cause)
- Update existing files rather than creating versioned copies
