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
uv run feature_engineering/rolling_features.py      # Recent performance, trends
uv run feature_engineering/matchup_features.py      # Opponent defense, pace
uv run feature_engineering/contextual_features.py   # Home/away, rest, timing
uv run feature_engineering/advanced_metrics.py      # CTG advanced stats
uv run feature_engineering/position_features.py     # Position normalization [HIGH IMPACT]
uv run feature_engineering/injury_features.py       # Injury tracking [HIGH IMPACT]

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
# Preview execution plan (dry run)
uv run feature_engineering/run_pipeline.py --dry-run

# Resume from last failure (skips completed stages)
uv run feature_engineering/run_pipeline.py --skip-completed

# Run only specific stages
uv run feature_engineering/run_pipeline.py --only rolling_features,matchup_features

# Run from a specific stage onward
uv run feature_engineering/run_pipeline.py --from contextual_features

# Run up to a specific stage
uv run feature_engineering/run_pipeline.py --to build_features

# Reset state and start fresh
uv run feature_engineering/run_pipeline.py --clean-start

# See all available options
uv run feature_engineering/run_pipeline.py --help
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
```

**Key configuration options:**
- `enabled`: Set to `false` to skip a stage entirely
- `critical`: Set to `true` to stop pipeline on failure (default: `false`)
- `depends_on`: Files that must exist before running
- `outputs`: Files that should be created
- `estimated_duration`: For progress tracking

**Global settings:**
```yaml
config:
  fail_fast: true  # Stop on first error (vs continue with warnings)
  log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_to_file: true  # Save logs to logs/ directory
  max_parallel_jobs: 6  # For future parallel execution
```

### State Management

**State file:** `.pipeline_state.json` (in project root)

The pipeline tracks:
- Which stages have completed successfully
- Execution timestamps and durations
- Output files created
- Error messages from failures

**Example state:**
```json
{
  "version": "1.0.0",
  "last_run_start": "2025-10-29T01:35:29",
  "last_run_complete": "2025-10-29T01:35:31",
  "stages": {
    "rolling_features": {
      "status": "completed",
      "completed_at": "2025-10-29T01:35:30",
      "duration_seconds": 0.5,
      "outputs": ["../data/feature_tables/rolling_features.parquet"]
    }
  }
}
```

**To reset state:**
```bash
rm .pipeline_state.json
# OR
uv run feature_engineering/run_pipeline.py --clean-start
```

### Execution Logs

**Location:** `logs/pipeline_YYYYMMDD_HHMMSS.log`

Each pipeline run creates a detailed log file with:
- Stage start/completion times with duration
- Output file paths and sizes
- Error messages and tracebacks
- Validation results
- Feature counts and data quality metrics

**To view latest log:**
```bash
# List logs (newest first)
ls -lt logs/

# Follow latest log in real-time
tail -f logs/pipeline_*.log
```

### Failure Handling and Recovery

**If a stage fails:**

1. **Check the log file** in `logs/` for error details
2. **Examine the error message** - pipeline provides actionable guidance
3. **Fix the underlying issue** (see Troubleshooting section below)
4. **Resume with `--skip-completed`:**

```bash
uv run feature_engineering/run_pipeline.py --skip-completed
```

**Example failure recovery:**
```bash
# Run 1: Fails at advanced_metrics
uv run feature_engineering/run_pipeline.py
# ❌ advanced_metrics FAILED (CTG data missing)
# ✅ rolling_features completed
# ✅ matchup_features completed
# ✅ contextual_features completed

# Fix: Add CTG CSV files to data/ctg_data_organized/

# Run 2: Resume from failure point
uv run feature_engineering/run_pipeline.py --skip-completed
# ⏭️  Skipping rolling_features (already completed)
# ⏭️  Skipping matchup_features (already completed)
# ⏭️  Skipping contextual_features (already completed)
# ▶️  Running advanced_metrics
# ▶️  Running position_features
# ▶️  Running injury_features
# ▶️  Running build_features
# ▶️  Running validate_features
```

### Common Workflow Examples

**Scenario 1: Fresh run with new data**
```bash
# 1. Collect data (6-8 hours for full dataset)
uv run feature_engineering/data_loader.py

# 2. Run full pipeline
uv run feature_engineering/run_pipeline.py
```

**Scenario 2: Testing with sample data**
```bash
# 1. Edit data_loader.py: SAMPLE_SIZE = 10
# 2. Collect sample data (2-5 min)
uv run feature_engineering/data_loader.py

# 3. Preview pipeline execution
uv run feature_engineering/run_pipeline.py --dry-run

# 4. Run pipeline on sample
uv run feature_engineering/run_pipeline.py
```

**Scenario 3: Debugging one feature module**
```bash
# Option 1: Run only that stage via pipeline
uv run feature_engineering/run_pipeline.py --only rolling_features

# Option 2: Run script directly for more control
uv run feature_engineering/rolling_features.py
```

**Scenario 4: Add new feature category**
```bash
# 1. Create new feature script
# feature_engineering/my_new_features.py

# 2. Add to pipeline.yaml (copy existing stage and modify)

# 3. Test new stage only
uv run feature_engineering/run_pipeline.py --only my_new_features

# 4. Rebuild master features
uv run feature_engineering/run_pipeline.py --from build_features
```

**Scenario 5: Pipeline failed mid-execution**
```bash
# Check state to see what completed
cat .pipeline_state.json

# Check logs for error details
tail -100 logs/pipeline_*.log

# Fix the issue, then resume
uv run feature_engineering/run_pipeline.py --skip-completed
```

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
- **advanced_metrics.py**: Usage, efficiency, playmaking from CTG data
- **position_features.py**: Position-specific z-scores, percentiles (3-5% RMSE improvement)
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

1. Create `feature_engineering/new_category_features.py`:
```python
from data_loader import load_player_gamelogs

def calculate_new_features(df):
    """Calculate features"""
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
    features = df[['player_id', 'game_id', 'game_date']].copy()

    # Add your features here
    # ALWAYS use .shift(1) for rolling operations
    features['new_feature'] = df.groupby('player_id')['stat'].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )

    return features

def build_new_features():
    df = load_player_gamelogs()
    features = calculate_new_features(df)

    output_path = FEATURE_DIR / "new_category_features.parquet"
    features.to_parquet(output_path, index=False)
    return features

if __name__ == "__main__":
    build_new_features()
```

2. Update `build_features.py`:
```python
feature_tables = [
    # ... existing tables ...
    load_feature_table("new_category_features.parquet"),  # Add your table
]
```

3. Run your script, then rebuild:
```bash
uv run feature_engineering/new_category_features.py
uv run feature_engineering/build_features.py
uv run feature_features.py
```

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

### Grain Integrity
- All feature tables MUST have exactly one row per `[player_id, game_id, game_date]`
- Merges use `validate='1:1'` to enforce this
- Duplicates will break the pipeline

### CTG Data Handling
- CTG data is season-level aggregates, not game-level
- When merging to game-level, same season values broadcast to all games
- This is intentional but means CTG features don't change game-to-game within a season
- To avoid leakage, consider using previous season's CTG data only

## Expected Performance

With all features (165+):
- **R² ~ 0.82-0.86**, **RMSE ~ 3.5-4.0** (with XGBoost)
- Position normalization adds 3-5% improvement
- Injury tracking adds 4-6% improvement
- Total improvement from baseline: 40-45%

## Troubleshooting

### Pipeline Failures

#### "Stage X failed" - How to resume
```bash
# Check the log file for error details
tail -100 logs/pipeline_*.log | grep -A 5 ERROR

# Fix the underlying issue, then resume from where it stopped
uv run feature_engineering/run_pipeline.py --skip-completed
```

#### Pipeline state corrupted or inconsistent
```bash
# Check current state
cat .pipeline_state.json

# Reset state and start fresh
rm .pipeline_state.json
uv run feature_engineering/run_pipeline.py
# OR
uv run feature_engineering/run_pipeline.py --clean-start
```

#### Dependency validation fails
```
ERROR: Dependency not found: ../data/nba_api/player_games.parquet
ERROR: Cannot proceed with stage 'rolling_features'
```

**Solution:** Run the missing dependency first:
```bash
# Collect NBA API data
uv run feature_engineering/data_loader.py
```

#### Stage keeps failing after fix
```bash
# 1. Reset just that stage in state file
# Edit .pipeline_state.json and remove the failed stage

# 2. Or force re-run specific stage
uv run feature_engineering/run_pipeline.py --only rolling_features

# 3. Or run the script directly for more detailed error messages
uv run feature_engineering/rolling_features.py
```

#### Pipeline runs but creates empty output files
```bash
# Check data loader created valid input
ls -lh data/nba_api/player_games.parquet

# Verify input data has rows
uv run python
>>> import pandas as pd
>>> df = pd.read_parquet('data/nba_api/player_games.parquet')
>>> len(df)  # Should be > 0

# Check if SAMPLE_SIZE is set too small in data_loader.py
```

### Feature Engineering Failures

#### "No module named 'data_loader'"
Feature engineering scripts import from each other. Run from project root:
```bash
cd /Users/diyagamah/Documents/NBA_PRA
uv run feature_engineering/rolling_features.py
```

### "Feature table not found"
Run feature scripts in order. Each depends on `player_games.parquet` existing.

### "Grain violation" / "Duplicate rows"
A feature calculation created multiple rows for same player-game. Debug:
```python
df.groupby(['player_id', 'game_id']).size().max()  # Should be 1
duplicates = df[df.duplicated(['player_id', 'game_id'], keep=False)]
```

### "Rate limit exceeded" (NBA API)
Increase delay in `data_loader.py`:
```python
gamelog = playergamelog.PlayerGameLog(...)
time.sleep(3.0)  # Increase from 2.0 to 3.0
```

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

## Current State

- ✅ Data collection pipeline complete
- ✅ 6 feature engineering modules implemented (165+ features)
- ✅ Automated pipeline orchestrator complete (YAML-based with state management)
- ✅ Validation framework in place
- ✅ Train/val/test split logic implemented
- ⏳ Model training pending
- ⏳ Backtesting framework pending

**Latest improvements:**
- **Pipeline orchestrator** (v1.0.0): Single command to run all 8 feature engineering stages with dependency checking, state management, and comprehensive logging
- **Position normalization features**: 14 position-specific features with z-scores and percentiles (expected 3-5% RMSE improvement)
- **Injury tracking features**: 16 DNP and recovery tracking features (expected 4-6% RMSE improvement)
- **Data leakage fixes**: Critical fixes in contextual_features.py and matchup_features.py

## Git Configuration

From `.claude/CLAUDE.md` (user preferences):
- When committing: Credit only diyagamah (hivin.diyagama@tabcorp.com.au)
- Never use "Co-Authored-By" attribution
- No fallback logic to bypass errors (prefer fixing root cause)
- Update existing files rather than creating versioned copies
