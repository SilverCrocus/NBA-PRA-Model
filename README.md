# NBA PRA Prediction Model

A clean, modular system for predicting NBA player PRA (Points + Rebounds + Assists) using advanced feature engineering and machine learning.

## Project Structure

```
NBA_PRA/
├── data/
│   ├── ctg_data_organized/      # Your CleaningTheGlass CSV files
│   ├── historical_odds/         # PRA betting odds data
│   ├── nba_api/                 # NBA API box scores (created by data_loader)
│   └── feature_tables/          # Individual feature tables (parquet)
│
├── feature_engineering/
│   ├── data_loader.py           # Fetch NBA API data & load CTG data
│   ├── rolling_features.py      # Player recent performance features
│   ├── matchup_features.py      # Opponent-based features
│   ├── contextual_features.py   # Game context features
│   ├── advanced_metrics.py      # CTG advanced statistics
│   └── build_features.py        # Joins all feature tables
│
├── model_training/
│   └── train_split.py           # Creates train/val/test splits
│
├── backtest/                    # (Future: backtesting framework)
└── requirements.txt             # Python dependencies
```

## Setup

### 1. Install Dependencies

Using **uv** (recommended - faster than pip):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### 2. Collect NBA API Data

**IMPORTANT:** This will take several hours due to API rate limiting (2 seconds between requests).

For **testing** (quick run with 10 players):
```bash
# Edit feature_engineering/data_loader.py
# Change line: SAMPLE_SIZE = None  →  SAMPLE_SIZE = 10
uv run feature_engineering/data_loader.py
```

For **production** (all players, ~6-8 hours):
```bash
# Keep SAMPLE_SIZE = None in data_loader.py
uv run feature_engineering/data_loader.py
```

This will create: `data/nba_api/player_games.parquet` with box score data and actual PRA values.

## Feature Engineering Pipeline

### Automated Pipeline (Recommended)

The easiest way to run all feature engineering steps is using the **pipeline orchestrator**:

```bash
# Run the complete feature engineering pipeline
uv run feature_engineering/run_pipeline.py
```

**Key Features:**
- ✅ Runs all 8 steps automatically (rolling → matchup → contextual → advanced → position → injury → build → validate)
- ✅ Dependency checking (verifies input files exist)
- ✅ Progress tracking with detailed logging
- ✅ State management (can resume from failures)
- ✅ Comprehensive validation
- ✅ Execution reports saved to `logs/`

**Common Usage:**

```bash
# Show execution plan without running
uv run feature_engineering/run_pipeline.py --dry-run

# Resume from last failure (skips completed stages)
uv run feature_engineering/run_pipeline.py --skip-completed

# Run only specific stages
uv run feature_engineering/run_pipeline.py --only rolling_features,matchup_features

# Run from a specific stage onward
uv run feature_engineering/run_pipeline.py --from contextual_features

# Clean start (reset state)
uv run feature_engineering/run_pipeline.py --clean-start

# See all options
uv run feature_engineering/run_pipeline.py --help
```

**Configuration:**
- Edit `feature_engineering/pipeline.yaml` to customize the pipeline
- Enable/disable stages, adjust parameters, modify logging settings
- State tracked in `.pipeline_state.json`

---

### Manual Pipeline (Individual Steps)

Alternatively, run these scripts **manually in order**:

### Step 1: Rolling Features
```bash
uv run feature_engineering/rolling_features.py
```
Creates: `data/feature_tables/rolling_features.parquet`
- Rolling averages (5, 10, 20 games) for PRA, points, rebounds, assists
- Exponentially weighted moving averages
- Trend features (improving/declining performance)
- Home/away splits
- **All features use `.shift(1)` to prevent data leakage**

### Step 2: Matchup Features
```bash
uv run feature_engineering/matchup_features.py
```
Creates: `data/feature_tables/matchup_features.parquet`
- Opponent defensive stats (points/PRA allowed)
- Opponent pace factor
- Player vs opponent history
- Rest days and back-to-back indicators
- Team strength features

### Step 3: Contextual Features
```bash
uv run feature_engineering/contextual_features.py
```
Creates: `data/feature_tables/contextual_features.parquet`
- Home/away indicators
- Rest and fatigue features
- Season timing (early/mid/late season)
- Day of week and weekend games
- Recent performance indicators
- Minutes context (starter vs bench role)

### Step 4: Advanced Metrics
```bash
uv run feature_engineering/advanced_metrics.py
```
Creates: `data/feature_tables/advanced_metrics.parquet`
- Usage rate from CTG data
- Playmaking features (assist rate, turnover rate)
- Shooting efficiency (eFG%, True Shooting %)
- Rebounding percentages
- Defensive metrics
- Role indicators

### Step 5: Position Features [HIGH IMPACT]
```bash
uv run feature_engineering/position_features.py
```
Creates: `data/feature_tables/position_features.parquet`
**Expected improvement: 3-5% RMSE reduction**
- Position-normalized PRA (z-scores)
- Position percentile rankings
- Position-specific role indicators
- Elite performer detection
- **Rationale**: 30 PRA for a center ≠ 30 PRA for a guard

### Step 6: Injury Features [HIGH IMPACT]
```bash
uv run feature_engineering/injury_features.py
```
Creates: `data/feature_tables/injury_features.parquet`
**Expected improvement: 4-6% RMSE reduction**
- Games since injury return
- DNP tracking and patterns
- Load management indicators
- Minutes recovery tracking
- Availability scores
- **Rationale**: Players follow predictable ramp-up after injuries

### Step 7: Build Master Features
```bash
uv run feature_engineering/build_features.py
```
Creates: `data/feature_tables/master_features.parquet`
- Joins all 6 feature tables
- Validates data grain (no duplicates)
- Quality checks and reporting
- Ready for model training

### Step 8: Validate Features
```bash
uv run feature_engineering/validate_features.py
```
**Comprehensive validation checks:**
- Data leakage detection
- Grain uniqueness verification
- Feature distribution analysis
- Missing value patterns
- Target variable completeness

## Model Training Pipeline

### Create Train/Val/Test Splits
```bash
uv run model_training/train_split.py
```
Creates:
- `data/processed/train.parquet` (games before 2022-10-01)
- `data/processed/val.parquet` (2022-10-01 to 2023-10-01)
- `data/processed/test.parquet` (games after 2023-10-01)

**Features:**
- Chronological splits (no random shuffling)
- Handles missing values (fit on train, transform on val/test)
- Validates no temporal leakage
- Separates features, target, and metadata

## Data Leakage Prevention

This project implements **strict temporal validation** to prevent data leakage:

### ✓ What We Do Right:
1. **All rolling features use `.shift(1)`** - excludes current game
2. **Chronological train/test splits** - no k-fold cross-validation
3. **One-way time flow** - features only use past data
4. **Proper imputation** - fit on train, transform on val/test
5. **Validation tests** - automated checks for leakage

### ✗ What We Avoid:
1. ❌ Including current game stats in features
2. ❌ Random shuffling of time-series data
3. ❌ K-fold CV on temporal data
4. ❌ Using future information (e.g., season averages mid-season)
5. ❌ Fitting transformers on test data

## Quick Start Example

```bash
# 1. Install dependencies
uv sync

# 2. Collect data (use SAMPLE_SIZE=10 for testing)
uv run feature_engineering/data_loader.py

# 3. Run feature engineering pipeline
uv run feature_engineering/rolling_features.py
uv run feature_engineering/matchup_features.py
uv run feature_engineering/contextual_features.py
uv run feature_engineering/advanced_metrics.py
uv run feature_engineering/position_features.py  # HIGH IMPACT
uv run feature_engineering/injury_features.py    # HIGH IMPACT
uv run feature_engineering/build_features.py

# 4. Validate features
uv run feature_engineering/validate_features.py

# 5. Ready for modeling!
```

## Feature Categories

### Rolling Features (50+ features)
- **Recent performance**: 5/10/20 game averages
- **Trends**: Linear slopes over recent games
- **Volatility**: Standard deviations
- **Splits**: Home vs away performance
- **EWMA**: Exponentially weighted averages

### Matchup Features (15+ features)
- **Opponent defense**: Points/PRA allowed by opponent
- **Head-to-head**: Player's history vs specific team
- **Pace**: Opponent pace factor
- **Rest**: Days since last game, back-to-backs
- **Team strength**: Win percentage

### Contextual Features (20+ features)
- **Location**: Home/away
- **Timing**: Season stage, day of week
- **Fatigue**: Games in last 7/14 days
- **Role**: Minutes trends, starter indicators
- **Form**: Hot/cold streaks

### Advanced Metrics (25+ features)
- **Usage**: Possession usage rate
- **Efficiency**: True shooting %, eFG%
- **Playmaking**: Assist rate, turnover rate
- **Rebounding**: Offensive/defensive rebound %
- **Defense**: Steals, blocks per 36 minutes
- **Role**: Position indicators

### Position Features (30+ features) [HIGH IMPACT]
- **Normalization**: Position-specific z-scores for PRA
- **Percentiles**: Rank within position
- **Role indicators**: Guard/Wing/Forward/Center flags
- **Elite detection**: Position-adjusted performance thresholds
- **Context**: Deviation from position averages

### Injury Features (20+ features) [HIGH IMPACT]
- **Return tracking**: Games since injury/absence
- **DNP patterns**: Load management indicators
- **Recovery**: Minutes ramp-up tracking
- **Availability**: Player health scores
- **Durability**: Consecutive games played

## Expected Performance

Based on research and industry benchmarks:

- **Baseline (7-game average)**: R² ~ 0.60-0.65, RMSE ~ 6-7
- **With basic features (4 categories)**: R² ~ 0.70-0.75, RMSE ~ 5.0-5.5
- **With position normalization (+3-5% improvement)**: R² ~ 0.75-0.78, RMSE ~ 4.5-4.9
- **With injury tracking (+4-6% improvement)**: R² ~ 0.78-0.82, RMSE ~ 3.9-4.4
- **With XGBoost tuning**: R² ~ 0.82-0.86, RMSE ~ 3.5-4.0
- **Production optimized (feature selection)**: R² ~ 0.85-0.88, RMSE ~ 3.0-3.5

**Total Expected Improvement from New Features: 10-15% RMSE reduction**

## Data Sources

1. **NBA API** (via nba_api library)
   - Player game logs (box scores)
   - Team game logs
   - Game metadata

2. **CleaningTheGlass** (your CSV files)
   - Advanced player statistics
   - On/off court metrics
   - Shooting efficiency by zone
   - Usage and playmaking data

3. **Betting Odds** (your historical_odds data)
   - PRA lines from multiple sportsbooks
   - Over/under pricing
   - Line movement data

## Key Design Principles

1. **Simple structure**: Flat hierarchy, easy to navigate
2. **Modular features**: Each category in its own file
3. **Separate tables**: Debug individual feature sets easily
4. **Clean interfaces**: Consistent grain (player-game-date)
5. **Validation first**: Automated checks prevent errors
6. **No leakage**: Temporal validation throughout

## Next Steps

1. ✅ Feature engineering complete
2. ✅ Train/val/test splits created
3. ⏳ Train baseline models
4. ⏳ Train XGBoost model with hyperparameter tuning
5. ⏳ Implement backtesting framework
6. ⏳ Production deployment

## Troubleshooting

### "NBA API rate limit exceeded"
- Ensure 2+ second delays between requests
- Use `SAMPLE_SIZE=10` for testing
- Run data collection during off-peak hours

### "Missing CTG data"
- Check that `data/ctg_data_organized/` contains your CSV files
- Verify folder structure matches expected format
- Some CTG features will be NaN if data not available

### "Feature table not found"
- Run feature scripts in order (rolling → matchup → contextual → advanced)
- Check for errors in previous steps
- Verify `data/feature_tables/` directory exists

### "Target variable has missing values"
- Ensure NBA API data collection completed successfully
- Check `data/nba_api/player_games.parquet` exists
- Verify PRA calculation: `points + rebounds + assists`

## Contributing

When adding new features:
1. Create a new feature file in `feature_engineering/`
2. Use same grain: `[player_id, game_id, game_date]`
3. Apply `.shift(1)` to prevent leakage
4. Add validation checks
5. Save to `data/feature_tables/`
6. Update `build_features.py` to include new table

## License

This project is for personal use in NBA analytics research.
