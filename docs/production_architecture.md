# Production Architecture

**Version:** 2.0.0
**Date:** 2025-11-01

---

## Design Principles

The production system follows clean architecture principles with clear separation of concerns:

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Dependency Injection**: Use abstractions (interfaces) over concrete implementations
3. **Testability**: All modules have comprehensive unit tests (80%+ coverage)
4. **Error Handling**: Graceful degradation with specific exception types
5. **Self-Contained**: No dependencies on backtest/ directory (production-ready)
6. **Configurability**: Centralized configuration with environment variables

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   NBA PRA PRODUCTION SYSTEM                  │
└─────────────────────────────────────────────────────────────┘

                        ┌──────────────┐
                        │  CLI (Click) │
                        │  nba-pra     │
                        └──────┬───────┘
                               │
                               ├─── predict
                               ├─── train
                               ├─── recommend
                               ├─── pipeline
                               └─── status
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌────────────────┐    ┌─────────────────┐
│ Model Trainer │    │   Predictor    │    │ Betting Engine  │
└───────┬───────┘    └────────┬───────┘    └────────┬────────┘
        │                     │                      │
        ▼                     ▼                      ▼
┌───────────────────────────────────────────────────────────┐
│                    CORE UTILITIES                          │
├───────────────────────────────────────────────────────────┤
│  • monte_carlo.py    - Probabilistic calculations         │
│  • config.py         - Configuration management           │
│  • logging_config.py - Centralized logging                │
│  • exceptions.py     - Custom error types                 │
└───────────────────────────────────────────────────────────┘
        │                     │                      │
        ▼                     ▼                      ▼
┌───────────────┐    ┌────────────────┐    ┌─────────────────┐
│ Feature Data  │    │ Odds Providers │    │     Ledger      │
│ (master_feat) │    │  (TheOddsAPI)  │    │  (CSV Store)    │
└───────────────┘    └────────────────┘    └─────────────────┘
```

---

## Module Responsibilities

### Core Modules

#### `config.py`
**Purpose:** Centralized configuration management

**Responsibilities:**
- Load environment variables (.env)
- Define training parameters (CV folds, window size)
- Set betting thresholds (Kelly fraction, min edge)
- Manage file paths (models/, outputs/)

**Key Constants:**
```python
TRAINING_WINDOW_YEARS = 3
CV_FOLDS = 19
KELLY_FRACTION = 0.25
MIN_EDGE_KELLY = 0.03
MIN_CONFIDENCE = 0.6
MAX_CV = 0.35
```

#### `monte_carlo.py`
**Purpose:** Self-contained probabilistic prediction utilities

**Responsibilities:**
- Fit Gamma distribution parameters from mean/variance
- Calculate P(PRA > line) using analytical methods
- Convert American odds to breakeven probability
- Calculate bet edge and Kelly criterion sizing

**Key Functions:**
```python
fit_gamma_parameters(mean, variance) -> (alpha, beta)
calculate_probability_over_line(mean, std_dev, line) -> float
american_odds_to_probability(odds) -> float
calculate_bet_edge(prob_win, odds) -> float
calculate_kelly_fraction(prob_win, odds) -> float
```

**Why extracted from backtest/:**
- Production system should be self-contained
- No dependencies on backtesting code
- Enables independent testing and deployment

#### `model_trainer.py`
**Purpose:** Model training and persistence

**Responsibilities:**
- Load 3-year rolling training window
- Create 19 time-series CV folds
- Train mean prediction models (XGBoost)
- Train variance models (for uncertainty quantification)
- Save/load ensemble artifacts

**Key Classes:**
```python
ProductionModelTrainer:
    - load_training_data()
    - create_cv_folds()
    - train_fold()
    - save_ensemble()
    - load_ensemble()
```

#### `predictor.py`
**Purpose:** Prediction generation with probabilistic forecasting

**Responsibilities:**
- Load upcoming games from master_features.parquet
- Generate ensemble predictions (average of 19 models)
- Predict variance for uncertainty quantification
- Fit Gamma distributions to predictions
- Calculate P(over) and P(under) for betting lines

**Key Classes:**
```python
ProductionPredictor:
    - load_upcoming_games()
    - predict_ensemble()
    - calculate_probabilities()
    - merge_with_betting_lines()
```

#### `betting_engine.py`
**Purpose:** Betting decision logic

**Responsibilities:**
- Apply Kelly criterion sizing
- Filter by confidence score (min 60%)
- Filter by edge (min 3%)
- Filter by volatility (CV < 35%)
- Select bet direction (OVER vs UNDER)
- Analyze bet quality

**Key Classes:**
```python
BettingEngine:
    - calculate_betting_decisions()
    - apply_kelly_sizing()
    - filter_by_confidence()
    - analyze_bet_quality()
```

#### `ledger.py`
**Purpose:** Bet tracking and performance monitoring

**Responsibilities:**
- Record bets to CSV ledger
- Update results when games complete
- Calculate win rate and ROI
- Generate performance summaries

**Key Functions:**
```python
add_bets_to_ledger(bets_df, bet_date)
update_bet_result(bet_id, actual_pra, result)
get_ledger_summary() -> dict
```

### Infrastructure Modules

#### `odds_providers/` (Abstraction Layer)

**Purpose:** Abstract interface for betting odds providers

**Files:**
- `base.py` - Abstract `OddsProvider` interface
- `theoddsapi.py` - TheOddsAPI concrete implementation

**Why abstraction:**
- Swap providers without changing production code
- Mock providers for testing
- Support multiple bookmakers

**Interface:**
```python
class OddsProvider(ABC):
    def fetch_upcoming_games(date) -> List[Dict]
    def fetch_player_props(event_id) -> Dict
    def parse_pra_lines(event_data) -> DataFrame
    def get_all_pra_lines(date) -> DataFrame
```

#### `cli.py`
**Purpose:** Unified command-line interface

**Commands:**
- `nba-pra predict` - Generate predictions
- `nba-pra train` - Train models
- `nba-pra recommend` - Show top bets
- `nba-pra pipeline` - Run full workflow
- `nba-pra status` - System status

**Benefits:**
- Single entry point for all operations
- No PYTHONPATH required
- Consistent error handling
- Built-in help and documentation

#### `exceptions.py`
**Purpose:** Custom exception hierarchy

**Exception Types:**
```python
ProductionError (base)
├── ModelNotFoundError
├── FeatureDataError
├── PredictionError
├── BettingEngineError
└── InsufficientDataError
```

**Why custom exceptions:**
- Specific error handling per failure type
- Better error messages for debugging
- Graceful degradation (e.g., skip odds fetch if API fails)

#### `logging_config.py`
**Purpose:** Centralized logging configuration

**Features:**
- Consistent logging format across all modules
- Console and file handlers
- Log rotation (daily logs/)
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)

**Usage:**
```python
from production.logging_config import setup_production_logging
logger = setup_production_logging('module_name')
```

### Legacy Modules (Deprecated)

The following scripts are **deprecated** and will be removed in v3.0:

- `run_daily.py` → Use `nba-pra predict`
- `run_full_pipeline.py` → Use `nba-pra pipeline`
- `recommend_bets.py` → Use `nba-pra recommend`

These scripts now show deprecation warnings on execution.

---

## Data Flow

### Daily Prediction Pipeline

```
1. FEATURE DATA (INPUT)
   └─ master_features.parquet (134 features, 587k games)
      ├─ Rolling features (EWMA, trends, volatility)
      ├─ Matchup features (opponent defense, pace)
      ├─ Contextual features (rest, home/away, timing)
      ├─ Advanced metrics (usage, efficiency, playmaking)
      ├─ Position features (z-scores, percentiles)
      └─ Injury features (DNP tracking, recovery)

2. MODEL TRAINING
   └─ ProductionModelTrainer
      ├─ Load 3-year rolling window
      ├─ Create 19 CV folds (time-series)
      ├─ Train mean models (XGBoost)
      ├─ Train variance models
      └─ Save ensemble → models/ensemble_YYYYMMDD.pkl

3. ODDS FETCHING
   └─ OddsFetcher (TheOddsAPI)
      ├─ Fetch tomorrow's NBA games
      ├─ Get player props (points, rebounds, assists)
      ├─ Calculate PRA lines (sum of markets)
      └─ Return DataFrame with lines

4. PREDICTION GENERATION
   └─ ProductionPredictor
      ├─ Load upcoming games (tomorrow)
      ├─ Predict mean (ensemble average)
      ├─ Predict variance (ensemble)
      ├─ Fit Gamma(α, β) distributions
      ├─ Calculate P(PRA > line)
      └─ Save → outputs/predictions/predictions_YYYY-MM-DD.csv

5. BETTING DECISIONS
   └─ BettingEngine
      ├─ Apply Kelly sizing
      ├─ Filter by confidence (≥60%)
      ├─ Filter by edge (≥3%)
      ├─ Filter by CV (≤35%)
      ├─ Select direction (OVER/UNDER)
      └─ Save → outputs/bets/bets_YYYY-MM-DD.csv

6. LEDGER TRACKING
   └─ Ledger
      ├─ Record bets to ledger CSV
      ├─ Update results after games
      ├─ Calculate win rate, ROI
      └─ Generate summaries
```

---

## Testing Strategy

### Test Coverage: 80%+ (Target)

#### Unit Tests
**Location:** `tests/production/`

**Coverage:**
- `test_monte_carlo.py` - Probabilistic calculations
- `test_model_trainer.py` - Training and CV folds
- `test_predictor.py` - Prediction generation
- `test_betting_engine.py` - Betting decisions
- `test_ledger.py` - Bet tracking
- `test_odds_providers.py` - Odds fetching
- `test_cli.py` - CLI commands

#### Integration Tests
**Location:** `tests/production/test_integration.py`

**Scenarios:**
- Full pipeline (train → predict → bet)
- Pipeline with missing odds (graceful degradation)
- Pipeline with insufficient data (filtering)
- Idempotency (running twice produces same results)

#### Fixtures
**Location:** `tests/production/conftest.py`

**Shared Fixtures:**
- `temp_production_dir` - Temporary directory structure
- `sample_features_df` - Mock feature data
- `sample_predictions_df` - Mock predictions
- `sample_odds_response` - Mock API response
- `mock_ensemble_data` - Mock trained models

#### Running Tests

```bash
# All tests
pytest tests/production/ -v

# Coverage report
pytest tests/production/ --cov=production --cov-report=html

# Integration tests only
pytest tests/production/ -m integration

# Specific module
pytest tests/production/test_monte_carlo.py -v
```

---

## Deployment

### Local Development

```bash
# Install dependencies
uv sync

# Train models
nba-pra train

# Run predictions
nba-pra predict

# View recommendations
nba-pra recommend
```

### Production (Cron)

```bash
# Daily automation (2 AM)
0 2 * * * cd /path/to/NBA_PRA && nba-pra pipeline --full >> logs/cron_daily.log 2>&1
```

**Runtime:** ~20-25 minutes
- Data fetch: ~5-10 min
- Feature engineering: ~5-10 min
- Model training: ~3-5 min
- Predictions: ~1 min

### Environment Variables

**Required:**
```bash
ODDS_API_FREE_KEY=your_api_key_here  # TheOddsAPI key
```

**Optional:**
```bash
PYTHONPATH=/path/to/NBA_PRA  # Only for legacy scripts
```

---

## Performance Monitoring

### Health Indicators

**✅ Healthy System:**
- Win rate: 68-75%
- ROI: 45-55%
- Edge calibration: Monotonic (higher edge → higher win rate)
- Bet volume: 40-50% of predictions

**⚠️ Warning Signs:**
- Win rate < 65% for 30 days
- ROI < 30%
- Edge not calibrated (random wins/losses)
- Bet volume < 20% or > 70%

### Monitoring Commands

```bash
# Check system status
nba-pra status

# View recent predictions
ls -lh production/outputs/predictions/

# Analyze betting performance
cat production/outputs/bets/bets_*.csv | python analyze_performance.py
```

---

## Error Handling

### Graceful Degradation

The system handles failures gracefully:

1. **Odds API fails** → Continue with predictions only (no betting analysis)
2. **Model not found** → Prompt user to train
3. **Feature data outdated** → Warn and continue (or fail fast with --strict)
4. **Insufficient player data** → Filter out players with <5 games

### Retry Logic

API calls use exponential backoff:

```python
for attempt in range(max_retries):
    try:
        return fetch_odds()
    except OddsAPIError:
        wait_time = 2 ** attempt
        time.sleep(wait_time)
```

### Logging

All errors logged to:
- Console (INFO level)
- File logs (DEBUG level): `logs/MODULE_NAME.log`

---

## Future Enhancements

### Planned for v3.0

1. **Database ledger** - Replace CSV with SQLite/PostgreSQL
2. **Multiple bookmakers** - Compare lines, find best edge
3. **Live betting** - Real-time predictions during games
4. **Web dashboard** - Flask/FastAPI frontend for monitoring
5. **Model versioning** - Track model performance over time
6. **A/B testing** - Compare different model architectures
7. **Alerting** - Slack/email notifications for high-edge bets

---

## References

- **Project docs:** `.claude/CLAUDE.md`
- **Backtest docs:** `backtest/README.md`
- **Monte Carlo docs:** `backtest/monte_carlo/README.md`
- **Implementation plan:** `docs/plans/2025-11-01-production-folder-refactor.md`

---

## Version History

### v2.0.0 (2025-11-01)
- Unified CLI (`nba-pra` command)
- Odds provider abstraction layer
- Self-contained Monte Carlo utilities
- Custom exception hierarchy
- Comprehensive test suite (80%+ coverage)
- Improved logging and error handling

### v1.0.0 (2025-10-31)
- Initial production release
- 19-fold CV ensemble
- Monte Carlo probabilistic predictions
- TheOddsAPI integration
- Kelly criterion bet sizing
