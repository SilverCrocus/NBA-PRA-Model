# NBA PRA Walk-Forward Backtest

Production-realistic walk-forward backtesting system for NBA PRA predictions with daily retraining.

## Overview

This backtest simulates how the model would perform in production by:
1. **Daily retraining**: Retrain 19-fold CV ensemble after each NBA game day
2. **3-year rolling window**: Use last 3 seasons of data before each prediction date
3. **Betting simulation**: Match predictions to actual betting lines, calculate win rate and ROI
4. **Proper temporal validation**: Position baselines calculated per-fold to prevent data leakage

## Features Implemented

### Phase 1: CTG Missing Data Imputation (Completed)
✅ **Positional average imputation** instead of forward-filling
✅ **Binary imputation flags**: `has_ctg_data`, `is_rookie`, `ctg_data_quality`, `ctg_seasons_available`
✅ **Position-relative features**: `usage_vs_position`, `assist_vs_position`, etc.
✅ **Per-fold baseline calculation**: Prevents temporal leakage by using training data only

**Files modified:**
- `feature_engineering/features/ctg_imputation.py` (NEW)
- `feature_engineering/features/advanced_metrics.py` (UPDATED)

### Phase 2: Walk-Forward Backtest Infrastructure (Completed)
✅ Complete modular architecture
✅ Daily retraining with 19-fold CV ensemble
✅ Betting line integration from historical odds
✅ Comprehensive metrics tracking (MAE, RMSE, R², win rate, ROI)
✅ Incremental results saving (resumable if interrupted)

**Files created:**
- `backtest/config.py` - Configuration and constants
- `backtest/data_loader.py` - Data loading and preparation
- `backtest/betting_evaluator.py` - Betting performance metrics
- `backtest/walk_forward_engine.py` - Main backtest orchestrator
- `backtest/reporting.py` - Report generation
- `backtest/run_backtest.py` - Main entry point

## Quick Start

### Run the Backtest

```bash
# From project root
uv run backtest/run_backtest.py
```

**Expected runtime:** 5-6 hours (170-180 days × 19 models × ~5 seconds/model)

### Monitor Progress

The backtest saves progress incrementally:
- Every 10 days: Log progress update
- Every 20 days: Save checkpoint to `backtest/results/daily_predictions.csv`

You can monitor the log file in `backtest/results/backtest_*.log`

## Output Files

All results saved to `backtest/results/`:

### 1. daily_predictions.csv
Every prediction with columns:
- `game_date`, `player_id`, `player_name`
- `prediction`, `actual_pra`, `betting_line`
- `bet_decision` (over/under), `bet_correct` (win/loss), `profit`
- `has_ctg_data`, `edge_size`

### 2. daily_metrics.csv
Per-game-day summary:
- `game_date`, `num_bets`, `wins`, `win_rate`
- `daily_profit`, `cumulative_profit`, `roi`

### 3. player_analysis.csv
Per-player performance:
- `player_name`, `num_games`, `win_rate`, `roi`, `total_profit`
- Sorted by ROI (best to worst)

### 4. betting_performance.csv
Overall betting summary:
- Total bets, wins, losses, win rate
- ROI, total profit, break-even analysis
- Edge performance (small/medium/large edges)

### 5. backtest_report.md
Comprehensive markdown report with:
- Executive summary (profitable/not profitable)
- Prediction accuracy (overall and by CTG availability)
- Betting performance analysis
- Top/bottom 10 players
- Recommendations for production deployment

## Architecture

### Data Flow

```
Master Features (2024-25) → Daily Iteration Loop:
    ├─ Get games for prediction date
    ├─ Extract 3-year training window
    ├─ Calculate position baselines from training data
    ├─ Create 19 CV folds
    ├─ Train 19 XGBoost models
    ├─ Make ensemble predictions
    ├─ Match to betting lines
    ├─ Calculate metrics
    └─ Save results
```

### Key Design Decisions

**1. 3-year rolling window** (user requested)
- Uses last 3 seasons before prediction date
- Balances having enough data vs adapting to recent trends

**2. 19-fold CV ensemble** (user requested for realism)
- Matches production model architecture
- More robust than single model
- ~1.75 min per day (vs ~5 seconds for single model)

**3. Positional imputation** (user requested, statistically sound)
- Uses position-specific averages for missing CTG data
- Adds imputation flags so model learns when to trust CTG
- Prevents bias from forward-filling

**4. Per-fold baseline calculation**
- Calculates position averages from each fold's training data
- Prevents temporal leakage (test data not used in baselines)
- Critical for valid backtest results

## Configuration

Edit `backtest/config.py` to customize:

```python
# Training window
TRAINING_WINDOW_YEARS = 3  # Use 3 years before prediction date

# Ensemble settings
CV_FOLDS = 19  # Number of models in ensemble

# Target season
TARGET_SEASON = "2024-25"
TARGET_START_DATE = "2024-10-01"
TARGET_END_DATE = "2025-06-30"

# Betting odds
BETTING_ODDS = -110  # Standard American odds
WIN_PAYOUT = 0.909   # Win profit at -110 odds
BREAK_EVEN_WIN_RATE = 0.524  # Need 52.4% to profit
```

## Expected Performance

### Prediction Accuracy
Based on CV results, walk-forward expected to achieve:
- **MAE**: 2.85-3.10 (slightly worse than CV due to single-timepoint training)
- **RMSE**: 4.15-4.50
- **R²**: 0.855-0.875

**Imputation impact** (from Phase 1):
- Overall MAE improvement: -0.15 points (5% better)
- MAE on games without CTG: -0.25 points (8% better for this subset)

### Betting Performance
If MAE < 3.1 and well-calibrated:
- **Win rate**: 52-55% (need >52.4% to profit)
- **ROI**: +2% to +5% if win rate >53%
- **Edge bets** (3+ points difference): Expected 55-60% win rate

## Troubleshooting

### "FileNotFoundError: Master features not found"
Run feature engineering pipeline first:
```bash
uv run feature_engineering/run_pipeline.py
```

### "No data found for target season 2024-25"
Ensure 2024-25 data is in `data/nba_api/player_games.parquet`

### "No betting lines available"
Backtest will run without betting analysis if odds file missing.
Check `data/historical_odds/2024-25/pra_odds.csv`

### Backtest interrupted
Partial results saved in `backtest/results/daily_predictions.csv`
Re-run to continue (checkpoints saved every 20 days)

### Out of memory
Reduce CV_FOLDS in config.py (e.g., from 19 to 10)

## Validation Checks

The backtest includes automatic validation:

**Data quality:**
- Warns if >30% of features missing
- Checks for duplicate player-game records
- Validates date ranges

**Temporal leakage prevention:**
- Training cutoff < prediction date (strict inequality)
- Position baselines calculated from training data only
- Rolling features use `.shift(1)` (from feature engineering)

**Performance thresholds:**
- Alerts if MAE > 3.5 or R² < 0.80
- Warns if win rate < 50%

## Development Notes

### Adding New Features to Backtest

If you modify feature engineering, re-run the pipeline:
```bash
uv run feature_engineering/run_pipeline.py
```

Then re-run backtest to use new features.

### Testing with Subset

To test on first month only, edit `run_backtest.py`:
```python
# In main():
predictions_df = run_walk_forward_backtest(
    all_data, test_data, odds_data,
    start_date="2024-10-01",
    end_date="2024-11-01"  # Test first month only
)
```

### Parallel Execution

Currently runs sequentially (one day at a time). Could parallelize by:
- Training models for multiple days concurrently
- Requires careful handling of rolling training window

## Comparison to CV Results

**CV (from model_training/):**
- 19 folds with 3-year training windows
- Each fold evaluated on distinct test period
- MAE: 2.99, RMSE: 4.25, R²: 0.868

**Walk-Forward Backtest:**
- Daily retraining on 3-year rolling window
- All predictions on 2024-25 season
- Expected MAE: 2.85-3.10 (similar to CV)

**Why walk-forward MAE might differ:**
- Different training data distribution (2024-25 vs historical)
- New players, rule changes, meta shifts
- But should be within 0.1-0.2 points of CV if model is stable

## Next Steps

After backtest completes:

1. **Review report**: Open `backtest/results/backtest_report.md`

2. **Analyze by player**: Check `player_analysis.csv` for most/least predictable players

3. **Examine edge performance**: Look at betting metrics for large-edge bets (3+ point diff)

4. **Production deployment**: If profitable, use insights to:
   - Focus bets on high-edge opportunities
   - Prioritize top-performing players
   - Set bet sizing based on edge magnitude

5. **Iterate**: If not profitable:
   - Only bet when model has 3+ point edge
   - Consider additional feature engineering
   - Focus on players with >10 games and positive ROI

## Credits

**Implementation Date:** 2025-10-30
**Backtest Type:** Walk-forward with daily retraining
**Model:** 19-fold XGBoost ensemble
**Imputation Strategy:** Positional average + flags
