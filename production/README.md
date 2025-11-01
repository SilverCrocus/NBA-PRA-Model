

# NBA PRA Production System

Production deployment for daily NBA player PRA (Points + Rebounds + Assists) predictions with Monte Carlo probabilistic forecasting and Kelly criterion bet sizing.

**Version:** 2.0.0
**Date:** 2025-11-01

---

## Overview

This production system automates daily NBA player prop betting predictions using:
- **3-year rolling training window** (last 3 seasons)
- **19-fold CV ensemble** (same methodology as backtesting)
- **Monte Carlo probabilistic predictions** (Gamma distributions)
- **Kelly criterion bet sizing** with confidence filtering
- **TheOddsAPI integration** for real-time betting lines

### Expected Performance

Based on backtesting results:
- **Win Rate:** 68-75% (conservative estimate)
- **ROI:** 47-55%
- **Edge:** 3-8% on actionable bets
- **Bet Volume:** 40-50% of all predictions (confidence filtering)

---

## Quick Start (NEW CLI - v2.0.0)

### 1. Install Dependencies

```bash
# Install with uv
uv sync
```

### 2. Configure API Key

Update `.env` file with your free TheOddsAPI key:
```bash
ODDS_API_FREE_KEY=your_api_key_here
```

### 3. Train Initial Models

```bash
nba-pra train
```

**Training time:** ~3-5 minutes (19 folds, 3 years of data)

### 4. Run Daily Pipeline

**🎯 RECOMMENDED: Single-Command Workflow**

```bash
# Complete daily pipeline (data → features → training → predictions)
nba-pra pipeline --full

# Or run just predictions (if data/features are current)
nba-pra predict
```

**What the full pipeline does:**
1. ✅ Fetches latest NBA game logs automatically
2. ✅ Regenerates all features (~5-10 min)
3. ✅ **Retrains models with new data** (~3-5 min)
4. ✅ Fetches tomorrow's betting lines
5. ✅ Generates predictions and bets

**Total time:** ~20-25 minutes (acceptable for overnight automation)

**Why retrain daily?**
- Your backtest used walk-forward with daily retraining → 77% win rate
- New data without retraining wastes that information
- Training only takes 3-5 minutes (worth it!)
- Captures latest player form, injuries, lineup changes

### 5. View Recommendations

```bash
# Top 10 bets with 5% minimum edge
nba-pra recommend --min-edge 0.05 --min-confidence 0.7

# Check system status
nba-pra status
```

---

## New CLI Commands (v2.0.0)

The production system now uses a unified CLI with the following commands:

### `nba-pra predict`
Generate predictions for upcoming games.

```bash
# Default (tomorrow's games)
nba-pra predict

# Specific date
nba-pra predict --date 2024-11-01

# Skip training (use cached model)
nba-pra predict --skip-training

# Dry run (preview)
nba-pra predict --dry-run
```

### `nba-pra train`
Train production models.

```bash
# Train 19-fold ensemble
nba-pra train

# Custom configuration
nba-pra train --cv-folds 19 --training-window 3
```

### `nba-pra recommend`
Recommend top bets based on edge and confidence.

```bash
# Default (top 10 with 5% edge)
nba-pra recommend

# Custom filters
nba-pra recommend --min-edge 0.07 --min-confidence 0.8 --top-n 5

# Specific date
nba-pra recommend --date 2024-11-01
```

### `nba-pra pipeline`
Run complete daily pipeline.

```bash
# Full pipeline (data + features + training + predictions)
nba-pra pipeline --full

# Skip specific steps
nba-pra pipeline --skip-data-update
nba-pra pipeline --skip-feature-engineering
nba-pra pipeline --skip-training
```

### `nba-pra status`
Show production system status.

```bash
nba-pra status
```

Displays:
- Latest model information
- Recent predictions
- Betting ledger summary

---

## Migration Guide (v1.0 → v2.0)

### OLD vs NEW Commands

```bash
# OLD (v1.0)
PYTHONPATH=/path/to/NBA_PRA uv run python production/run_daily.py

# NEW (v2.0)
nba-pra predict

# ──────────────────────────────────────────────────

# OLD
PYTHONPATH=/path/to/NBA_PRA uv run python production/run_full_pipeline.py --auto-fetch-data

# NEW
nba-pra pipeline --full

# ──────────────────────────────────────────────────

# OLD
PYTHONPATH=/path/to/NBA_PRA uv run python production/model_trainer.py

# NEW
nba-pra train

# ──────────────────────────────────────────────────

# OLD
PYTHONPATH=/path/to/NBA_PRA uv run python production/recommend_bets.py --min-edge 0.05

# NEW
nba-pra recommend --min-edge 0.05
```

### Benefits of v2.0 CLI

✅ **No PYTHONPATH required** - CLI handles imports automatically
✅ **Shorter commands** - `nba-pra predict` vs long paths
✅ **Unified interface** - Single entry point for all operations
✅ **Better help** - `nba-pra --help` shows all commands
✅ **Improved error handling** - Consistent logging and exceptions

### Legacy Scripts (Deprecated)

The following scripts still work but are **deprecated** and will be removed in v3.0:

- `production/run_daily.py` → Use `nba-pra predict`
- `production/run_full_pipeline.py` → Use `nba-pra pipeline`
- `production/recommend_bets.py` → Use `nba-pra recommend`

All legacy scripts now show deprecation warnings.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  DAILY PRODUCTION WORKFLOW                   │
└─────────────────────────────────────────────────────────────┘

STEP 1: Model Training (or load cached)
   ├─ Load last 3 years of data
   ├─ Create 19 time-series CV folds
   ├─ Train 19 XGBoost models (mean predictions)
   ├─ Train 19 variance models (uncertainty)
   └─ Save ensemble → models/ensemble_YYYYMMDD.pkl

STEP 2: Fetch Betting Lines
   ├─ Get tomorrow's NBA games from TheOddsAPI
   ├─ Fetch player_points, player_rebounds, player_assists
   ├─ Calculate PRA lines (sum of three markets)
   └─ Return DataFrame with lines

STEP 3: Generate Predictions
   ├─ Load tomorrow's games from master_features.parquet
   ├─ Predict using 19-model ensemble → mean_pred
   ├─ Predict variance using variance ensemble → var_pred
   ├─ Fit Gamma(α, β) distributions
   ├─ Calculate P(PRA > line) for each betting line
   └─ Save predictions → predictions/predictions_YYYY-MM-DD.csv

STEP 4: Calculate Betting Decisions
   ├─ Apply Kelly criterion sizing
   ├─ Filter by confidence (min 60%, max CV 35%)
   ├─ Filter by edge (min 3%)
   ├─ Analyze bet quality
   └─ Save bets → bets/bets_YYYY-MM-DD.csv
```

---

## Modules

### 1. `config.py`

Centralized configuration:
- API keys and endpoints
- Training parameters (3-year window, 19 CV folds)
- Betting thresholds (Kelly fraction, min edge, confidence)
- File paths

**Key parameters:**
```python
TRAINING_WINDOW_YEARS = 3
CV_FOLDS = 19
KELLY_FRACTION = 0.25       # Quarter Kelly
MIN_EDGE_KELLY = 0.03       # 3% minimum edge
MIN_CONFIDENCE = 0.6        # 60% confidence
MAX_CV = 0.35               # 35% max coefficient of variation
```

### 2. `odds_fetcher.py`

TheOddsAPI integration:
- Fetch upcoming NBA games
- Get player props (points, rebounds, assists)
- Calculate PRA lines from individual markets
- Rate limiting for free tier (500 requests/month)

**Usage:**
```python
from production.odds_fetcher import fetch_tomorrow_odds

# Fetch tomorrow's PRA lines
pra_lines = fetch_tomorrow_odds()
print(pra_lines[['player_name', 'pra_line', 'bookmaker']])
```

### 3. `model_trainer.py`

Model training with CV ensemble:
- 3-year rolling window
- 19 time-series CV folds
- Mean + variance models
- Model persistence

**Usage:**
```python
from production.model_trainer import train_production_models

# Train and save ensemble
mean_models, variance_models = train_production_models(save=True)
```

### 4. `predictor.py`

Prediction generation:
- Load upcoming games
- Ensemble predictions (mean + variance)
- Gamma distribution fitting
- P(PRA > line) calculations

**Usage:**
```python
from production.predictor import predict_tomorrow
from production.model_trainer import get_latest_model_path

model_path = get_latest_model_path()
predictions = predict_tomorrow(str(model_path), odds_df)
print(predictions[['player_name', 'mean_pred', 'prob_over', 'edge_over']])
```

### 5. `betting_engine.py`

Betting decision calculator:
- Kelly criterion sizing
- Confidence filtering
- Edge-based filtering
- Bet quality analysis

**Usage:**
```python
from production.betting_engine import generate_bets_from_predictions

# Convert predictions to bets
bets = generate_bets_from_predictions(predictions)
print(bets[['player_name', 'direction', 'edge', 'kelly_size']])
```

### 6. `run_daily.py`

Main orchestrator:
- Runs 4 steps: train → fetch odds → predict → bets
- Command-line interface
- Error handling
- Logging

**Usage:**
```bash
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python production/run_daily.py [options]
```

**Note:** Does NOT update NBA data or regenerate features. Use `run_full_pipeline.py` for that.

### 7. `run_full_pipeline.py`

Complete pipeline orchestrator:
- Prompts for data updates
- Runs feature engineering
- Runs production pipeline

**Usage:**
```bash
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python production/run_full_pipeline.py [options]
```

---

## Output Files

### Predictions

**Location:** `production/outputs/predictions/predictions_YYYY-MM-DD.csv`

**Columns:**
- `player_name`, `game_date`, `team_abbreviation`
- `mean_pred` - Ensemble mean prediction
- `std_dev` - Standard deviation (uncertainty)
- `betting_line` - PRA line from bookmaker
- `prob_over` - P(PRA > line)
- `prob_under` - P(PRA ≤ line)
- `edge_over` - Edge on OVER bet
- `edge_under` - Edge on UNDER bet
- `confidence_score` - Confidence metric (0-1)
- `cv` - Coefficient of variation

### Bets

**Location:** `production/outputs/bets/bets_YYYY-MM-DD.csv`

**Columns:**
- `player_name`, `game_date`, `team_abbreviation`
- `betting_line` - PRA line
- `direction` - 'OVER' or 'UNDER'
- `mean_pred` - Model prediction
- `prob_win` - Win probability
- `edge` - Edge over breakeven
- `kelly_size` - Recommended bet size (fraction of bankroll)
- `confidence_score` - Confidence metric
- `bookmaker` - Sportsbook name

**Interpretation:**
- `kelly_size` = 0.05 means bet 5% of bankroll
- `edge` = 0.06 means 6% edge over breakeven (52.4%)
- Higher `edge` = better bet quality

### Models

**Location:** `production/models/ensemble_YYYYMMDD_HHMMSS.pkl`

Saved ensemble data:
- 19 mean prediction models
- 19 variance models (if Monte Carlo enabled)
- Feature names
- Training metrics

---

## Daily Automation

### Cron Job Setup (RECOMMENDED)

**🎯 Single-Command Daily Automation** - Matches backtest methodology with daily retraining:

```bash
# Edit crontab
crontab -e

# Add this line (runs at 2 AM after games finish):
0 2 * * * cd /Users/diyagamah/Documents/NBA_PRA && PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA /usr/local/bin/uv run python production/run_full_pipeline.py --auto-fetch-data >> logs/cron_daily.log 2>&1
```

**What this does every night:**
1. Fetches latest NBA game logs
2. Regenerates features
3. **Retrains models** (matches your walk-forward backtest)
4. Generates predictions and bets for tomorrow

**Runtime:** ~20-25 minutes total
- Data fetch: ~5-10 min
- Feature engineering: ~5-10 min
- Model training: ~3-5 min
- Predictions: ~1 min

**Why daily retraining?**
- Your backtest retrains after every game day → 77% win rate
- Captures latest player performance, injuries, form
- Only adds 3-5 minutes (worth it for up-to-date model)

### Alternative: Weekly Retraining (NOT RECOMMENDED)

If compute resources are limited, you could retrain weekly (though this deviates from your proven backtest approach):

```bash
# Sunday 2 AM - full pipeline with retraining
0 2 * * 0 cd /Users/diyagamah/Documents/NBA_PRA && PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA /usr/local/bin/uv run python production/run_full_pipeline.py --auto-fetch-data >> logs/cron_sunday.log 2>&1

# Monday-Saturday 2 AM - skip training (uses stale model)
0 2 * * 1-6 cd /Users/diyagamah/Documents/NBA_PRA && PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA /usr/local/bin/uv run python production/run_full_pipeline.py --auto-fetch-data --skip-training >> logs/cron_daily.log 2>&1
```

**Note:** This is less optimal than daily retraining.

---

## Betting Strategy

### Kelly Criterion

The system uses **fractional Kelly (25%)** for risk management:

```
Bet size = 0.25 × Kelly fraction
         = 0.25 × (edge / variance)
```

**Example:**
- Edge = 6%
- Variance = 0.2
- Kelly size = 0.25 × (0.06 / 0.2) = 0.075 (7.5% of bankroll)

**Risk management:**
- Never bet more than 25% of bankroll on single bet
- Only bet when edge ≥ 3%
- Filter low-confidence bets

### Confidence Filtering

Bets must pass **three criteria**:

1. **Edge ≥ 3%** - Probability exceeds breakeven by at least 3%
2. **Confidence ≥ 60%** - Probability far from 50/50
3. **CV ≤ 35%** - Relative uncertainty (std/mean) below threshold

This reduces bet volume by ~50% but improves win rate by ~5-8%.

### Bankroll Management

**Recommended:**
- Start with $1,000 bankroll
- Bet according to Kelly sizes (typically 2-8% per bet)
- Track 30-day rolling win rate
- Stop if win rate < 65% for 30 days
- Reassess model if performance degrades

---

## Troubleshooting

### Issue: No trained models found

**Solution:**
```bash
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python production/model_trainer.py
```

### Issue: TheOddsAPI quota exceeded

**Symptoms:** "Rate limit exceeded" error

**Solution:**
- Free tier: 500 requests/month
- Each game uses ~1 request
- Reduce frequency or upgrade to paid tier

### Issue: No betting lines found

**Possible causes:**
1. No games scheduled for target date
2. API key invalid
3. Player props not available yet (lines post ~24h before game)

**Check:**
```bash
# Test odds fetcher
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python production/odds_fetcher.py
```

### Issue: ModuleNotFoundError

**Solution:** Always set PYTHONPATH:
```bash
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python production/...
```

### Issue: Feature data outdated

**Solution:** Run feature engineering pipeline:
```bash
uv run feature_engineering/run_pipeline.py
```

---

## Performance Monitoring

### Daily Checks

1. **Check logs:** `logs/run_daily.log`
2. **Review bets:** `production/outputs/bets/bets_YYYY-MM-DD.csv`
3. **Validate predictions:** Compare to actual results next day

### Weekly Analysis

1. **Win rate tracking:**
   ```python
   import pandas as pd

   bets = pd.read_csv('production/outputs/bets/bets_2025-11-01.csv')
   # After games complete, add actual results
   # Calculate win_rate = (correct_bets / total_bets)
   ```

2. **Edge calibration:**
   - High edge (>5%) → High win rate (>75%)
   - Medium edge (3-5%) → Medium win rate (65-75%)
   - If not calibrated, retrain models

3. **ROI calculation:**
   ```
   ROI = (total_profit / total_wagered) × 100
   ```

### Model Performance Indicators

**✅ Healthy model:**
- Win rate: 68-75%
- ROI: 45-55%
- Edge calibration: Monotonic (higher edge → higher win rate)
- Bet volume: 40-50% of predictions

**⚠️ Warning signs:**
- Win rate < 65% for 30 days
- ROI < 30%
- Edge not calibrated (random wins/losses)
- Bet volume < 20% (too conservative) or > 70% (too aggressive)

**🔧 Actions:**
- Retrain with latest data
- Adjust confidence thresholds
- Review feature engineering pipeline

---

## API Costs

### TheOddsAPI Free Tier

- **Quota:** 500 requests/month
- **Cost per request:** ~1 request per game
- **Typical usage:**
  - 10 games/day × 30 days = 300 requests/month
  - **Fits within free tier ✓**

### Upgrading to Paid Tier

If you need more quota:
- **Starter:** $39/month, 10,000 requests
- **Pro:** $99/month, 50,000 requests
- [Pricing details](https://the-odds-api.com/pricing)

---

## Advanced Usage

### Custom Betting Thresholds

Edit `production/config.py`:
```python
KELLY_FRACTION = 0.20        # More aggressive (20% Kelly)
MIN_EDGE_KELLY = 0.05        # Higher edge requirement (5%)
MIN_CONFIDENCE = 0.65        # Higher confidence requirement
MAX_CV = 0.30                # Lower variance tolerance
```

### Multiple Bookmakers

The odds fetcher supports multiple bookmakers:
```python
# In odds_fetcher.py
ODDS_REGIONS = "us,us2"  # Multiple regions
```

Compare lines across bookmakers and choose best edge.

### Historical Backtesting

To validate on new data:
```bash
cd backtest
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python run_backtest.py
```

---

## File Structure

```
production/
├── __init__.py              # Package initialization
├── README.md                # This file
├── config.py                # Configuration
├── odds_fetcher.py          # TheOddsAPI integration
├── model_trainer.py         # Model training
├── predictor.py             # Prediction generation
├── betting_engine.py        # Betting decisions
├── run_daily.py             # Main orchestrator (no data updates)
├── run_full_pipeline.py     # Complete pipeline (includes feature engineering)
├── models/                  # Saved models
│   ├── ensemble_20251031_143022.pkl
│   └── training_metrics_20251031_143022.csv
└── outputs/
    ├── predictions/         # Daily predictions
    │   └── predictions_2025-11-01.csv
    └── bets/               # Daily bets
        └── bets_2025-11-01.csv
```

---

## Support

### Logs

All modules log to `logs/` directory:
- `logs/model_trainer.log` - Training logs
- `logs/odds_fetcher.log` - API fetch logs
- `logs/predictor.log` - Prediction logs
- `logs/betting_engine.log` - Betting decision logs
- `logs/run_daily.log` - Pipeline orchestration logs

### Common Issues

See **Troubleshooting** section above.

### Documentation

- **Project docs:** `/Users/diyagamah/Documents/NBA_PRA/.claude/CLAUDE.md`
- **Backtest docs:** `/Users/diyagamah/Documents/NBA_PRA/backtest/README.md`
- **Monte Carlo docs:** `/Users/diyagamah/Documents/NBA_PRA/backtest/monte_carlo/README.md`

---

## Version History

**v1.0.0** (2025-10-31)
- Initial production release
- 19-fold CV ensemble
- Monte Carlo probabilistic predictions
- TheOddsAPI integration
- Kelly criterion bet sizing
- Automated daily pipeline

---

## License

Internal use only.

## Author

NBA PRA Prediction System
Date: 2025-10-31
