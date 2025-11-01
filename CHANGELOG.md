# Changelog

All notable changes to the NBA PRA Production System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2025-11-01

### Added
- **Unified CLI** (`nba-pra` command) with 5 subcommands:
  - `nba-pra predict` - Generate predictions for upcoming games
  - `nba-pra train` - Train production models
  - `nba-pra recommend` - Show top betting recommendations
  - `nba-pra pipeline` - Run complete daily workflow
  - `nba-pra status` - Display system status
- **Odds Provider Abstraction Layer**:
  - `OddsProvider` abstract interface
  - `TheOddsAPIProvider` concrete implementation
  - Support for swappable odds providers
- **Self-Contained Monte Carlo Utilities** (`production/monte_carlo.py`):
  - Gamma distribution parameter fitting
  - Probability calculations (P(PRA > line))
  - American odds conversion
  - Bet edge calculation
  - Kelly criterion sizing
- **Custom Exception Hierarchy** (`production/exceptions.py`):
  - `ProductionError` (base)
  - `ModelNotFoundError`
  - `FeatureDataError`
  - `PredictionError`
  - `BettingEngineError`
  - `InsufficientDataError`
- **Centralized Logging** (`production/logging_config.py`):
  - Consistent logging format
  - Console and file handlers
  - Configurable log levels
- **Comprehensive Test Suite** (80%+ coverage):
  - Unit tests for all core modules
  - Integration tests for full pipeline
  - Pytest fixtures for reusable test data
  - 99/101 tests passing (98% pass rate)
- **Production Architecture Documentation** (`docs/production_architecture.md`):
  - Design principles
  - Module responsibilities
  - Data flow diagrams
  - Testing strategy
  - Deployment guide

### Changed
- **Production folder structure** - Improved modularity and separation of concerns
- **Error handling** - Graceful degradation with specific exception types
- **Logging** - Centralized configuration replacing scattered print statements
- **Module organization** - Clear separation between core, infrastructure, and legacy code
- **Documentation** - Updated `production/README.md` with v2.0 CLI commands and migration guide

### Deprecated
- `production/run_daily.py` → Use `nba-pra predict`
- `production/run_full_pipeline.py` → Use `nba-pra pipeline`
- `production/recommend_bets.py` → Use `nba-pra recommend`
- All deprecated scripts now show warnings with migration instructions
- **Scheduled for removal in v3.0.0**

### Fixed
- Removed cross-module dependencies on `backtest/` directory
- Production system is now self-contained and deployment-ready
- Improved testability with dependency injection pattern
- No PYTHONPATH required for CLI commands

### Technical Details
- **Test Coverage**: 99/101 tests passing (98% pass rate)
- **Linting**: Ruff check passes with minor warnings (wildcard imports, import order)
- **CLI Framework**: Click 8.x
- **Python**: 3.11+
- **Package Manager**: uv (recommended) or pip

---

## [1.0.0] - 2025-10-31

### Added
- Initial production release
- **19-fold CV ensemble** with time-series cross-validation
- **Monte Carlo probabilistic predictions** using Gamma distributions
- **TheOddsAPI integration** for real-time betting lines
- **Kelly criterion bet sizing** with confidence filtering
- **Daily pipeline orchestrator** (`run_daily.py`)
- **Full pipeline orchestrator** (`run_full_pipeline.py`)
- **Bet recommendation script** (`recommend_bets.py`)
- **Production modules**:
  - `config.py` - Centralized configuration
  - `model_trainer.py` - 3-year rolling window training
  - `predictor.py` - Ensemble predictions
  - `betting_engine.py` - Bet decision logic
  - `ledger.py` - Bet tracking
  - `odds_fetcher.py` - TheOddsAPI client
- **Expected Performance** (from backtesting):
  - Win Rate: 68-75%
  - ROI: 47-55%
  - Edge: 3-8% on actionable bets

### Features
- 3-year rolling training window
- 19 time-series CV folds (matches backtest methodology)
- Mean + variance predictions for uncertainty quantification
- Confidence filtering (≥60%)
- Edge filtering (≥3%)
- Volatility filtering (CV ≤35%)
- Automated daily pipeline
- Cron job support for overnight automation

---

## Migration Guide (v1.0 → v2.0)

### Command Changes

| Old Command (v1.0) | New Command (v2.0) |
|-------------------|-------------------|
| `PYTHONPATH=/path uv run python production/run_daily.py` | `nba-pra predict` |
| `PYTHONPATH=/path uv run python production/run_full_pipeline.py --auto-fetch-data` | `nba-pra pipeline --full` |
| `PYTHONPATH=/path uv run python production/model_trainer.py` | `nba-pra train` |
| `PYTHONPATH=/path uv run python production/recommend_bets.py --min-edge 0.05` | `nba-pra recommend --min-edge 0.05` |

### Benefits of v2.0

✅ **No PYTHONPATH required** - CLI handles imports automatically
✅ **Shorter commands** - `nba-pra predict` vs long paths
✅ **Unified interface** - Single entry point for all operations
✅ **Better help** - `nba-pra --help` shows all commands
✅ **Improved error handling** - Consistent logging and exceptions
✅ **Self-contained** - No dependencies on backtest/ directory
✅ **Better testability** - 80%+ test coverage

### Updating Your Automation

If you have cron jobs using v1.0 commands:

```bash
# OLD (v1.0)
0 2 * * * cd /path/to/NBA_PRA && PYTHONPATH=/path/to/NBA_PRA /usr/local/bin/uv run python production/run_full_pipeline.py --auto-fetch-data

# NEW (v2.0)
0 2 * * * cd /path/to/NBA_PRA && /usr/local/bin/uv run nba-pra pipeline --full
```

---

## Future Plans (v3.0)

- **Remove deprecated scripts** (run_daily.py, run_full_pipeline.py, recommend_bets.py)
- **Database ledger** - Replace CSV with SQLite/PostgreSQL
- **Multiple bookmakers** - Compare lines across providers
- **Live betting** - Real-time predictions during games
- **Web dashboard** - Flask/FastAPI frontend
- **Model versioning** - Track model performance over time
- **A/B testing** - Compare model architectures
- **Alerting** - Slack/email notifications for high-edge bets

---

## Support

- **Documentation**: `production/README.md`, `docs/production_architecture.md`
- **Issues**: File issues for bugs or feature requests
- **Logs**: Check `logs/` directory for detailed execution logs

---

## Contributors

NBA PRA Prediction System
Developed by: Hivin Diyagama (diyagamah)

---

## License

Internal use only.
