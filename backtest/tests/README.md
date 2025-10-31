# Backtest Tests

Testing and validation tools for the NBA PRA backtesting system.

## Test Files

### test_mc_integration.py
**Purpose**: Integration testing for Monte Carlo system

**Tests:**
- Backwards compatibility (MC disabled = original behavior)
- MC predictions generate correctly when enabled
- Calibration works properly
- No errors in walk-forward flow

**Usage:**
```bash
# Test backwards compatibility (MC disabled)
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/test_mc_integration.py --mode disabled

# Test MC enabled
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/test_mc_integration.py --mode enabled

# Test both modes and compare
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/test_mc_integration.py --mode both
```

**When to run:**
- Before committing changes to MC code
- After updating dependencies
- To verify system health

---

### validate_mc_predictions.py
**Purpose**: Validate Monte Carlo prediction quality

**Validates:**
- Calibration quality (ECE < 0.05 = well-calibrated)
- Coverage (95% intervals contain 93-97% of outcomes)
- PIT uniformity (predictions are unbiased)
- Variance prediction accuracy
- Edge monotonicity (higher edge → higher win rate)
- Betting performance comparison (MC vs baseline)

**Usage:**
```bash
# Validate predictions from a backtest run
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/validate_mc_predictions.py \
    --predictions backtest/results/daily_predictions.csv

# Generate full calibration report with visualizations
PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/validate_mc_predictions.py \
    --predictions backtest/results/daily_predictions.csv \
    --generate-visuals
```

**When to run:**
- After every MC-enabled backtest
- When tuning MC parameters
- To assess production readiness

**Output:**
- Console report with validation metrics
- Visualizations saved to `backtest/monte_carlo/visuals/`:
  - `calibration_curve.png` - Reliability diagram
  - `pit_histogram.png` - Uniformity check
  - `coverage_analysis.png` - Interval coverage
  - `edge_calibration.png` - Edge vs win rate
  - `variance_analysis.png` - Variance prediction quality

---

## Quick Reference

### Pass/Fail Criteria

**Calibration:**
- ✓ ECE < 0.05
- ✓ Coverage: 93-97%
- ✓ PIT KS p-value > 0.05

**Variance:**
- ✓ Correlation > 0.5
- ✓ Ratio (predicted/actual): 0.8-1.2

**Edge:**
- ✓ Monotonic (higher edge → higher win rate)

**Betting:**
- ✓ MC ROI ≥ Baseline ROI

### Troubleshooting

**ECE > 0.05 (poorly calibrated):**
- Increase calibration dataset size
- Apply conformal calibration
- Adjust variance scaling

**Coverage < 93% (intervals too narrow):**
- Increase `MC_MIN_VARIANCE` in config.py
- Apply conformal calibration

**Variance correlation < 0.5:**
- Add variance-specific features
- Retrain variance model
- Check for data leakage

**Edge not monotonic:**
- Recalibrate probabilities
- Adjust confidence thresholds
- Review variance model
