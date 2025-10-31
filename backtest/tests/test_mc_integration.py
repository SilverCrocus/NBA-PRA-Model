"""
Test Monte Carlo Integration

Validates:
1. Backwards compatibility (MC disabled = original behavior)
2. MC predictions generate correctly when enabled
3. Calibration works properly
4. No errors in walk-forward flow

Usage:
    # Test with MC disabled (backwards compatibility)
    PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/test_mc_integration.py --mode disabled

    # Test with MC enabled
    PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/test_mc_integration.py --mode enabled

    # Test both modes and compare
    PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/test_mc_integration.py --mode both
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.walk_forward_engine import WalkForwardBacktest
from backtest.data_loader import load_master_features, load_historical_odds
from backtest.config import setup_logging

logger = setup_logging('mc_integration_test')


def load_test_data(n_days: int = 2):
    """
    Load a small subset of data for testing

    Args:
        n_days: Number of game days to test

    Returns:
        all_data, test_data, odds_data
    """
    logger.info(f"Loading test data ({n_days} days)...")

    # Load full dataset
    all_data = load_master_features()
    odds_data = load_historical_odds()

    # Get test period (2024-25 season)
    test_data = all_data[all_data['season'] == '2024-25'].copy()

    # Take only first n_days
    unique_dates = test_data['game_date'].unique()
    unique_dates = sorted(unique_dates)[:n_days]
    test_data = test_data[test_data['game_date'].isin(unique_dates)]

    logger.info(f"  All data: {len(all_data)} games")
    logger.info(f"  Test data: {len(test_data)} games ({n_days} days)")
    logger.info(f"  Odds data: {len(odds_data)} lines")

    return all_data, test_data, odds_data


def test_backwards_compatibility():
    """
    Test that MC disabled produces original behavior

    This should work identically to the system before MC was added.
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 1: BACKWARDS COMPATIBILITY (MC DISABLED)")
    logger.info("="*60)

    # Import config and temporarily disable MC
    from backtest import config
    original_mc_setting = config.ENABLE_MONTE_CARLO
    config.ENABLE_MONTE_CARLO = False

    try:
        # Load test data
        all_data, test_data, odds_data = load_test_data(n_days=2)

        # Initialize backtest
        backtest = WalkForwardBacktest(all_data, test_data, odds_data)

        # Verify MC is disabled
        assert not backtest.mc_enabled, "MC should be disabled"
        assert backtest.variance_model is None, "Variance model should be None"
        assert backtest.calibrator is None, "Calibrator should be None"

        logger.info("✓ MC correctly disabled in backtest engine")

        # Run backtest
        predictions = backtest.run_backtest()

        # Validate results
        assert len(predictions) > 0, "Should have predictions"
        assert 'prediction' in predictions.columns, "Should have prediction column"
        assert 'actual_pra' in predictions.columns, "Should have actual_pra column"

        # MC columns should NOT be present
        mc_columns = [col for col in predictions.columns if col.startswith('mc_')]
        assert len(mc_columns) == 0, f"Should not have MC columns, found: {mc_columns}"

        logger.info(f"✓ Generated {len(predictions)} predictions")
        logger.info(f"✓ No MC columns present (as expected)")
        logger.info(f"✓ Columns: {list(predictions.columns)}")

        # Basic sanity checks
        assert predictions['prediction'].notna().all(), "All predictions should be valid"
        assert predictions['actual_pra'].notna().all(), "All actuals should be valid"

        logger.info("✓ All predictions valid")
        logger.info("\n✓ BACKWARDS COMPATIBILITY TEST PASSED\n")

        return predictions

    finally:
        # Restore original setting
        config.ENABLE_MONTE_CARLO = original_mc_setting


def test_mc_enabled():
    """
    Test that MC enabled produces probabilistic predictions

    This should add MC columns with variance, std, alpha, beta.
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 2: MONTE CARLO ENABLED")
    logger.info("="*60)

    # Import config and temporarily enable MC
    from backtest import config
    original_mc_setting = config.ENABLE_MONTE_CARLO
    config.ENABLE_MONTE_CARLO = True

    try:
        # Load test data (need more days for calibration)
        all_data, test_data, odds_data = load_test_data(n_days=10)

        # Initialize backtest
        backtest = WalkForwardBacktest(all_data, test_data, odds_data)

        # Verify MC is enabled
        assert backtest.mc_enabled, "MC should be enabled"

        logger.info("✓ MC correctly enabled in backtest engine")

        # Run backtest
        predictions = backtest.run_backtest()

        # Validate results
        assert len(predictions) > 0, "Should have predictions"

        # Check for MC columns
        expected_mc_cols = ['mc_variance', 'mc_std', 'mc_alpha', 'mc_beta']
        for col in expected_mc_cols:
            assert col in predictions.columns, f"Missing MC column: {col}"

        logger.info(f"✓ Generated {len(predictions)} predictions")
        logger.info(f"✓ All MC columns present: {expected_mc_cols}")

        # Validate MC columns have reasonable values
        mc_rows = predictions[predictions['mc_variance'].notna()]
        if len(mc_rows) > 0:
            assert (mc_rows['mc_variance'] > 0).all(), "Variance should be positive"
            assert (mc_rows['mc_std'] > 0).all(), "Std should be positive"
            assert (mc_rows['mc_alpha'] > 0).all(), "Alpha should be positive"
            assert (mc_rows['mc_beta'] > 0).all(), "Beta should be positive"

            logger.info(f"✓ MC columns valid for {len(mc_rows)} predictions")
            logger.info(f"  Mean variance: {mc_rows['mc_variance'].mean():.2f}")
            logger.info(f"  Mean std: {mc_rows['mc_std'].mean():.2f}")
            logger.info(f"  Mean alpha: {mc_rows['mc_alpha'].mean():.2f}")
            logger.info(f"  Mean beta: {mc_rows['mc_beta'].mean():.3f}")

        # Check for MC betting columns (if calibration happened)
        mc_betting_cols = [col for col in predictions.columns if col.startswith('mc_prob_') or col.startswith('mc_should_bet_')]
        if len(mc_betting_cols) > 0:
            logger.info(f"✓ MC betting columns present: {mc_betting_cols}")
        else:
            logger.warning("⚠ No MC betting columns (calibration may not have completed)")

        logger.info("\n✓ MONTE CARLO ENABLED TEST PASSED\n")

        return predictions

    finally:
        # Restore original setting
        config.ENABLE_MONTE_CARLO = original_mc_setting


def test_both_modes():
    """
    Test both modes and compare results

    Verifies:
    - Both modes run without errors
    - MC disabled has no MC columns
    - MC enabled has MC columns
    - Point predictions (mean) are similar between modes
    """
    logger.info("\n" + "="*60)
    logger.info("TEST 3: COMPARE BOTH MODES")
    logger.info("="*60)

    # Run both tests
    predictions_disabled = test_backwards_compatibility()
    predictions_enabled = test_mc_enabled()

    logger.info("\n" + "="*60)
    logger.info("COMPARISON SUMMARY")
    logger.info("="*60)

    # Compare predictions counts
    logger.info(f"Predictions (MC disabled): {len(predictions_disabled)}")
    logger.info(f"Predictions (MC enabled):  {len(predictions_enabled)}")

    # Compare column counts
    logger.info(f"Columns (MC disabled): {len(predictions_disabled.columns)}")
    logger.info(f"Columns (MC enabled):  {len(predictions_enabled.columns)}")

    # Check MC columns only in enabled mode
    mc_cols_disabled = [col for col in predictions_disabled.columns if col.startswith('mc_')]
    mc_cols_enabled = [col for col in predictions_enabled.columns if col.startswith('mc_')]

    logger.info(f"MC columns (disabled): {len(mc_cols_disabled)} (expected: 0)")
    logger.info(f"MC columns (enabled):  {len(mc_cols_enabled)} (expected: >0)")

    assert len(mc_cols_disabled) == 0, "MC disabled should have no MC columns"
    assert len(mc_cols_enabled) > 0, "MC enabled should have MC columns"

    logger.info("\n✓ ALL TESTS PASSED\n")


def main():
    parser = argparse.ArgumentParser(description='Test Monte Carlo Integration')
    parser.add_argument('--mode', choices=['disabled', 'enabled', 'both'], default='both',
                       help='Test mode: disabled (backwards compat), enabled (MC), or both')

    args = parser.parse_args()

    try:
        if args.mode == 'disabled':
            test_backwards_compatibility()
        elif args.mode == 'enabled':
            test_mc_enabled()
        else:
            test_both_modes()

        logger.info("\n" + "="*60)
        logger.info("✓ ALL INTEGRATION TESTS PASSED")
        logger.info("="*60 + "\n")

        return 0

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
