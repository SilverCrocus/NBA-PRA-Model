"""
Validate Monte Carlo Predictions

Comprehensive validation of MC predictions including:
- Calibration quality (ECE, coverage, PIT)
- Variance prediction accuracy
- Edge monotonicity
- Betting performance comparison
- Diagnostic visualizations

Usage:
    # Validate MC predictions from a backtest run
    PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/validate_mc_predictions.py \
        --predictions backtest/results/daily_predictions.csv

    # Generate full calibration report with visualizations
    PYTHONPATH=/Users/diyagamah/Documents/NBA_PRA uv run python backtest/tests/validate_mc_predictions.py \
        --predictions backtest/results/daily_predictions.csv \
        --generate-visuals

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.config import setup_logging
from backtest.monte_carlo.calibration import evaluate_calibration
from backtest.monte_carlo.visualization import create_calibration_report
from backtest.monte_carlo.distribution_fitting import calculate_probability_over_line

logger = setup_logging('mc_validation')


def load_predictions(predictions_path: Path) -> pd.DataFrame:
    """
    Load predictions from CSV

    Args:
        predictions_path: Path to daily_predictions.csv

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading predictions from {predictions_path}")

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    df = pd.read_csv(predictions_path)
    logger.info(f"  Loaded {len(df)} predictions")

    # Check for MC columns
    mc_cols = [col for col in df.columns if col.startswith('mc_')]
    if len(mc_cols) == 0:
        raise ValueError("No MC columns found. Was Monte Carlo enabled during backtest?")

    logger.info(f"  Found {len(mc_cols)} MC columns")

    return df


def validate_calibration(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate calibration quality

    Checks:
    - Expected Calibration Error (ECE < 0.05 = well-calibrated)
    - Coverage (95% intervals should contain 95% of outcomes)
    - PIT uniformity (KS test)

    Args:
        predictions: DataFrame with MC predictions

    Returns:
        Dictionary with calibration metrics
    """
    logger.info("\n" + "="*60)
    logger.info("CALIBRATION VALIDATION")
    logger.info("="*60)

    # Filter to predictions with MC columns
    mc_predictions = predictions[
        predictions[['mc_alpha', 'mc_beta', 'actual_pra']].notna().all(axis=1)
    ].copy()

    if len(mc_predictions) == 0:
        logger.warning("No MC predictions with actuals found")
        return {}

    logger.info(f"Validating {len(mc_predictions)} MC predictions")

    # Evaluate calibration
    metrics = evaluate_calibration(
        y_true=mc_predictions['actual_pra'].values,
        alpha=mc_predictions['mc_alpha'].values,
        beta=mc_predictions['mc_beta'].values
    )

    # Display results
    logger.info("\nCalibration Metrics:")
    logger.info(f"  ECE (Expected Calibration Error): {metrics['ece']:.4f}")
    logger.info(f"    {'✓ Well-calibrated' if metrics['ece'] < 0.05 else '✗ Poorly calibrated'} (target: < 0.05)")

    logger.info(f"\n  Coverage (95% interval): {metrics['coverage']:.1%}")
    logger.info(f"    {'✓ Good coverage' if 0.93 <= metrics['coverage'] <= 0.97 else '✗ Poor coverage'} (target: 93-97%)")

    logger.info(f"\n  PIT KS statistic: {metrics['ks_statistic']:.4f}")
    logger.info(f"  PIT KS p-value: {metrics['ks_pvalue']:.4f}")
    logger.info(f"    {'✓ Uniform' if metrics['ks_pvalue'] > 0.05 else '✗ Non-uniform'} (target: p > 0.05)")

    logger.info(f"\n  Is calibrated: {metrics['is_calibrated']}")

    # Overall assessment
    if metrics['is_calibrated']:
        logger.info("\n✓ CALIBRATION PASSED - Model is well-calibrated")
    else:
        logger.warning("\n✗ CALIBRATION FAILED - Model needs recalibration")
        logger.warning("  Consider:")
        logger.warning("  1. Increasing calibration dataset size")
        logger.warning("  2. Adjusting variance scaling")
        logger.warning("  3. Using different distribution family")

    return metrics


def validate_variance_predictions(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate variance prediction accuracy

    Checks correlation between predicted variance and actual squared residuals.

    Args:
        predictions: DataFrame with MC predictions

    Returns:
        Dictionary with variance metrics
    """
    logger.info("\n" + "="*60)
    logger.info("VARIANCE PREDICTION VALIDATION")
    logger.info("="*60)

    # Filter to predictions with MC columns
    mc_predictions = predictions[
        predictions[['mc_variance', 'prediction', 'actual_pra']].notna().all(axis=1)
    ].copy()

    if len(mc_predictions) == 0:
        logger.warning("No MC predictions with actuals found")
        return {}

    # Calculate actual squared residuals
    mc_predictions['squared_residual'] = (
        mc_predictions['actual_pra'] - mc_predictions['prediction']
    ) ** 2

    # Calculate correlation
    correlation = mc_predictions[['mc_variance', 'squared_residual']].corr().iloc[0, 1]

    # Calculate metrics
    mean_pred_var = mc_predictions['mc_variance'].mean()
    mean_actual_var = mc_predictions['squared_residual'].mean()
    var_ratio = mean_pred_var / mean_actual_var

    logger.info(f"Variance correlation: {correlation:.3f}")
    logger.info(f"  {'✓ Good correlation' if correlation > 0.5 else '✗ Weak correlation'} (target: > 0.5)")

    logger.info(f"\nMean predicted variance: {mean_pred_var:.2f}")
    logger.info(f"Mean actual variance: {mean_actual_var:.2f}")
    logger.info(f"Ratio (predicted/actual): {var_ratio:.2f}")
    logger.info(f"  {'✓ Well-scaled' if 0.8 <= var_ratio <= 1.2 else '✗ Needs scaling'} (target: 0.8-1.2)")

    metrics = {
        'correlation': correlation,
        'mean_pred_variance': mean_pred_var,
        'mean_actual_variance': mean_actual_var,
        'variance_ratio': var_ratio,
        'is_well_correlated': correlation > 0.5,
        'is_well_scaled': 0.8 <= var_ratio <= 1.2
    }

    if metrics['is_well_correlated'] and metrics['is_well_scaled']:
        logger.info("\n✓ VARIANCE PREDICTION PASSED")
    else:
        logger.warning("\n✗ VARIANCE PREDICTION NEEDS IMPROVEMENT")
        if not metrics['is_well_correlated']:
            logger.warning("  Low correlation - consider adding variance-specific features")
        if not metrics['is_well_scaled']:
            logger.warning(f"  Poor scaling - apply correction factor: {1/var_ratio:.2f}x")

    return metrics


def validate_edge_monotonicity(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate edge calibration

    Higher predicted edge should yield higher win rate.

    Args:
        predictions: DataFrame with MC predictions

    Returns:
        Dictionary with edge metrics
    """
    logger.info("\n" + "="*60)
    logger.info("EDGE MONOTONICITY VALIDATION")
    logger.info("="*60)

    # Filter to bets with betting lines
    bets = predictions[
        predictions['betting_line'].notna() &
        predictions[['mc_alpha', 'mc_beta']].notna().all(axis=1)
    ].copy()

    if len(bets) == 0:
        logger.warning("No bets with MC predictions found")
        return {}

    # Calculate P(PRA > line)
    bets['prob_over'] = calculate_probability_over_line(
        alpha=bets['mc_alpha'].values,
        beta=bets['mc_beta'].values,
        betting_line=bets['betting_line'].values
    )

    # Calculate edge (prob - breakeven)
    bets['edge'] = bets['prob_over'] - 0.524

    # Calculate actual outcome
    bets['actual_over'] = (bets['actual_pra'] > bets['betting_line']).astype(int)

    # Bin by edge and calculate win rate
    bets['edge_bin'] = pd.cut(bets['edge'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    edge_analysis = bets.groupby('edge_bin', observed=True).agg({
        'edge': 'mean',
        'actual_over': 'mean',
        'prob_over': 'mean',
        'player_id': 'count'
    }).rename(columns={'player_id': 'n_bets'})

    logger.info("\nEdge vs Win Rate:")
    logger.info(edge_analysis.to_string())

    # Check monotonicity
    win_rates = edge_analysis['actual_over'].values
    is_monotonic = all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1))

    metrics = {
        'edge_analysis': edge_analysis,
        'is_monotonic': is_monotonic,
        'n_bets': len(bets)
    }

    if is_monotonic:
        logger.info("\n✓ EDGE MONOTONICITY PASSED - Higher edge → higher win rate")
    else:
        logger.warning("\n✗ EDGE MONOTONICITY FAILED - Edge not well-calibrated")
        logger.warning("  Consider retraining variance model or adjusting calibration")

    return metrics


def compare_betting_performance(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Compare original vs MC betting performance

    Args:
        predictions: DataFrame with both original and MC betting columns

    Returns:
        Dictionary with performance comparison
    """
    logger.info("\n" + "="*60)
    logger.info("BETTING PERFORMANCE COMPARISON")
    logger.info("="*60)

    # Filter to bets with betting lines
    bets = predictions[predictions['betting_line'].notna()].copy()

    if len(bets) == 0:
        logger.warning("No bets found")
        return {}

    # Original betting performance
    original_bets = bets[bets['bet_decision'].notna()]
    if len(original_bets) > 0:
        original_win_rate = original_bets['bet_correct'].mean()
        original_roi = (original_bets['profit'].sum() / len(original_bets)) * 100
        logger.info("\nOriginal Betting (point estimates):")
        logger.info(f"  Bets: {len(original_bets)}")
        logger.info(f"  Win Rate: {original_win_rate:.1%}")
        logger.info(f"  ROI: {original_roi:.2f}%")
    else:
        logger.warning("No original bets found")
        original_win_rate = np.nan
        original_roi = np.nan

    # MC betting performance
    mc_bets = bets[
        (bets['mc_should_bet_over'] == True) | (bets['mc_should_bet_under'] == True)
    ] if 'mc_should_bet_over' in bets.columns else pd.DataFrame()

    if len(mc_bets) > 0:
        # Calculate MC bet outcomes
        mc_bets['mc_correct'] = np.where(
            mc_bets['mc_should_bet_over'],
            mc_bets['actual_pra'] > mc_bets['betting_line'],
            mc_bets['actual_pra'] <= mc_bets['betting_line']
        )
        mc_bets['mc_profit'] = np.where(
            mc_bets['mc_correct'],
            0.909,  # Win
            -1.0    # Loss
        )

        mc_win_rate = mc_bets['mc_correct'].mean()
        mc_roi = (mc_bets['mc_profit'].sum() / len(mc_bets)) * 100

        logger.info("\nMonte Carlo Betting (probabilistic):")
        logger.info(f"  Bets: {len(mc_bets)}")
        logger.info(f"  Win Rate: {mc_win_rate:.1%}")
        logger.info(f"  ROI: {mc_roi:.2f}%")

        # Comparison
        if not np.isnan(original_win_rate):
            logger.info("\nComparison:")
            logger.info(f"  Win Rate: {mc_win_rate:.1%} vs {original_win_rate:.1%} (MC vs Original)")
            logger.info(f"  ROI: {mc_roi:.2f}% vs {original_roi:.2f}%")
            logger.info(f"  Bet Volume: {len(mc_bets)} vs {len(original_bets)} ({len(mc_bets)/len(original_bets)*100:.1f}% of original)")

            if mc_roi > original_roi:
                logger.info("\n✓ MC BETTING OUTPERFORMS - Higher ROI through selectivity")
            else:
                logger.warning("\n⚠ MC BETTING UNDERPERFORMS - May need tuning")
    else:
        logger.warning("No MC bets found (calibration may not have completed)")
        mc_win_rate = np.nan
        mc_roi = np.nan

    metrics = {
        'original_n_bets': len(original_bets) if len(original_bets) > 0 else 0,
        'original_win_rate': original_win_rate,
        'original_roi': original_roi,
        'mc_n_bets': len(mc_bets),
        'mc_win_rate': mc_win_rate,
        'mc_roi': mc_roi
    }

    return metrics


def generate_visualizations(predictions: pd.DataFrame, output_dir: Path):
    """
    Generate full calibration report with visualizations

    Args:
        predictions: DataFrame with MC predictions
        output_dir: Directory to save visualizations
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING CALIBRATION VISUALIZATIONS")
    logger.info("="*60)

    # Filter to predictions with MC columns and betting lines
    mc_predictions = predictions[
        predictions[['mc_alpha', 'mc_beta', 'actual_pra', 'betting_line']].notna().all(axis=1)
    ].copy()

    if len(mc_predictions) == 0:
        logger.warning("No MC predictions with betting lines found")
        return

    logger.info(f"Generating visuals for {len(mc_predictions)} predictions")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate report
    report = create_calibration_report(
        y_true=mc_predictions['actual_pra'].values,
        alpha=mc_predictions['mc_alpha'].values,
        beta=mc_predictions['mc_beta'].values,
        betting_lines=mc_predictions['betting_line'].values,
        mean_pred=mc_predictions['prediction'].values,
        var_pred=mc_predictions['mc_variance'].values,
        output_dir=str(output_dir)
    )

    logger.info(f"\n✓ Visualizations saved to {output_dir}")
    logger.info("  Files created:")
    for plot_file in output_dir.glob("*.png"):
        logger.info(f"    - {plot_file.name}")


def main():
    parser = argparse.ArgumentParser(description='Validate Monte Carlo Predictions')
    parser.add_argument('--predictions', type=Path,
                       default=Path('backtest/results/daily_predictions.csv'),
                       help='Path to predictions CSV')
    parser.add_argument('--generate-visuals', action='store_true',
                       help='Generate calibration visualizations')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('backtest/monte_carlo/visuals'),
                       help='Output directory for visualizations')

    args = parser.parse_args()

    try:
        # Load predictions
        predictions = load_predictions(args.predictions)

        # Run validations
        calibration_metrics = validate_calibration(predictions)
        variance_metrics = validate_variance_predictions(predictions)
        edge_metrics = validate_edge_monotonicity(predictions)
        betting_metrics = compare_betting_performance(predictions)

        # Generate visualizations if requested
        if args.generate_visuals:
            generate_visualizations(predictions, args.output_dir)

        # Final summary
        logger.info("\n" + "="*60)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*60)

        passed_checks = []
        failed_checks = []

        if calibration_metrics.get('is_calibrated', False):
            passed_checks.append("✓ Calibration")
        else:
            failed_checks.append("✗ Calibration")

        if variance_metrics.get('is_well_correlated', False) and variance_metrics.get('is_well_scaled', False):
            passed_checks.append("✓ Variance Prediction")
        else:
            failed_checks.append("✗ Variance Prediction")

        if edge_metrics.get('is_monotonic', False):
            passed_checks.append("✓ Edge Monotonicity")
        else:
            failed_checks.append("✗ Edge Monotonicity")

        logger.info("\nPassed Checks:")
        for check in passed_checks:
            logger.info(f"  {check}")

        if failed_checks:
            logger.info("\nFailed Checks:")
            for check in failed_checks:
                logger.info(f"  {check}")

        if len(failed_checks) == 0:
            logger.info("\n✓ ALL VALIDATIONS PASSED - MC system is production-ready")
            return 0
        else:
            logger.warning(f"\n⚠ {len(failed_checks)} VALIDATION(S) FAILED - Review and retune")
            return 1

    except Exception as e:
        logger.error(f"\n✗ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
