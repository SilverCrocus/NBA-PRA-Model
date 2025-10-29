"""
Main Entry Point for Walk-Forward Backtest

Run complete walk-forward backtest on 2024-25 NBA season.

Usage:
    uv run backtest/run_backtest.py

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.config import setup_logging
from backtest.data_loader import load_backtest_data, validate_data_quality
from backtest.walk_forward_engine import run_walk_forward_backtest
from backtest.reporting import generate_backtest_report

logger = setup_logging('backtest_main')


def main():
    """
    Main execution function

    Steps:
    1. Load all data (master features + betting lines)
    2. Run walk-forward backtest (daily retraining)
    3. Generate summary report
    """
    try:
        logger.info("\n" + "="*70)
        logger.info("NBA PRA WALK-FORWARD BACKTEST")
        logger.info("="*70)

        # Step 1: Load data
        logger.info("\nStep 1/3: Loading data...")
        all_data, test_data, odds_data = load_backtest_data()

        # Validate data quality
        validate_data_quality(test_data)

        # Step 2: Run backtest
        logger.info("\nStep 2/3: Running walk-forward backtest...")
        logger.info("This will take ~5-6 hours (daily retraining with 19-fold CV ensemble)")
        logger.info("Progress will be saved incrementally to backtest/results/\n")

        predictions_df = run_walk_forward_backtest(all_data, test_data, odds_data)

        # Step 3: Generate report
        logger.info("\nStep 3/3: Generating summary report...")
        generate_backtest_report(predictions_df)

        # Success message
        logger.info("\n" + "="*70)
        logger.info("✓ BACKTEST COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: backtest/results/")
        logger.info(f"  - daily_predictions.csv : All predictions and outcomes")
        logger.info(f"  - daily_metrics.csv : Per-day performance metrics")
        logger.info(f"  - player_analysis.csv : Per-player performance")
        logger.info(f"  - betting_performance.csv : Overall betting summary")
        logger.info(f"  - backtest_report.md : Comprehensive analysis report")
        logger.info(f"\nOpen backtest/results/backtest_report.md to view detailed analysis.\n")

    except KeyboardInterrupt:
        logger.warning("\n\nBacktest interrupted by user")
        logger.info("Partial results may be saved in backtest/results/")
        sys.exit(1)

    except Exception as e:
        logger.error(f"\n\nERROR: Backtest failed with exception:")
        logger.error(f"  {type(e).__name__}: {str(e)}")
        logger.error("\nCheck logs for details")
        raise


if __name__ == "__main__":
    main()
