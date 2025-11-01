"""
Full Production Pipeline

⚠️ DEPRECATED: This script is deprecated as of v2.0.0 and will be removed in v3.0.0

Use the new unified CLI instead:
    OLD: PYTHONPATH=/path/to/NBA_PRA uv run python production/run_full_pipeline.py --auto-fetch-data
    NEW: nba-pra pipeline --full

Migration guide:
    python production/run_full_pipeline.py --auto-fetch-data
    →  nba-pra pipeline --full

    python production/run_full_pipeline.py --skip-data-update
    →  nba-pra pipeline --skip-data-update

    python production/run_full_pipeline.py --skip-training
    →  nba-pra pipeline --skip-training

For more information, see production/README.md or docs/production_architecture.md

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import argparse
import warnings

# Show deprecation warning
warnings.warn(
    "\n\n"
    "=" * 80 + "\n"
    "⚠️  DEPRECATION WARNING\n"
    "=" * 80 + "\n"
    "This script (run_full_pipeline.py) is deprecated and will be removed in v3.0.0\n\n"
    "Please use the new unified CLI instead:\n"
    "  OLD: PYTHONPATH=/path uv run python production/run_full_pipeline.py --auto-fetch-data\n"
    "  NEW: nba-pra pipeline --full\n\n"
    "See production/README.md for migration guide.\n"
    "=" * 80 + "\n",
    DeprecationWarning,
    stacklevel=2
)
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.config import setup_logging

logger = setup_logging('full_pipeline')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run complete NBA PRA production pipeline'
    )

    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Target date for predictions (YYYY-MM-DD), defaults to tomorrow'
    )

    parser.add_argument(
        '--skip-data-update',
        action='store_true',
        help='Skip NBA data fetching (use existing data)'
    )

    parser.add_argument(
        '--auto-fetch-data',
        action='store_true',
        help='Automatically fetch latest NBA data (runs data_loader.py)'
    )

    parser.add_argument(
        '--skip-feature-engineering',
        action='store_true',
        help='Skip feature regeneration (use existing features)'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use latest cached model)'
    )

    return parser.parse_args()


def run_command(cmd: list, description: str, cwd: Path = None) -> bool:
    """
    Run a shell command and return success status

    Args:
        cmd: Command as list of strings
        description: Human-readable description
        cwd: Working directory (defaults to project root)

    Returns:
        True if successful, False otherwise
    """
    cwd = cwd or project_root

    logger.info(f"\n{'='*70}")
    logger.info(f"{description}")
    logger.info(f"{'='*70}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Working directory: {cwd}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True
        )

        logger.info(f"✓ {description} completed successfully\n")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        logger.error(f"Error: {e}\n")
        return False


def main():
    """Main pipeline orchestrator"""
    args = parse_arguments()

    # Set target date
    if args.date:
        target_date = args.date
    else:
        tomorrow = datetime.now() + timedelta(days=1)
        target_date = tomorrow.strftime('%Y-%m-%d')

    logger.info(f"\n{'#'*70}")
    logger.info(f"# FULL PRODUCTION PIPELINE - {target_date}")
    logger.info(f"{'#'*70}\n")

    start_time = datetime.now()
    steps_completed = 0
    total_steps = 3

    # STEP 1: Update NBA Data
    if args.auto_fetch_data:
        logger.info("STEP 1/3: Update NBA Data (Auto-Fetch)")

        cmd = ['uv', 'run', 'feature_engineering/data_loader.py']

        if not run_command(cmd, "Fetch Latest NBA Data", cwd=project_root):
            logger.error("Data fetching failed. Aborting.")
            sys.exit(1)

        steps_completed += 1

    elif not args.skip_data_update:
        logger.info("STEP 1/3: Update NBA Data (Manual Confirmation)")

        logger.warning("⚠️  Data update step requires confirmation")
        logger.warning("    Run: uv run feature_engineering/data_loader.py")
        logger.warning("    OR: Use --auto-fetch-data to fetch automatically")
        logger.warning("    OR: Use --skip-data-update if data is current\n")

        response = input("Have you updated NBA data already? (y/n): ").lower()
        if response != 'y':
            logger.error("Please update NBA data first, then re-run this script")
            logger.error("  Option 1: Run data_loader.py manually, then use --skip-data-update")
            logger.error("  Option 2: Re-run with --auto-fetch-data flag")
            sys.exit(1)

        steps_completed += 1
    else:
        logger.info("STEP 1/3: Update NBA Data [SKIPPED]")
        steps_completed += 1

    # STEP 2: Feature Engineering
    if not args.skip_feature_engineering:
        logger.info(f"\nSTEP 2/3: Feature Engineering")

        cmd = ['uv', 'run', 'feature_engineering/run_pipeline.py']

        if not run_command(cmd, "Feature Engineering Pipeline", cwd=project_root):
            logger.error("Feature engineering failed. Aborting.")
            sys.exit(1)

        steps_completed += 1
    else:
        logger.info("STEP 2/3: Feature Engineering [SKIPPED]")
        steps_completed += 1

    # STEP 3: Production Pipeline (train + predict + bets)
    logger.info(f"\nSTEP 3/3: Production Pipeline (Train + Predict + Bets)")

    cmd = [
        'uv', 'run', 'python', 'production/run_daily.py',
        '--date', target_date
    ]

    if args.skip_training:
        cmd.append('--skip-training')

    # Set PYTHONPATH
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            check=True,
            capture_output=False,
            text=True
        )

        logger.info(f"✓ Production pipeline completed successfully\n")
        steps_completed += 1

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Production pipeline failed with exit code {e.returncode}\n")
        sys.exit(1)

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info(f"\n{'#'*70}")
    logger.info(f"# PIPELINE COMPLETE")
    logger.info(f"{'#'*70}")
    logger.info(f"Target date: {target_date}")
    logger.info(f"Steps completed: {steps_completed}/{total_steps}")
    logger.info(f"Total runtime: {elapsed/60:.1f} minutes")
    logger.info(f"{'#'*70}\n")

    logger.info("📁 Output files:")
    logger.info(f"   Predictions: production/outputs/predictions/predictions_{target_date}.csv")
    logger.info(f"   Bets: production/outputs/bets/bets_{target_date}.csv\n")


if __name__ == "__main__":
    main()
