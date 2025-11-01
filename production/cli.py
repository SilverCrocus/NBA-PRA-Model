"""
Unified Production CLI

Single command-line interface for all production operations.

Usage:
    nba-pra predict --date 2024-11-01
    nba-pra train --cv-folds 19
    nba-pra recommend --min-edge 0.05
    nba-pra pipeline --full  # Complete pipeline
    nba-pra status

Author: NBA PRA Prediction System
Date: 2025-11-01
"""
import click
from pathlib import Path
from datetime import datetime, timedelta
import sys
import pytz
import pandas as pd
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.config import (
    PREDICTIONS_DIR,
    BETS_DIR,
    LOGS_DIR
)
from production.logging_config import setup_production_logging
from production.model_trainer import (
    train_production_models,
    get_latest_model_path,
    ProductionModelTrainer
)
from production.odds_fetcher import OddsFetcher
from production.predictor import ProductionPredictor
from production.betting_engine import BettingEngine

logger = setup_production_logging('cli')


@click.group()
@click.version_option(version='1.0.0')
def cli():
    """
    NBA PRA Production Pipeline

    Unified command-line interface for NBA player prop predictions.
    """
    pass


@cli.command()
@click.option('--date', type=str, default=None, help='Target date (YYYY-MM-DD), defaults to tomorrow in US/Eastern')
@click.option('--skip-training', is_flag=True, help='Skip model training (use latest model)')
@click.option('--skip-odds', is_flag=True, help='Skip odds fetching')
@click.option('--dry-run', is_flag=True, help='Preview without executing')
def predict(date, skip_training, skip_odds, dry_run):
    """
    Generate predictions for upcoming games.

    Steps:
    1. Train models (optional)
    2. Fetch betting lines
    3. Generate predictions with probabilities
    4. Export predictions CSV
    """
    logger.info("Running PREDICT command")

    if dry_run:
        click.echo(f"[DRY RUN] Would generate predictions for date: {date or 'tomorrow (US/Eastern)'}")
        click.echo(f"[DRY RUN] Skip training: {skip_training}")
        click.echo(f"[DRY RUN] Skip odds: {skip_odds}")
        return

    # Set target date
    if date:
        target_date = date
    else:
        # Default to tomorrow in US/Eastern timezone (NBA schedule timezone)
        eastern = pytz.timezone('US/Eastern')
        tomorrow = datetime.now(eastern) + timedelta(days=1)
        target_date = tomorrow.strftime('%Y-%m-%d')

    logger.info(f"Target date: {target_date}")

    try:
        # Step 1: Train models (if not skipped)
        if not skip_training:
            logger.info("Step 1/4: Training models...")
            train_production_models(save=True)
        else:
            logger.info("Step 1/4: Skipping training (using cached model)")
            model_path = get_latest_model_path()
            if not model_path:
                logger.error("No trained model found. Run with --skip-training=False or train separately.")
                sys.exit(1)
            logger.info(f"Using model: {model_path.name}")

        # Step 2: Fetch odds (if not skipped)
        betting_lines = None
        if not skip_odds:
            logger.info("Step 2/4: Fetching betting lines...")
            try:
                fetcher = OddsFetcher()
                betting_lines = fetcher.fetch_all_pra_lines(target_date=target_date)
                logger.info(f"Fetched {len(betting_lines)} betting lines")
            except Exception as e:
                logger.warning(f"Failed to fetch odds: {e}")
                logger.info("Continuing with predictions only (no betting analysis)")
        else:
            logger.info("Step 2/4: Skipping odds fetching")

        # Step 3: Generate predictions
        logger.info("Step 3/4: Generating predictions...")
        predictor = ProductionPredictor(target_date=target_date)
        predictions_df = predictor.generate_predictions(betting_lines=betting_lines)

        # Step 4: Export results
        logger.info("Step 4/4: Exporting predictions...")
        output_file = PREDICTIONS_DIR / f"predictions_{target_date}.csv"
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"✓ Predictions saved: {output_file}")

        # Summary
        logger.info("\n" + "="*60)
        logger.info("PREDICTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Date: {target_date}")
        logger.info(f"Total predictions: {len(predictions_df)}")
        if betting_lines is not None and len(betting_lines) > 0:
            matched = predictions_df['betting_line'].notna().sum()
            logger.info(f"Matched to betting lines: {matched}")
        logger.info(f"Output: {output_file}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--cv-folds', type=int, default=19, help='Number of CV folds')
@click.option('--training-window', type=int, default=3, help='Training window (years)')
@click.option('--dry-run', is_flag=True, help='Preview without executing')
def train(cv_folds, training_window, dry_run):
    """
    Train production models.

    Creates ensemble of CV models with mean and variance predictions.
    """
    logger.info("Running TRAIN command")

    if dry_run:
        click.echo(f"[DRY RUN] Would train {cv_folds}-fold ensemble")
        click.echo(f"[DRY RUN] Training window: {training_window} years")
        return

    logger.info(f"Training {cv_folds}-fold ensemble (window: {training_window} years)")

    try:
        # Train models
        train_production_models(save=True)
        logger.info("✓ Training complete")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--date', type=str, default=None, help='Target date (YYYY-MM-DD)')
@click.option('--min-edge', type=float, default=0.05, help='Minimum edge threshold')
@click.option('--min-confidence', type=float, default=0.7, help='Minimum confidence')
@click.option('--top-n', type=int, default=10, help='Show top N bets')
@click.option('--dry-run', is_flag=True, help='Preview without executing')
def recommend(date, min_edge, min_confidence, top_n, dry_run):
    """
    Recommend top bets for target date.

    Filters predictions by edge and confidence, displays best opportunities.
    """
    logger.info("Running RECOMMEND command")

    if dry_run:
        click.echo(f"[DRY RUN] Would show top {top_n} recommendations")
        click.echo(f"[DRY RUN] Min edge: {min_edge:.2%}, Min confidence: {min_confidence:.2%}")
        return

    try:
        # Find predictions file
        if date:
            pred_file = PREDICTIONS_DIR / f"predictions_{date}.csv"
        else:
            # Find latest
            files = list(PREDICTIONS_DIR.glob("predictions_*.csv"))
            if not files:
                logger.error("No prediction files found. Run 'nba-pra predict' first.")
                sys.exit(1)
            pred_file = max(files, key=lambda p: p.stat().st_mtime)

        if not pred_file.exists():
            logger.error(f"Predictions not found: {pred_file}")
            sys.exit(1)

        logger.info(f"Loading predictions from: {pred_file.name}")
        predictions = pd.read_csv(pred_file)

        # Filter and analyze
        engine = BettingEngine()
        recommendations = engine.calculate_betting_decisions(
            predictions,
            min_edge=min_edge,
            min_confidence=min_confidence
        )

        if len(recommendations) == 0:
            logger.warning("No bets meet the criteria")
            return

        # Sort by edge and take top N
        recommendations = recommendations.sort_values('edge', ascending=False).head(top_n)

        # Display recommendations
        logger.info("\n" + "="*80)
        logger.info(f"TOP {len(recommendations)} BETTING RECOMMENDATIONS")
        logger.info("="*80)

        for idx, row in recommendations.iterrows():
            logger.info(f"\n{row['player_name']} ({row['team_abbreviation']} vs {row['opponent']})")
            logger.info(f"  Direction: {row['direction']}")
            logger.info(f"  Line: {row['betting_line']:.1f}")
            logger.info(f"  Prediction: {row['mean_pred']:.1f} ± {row['std_dev']:.1f}")
            logger.info(f"  Edge: {row['edge']:.2%}")
            logger.info(f"  Confidence: {row['confidence_score']:.2%}")
            logger.info(f"  Kelly Size: {row['kelly_size']:.2%}")

        logger.info("\n" + "="*80)

    except Exception as e:
        logger.error(f"Recommendation failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
@click.option('--full', is_flag=True, help='Run full pipeline (data + features + training)')
@click.option('--date', type=str, default=None, help='Target date')
@click.option('--skip-data-update', is_flag=True, help='Skip NBA data fetch')
@click.option('--skip-feature-engineering', is_flag=True, help='Skip feature regeneration')
@click.option('--skip-training', is_flag=True, help='Skip model retraining')
@click.option('--dry-run', is_flag=True, help='Preview without executing')
def pipeline(full, date, skip_data_update, skip_feature_engineering, skip_training, dry_run):
    """
    Run complete daily pipeline.

    Orchestrates all steps: data → features → training → predictions → bets
    """
    logger.info("Running PIPELINE command")

    if dry_run:
        click.echo("[DRY RUN] Pipeline execution plan:")
        if full:
            click.echo("  - Fetch NBA data (feature_engineering/data_loader.py)")
            click.echo("  - Regenerate features (feature_engineering/run_pipeline.py)")
        else:
            if not skip_data_update:
                click.echo("  - Update NBA data")
            if not skip_feature_engineering:
                click.echo("  - Regenerate features")
        if not skip_training:
            click.echo("  - Train models")
        click.echo("  - Generate predictions")
        click.echo("  - Calculate betting decisions")
        return

    try:
        import subprocess

        if full or not skip_data_update:
            logger.info("Step 1: Updating NBA data...")
            result = subprocess.run(
                ['python', 'feature_engineering/data_loader.py'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Data update failed: {result.stderr}")
                sys.exit(1)

        if full or not skip_feature_engineering:
            logger.info("Step 2: Regenerating features...")
            result = subprocess.run(
                ['python', 'feature_engineering/run_pipeline.py'],
                cwd=project_root,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.error(f"Feature engineering failed: {result.stderr}")
                sys.exit(1)

        # Run prediction pipeline
        logger.info("Step 3: Running prediction pipeline...")
        from production.run_daily import DailyPipeline
        import argparse

        # Create args object
        args = argparse.Namespace(
            date=date,
            skip_training=skip_training,
            skip_odds=False,
            retrain_only=False
        )

        pipeline_obj = DailyPipeline(args)
        pipeline_obj.run()

        logger.info("✓ Pipeline complete")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


@cli.command()
def status():
    """
    Show production system status.

    Displays:
    - Latest model info
    - Recent predictions
    - Ledger summary
    """
    click.echo("\n" + "="*60)
    click.echo("NBA PRA PRODUCTION SYSTEM STATUS")
    click.echo("="*60)

    # Latest model
    model_path = get_latest_model_path()
    if model_path:
        click.echo(f"\n📊 Latest Model: {model_path.name}")
        click.echo(f"   Location: {model_path.parent}")
    else:
        click.echo("\n⚠️  No trained models found")
        click.echo("   Run: nba-pra train")

    # Recent predictions
    pred_files = sorted(PREDICTIONS_DIR.glob("predictions_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pred_files:
        click.echo(f"\n📈 Recent Predictions:")
        for pred_file in pred_files[:3]:
            click.echo(f"   - {pred_file.name}")
    else:
        click.echo("\n📈 No predictions found")
        click.echo("   Run: nba-pra predict")

    # Ledger summary
    try:
        from production.ledger import get_ledger_summary
        summary = get_ledger_summary()

        if summary and summary.get('total_bets', 0) > 0:
            click.echo(f"\n💰 Betting Ledger:")
            click.echo(f"   Total bets: {summary['total_bets']}")
            click.echo(f"   Win rate: {summary.get('win_rate', 0):.1%}")
            click.echo(f"   ROI: {summary.get('roi', 0):.1f}%")
        else:
            click.echo("\n💰 No betting history yet")
    except Exception as e:
        logger.debug(f"Could not load ledger: {e}")

    click.echo("\n" + "="*60)


if __name__ == '__main__':
    cli()
