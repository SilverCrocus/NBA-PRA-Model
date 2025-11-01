"""
Daily Production Pipeline Orchestrator

Main script to run daily NBA PRA prediction and betting workflow.

Usage:
    python production/run_daily.py [--skip-training] [--skip-odds] [--date YYYY-MM-DD]

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.config import (
    PREDICTIONS_DIR,
    BETS_DIR,
    setup_logging
)
from production.model_trainer import (
    train_production_models,
    get_latest_model_path,
    ProductionModelTrainer
)
from production.odds_fetcher import OddsFetcher
from production.predictor import ProductionPredictor
from production.betting_engine import BettingEngine

logger = setup_logging('run_daily')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run daily NBA PRA prediction pipeline'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use latest trained model)'
    )

    parser.add_argument(
        '--skip-odds',
        action='store_true',
        help='Skip odds fetching (generate predictions without betting lines)'
    )

    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Target date for predictions (YYYY-MM-DD), defaults to tomorrow in US/Eastern timezone'
    )

    parser.add_argument(
        '--retrain-only',
        action='store_true',
        help='Only retrain models, skip predictions'
    )

    return parser.parse_args()


class DailyPipeline:
    """
    Orchestrates daily production workflow

    Steps:
    1. Train models (optional, can use cached)
    2. Fetch betting lines from TheOddsAPI
    3. Generate predictions with probabilities
    4. Calculate betting decisions
    5. Export results
    """

    def __init__(self, args):
        """
        Initialize pipeline

        Args:
            args: Command line arguments
        """
        self.args = args

        # Set target date (use EST timezone for NBA games)
        if args.date:
            self.target_date = args.date
        else:
            # Default to today in US/Eastern timezone (where odds are usually available)
            est = pytz.timezone('US/Eastern')
            today = datetime.now(est)
            self.target_date = today.strftime('%Y-%m-%d')
            logger.info(f"No date specified, using today EST: {self.target_date}")

        logger.info(f"\n{'='*70}")
        logger.info(f"DAILY PRODUCTION PIPELINE - {self.target_date}")
        logger.info(f"{'='*70}\n")

    def step_1_train_models(self):
        """Step 1: Train models (or load latest)"""
        logger.info("STEP 1: Model Training")
        logger.info("-" * 70)

        if self.args.skip_training:
            logger.info("Skipping training (using latest model)...")
            model_path = get_latest_model_path()

            if model_path is None:
                logger.error("No trained models found! Please train first.")
                sys.exit(1)

            logger.info(f"Using model: {model_path.name}")
            return model_path

        else:
            logger.info("Training new ensemble...")
            train_production_models(save=True)

            model_path = get_latest_model_path()
            logger.info(f"✓ Models trained and saved: {model_path.name}\n")

            return model_path

    def step_2_fetch_odds(self):
        """Step 2: Fetch betting lines from TheOddsAPI"""
        logger.info("STEP 2: Fetch Betting Lines")
        logger.info("-" * 70)

        if self.args.skip_odds:
            logger.info("Skipping odds fetching\n")
            return None

        try:
            fetcher = OddsFetcher()
            pra_lines = fetcher.get_all_pra_lines(target_date=self.target_date)

            if pra_lines.empty:
                logger.warning(f"No betting lines found for {self.target_date}")
                logger.warning("Predictions will be generated without lines\n")
                return None

            logger.info(f"✓ Fetched {len(pra_lines)} PRA lines")
            logger.info(f"  Bookmakers: {pra_lines['bookmaker'].nunique()}")
            logger.info(f"  Players: {pra_lines['player_name'].nunique()}\n")

            return pra_lines

        except Exception as e:
            logger.error(f"Error fetching odds: {e}")
            logger.warning("Continuing without betting lines\n")
            return None

    def step_3_generate_predictions(self, model_path, odds_df):
        """Step 3: Generate predictions with probabilities"""
        logger.info("STEP 3: Generate Predictions")
        logger.info("-" * 70)

        try:
            # Load model
            ensemble_data = ProductionModelTrainer.load_ensemble(str(model_path))

            # Create predictor
            predictor = ProductionPredictor(ensemble_data)

            # Generate predictions
            predictions = predictor.predict_with_probabilities(
                self.target_date,
                betting_lines=odds_df
            )

            if predictions.empty:
                logger.warning(f"No predictions generated for {self.target_date}")
                return None

            logger.info(f"✓ Generated {len(predictions)} predictions")

            # Save predictions
            pred_filename = f"predictions_{self.target_date}.csv"
            pred_path = PREDICTIONS_DIR / pred_filename
            predictions.to_csv(pred_path, index=False)

            logger.info(f"  Saved to: {pred_filename}\n")

            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            raise

    def step_4_calculate_bets(self, predictions):
        """Step 4: Calculate betting decisions"""
        logger.info("STEP 4: Calculate Betting Decisions")
        logger.info("-" * 70)

        if predictions is None or predictions.empty:
            logger.warning("No predictions to process")
            return None

        # Check if we have probabilities
        if 'prob_over' not in predictions.columns:
            logger.warning("No probabilities in predictions (Monte Carlo disabled?)")
            return None

        # Check if we have betting lines
        if 'pra_line' not in predictions.columns or predictions['pra_line'].isna().all():
            logger.warning("No betting lines available")
            return None

        try:
            # Create betting engine
            engine = BettingEngine()

            # Process predictions → bets
            bets = engine.process_predictions(
                predictions,
                export=True,
                filename=f"bets_{self.target_date}.csv"
            )

            if bets.empty:
                logger.warning("No bets passed filtering criteria")
                return None

            logger.info(f"✓ Generated {len(bets)} betting recommendations\n")

            return bets

        except Exception as e:
            logger.error(f"Error calculating bets: {e}")
            raise

    def run(self):
        """Execute full pipeline"""
        start_time = datetime.now()

        try:
            # Step 1: Train models
            model_path = self.step_1_train_models()

            if self.args.retrain_only:
                logger.info("✓ Retraining complete. Exiting.\n")
                return

            # Step 2: Fetch odds
            odds_df = self.step_2_fetch_odds()

            # Step 3: Generate predictions
            predictions = self.step_3_generate_predictions(model_path, odds_df)

            # Step 4: Calculate bets
            bets = self.step_4_calculate_bets(predictions)

            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()

            logger.info(f"{'='*70}")
            logger.info(f"PIPELINE COMPLETE")
            logger.info(f"{'='*70}")
            logger.info(f"Target date: {self.target_date}")
            logger.info(f"Predictions: {len(predictions) if predictions is not None else 0}")
            logger.info(f"Bets: {len(bets) if bets is not None else 0}")
            logger.info(f"Runtime: {elapsed:.1f}s")
            logger.info(f"{'='*70}\n")

            # Display best bets
            if bets is not None and not bets.empty:
                logger.info("TOP 10 BETTING RECOMMENDATIONS:")
                logger.info("-" * 70)

                top_bets = bets.head(10)

                for idx, bet in top_bets.iterrows():
                    logger.info(
                        f"{bet['player_name']:20s} | {bet['direction']:5s} {bet['betting_line']:5.1f} | "
                        f"Pred: {bet['mean_pred']:5.1f} | Edge: {bet['edge']:5.1%} | "
                        f"Win%: {bet['prob_win']:5.1%} | Kelly: {bet['kelly_size']:.3f}"
                    )

                logger.info(f"{'='*70}\n")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            sys.exit(1)


def main():
    """Main entry point"""
    args = parse_arguments()

    pipeline = DailyPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
