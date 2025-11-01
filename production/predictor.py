"""
Predictor Module

Generates PRA predictions with Monte Carlo probabilistic forecasts.

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.config import (
    MASTER_FEATURES_PATH,
    EXCLUDE_COLUMNS,
    ENABLE_MONTE_CARLO,
    MC_PROBABILITY_METHOD,
    MIN_CAREER_GAMES,
    MIN_RECENT_GAMES,
    setup_logging
)
from production.model_trainer import ProductionModelTrainer
from production.upcoming_games_fetcher import get_upcoming_game_features

# Monte Carlo imports
if ENABLE_MONTE_CARLO:
    from production.monte_carlo import (
        fit_gamma_parameters,
        calculate_probability_over_line,
        calculate_std_dev
    )

logger = setup_logging('predictor')


class ProductionPredictor:
    """
    Generate PRA predictions for upcoming games

    Features:
    - Ensemble mean predictions (19 models)
    - Variance predictions for uncertainty
    - Gamma distribution fitting
    - P(PRA > line) calculations
    - Player filtering (minimum games)
    """

    def __init__(self, ensemble_data: Dict):
        """
        Initialize predictor with trained ensemble

        Args:
            ensemble_data: Loaded ensemble from model_trainer
        """
        self.mean_models = ensemble_data['mean_models']
        self.variance_models = ensemble_data.get('variance_models')
        self.feature_names = ensemble_data['feature_names']
        self.n_folds = ensemble_data['n_folds']

        logger.info(f"Predictor initialized with {self.n_folds}-fold ensemble")

    def load_upcoming_games(self, target_date: str) -> pd.DataFrame:
        """
        Load features for upcoming games

        Args:
            target_date: Date to predict (YYYY-MM-DD)

        Returns:
            DataFrame with features for upcoming games
        """
        logger.info(f"Loading upcoming games for {target_date}...")

        # Fetch upcoming games and generate features
        df = get_upcoming_game_features(target_date)

        logger.info(f"Found {len(df)} player-games on {target_date}")

        return df

    def filter_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter players based on minimum games criteria

        Args:
            df: DataFrame with player games

        Returns:
            Filtered DataFrame
        """
        initial_count = len(df)

        # Filter by career games (if column exists)
        if 'games_played' in df.columns:
            df = df[df['games_played'] >= MIN_CAREER_GAMES].copy()
            logger.info(f"Filtered by career games: {initial_count} → {len(df)}")

        # Filter by recent activity (if column exists)
        if 'games_last_30' in df.columns:
            df = df[df['games_last_30'] >= MIN_RECENT_GAMES].copy()
            logger.info(f"Filtered by recent games: {initial_count} → {len(df)}")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for prediction

        Args:
            df: Raw data

        Returns:
            X (features), metadata (player info)
        """
        # Extract metadata
        metadata_cols = ['player_id', 'player_name', 'game_date', 'team_abbreviation']
        metadata_cols = [col for col in metadata_cols if col in df.columns]
        metadata = df[metadata_cols].copy()

        # Extract features
        feature_cols = [col for col in self.feature_names if col in df.columns]

        if len(feature_cols) != len(self.feature_names):
            missing = set(self.feature_names) - set(feature_cols)
            logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")

        X = df[feature_cols].copy()

        # Fill missing values
        X = X.fillna(0)

        logger.info(f"Prepared {len(feature_cols)} features for {len(X)} games")

        return X, metadata

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate mean predictions from ensemble

        Args:
            X: Features

        Returns:
            Mean predictions (average of 19 models)
        """
        logger.info("Generating mean predictions...")

        # Get predictions from each model
        predictions = np.array([model.predict(X) for model in self.mean_models])

        # Ensemble average
        mean_pred = predictions.mean(axis=0)

        logger.info(f"Mean predictions: {mean_pred.mean():.2f} ± {mean_pred.std():.2f}")

        return mean_pred

    def predict_variance(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate variance predictions for uncertainty

        Args:
            X: Features

        Returns:
            Variance predictions
        """
        if not ENABLE_MONTE_CARLO or self.variance_models is None:
            logger.warning("Variance models not available")
            return None

        logger.info("Generating variance predictions...")

        # Get variance from each model
        variances = np.array([model.predict(X) for model in self.variance_models])

        # Ensemble average
        var_pred = variances.mean(axis=0)

        logger.info(f"Variance predictions: {var_pred.mean():.2f} ± {var_pred.std():.2f}")

        return var_pred

    def calculate_probabilities(self, mean_pred: np.ndarray, var_pred: np.ndarray,
                              betting_lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate P(PRA > line) using Gamma distribution

        Args:
            mean_pred: Mean predictions
            var_pred: Variance predictions
            betting_lines: Betting lines

        Returns:
            (prob_over, alpha, beta)
        """
        if not ENABLE_MONTE_CARLO:
            logger.warning("Monte Carlo disabled")
            return None, None, None

        logger.info("Calculating probabilities...")

        # Fit Gamma distribution
        alpha, beta = fit_gamma_parameters(mean_pred, var_pred)

        # Calculate P(PRA > line)
        prob_over = calculate_probability_over_line(
            alpha, beta, betting_lines, method=MC_PROBABILITY_METHOD
        )

        logger.info(f"Probabilities: {prob_over.mean():.3f} ± {prob_over.std():.3f}")

        return prob_over, alpha, beta

    def predict_with_probabilities(self, target_date: str,
                                  betting_lines: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate predictions with probabilities for target date

        Args:
            target_date: Date to predict (YYYY-MM-DD)
            betting_lines: DataFrame with betting lines (must have player_name and pra_line columns)

        Returns:
            DataFrame with predictions and probabilities
        """
        logger.info(f"Generating predictions for {target_date}")

        # Load upcoming games
        df = self.load_upcoming_games(target_date)

        if df.empty:
            logger.warning(f"No games found for {target_date}")
            return pd.DataFrame()

        # Filter players
        df = self.filter_players(df)

        if df.empty:
            logger.warning("No players passed filtering criteria")
            return pd.DataFrame()

        # Prepare features
        X, metadata = self.prepare_features(df)

        # Generate mean predictions
        mean_pred = self.predict_mean(X)

        # Generate variance predictions
        var_pred = self.predict_variance(X) if ENABLE_MONTE_CARLO else None
        std_dev = np.sqrt(var_pred) if var_pred is not None else None

        # Add predictions to metadata
        metadata['mean_pred'] = mean_pred
        metadata['std_dev'] = std_dev if std_dev is not None else np.nan

        # Merge with betting lines if provided
        if betting_lines is not None and not betting_lines.empty:
            # DEBUG: Log player names for comparison
            logger.info(f"Betting lines available for {len(betting_lines)} rows, {betting_lines['player_name'].nunique()} unique players")
            logger.info(f"Sample betting line player names: {sorted(betting_lines['player_name'].unique())[:5]}")
            logger.info(f"Prediction player names: {sorted(metadata['player_name'].unique())[:5]}")

            # Merge on player_name (creates multiple rows per player if multiple bookmakers)
            # Include pra_odds if available
            merge_cols = ['player_name', 'pra_line', 'bookmaker']
            if 'pra_odds' in betting_lines.columns:
                merge_cols.append('pra_odds')

            results = metadata.merge(
                betting_lines[merge_cols],
                on='player_name',
                how='left'
            )

            logger.info(f"After merge: {results['pra_line'].notna().sum()} rows with odds out of {len(results)} total")

            # Use betting line where available, otherwise use mean prediction
            results['line_for_prob'] = results['pra_line'].fillna(results['mean_pred'])

            # Set default odds if not available
            if 'pra_odds' not in results.columns:
                results['pra_odds'] = -110
            else:
                results['pra_odds'] = results['pra_odds'].fillna(-110)
        else:
            # No betting lines - use mean predictions as dummy lines
            results = metadata.copy()
            results['pra_line'] = np.nan
            results['bookmaker'] = None
            results['line_for_prob'] = results['mean_pred']
            results['pra_odds'] = -110

        # Calculate probabilities using the line for each row
        if ENABLE_MONTE_CARLO and var_pred is not None:
            # Expand mean_pred and var_pred to match results length (for multiple bookmakers)
            expanded_mean = results['mean_pred'].values
            expanded_std = results['std_dev'].values
            expanded_var = expanded_std ** 2

            prob_over, alpha, beta = self.calculate_probabilities(
                expanded_mean, expanded_var, results['line_for_prob'].values
            )
            prob_under = 1 - prob_over

            # Calculate breakeven probability from actual odds
            def calculate_breakeven(odds):
                """Convert American odds to breakeven probability"""
                if odds < 0:
                    return abs(odds) / (abs(odds) + 100)
                else:
                    return 100 / (odds + 100)

            breakeven_prob = results['pra_odds'].apply(calculate_breakeven)

            results['prob_over'] = prob_over
            results['prob_under'] = prob_under
            results['breakeven_prob'] = breakeven_prob
            results['edge_over'] = prob_over - breakeven_prob
            results['edge_under'] = prob_under - breakeven_prob
            results['confidence_score'] = np.abs(prob_over - 0.5) / 0.5
            results['cv'] = expanded_std / (expanded_mean + 1e-6)

        # Filter results: only keep players with betting lines
        if betting_lines is not None and not betting_lines.empty:
            initial_count = len(results)
            results = results[results['pra_line'].notna()].copy()
            logger.info(f"Filtered to players with odds: {initial_count} → {len(results)}")

            # For each player, keep only preferred bookmaker (fanduel > draftkings ONLY)
            def select_preferred_bookmaker(group):
                if 'fanduel' in group['bookmaker'].str.lower().values:
                    return group[group['bookmaker'].str.lower() == 'fanduel'].iloc[0:1]
                elif 'draftkings' in group['bookmaker'].str.lower().values:
                    return group[group['bookmaker'].str.lower() == 'draftkings'].iloc[0:1]
                else:
                    # No FanDuel or DraftKings - exclude this player
                    return pd.DataFrame()

            if not results.empty and 'player_name' in results.columns:
                before_dedup = len(results)
                results = results.groupby('player_name', group_keys=False).apply(select_preferred_bookmaker).reset_index(drop=True)
                logger.info(f"Deduplicated by bookmaker: {before_dedup} → {len(results)}")

        logger.info(f"Generated {len(results)} predictions")

        return results

    def export_predictions(self, predictions: pd.DataFrame, output_path: str):
        """
        Export predictions to CSV

        Args:
            predictions: Predictions DataFrame
            output_path: Output file path
        """
        if predictions.empty:
            logger.warning("No predictions to export")
            return

        predictions.to_csv(output_path, index=False)
        logger.info(f"Exported {len(predictions)} predictions to {output_path}")


def predict_tomorrow(ensemble_path: str, odds_df: Optional[pd.DataFrame] = None,
                     output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to predict tomorrow's games

    Args:
        ensemble_path: Path to trained ensemble
        odds_df: Optional DataFrame with betting lines
        output_path: Optional CSV path to save results

    Returns:
        DataFrame with predictions
    """
    # Load ensemble
    ensemble_data = ProductionModelTrainer.load_ensemble(ensemble_path)

    # Create predictor
    predictor = ProductionPredictor(ensemble_data)

    # Get tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)
    target_date = tomorrow.strftime('%Y-%m-%d')

    # Generate predictions
    predictions = predictor.predict_with_probabilities(target_date, odds_df)

    # Export if requested
    if output_path and not predictions.empty:
        predictor.export_predictions(predictions, output_path)

    return predictions


if __name__ == "__main__":
    """Test the predictor"""
    from production.model_trainer import get_latest_model_path

    print("Testing Predictor...")
    print("-" * 60)

    # Get latest model
    model_path = get_latest_model_path()

    if model_path is None:
        print("No trained models found. Please run model_trainer.py first.")
    else:
        # Generate predictions
        predictions = predict_tomorrow(str(model_path))

        if not predictions.empty:
            print(f"\nGenerated {len(predictions)} predictions")
            print("\nSample:")
            print(predictions.head(10))

            if 'prob_over' in predictions.columns:
                print("\nProbability distribution:")
                print(predictions['prob_over'].describe())
        else:
            print("\nNo predictions generated (possibly no games tomorrow)")
