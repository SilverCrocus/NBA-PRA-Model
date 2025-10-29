"""
Walk-Forward Backtest Engine

Main orchestrator for walk-forward backtesting with daily retraining.

Simulates production deployment: retrain model after each NBA game day,
predict next day's games.

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from typing import List, Tuple, Dict
import logging
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.config import (
    XGBOOST_PARAMS, CV_FOLDS, CV_VALIDATION_SPLIT,
    DAILY_PREDICTIONS_PATH, DAILY_METRICS_PATH,
    PROGRESS_LOG_INTERVAL, SAVE_CHECKPOINT_INTERVAL,
    setup_logging
)
from backtest.data_loader import (
    get_training_window,
    prepare_features_for_training,
    match_predictions_to_betting_lines,
    get_unique_game_days
)
from backtest.betting_evaluator import (
    calculate_bet_decisions,
    calculate_bet_outcomes,
    calculate_win_rate,
    calculate_roi
)
from feature_engineering.features.ctg_imputation import (
    calculate_position_baselines,
    impute_missing_ctg,
    create_position_relative_features
)

logger = setup_logging('walk_forward_backtest')


class WalkForwardBacktest:
    """
    Walk-forward backtesting engine with daily retraining

    Simulates production deployment by:
    1. Training on N-year rolling window before each game day
    2. Creating 19-fold CV ensemble for robustness
    3. Predicting that day's games
    4. Evaluating against actual results and betting lines
    5. Adding day's results to training data for next iteration
    """

    def __init__(self,
                 all_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 odds_data: pd.DataFrame):
        """
        Initialize backtest engine

        Args:
            all_data: Full historical dataset (for training windows)
            test_data: Target season data (2024-25)
            odds_data: Historical betting lines
        """
        self.all_data = all_data
        self.test_data = test_data
        self.odds_data = odds_data

        self.daily_predictions = []
        self.daily_metrics = []

        logger.info("Walk-Forward Backtest Engine initialized")
        logger.info(f"  Historical data: {len(all_data)} games")
        logger.info(f"  Test data: {len(test_data)} games")
        logger.info(f"  Betting lines: {len(odds_data)} lines")

    def create_cv_folds(self,
                       train_data: pd.DataFrame,
                       n_folds: int = CV_FOLDS) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time-series CV folds from training data

        Uses chronological splits with validation split within each fold.

        Args:
            train_data: Training data
            n_folds: Number of folds to create

        Returns:
            List of (train_fold, val_fold) tuples
        """
        # Sort by date
        train_data = train_data.sort_values('game_date')

        # Split into n_folds
        fold_size = len(train_data) // n_folds
        folds = []

        for i in range(n_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(train_data)

            fold_data = train_data.iloc[start_idx:end_idx]

            # Split fold into train/val
            val_size = int(len(fold_data) * CV_VALIDATION_SPLIT)
            train_fold = fold_data.iloc[:-val_size]
            val_fold = fold_data.iloc[-val_size:]

            folds.append((train_fold, val_fold))

        logger.debug(f"  Created {len(folds)} CV folds")

        return folds

    def train_single_model(self,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: pd.DataFrame,
                          y_val: pd.Series) -> XGBRegressor:
        """
        Train a single XGBoost model with early stopping

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Trained XGBoost model
        """
        model = XGBRegressor(**XGBOOST_PARAMS)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        return model

    def train_ensemble(self,
                      train_data: pd.DataFrame) -> List[XGBRegressor]:
        """
        Train 19-fold CV ensemble

        Args:
            train_data: Training data with features and target

        Returns:
            List of trained XGBoost models
        """
        logger.debug("  Training 19-fold CV ensemble...")

        # Create CV folds
        folds = self.create_cv_folds(train_data, n_folds=CV_FOLDS)

        # Train model on each fold
        models = []
        for fold_idx, (train_fold, val_fold) in enumerate(folds):
            # Calculate position baselines from THIS FOLD'S training data
            # This prevents temporal leakage
            if 'position' in train_fold.columns:
                position_baselines = calculate_position_baselines(train_fold)

                # Apply imputation to train and val using fold-specific baselines
                train_fold = impute_missing_ctg(train_fold, position_baselines)
                val_fold = impute_missing_ctg(val_fold, position_baselines)

                # Add position-relative features
                pos_rel_train = create_position_relative_features(train_fold, position_baselines)
                pos_rel_val = create_position_relative_features(val_fold, position_baselines)

                # Merge position-relative features
                train_fold = train_fold.merge(pos_rel_train, on=['player_id', 'game_id', 'game_date'], how='left')
                val_fold = val_fold.merge(pos_rel_val, on=['player_id', 'game_id', 'game_date'], how='left')

            # Prepare features
            X_train, y_train = prepare_features_for_training(train_fold)
            X_val, y_val = prepare_features_for_training(val_fold)

            # Train model
            model = self.train_single_model(X_train, y_train, X_val, y_val)
            models.append(model)

        logger.debug(f"  Trained {len(models)} models in ensemble")

        return models

    def predict_with_ensemble(self,
                             models: List[XGBRegressor],
                             X_test: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble (average of all models)

        Args:
            models: List of trained models
            X_test: Test features

        Returns:
            Array of predictions
        """
        # Get predictions from each model
        all_predictions = np.array([model.predict(X_test) for model in models])

        # Average predictions
        ensemble_predictions = all_predictions.mean(axis=0)

        return ensemble_predictions

    def predict_single_day(self,
                          prediction_date: pd.Timestamp,
                          position_baselines: pd.DataFrame) -> pd.DataFrame:
        """
        Predict games for a single day

        Args:
            prediction_date: Date to predict
            position_baselines: Pre-calculated position baselines from training data

        Returns:
            DataFrame with predictions and actuals
        """
        # Get games for this date
        games_today = self.test_data[
            self.test_data['game_date'] == prediction_date
        ].copy()

        if len(games_today) == 0:
            logger.warning(f"  No games on {prediction_date.date()}")
            return pd.DataFrame()

        # Get training data (3-year rolling window)
        train_data = get_training_window(self.all_data, prediction_date)

        if len(train_data) < 1000:
            logger.warning(f"  Insufficient training data ({len(train_data)} games), skipping day")
            return pd.DataFrame()

        # Train ensemble
        models = self.train_ensemble(train_data)

        # Apply imputation to test games using training baselines
        games_today_imputed = impute_missing_ctg(games_today, position_baselines)

        # Add position-relative features
        pos_rel_test = create_position_relative_features(games_today_imputed, position_baselines)
        games_today_imputed = games_today_imputed.merge(
            pos_rel_test,
            on=['player_id', 'game_id', 'game_date'],
            how='left'
        )

        # Prepare features
        X_test, y_test = prepare_features_for_training(games_today_imputed)

        # Make predictions
        predictions = self.predict_with_ensemble(models, X_test)

        # Create results dataframe
        results = pd.DataFrame({
            'game_date': games_today['game_date'],
            'player_id': games_today['player_id'],
            'player_name': games_today.get('player_name', 'Unknown'),
            'prediction': predictions,
            'actual_pra': y_test.values
        })

        return results

    def run_backtest(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Run full walk-forward backtest

        Args:
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with all predictions
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING WALK-FORWARD BACKTEST")
        logger.info("="*60)

        # Get unique game days in test period
        game_days = get_unique_game_days(self.test_data, start_date, end_date)

        logger.info(f"\nBacktesting {len(game_days)} game days")
        logger.info(f"  From: {game_days[0].date()}")
        logger.info(f"  To: {game_days[-1].date()}")

        # Calculate position baselines from ALL historical data before test period
        # (In a truly realistic setup, this would be calculated before the season)
        logger.info("\nCalculating position baselines from historical data...")
        if 'position' in self.all_data.columns:
            historical_data_before_test = self.all_data[
                self.all_data['game_date'] < game_days[0]
            ]
            position_baselines = calculate_position_baselines(historical_data_before_test)
        else:
            position_baselines = pd.DataFrame()

        # Iterate through each game day
        all_predictions = []

        for day_idx, prediction_date in enumerate(game_days, 1):
            logger.info(f"\n[{day_idx}/{len(game_days)}] Predicting {prediction_date.date()}...")

            # Predict this day
            day_predictions = self.predict_single_day(prediction_date, position_baselines)

            if day_predictions.empty:
                continue

            # Match to betting lines
            day_predictions = match_predictions_to_betting_lines(
                day_predictions,
                self.odds_data
            )

            # Calculate bet decisions and outcomes
            bet_decisions = calculate_bet_decisions(
                day_predictions['prediction'],
                day_predictions['betting_line']
            )
            bet_outcomes = calculate_bet_outcomes(
                day_predictions['actual_pra'],
                day_predictions['betting_line'],
                bet_decisions['bet_decision']
            )

            # Merge betting results
            day_predictions = pd.concat([day_predictions, bet_decisions, bet_outcomes], axis=1)

            # Calculate daily metrics
            mae = mean_absolute_error(day_predictions['actual_pra'], day_predictions['prediction'])
            rmse = np.sqrt(mean_squared_error(day_predictions['actual_pra'], day_predictions['prediction']))
            r2 = r2_score(day_predictions['actual_pra'], day_predictions['prediction'])

            has_line = ~day_predictions['betting_line'].isna()
            if has_line.sum() > 0:
                win_rate = calculate_win_rate(day_predictions['bet_correct'], has_line)
                roi = calculate_roi(day_predictions['profit'], has_line)
            else:
                win_rate, roi = np.nan, np.nan

            # Log progress
            if day_idx % PROGRESS_LOG_INTERVAL == 0 or day_idx == len(game_days):
                logger.info(f"  Predictions: {len(day_predictions)} | MAE: {mae:.3f} | Win Rate: {win_rate:.1%} | ROI: {roi:.2f}%")

            # Store results
            all_predictions.append(day_predictions)

            # Save checkpoint
            if day_idx % SAVE_CHECKPOINT_INTERVAL == 0:
                checkpoint_df = pd.concat(all_predictions, ignore_index=True)
                checkpoint_df.to_csv(DAILY_PREDICTIONS_PATH, index=False)
                logger.info(f"  Checkpoint saved: {len(checkpoint_df)} total predictions")

        # Combine all predictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)

        # Save final results
        final_predictions.to_csv(DAILY_PREDICTIONS_PATH, index=False)
        logger.info(f"\n✓ Backtest complete! Saved {len(final_predictions)} predictions to {DAILY_PREDICTIONS_PATH}")

        return final_predictions


def run_walk_forward_backtest(all_data: pd.DataFrame,
                               test_data: pd.DataFrame,
                               odds_data: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for walk-forward backtest

    Args:
        all_data: Full historical dataset
        test_data: Target season data
        odds_data: Historical betting lines

    Returns:
        DataFrame with all predictions and results
    """
    backtest = WalkForwardBacktest(all_data, test_data, odds_data)
    predictions_df = backtest.run_backtest()

    return predictions_df
