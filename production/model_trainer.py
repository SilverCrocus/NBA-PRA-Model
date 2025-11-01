"""
Model Trainer Module

Trains 19-fold CV ensemble of XGBoost models for PRA prediction.
Also trains variance models for Monte Carlo probabilistic predictions.

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional
import pickle
import logging
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.config import (
    MASTER_FEATURES_PATH,
    MODELS_DIR,
    TRAINING_WINDOW_YEARS,
    CV_FOLDS,
    CV_VALIDATION_SPLIT,
    EXCLUDE_COLUMNS,
    XGBOOST_PARAMS,
    VARIANCE_MODEL_PARAMS,
    ENABLE_MONTE_CARLO,
    setup_logging
)

# Import Monte Carlo modules if enabled
if ENABLE_MONTE_CARLO:
    from backtest.monte_carlo.variance_model import VarianceModel

logger = setup_logging('model_trainer')


class ProductionModelTrainer:
    """
    Trains production-ready ensemble of XGBoost models

    Features:
    - 3-year rolling training window
    - 19-fold time-series cross-validation
    - Mean prediction ensemble
    - Variance prediction for Monte Carlo (optional)
    - Model persistence
    """

    def __init__(self):
        """Initialize model trainer"""
        self.mean_models = []
        self.variance_models = [] if ENABLE_MONTE_CARLO else None
        self.feature_names = None
        self.training_metrics = {}

        logger.info("ProductionModelTrainer initialized")

    def load_training_data(self, cutoff_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load training data (last 3 years)

        Args:
            cutoff_date: Cutoff date for training (defaults to today)

        Returns:
            DataFrame with features and target
        """
        logger.info("Loading master features...")
        df = pd.read_parquet(MASTER_FEATURES_PATH)

        # Set cutoff date (defaults to today)
        if cutoff_date is None:
            cutoff_date = datetime.now().strftime('%Y-%m-%d')

        cutoff_dt = pd.to_datetime(cutoff_date)
        start_dt = cutoff_dt - timedelta(days=TRAINING_WINDOW_YEARS * 365)

        # Filter to training window
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df[(df['game_date'] >= start_dt) & (df['game_date'] < cutoff_dt)].copy()

        logger.info(f"Training window: {start_dt.date()} to {cutoff_dt.date()}")
        logger.info(f"Loaded {len(df)} games for training")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target

        Args:
            df: Raw data with features and target

        Returns:
            X (features), y (target)
        """
        # Target column (master_features uses 'target_pra')
        target_col = 'target_pra' if 'target_pra' in df.columns else 'pra'

        # Drop excluded columns (including target)
        exclude_cols = EXCLUDE_COLUMNS + [target_col]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Drop object/string columns (XGBoost can't handle them)
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            logger.warning(f"Dropping {len(object_cols)} object columns: {object_cols}")
            X = X.drop(columns=object_cols)

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle missing values (fill with 0)
        X = X.fillna(0)

        logger.info(f"Prepared {len(X.columns)} features")
        logger.debug(f"Features: {', '.join(X.columns.tolist()[:10])}...")

        return X, y

    def create_cv_splits(self, df: pd.DataFrame) -> List[Tuple]:
        """
        Create time-series CV splits

        Args:
            df: Data with game_date column

        Returns:
            List of (train_idx, val_idx) tuples
        """
        # Sort by date
        df = df.sort_values('game_date').reset_index(drop=True)

        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=CV_FOLDS)

        splits = []
        for train_idx, test_idx in tscv.split(df):
            # Further split test into validation
            val_size = int(len(test_idx) * CV_VALIDATION_SPLIT)
            val_idx = test_idx[:val_size]
            # Keep rest for future testing if needed

            splits.append((train_idx, val_idx))

        logger.info(f"Created {len(splits)} CV folds")
        return splits

    def train_mean_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        fold: int) -> XGBRegressor:
        """
        Train a single mean prediction model

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            fold: Fold number

        Returns:
            Trained XGBoost model
        """
        logger.info(f"Training mean model for fold {fold}...")

        # Create model
        model = XGBRegressor(**XGBOOST_PARAMS)

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)

        logger.info(f"Fold {fold} - Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f}, "
                   f"Val RMSE: {val_rmse:.3f}, Val R²: {val_r2:.3f}")

        # Store metrics
        self.training_metrics[f'fold_{fold}'] = {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }

        return model

    def train_variance_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           mean_pred_train: np.ndarray,
                           fold: int) -> 'VarianceModel':
        """
        Train variance model for uncertainty quantification

        Args:
            X_train: Training features
            y_train: Training target
            mean_pred_train: Mean predictions from mean model
            fold: Fold number

        Returns:
            Trained variance model
        """
        if not ENABLE_MONTE_CARLO:
            return None

        logger.info(f"Training variance model for fold {fold}...")

        # Create variance model
        variance_model = VarianceModel(params=VARIANCE_MODEL_PARAMS)

        # Train on squared residuals
        variance_model.fit(X_train, y_train, mean_pred_train)

        logger.info(f"Fold {fold} - Variance model trained")

        return variance_model

    def train_ensemble(self) -> Tuple[List[XGBRegressor], List]:
        """
        Train full 19-fold ensemble

        Returns:
            (mean_models, variance_models)
        """
        logger.info("Starting ensemble training...")

        # Load data
        df = self.load_training_data()

        # Prepare features
        X, y = self.prepare_features(df)

        # Create CV splits
        splits = self.create_cv_splits(df)

        # Train each fold
        mean_models = []
        variance_models = [] if ENABLE_MONTE_CARLO else None

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train mean model
            mean_model = self.train_mean_model(X_train, y_train, X_val, y_val, fold)
            mean_models.append(mean_model)

            # Train variance model
            if ENABLE_MONTE_CARLO:
                mean_pred_train = mean_model.predict(X_train)
                variance_model = self.train_variance_model(
                    X_train, y_train, mean_pred_train, fold
                )
                variance_models.append(variance_model)

        # Store models
        self.mean_models = mean_models
        self.variance_models = variance_models

        # Log summary
        avg_val_mae = np.mean([m['val_mae'] for m in self.training_metrics.values()])
        avg_val_rmse = np.mean([m['val_rmse'] for m in self.training_metrics.values()])
        avg_val_r2 = np.mean([m['val_r2'] for m in self.training_metrics.values()])

        logger.info(f"Ensemble training complete!")
        logger.info(f"Average metrics - MAE: {avg_val_mae:.3f}, RMSE: {avg_val_rmse:.3f}, R²: {avg_val_r2:.3f}")

        return mean_models, variance_models

    def save_models(self, timestamp: Optional[str] = None):
        """
        Save trained models to disk

        Args:
            timestamp: Optional timestamp string (defaults to current time)
        """
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save ensemble
        ensemble_path = MODELS_DIR / f"ensemble_{timestamp}.pkl"

        ensemble_data = {
            'mean_models': self.mean_models,
            'variance_models': self.variance_models,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'timestamp': timestamp,
            'n_folds': len(self.mean_models),
            'enable_monte_carlo': ENABLE_MONTE_CARLO
        }

        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_data, f)

        logger.info(f"Models saved to: {ensemble_path}")

        # Save metrics to CSV
        metrics_df = pd.DataFrame(self.training_metrics).T
        metrics_path = MODELS_DIR / f"training_metrics_{timestamp}.csv"
        metrics_df.to_csv(metrics_path)

        logger.info(f"Metrics saved to: {metrics_path}")

        return ensemble_path

    @staticmethod
    def load_ensemble(model_path: str):
        """
        Load trained ensemble from disk

        Args:
            model_path: Path to saved ensemble

        Returns:
            Dictionary with models and metadata
        """
        logger.info(f"Loading ensemble from: {model_path}")

        with open(model_path, 'rb') as f:
            ensemble_data = pickle.load(f)

        logger.info(f"Loaded {ensemble_data['n_folds']}-fold ensemble from {ensemble_data['timestamp']}")

        return ensemble_data


def train_production_models(save: bool = True) -> Tuple[List[XGBRegressor], List]:
    """
    Convenience function to train production models

    Args:
        save: Whether to save models to disk

    Returns:
        (mean_models, variance_models)
    """
    trainer = ProductionModelTrainer()

    # Train ensemble
    mean_models, variance_models = trainer.train_ensemble()

    # Save if requested
    if save:
        trainer.save_models()

    return mean_models, variance_models


def get_latest_model_path() -> Optional[Path]:
    """
    Get path to most recent trained model

    Returns:
        Path to latest ensemble pickle file
    """
    ensemble_files = list(MODELS_DIR.glob("ensemble_*.pkl"))

    if not ensemble_files:
        logger.warning("No trained models found")
        return None

    # Sort by modification time
    latest = max(ensemble_files, key=lambda p: p.stat().st_mtime)

    logger.info(f"Latest model: {latest.name}")
    return latest


if __name__ == "__main__":
    """Train models"""

    print("Training production models...")
    print("-" * 60)

    # Train and save
    mean_models, variance_models = train_production_models(save=True)

    print(f"\nTrained {len(mean_models)} mean models")
    if variance_models:
        print(f"Trained {len(variance_models)} variance models")

    print("\nDone!")
