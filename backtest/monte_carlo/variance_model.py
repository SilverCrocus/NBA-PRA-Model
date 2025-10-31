"""
Variance Model for Heteroskedastic Uncertainty Estimation

This module trains a separate XGBoost model to predict player-specific
variance in PRA predictions. The variance model captures:
- Recent performance volatility (pra_std_last10)
- Minutes uncertainty (minutes_std_last10)
- Injury risk (dnp_rate_last30, games_since_return)
- Context factors (back-to-back, opponent strength)

The variance model is trained on squared residuals from the mean model,
enabling heteroskedastic predictions where variance depends on features.
"""

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import logging
from typing import Optional, Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)


class VarianceModel:
    """
    Heteroskedastic variance model for PRA predictions.

    Predicts conditional variance σ²(X) as a function of features,
    capturing player-specific and context-dependent uncertainty.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize variance model.

        Args:
            params: XGBoost hyperparameters. If None, uses defaults
                    optimized for variance prediction.
        """
        if params is None:
            # Default parameters for variance model
            # Use slightly more conservative settings than mean model
            params = {
                'objective': 'reg:squarederror',
                'max_depth': 5,  # Shallower trees to prevent overfitting variance
                'learning_rate': 0.05,
                'n_estimators': 200,  # Fewer trees than mean model
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 10,  # Higher to ensure stable variance estimates
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'random_state': 42,
                'n_jobs': -1
            }

        self.model = XGBRegressor(**params)
        self.params = params
        self.is_fitted = False
        self.mean_variance = None  # Population mean variance
        self.min_variance = 0.01  # Minimum variance to prevent zero

        logger.info(f"Initialized VarianceModel with params: {params}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        mean_predictions: np.ndarray,
        eval_set: Optional[tuple] = None
    ) -> 'VarianceModel':
        """
        Train variance model on squared residuals.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: True PRA values (n_samples,)
            mean_predictions: Predictions from mean model (n_samples,)
            eval_set: Optional (X_val, y_val, mean_pred_val) for early stopping

        Returns:
            self (fitted model)
        """
        logger.info(f"Training variance model on {len(X)} samples")

        # Calculate squared residuals (target for variance model)
        residuals = y - mean_predictions
        squared_residuals = residuals ** 2

        # Store mean variance for reference
        self.mean_variance = squared_residuals.mean()

        # Log diagnostics
        logger.info(f"Mean squared residual: {self.mean_variance:.3f}")
        logger.info(f"Std of squared residuals: {squared_residuals.std():.3f}")
        logger.info(f"Min/Max squared residuals: {squared_residuals.min():.3f} / {squared_residuals.max():.3f}")

        # Prepare evaluation set if provided
        eval_list = None
        if eval_set is not None:
            X_val, y_val, mean_pred_val = eval_set
            residuals_val = y_val - mean_pred_val
            squared_residuals_val = residuals_val ** 2
            eval_list = [(X_val, squared_residuals_val)]

        # Train model on squared residuals
        fit_params = {
            'verbose': False
        }

        if eval_list is not None:
            fit_params['eval_set'] = eval_list
            fit_params['early_stopping_rounds'] = 20

        self.model.fit(X, squared_residuals, **fit_params)
        self.is_fitted = True

        # Validate predictions are reasonable
        var_pred = self.model.predict(X)

        # Check for negative predictions
        negative_var = (var_pred < 0).sum()
        if negative_var > 0:
            logger.warning(f"{negative_var} negative variance predictions ({negative_var/len(var_pred)*100:.1f}%)")

        # Check for extreme predictions
        extreme_var = (var_pred > 10 * self.mean_variance).sum()
        if extreme_var > 0:
            logger.warning(f"{extreme_var} extreme variance predictions (>10x mean)")

        # Calculate training metrics
        train_rmse = np.sqrt(mean_squared_error(squared_residuals, var_pred))
        logger.info(f"Training RMSE (variance prediction): {train_rmse:.3f}")

        if eval_list is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(squared_residuals_val, val_pred))
            logger.info(f"Validation RMSE (variance prediction): {val_rmse:.3f}")

        logger.info("Variance model training complete")

        return self

    def predict(
        self,
        X: np.ndarray,
        min_variance: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict conditional variance for new data.

        Args:
            X: Feature matrix (n_samples, n_features)
            min_variance: Minimum variance (default: self.min_variance)

        Returns:
            Predicted variances (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        if min_variance is None:
            min_variance = self.min_variance

        # Predict variance
        var_pred = self.model.predict(X)

        # Ensure positive variance (clip to minimum)
        var_pred = np.maximum(var_pred, min_variance)

        # Log diagnostics for large prediction sets
        if len(var_pred) > 100:
            logger.debug(f"Predicted variance - Mean: {var_pred.mean():.3f}, "
                        f"Std: {var_pred.std():.3f}, "
                        f"Min/Max: {var_pred.min():.3f}/{var_pred.max():.3f}")

        return var_pred

    def get_feature_importance(
        self,
        feature_names: Optional[list] = None,
        importance_type: str = 'gain',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance for variance prediction.

        Args:
            feature_names: Names of features (optional)
            importance_type: 'gain', 'weight', or 'cover'
            top_n: Number of top features to return

        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get importance scores
        importance = self.model.get_booster().get_score(importance_type=importance_type)

        # Convert to DataFrame
        if feature_names is not None:
            # Map feature indices to names
            importance_df = pd.DataFrame([
                {'feature': feature_names[int(k.replace('f', ''))],
                 'importance': v}
                for k, v in importance.items()
            ])
        else:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in importance.items()
            ])

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        # Return top N
        return importance_df.head(top_n)

    def save(self, filepath: str) -> None:
        """Save variance model to file."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        self.model.save_model(filepath)
        logger.info(f"Variance model saved to {filepath}")

    def load(self, filepath: str) -> 'VarianceModel':
        """Load variance model from file."""
        self.model.load_model(filepath)
        self.is_fitted = True
        logger.info(f"Variance model loaded from {filepath}")
        return self

    def diagnose(self, X: np.ndarray, y: np.ndarray, mean_predictions: np.ndarray) -> Dict[str, Any]:
        """
        Diagnose variance model performance.

        Args:
            X: Feature matrix
            y: True values
            mean_predictions: Mean model predictions

        Returns:
            Dictionary with diagnostic metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Calculate true variance
        residuals = y - mean_predictions
        true_variance = residuals ** 2

        # Predict variance
        pred_variance = self.predict(X)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(true_variance, pred_variance))
        mae = np.mean(np.abs(true_variance - pred_variance))

        # Correlation between predicted and actual variance
        correlation = np.corrcoef(true_variance, pred_variance)[0, 1]

        # Coverage analysis (are prediction intervals well-calibrated?)
        std_pred = np.sqrt(pred_variance)

        # 95% interval: [mean - 1.96*std, mean + 1.96*std]
        lower_95 = mean_predictions - 1.96 * std_pred
        upper_95 = mean_predictions + 1.96 * std_pred
        coverage_95 = ((y >= lower_95) & (y <= upper_95)).mean()

        # 68% interval (1 std)
        lower_68 = mean_predictions - std_pred
        upper_68 = mean_predictions + std_pred
        coverage_68 = ((y >= lower_68) & (y <= upper_68)).mean()

        diagnostics = {
            'rmse': rmse,
            'mae': mae,
            'correlation': correlation,
            'mean_true_variance': true_variance.mean(),
            'mean_pred_variance': pred_variance.mean(),
            'coverage_68': coverage_68,
            'coverage_95': coverage_95,
            'target_coverage_68': 0.68,
            'target_coverage_95': 0.95,
            'coverage_error_68': abs(coverage_68 - 0.68),
            'coverage_error_95': abs(coverage_95 - 0.95),
        }

        # Log results
        logger.info(f"Variance Model Diagnostics:")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  Correlation: {correlation:.3f}")
        logger.info(f"  68% Coverage: {coverage_68:.1%} (target: 68%)")
        logger.info(f"  95% Coverage: {coverage_95:.1%} (target: 95%)")

        return diagnostics
