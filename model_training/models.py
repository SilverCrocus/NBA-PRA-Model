"""
Model Wrapper Classes
Provides clean abstractions for XGBoost/LightGBM with consistent interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import lightgbm as lgb

from model_training.config import XGBOOST_PARAMS, LIGHTGBM_PARAMS


class BaseModel(ABC):
    """
    Abstract base for all models - ensures consistent interface
    """

    def __init__(self, params: Dict[str, Any], name: str):
        """
        Initialize model with parameters

        Args:
            params: Model hyperparameters
            name: Model name for logging
        """
        self.params = params.copy()  # Copy to avoid mutation
        self.name = name
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.training_history: Optional[Dict[str, List[float]]] = None

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train model with validation set for early stopping

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Dictionary with training metrics and metadata
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores

        Returns:
            DataFrame with feature names and importance scores
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save model to disk

        Args:
            path: Path to save model (will create parent directories)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model object, parameters, and metadata
        model_data = {
            'model': self.model,
            'params': self.params,
            'name': self.name,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }

        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """
        Load model from disk with validation

        Args:
            path: Path to saved model

        Returns:
            Loaded model instance

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model file is invalid or corrupted
        """
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)

        # Validate model_data structure
        required_keys = ['model', 'params', 'feature_names', 'training_history']
        missing_keys = [k for k in required_keys if k not in model_data]
        if missing_keys:
            raise ValueError(
                f"Invalid model file: missing required keys {missing_keys}. "
                f"Found keys: {list(model_data.keys())}"
            )

        # Validate model is not None
        if model_data['model'] is None:
            raise ValueError("Model file contains None model - file may be corrupted")

        # Validate feature_names
        if not isinstance(model_data['feature_names'], list):
            raise ValueError(
                f"feature_names must be a list, got {type(model_data['feature_names'])}"
            )

        if len(model_data['feature_names']) == 0:
            raise ValueError("feature_names is empty - model has no features")

        # Validate params
        if not isinstance(model_data['params'], dict):
            raise ValueError(
                f"params must be a dict, got {type(model_data['params'])}"
            )

        # Recreate model instance
        instance = cls(params=model_data['params'])
        instance.name = model_data.get('name', instance.name)  # Restore saved name or use default
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_history = model_data['training_history']

        return instance


class XGBoostModel(BaseModel):
    """
    XGBoost regression model wrapper
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model

        Args:
            params: Optional custom hyperparameters (uses config defaults if None)
        """
        params = params or XGBOOST_PARAMS.copy()
        super().__init__(params, name="XGBoost")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train XGBoost with early stopping on validation set

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Dictionary with training metrics and best iteration
        """
        # Store feature names
        self.feature_names = list(X_train.columns)

        # Extract early stopping parameter
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)
        n_estimators = self.params.pop("n_estimators", 500)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        # Setup evaluation sets
        evals = [(dtrain, "train"), (dval, "val")]
        evals_result = {}

        # Train model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=n_estimators,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False  # Suppress per-iteration output
        )

        # Store training history
        self.training_history = evals_result

        # Extract final metrics
        best_iteration = self.model.best_iteration
        train_rmse = evals_result["train"]["rmse"][best_iteration]
        val_rmse = evals_result["val"]["rmse"][best_iteration]

        # Calculate MAE if available
        train_mae = evals_result["train"]["mae"][best_iteration] if "mae" in evals_result["train"] else None
        val_mae = evals_result["val"]["mae"][best_iteration] if "mae" in evals_result["val"] else None

        training_info = {
            "best_iteration": int(best_iteration),
            "n_estimators_used": int(best_iteration + 1),
            "train_rmse_final": float(train_rmse),
            "val_rmse_final": float(val_rmse),
        }

        if train_mae is not None:
            training_info["train_mae_final"] = float(train_mae)
        if val_mae is not None:
            training_info["val_mae_final"] = float(val_mae)

        # Restore params for saving
        self.params["early_stopping_rounds"] = early_stopping_rounds
        self.params["n_estimators"] = n_estimators

        return training_info

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions

        Raises:
            ValueError: If model not trained or feature mismatch
        """
        if self.model is None:
            raise ValueError("Model not trained! Call .train() first.")

        if self.feature_names is None:
            raise ValueError("Feature names not set! Model may not be properly trained.")

        # Validate feature names match
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch! Expected {len(self.feature_names)} features, "
                f"got {len(X.columns)}.\n"
                f"Missing: {set(self.feature_names) - set(X.columns)}\n"
                f"Extra: {set(X.columns) - set(self.feature_names)}"
            )

        # Create DMatrix and predict
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dmatrix)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores

        Returns:
            DataFrame with columns: ['feature', 'importance_gain', 'importance_weight', 'importance_cover']
            Sorted by gain importance (most important first)

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained! Call .train() first.")

        # Get importance scores (multiple types)
        importance_gain = self.model.get_score(importance_type='gain')
        importance_weight = self.model.get_score(importance_type='weight')
        importance_cover = self.model.get_score(importance_type='cover')

        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': self.feature_names
        })

        # Map importance scores (features not used will have 0)
        feature_df['importance_gain'] = feature_df['feature'].map(importance_gain).fillna(0)
        feature_df['importance_weight'] = feature_df['feature'].map(importance_weight).fillna(0)
        feature_df['importance_cover'] = feature_df['feature'].map(importance_cover).fillna(0)

        # Sort by gain (most interpretable)
        feature_df = feature_df.sort_values('importance_gain', ascending=False)

        return feature_df


class LightGBMModel(BaseModel):
    """
    LightGBM regression model wrapper
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LightGBM model

        Args:
            params: Optional custom hyperparameters (uses config defaults if None)
        """
        params = params or LIGHTGBM_PARAMS.copy()
        super().__init__(params, name="LightGBM")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Train LightGBM with early stopping on validation set

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Dictionary with training metrics and best iteration
        """
        # Store feature names
        self.feature_names = list(X_train.columns)

        # Extract early stopping parameter
        early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)
        n_estimators = self.params.pop("n_estimators", 500)

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names, reference=train_data)

        # Setup callbacks
        evals_result = {}
        callbacks = [
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        ]

        # Train model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )

        # Store training history
        self.training_history = evals_result

        # Extract final metrics
        best_iteration = self.model.best_iteration
        train_rmse = evals_result["train"]["rmse"][best_iteration - 1]  # 0-indexed
        val_rmse = evals_result["val"]["rmse"][best_iteration - 1]

        # Calculate MAE if available
        train_mae = evals_result["train"]["mae"][best_iteration - 1] if "mae" in evals_result["train"] else None
        val_mae = evals_result["val"]["mae"][best_iteration - 1] if "mae" in evals_result["val"] else None

        training_info = {
            "best_iteration": int(best_iteration),
            "n_estimators_used": int(best_iteration),
            "train_rmse_final": float(train_rmse),
            "val_rmse_final": float(val_rmse),
        }

        if train_mae is not None:
            training_info["train_mae_final"] = float(train_mae)
        if val_mae is not None:
            training_info["val_mae_final"] = float(val_mae)

        # Restore params for saving
        self.params["early_stopping_rounds"] = early_stopping_rounds
        self.params["n_estimators"] = n_estimators

        return training_info

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions

        Raises:
            ValueError: If model not trained or feature mismatch
        """
        if self.model is None:
            raise ValueError("Model not trained! Call .train() first.")

        if self.feature_names is None:
            raise ValueError("Feature names not set! Model may not be properly trained.")

        # Validate feature names match
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch! Expected {len(self.feature_names)} features, "
                f"got {len(X.columns)}.\n"
                f"Missing: {set(self.feature_names) - set(X.columns)}\n"
                f"Extra: {set(X.columns) - set(self.feature_names)}"
            )

        # Predict
        predictions = self.model.predict(X)

        return predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores

        Returns:
            DataFrame with columns: ['feature', 'importance_gain', 'importance_split']
            Sorted by gain importance (most important first)

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained! Call .train() first.")

        # Get importance scores
        importance_gain = self.model.feature_importance(importance_type='gain')
        importance_split = self.model.feature_importance(importance_type='split')

        # Create DataFrame
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_gain': importance_gain,
            'importance_split': importance_split
        })

        # Sort by gain
        feature_df = feature_df.sort_values('importance_gain', ascending=False)

        return feature_df


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple fold models - averages predictions
    Used for time-series CV where each fold model captures different temporal patterns
    """

    def __init__(self, fold_models: List[BaseModel]):
        """
        Initialize ensemble with list of trained fold models

        Args:
            fold_models: List of trained BaseModel instances

        Raises:
            ValueError: If fold_models is empty or models have inconsistent features
        """
        if not fold_models:
            raise ValueError("fold_models list cannot be empty!")

        # Validate all models have same features
        base_features = fold_models[0].feature_names
        for i, model in enumerate(fold_models[1:], 1):
            if model.feature_names != base_features:
                raise ValueError(
                    f"Feature mismatch in fold model {i}! "
                    f"Expected {len(base_features)} features, got {len(model.feature_names)}"
                )

        # Initialize base with representative params
        base_model_type = fold_models[0].__class__.__name__
        ensemble_name = f"Ensemble_{base_model_type}_n{len(fold_models)}"
        params = fold_models[0].params.copy()

        super().__init__(params, name=ensemble_name)

        self.fold_models = fold_models
        self.feature_names = base_features.copy() if isinstance(base_features, list) else list(base_features)
        self.model = None  # Ensemble doesn't have single model object

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, Any]:
        """
        Ensemble models are pre-trained - this method is not used

        Raises:
            NotImplementedError: Ensembles are created from pre-trained models
        """
        raise NotImplementedError(
            "Ensemble models are created from pre-trained fold models. "
            "Train individual fold models first, then create ensemble."
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate ensemble predictions by averaging all fold model predictions

        Args:
            X: Feature DataFrame

        Returns:
            Array of averaged predictions

        Raises:
            ValueError: If feature mismatch with any fold model
        """
        # Validate features match
        if list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch! Expected {len(self.feature_names)} features, "
                f"got {len(X.columns)}.\n"
                f"Missing: {set(self.feature_names) - set(X.columns)}\n"
                f"Extra: {set(X.columns) - set(self.feature_names)}"
            )

        # Collect predictions from all fold models
        predictions_list = []
        for fold_model in self.fold_models:
            fold_preds = fold_model.predict(X)
            predictions_list.append(fold_preds)

        # Average predictions
        predictions_array = np.array(predictions_list)
        ensemble_predictions = predictions_array.mean(axis=0)

        return ensemble_predictions

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get average feature importance across all fold models

        Returns:
            DataFrame with averaged importance scores
            Columns vary by model type:
            - XGBoost: ['feature', 'importance_gain', 'importance_weight', 'importance_cover']
            - LightGBM: ['feature', 'importance_gain', 'importance_split']

        Raises:
            ValueError: If no fold models available
        """
        if not self.fold_models:
            raise ValueError("No fold models available!")

        # Get importance from each fold model
        importance_dfs = []
        for fold_model in self.fold_models:
            fold_importance = fold_model.get_feature_importance()
            importance_dfs.append(fold_importance)

        # Average importance scores
        # Start with first model's structure
        ensemble_importance = importance_dfs[0].copy()

        # Get numeric columns (importance scores)
        importance_cols = [col for col in ensemble_importance.columns if col.startswith('importance_')]

        # Average each importance type across folds
        for col in importance_cols:
            col_values = []
            for df in importance_dfs:
                col_values.append(df[col].values)

            # Average across folds
            ensemble_importance[col] = np.array(col_values).mean(axis=0)

        # Re-sort by primary importance (gain)
        if 'importance_gain' in ensemble_importance.columns:
            ensemble_importance = ensemble_importance.sort_values('importance_gain', ascending=False)

        return ensemble_importance

    def save(self, path: Path) -> None:
        """
        Save ensemble model to disk
        Saves list of fold models and metadata

        Args:
            path: Path to save ensemble (will create parent directories)
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save ensemble metadata and fold models
        ensemble_data = {
            'fold_models': self.fold_models,
            'params': self.params,
            'name': self.name,
            'feature_names': self.feature_names,
            'n_folds': len(self.fold_models),
            'base_model_type': self.fold_models[0].__class__.__name__
        }

        joblib.dump(ensemble_data, path)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        """
        Load ensemble model from disk

        Args:
            path: Path to saved ensemble

        Returns:
            Loaded EnsembleModel instance

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not path.exists():
            raise FileNotFoundError(f"Ensemble model file not found: {path}")

        ensemble_data = joblib.load(path)

        # Recreate ensemble from fold models
        instance = cls(fold_models=ensemble_data['fold_models'])
        instance.name = ensemble_data['name']
        instance.params = ensemble_data['params']
        instance.feature_names = ensemble_data['feature_names']

        return instance
