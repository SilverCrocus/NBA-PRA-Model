"""
MLflow Configuration and Utilities for NBA PRA Prediction

This module provides MLflow setup, logging utilities, and best practices
for experiment tracking in time-series ML projects.

Key Design Decisions:
- One experiment per model type (XGBoost, LightGBM, etc.)
- Nested runs for hyperparameter tuning
- Comprehensive artifact logging (models, plots, feature importance)
- Time-series specific metadata tracking (split dates, temporal validation)
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class MLflowConfig:
    """
    MLflow configuration and setup for NBA PRA prediction experiments

    This class handles:
    - Experiment creation and management
    - Run naming conventions
    - Tag standardization
    - Artifact organization
    """

    # Experiment names by model type
    EXPERIMENTS = {
        'xgboost': 'NBA_PRA_XGBoost',
        'lightgbm': 'NBA_PRA_LightGBM',
        'random_forest': 'NBA_PRA_RandomForest',
        'linear': 'NBA_PRA_Linear',
        'ensemble': 'NBA_PRA_Ensemble',
        'baseline': 'NBA_PRA_Baseline',
    }

    # Standard tags for all runs
    STANDARD_TAGS = {
        'project': 'nba_pra_prediction',
        'problem_type': 'regression',
        'target_variable': 'pra',
        'data_type': 'time_series',
    }

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize MLflow configuration

        Args:
            tracking_uri: MLflow tracking URI (default: local file store)
        """
        if tracking_uri is None:
            tracking_uri = f"file://{MLFLOW_DIR}"

        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Create artifacts directory
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        print(f"MLflow tracking URI: {tracking_uri}")

    def get_or_create_experiment(self, model_type: str) -> str:
        """
        Get or create experiment for a specific model type

        Args:
            model_type: Type of model ('xgboost', 'lightgbm', etc.)

        Returns:
            Experiment ID
        """
        experiment_name = self.EXPERIMENTS.get(model_type, f"NBA_PRA_{model_type.title()}")

        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                tags={
                    'model_type': model_type,
                    'created_at': datetime.now().isoformat(),
                    'description': f'NBA PRA prediction using {model_type}',
                }
            )
            print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

        return experiment_id

    def generate_run_name(self, model_type: str, description: Optional[str] = None) -> str:
        """
        Generate standardized run name

        Args:
            model_type: Type of model
            description: Optional description

        Returns:
            Formatted run name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if description:
            return f"{model_type}_{description}_{timestamp}"
        else:
            return f"{model_type}_{timestamp}"


class MLflowLogger:
    """
    MLflow logging utilities for NBA PRA prediction

    Handles:
    - Parameter logging
    - Metric tracking
    - Artifact management
    - Model registration
    """

    def __init__(self, experiment_id: str, run_name: str, tags: Optional[Dict] = None):
        """
        Initialize MLflow logger

        Args:
            experiment_id: MLflow experiment ID
            run_name: Name for this run
            tags: Additional tags
        """
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.tags = {**MLflowConfig.STANDARD_TAGS, **(tags or {})}

    def log_data_split_info(self,
                           train_df: pd.DataFrame,
                           val_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           split_config: Dict[str, str]):
        """
        Log time-series split information

        CRITICAL: Log split dates and sizes for reproducibility

        Args:
            train_df, val_df, test_df: DataFrames for each split
            split_config: Dict with train_end and val_end dates
        """
        # Log split configuration
        mlflow.log_params({
            'split_train_end_date': split_config['train_end'],
            'split_val_end_date': split_config['val_end'],
            'split_strategy': 'chronological',
        })

        # Log split sizes
        mlflow.log_metrics({
            'data_train_samples': len(train_df),
            'data_val_samples': len(val_df),
            'data_test_samples': len(test_df),
            'data_total_samples': len(train_df) + len(val_df) + len(test_df),
        })

        # Log date ranges
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if 'game_date' in df.columns:
                mlflow.log_params({
                    f'data_{name}_start_date': str(df['game_date'].min()),
                    f'data_{name}_end_date': str(df['game_date'].max()),
                })

        # Log unique players per split
        if 'player_id' in train_df.columns:
            mlflow.log_metrics({
                'data_train_unique_players': train_df['player_id'].nunique(),
                'data_val_unique_players': val_df['player_id'].nunique(),
                'data_test_unique_players': test_df['player_id'].nunique(),
            })

        # Create and log split visualization
        self._log_split_visualization(train_df, val_df, test_df)

    def _log_split_visualization(self, train_df, val_df, test_df):
        """Create visualization of train/val/test split"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Sample counts
        splits = ['Train', 'Validation', 'Test']
        counts = [len(train_df), len(val_df), len(test_df)]

        ax1.bar(splits, counts, color=['#2E86AB', '#A23B72', '#F18F01'])
        ax1.set_ylabel('Number of Samples')
        ax1.set_title('Train/Validation/Test Split Sizes')
        ax1.grid(axis='y', alpha=0.3)

        for i, (split, count) in enumerate(zip(splits, counts)):
            ax1.text(i, count + 100, f'{count:,}', ha='center', va='bottom')

        # Plot 2: Timeline
        if 'game_date' in train_df.columns:
            train_dates = pd.to_datetime(train_df['game_date'])
            val_dates = pd.to_datetime(val_df['game_date'])
            test_dates = pd.to_datetime(test_df['game_date'])

            ax2.hist([train_dates, val_dates, test_dates],
                    bins=50,
                    label=splits,
                    color=['#2E86AB', '#A23B72', '#F18F01'],
                    alpha=0.7)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Number of Games')
            ax2.set_title('Temporal Distribution of Splits')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        # Save and log
        viz_path = ARTIFACTS_DIR / "data_split_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact(str(viz_path), artifact_path="data_info")

    def log_feature_info(self, X_train: pd.DataFrame, feature_importance: Optional[pd.DataFrame] = None):
        """
        Log feature engineering information

        Args:
            X_train: Training features
            feature_importance: Optional feature importance DataFrame
        """
        # Log feature counts
        mlflow.log_metrics({
            'features_total_count': X_train.shape[1],
            'features_missing_ratio': X_train.isnull().sum().sum() / (X_train.shape[0] * X_train.shape[1]),
        })

        # Log feature list as artifact
        feature_info = pd.DataFrame({
            'feature_name': X_train.columns,
            'dtype': X_train.dtypes.values,
            'missing_count': X_train.isnull().sum().values,
            'missing_pct': (X_train.isnull().sum() / len(X_train) * 100).values,
        })

        feature_info_path = ARTIFACTS_DIR / "feature_info.csv"
        feature_info.to_csv(feature_info_path, index=False)
        mlflow.log_artifact(str(feature_info_path), artifact_path="features")

        # Log feature importance if provided
        if feature_importance is not None:
            self.log_feature_importance(feature_importance)

    def log_feature_importance(self, feature_importance: pd.DataFrame, top_n: int = 30):
        """
        Log and visualize feature importance

        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to visualize
        """
        # Save full feature importance
        fi_path = ARTIFACTS_DIR / "feature_importance.csv"
        feature_importance.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path), artifact_path="features")

        # Log top feature importance as metrics
        top_features = feature_importance.nlargest(10, 'importance')
        for idx, row in top_features.iterrows():
            mlflow.log_metric(f"feature_importance_{row['feature']}", row['importance'])

        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 12))

        top_n_features = feature_importance.nlargest(top_n, 'importance')

        ax.barh(range(len(top_n_features)), top_n_features['importance'].values)
        ax.set_yticks(range(len(top_n_features)))
        ax.set_yticklabels(top_n_features['feature'].values)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        viz_path = ARTIFACTS_DIR / "feature_importance_plot.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact(str(viz_path), artifact_path="features")

    def log_model_performance(self,
                             y_true: pd.Series,
                             y_pred: np.ndarray,
                             split_name: str = 'test',
                             save_predictions: bool = True):
        """
        Log model performance metrics and visualizations

        Args:
            y_true: True target values
            y_pred: Predicted values
            split_name: Name of split ('train', 'val', 'test')
            save_predictions: Whether to save predictions as artifact
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Log metrics
        mlflow.log_metrics({
            f'{split_name}_rmse': rmse,
            f'{split_name}_mae': mae,
            f'{split_name}_r2': r2,
        })

        # Calculate residuals
        residuals = y_true - y_pred

        mlflow.log_metrics({
            f'{split_name}_mean_residual': np.mean(residuals),
            f'{split_name}_std_residual': np.std(residuals),
            f'{split_name}_median_abs_error': np.median(np.abs(residuals)),
        })

        # Save predictions
        if save_predictions:
            predictions_df = pd.DataFrame({
                'y_true': y_true,
                'y_pred': y_pred,
                'residual': residuals,
                'abs_error': np.abs(residuals),
            })

            pred_path = ARTIFACTS_DIR / f"{split_name}_predictions.csv"
            predictions_df.to_csv(pred_path, index=False)
            mlflow.log_artifact(str(pred_path), artifact_path="predictions")

        # Create performance visualizations
        self._log_performance_plots(y_true, y_pred, residuals, split_name)

        return {'rmse': rmse, 'mae': mae, 'r2': r2}

    def _log_performance_plots(self, y_true, y_pred, residuals, split_name):
        """Create comprehensive performance visualization plots"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Predicted vs Actual
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(y_true, y_pred, alpha=0.5, s=20)

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        ax1.set_xlabel('Actual PRA')
        ax1.set_ylabel('Predicted PRA')
        ax1.set_title(f'{split_name.title()} Set: Predicted vs Actual')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Residual Plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(y_pred, residuals, alpha=0.5, s=20)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Predicted PRA')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{split_name.title()} Set: Residual Plot')
        ax2.grid(alpha=0.3)

        # 3. Residual Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'{split_name.title()} Set: Residual Distribution')
        ax3.grid(axis='y', alpha=0.3)

        # 4. Error Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        abs_errors = np.abs(residuals)
        ax4.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        ax4.set_xlabel('Absolute Error')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'{split_name.title()} Set: Absolute Error Distribution')
        ax4.grid(axis='y', alpha=0.3)

        # 5. Q-Q Plot for residuals
        ax5 = fig.add_subplot(gs[2, 0])
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax5)
        ax5.set_title(f'{split_name.title()} Set: Q-Q Plot (Normality Check)')
        ax5.grid(alpha=0.3)

        # 6. Cumulative Error
        ax6 = fig.add_subplot(gs[2, 1])
        sorted_errors = np.sort(abs_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        ax6.plot(sorted_errors, cumulative, lw=2)
        ax6.set_xlabel('Absolute Error')
        ax6.set_ylabel('Cumulative Percentage (%)')
        ax6.set_title(f'{split_name.title()} Set: Cumulative Error Distribution')
        ax6.grid(alpha=0.3)

        # Add percentile lines
        for pct in [50, 75, 90, 95]:
            error_at_pct = sorted_errors[int(len(sorted_errors) * pct / 100)]
            ax6.axhline(y=pct, color='gray', linestyle='--', alpha=0.5)
            ax6.axvline(x=error_at_pct, color='gray', linestyle='--', alpha=0.5)
            ax6.text(error_at_pct, pct + 2, f'{pct}%: {error_at_pct:.2f}', fontsize=8)

        plt.suptitle(f'Model Performance Analysis - {split_name.title()} Set', fontsize=14, y=0.995)

        # Save and log
        viz_path = ARTIFACTS_DIR / f"{split_name}_performance_plots.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        mlflow.log_artifact(str(viz_path), artifact_path="performance")

    def log_training_info(self,
                         training_time: float,
                         model_params: Dict[str, Any],
                         framework_versions: Optional[Dict[str, str]] = None):
        """
        Log training metadata

        Args:
            training_time: Time taken to train (seconds)
            model_params: Model hyperparameters
            framework_versions: Library versions
        """
        # Log training time
        mlflow.log_metric('training_time_seconds', training_time)

        # Log model hyperparameters
        # Flatten nested parameters and convert to strings for MLflow
        flat_params = self._flatten_params(model_params)
        mlflow.log_params(flat_params)

        # Log framework versions
        if framework_versions:
            for lib, version in framework_versions.items():
                mlflow.set_tag(f'framework_{lib}', version)

    def _flatten_params(self, params: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, str]:
        """Flatten nested parameter dictionary"""
        items = []
        for k, v in params.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_params(v, new_key, sep=sep).items())
            else:
                # Convert to string to handle all types
                items.append((new_key, str(v)))

        return dict(items)

    def log_model(self,
                  model,
                  model_type: str,
                  signature=None,
                  input_example=None,
                  registered_model_name: Optional[str] = None):
        """
        Log model to MLflow

        Args:
            model: Trained model
            model_type: Type of model ('xgboost', 'lightgbm', etc.)
            signature: MLflow model signature
            input_example: Example input for model
            registered_model_name: Name for model registry
        """
        # Log model based on type
        if model_type == 'xgboost':
            mlflow.xgboost.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        elif model_type == 'lightgbm':
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )
        else:
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name
            )

        print(f"Model logged to MLflow (type: {model_type})")


def create_model_signature(X_train: pd.DataFrame, y_pred: np.ndarray):
    """
    Create MLflow model signature for input/output schema

    Args:
        X_train: Training features
        y_pred: Sample predictions

    Returns:
        MLflow ModelSignature
    """
    from mlflow.models.signature import infer_signature

    # Use a sample for signature inference
    sample_input = X_train.head(100)
    sample_output = y_pred[:100]

    signature = infer_signature(sample_input, sample_output)

    return signature


def get_framework_versions() -> Dict[str, str]:
    """
    Get versions of ML frameworks

    Returns:
        Dictionary of framework versions
    """
    import sklearn
    import xgboost
    import lightgbm

    versions = {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'sklearn': sklearn.__version__,
        'xgboost': xgboost.__version__,
        'lightgbm': lightgbm.__version__,
        'mlflow': mlflow.__version__,
    }

    return versions


# Utility functions for model registry

def register_best_model(experiment_id: str,
                       model_name: str,
                       metric: str = 'test_rmse',
                       minimize: bool = True) -> str:
    """
    Register the best model from an experiment to the model registry

    Args:
        experiment_id: MLflow experiment ID
        model_name: Name for registered model
        metric: Metric to optimize
        minimize: Whether to minimize metric (True for RMSE, False for R²)

    Returns:
        Run ID of registered model
    """
    client = MlflowClient()

    # Get all runs from experiment
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if minimize else 'DESC'}"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No runs found in experiment {experiment_id}")

    best_run = runs[0]

    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            'best_metric': metric,
            'best_value': best_run.data.metrics.get(metric),
            'registered_at': datetime.now().isoformat(),
        }
    )

    print(f"Registered model '{model_name}' (version {result.version})")
    print(f"Run ID: {best_run.info.run_id}")
    print(f"Best {metric}: {best_run.data.metrics.get(metric):.4f}")

    return best_run.info.run_id


def transition_model_stage(model_name: str,
                          version: int,
                          stage: str,
                          archive_existing: bool = True):
    """
    Transition model to a different stage

    Args:
        model_name: Registered model name
        version: Model version
        stage: Target stage ('Staging', 'Production', 'Archived')
        archive_existing: Whether to archive existing production models
    """
    client = MlflowClient()

    # Archive existing production models if requested
    if stage == 'Production' and archive_existing:
        prod_versions = client.get_latest_versions(model_name, stages=['Production'])
        for prod_version in prod_versions:
            client.transition_model_version_stage(
                name=model_name,
                version=prod_version.version,
                stage='Archived'
            )
            print(f"Archived previous production version: {prod_version.version}")

    # Transition to new stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )

    print(f"Transitioned model '{model_name}' version {version} to {stage}")


if __name__ == "__main__":
    # Example usage
    print("MLflow Configuration Module")
    print("=" * 60)

    config = MLflowConfig()

    # Create experiments for each model type
    for model_type in ['xgboost', 'lightgbm', 'baseline']:
        exp_id = config.get_or_create_experiment(model_type)
        print(f"{model_type}: {exp_id}")

    print("\n" + "=" * 60)
    print("Framework Versions:")
    for lib, version in get_framework_versions().items():
        print(f"  {lib}: {version}")
