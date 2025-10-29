"""
Model Training with MLflow Integration

This script trains NBA PRA prediction models with comprehensive MLflow tracking.

Features:
- Multiple model types (XGBoost, LightGBM, etc.)
- Hyperparameter tracking
- Performance metrics and visualizations
- Feature importance logging
- Model versioning and registration
"""

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from datetime import datetime
import time
import argparse
import json

# ML libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Local imports
from mlflow_config import (
    MLflowConfig,
    MLflowLogger,
    create_model_signature,
    get_framework_versions,
    register_best_model,
    transition_model_stage
)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_train_val_test():
    """
    Load preprocessed train/validation/test splits

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        raise FileNotFoundError(
            "Train/val/test splits not found!\n"
            "Please run: python model_training/train_split.py"
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    print("Loaded data splits:")
    print(f"  Train: {train_df.shape}")
    print(f"  Validation: {val_df.shape}")
    print(f"  Test: {test_df.shape}")

    return train_df, val_df, test_df


def prepare_features_target(df):
    """
    Separate features, target, and metadata

    Args:
        df: DataFrame with all columns

    Returns:
        Tuple of (X, y, metadata)
    """
    # Metadata columns
    metadata_cols = ['player_id', 'game_id', 'game_date', 'player_name', 'season']
    metadata = df[metadata_cols].copy()

    # Target
    y = df['target_pra'].copy()

    # Features (everything except target and metadata)
    exclude_cols = metadata_cols + ['target_pra']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    return X, y, metadata


def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """
    Train XGBoost model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: XGBoost hyperparameters

    Returns:
        Trained model and feature importance DataFrame
    """
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': 42,
            'tree_method': 'hist',
            'eval_metric': 'rmse',
        }

    print("\nTraining XGBoost model...")
    print(f"Parameters: {params}")

    # Create model
    model = xgb.XGBRegressor(**params)

    # Train with early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=50
    )

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, feature_importance


def train_lightgbm(X_train, y_train, X_val, y_val, params=None):
    """
    Train LightGBM model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: LightGBM hyperparameters

    Returns:
        Trained model and feature importance DataFrame
    """
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'random_state': 42,
            'verbose': -1,
        }

    print("\nTraining LightGBM model...")
    print(f"Parameters: {params}")

    # Create model
    model = lgb.LGBMRegressor(**params)

    # Train with early stopping
    eval_set = [(X_train, y_train), (X_val, y_val)]

    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        eval_metric='rmse',
        callbacks=[lgb.log_evaluation(period=50)]
    )

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, feature_importance


def train_random_forest(X_train, y_train, X_val, y_val, params=None):
    """
    Train Random Forest model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (not used for training)
        params: Random Forest hyperparameters

    Returns:
        Trained model and feature importance DataFrame
    """
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'random_state': 42,
            'n_jobs': -1,
        }

    print("\nTraining Random Forest model...")
    print(f"Parameters: {params}")

    # Create and train model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, feature_importance


def train_baseline(X_train, y_train, X_val, y_val, params=None):
    """
    Train baseline Ridge regression model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data (not used for training)
        params: Ridge hyperparameters

    Returns:
        Trained model and feature importance DataFrame
    """
    if params is None:
        params = {
            'alpha': 1.0,
            'random_state': 42,
        }

    print("\nTraining Ridge Regression baseline...")
    print(f"Parameters: {params}")

    # Create and train model
    model = Ridge(**params)
    model.fit(X_train, y_train)

    # Get feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False)

    return model, feature_importance


def evaluate_model(model, X, y, split_name='test'):
    """
    Evaluate model performance

    Args:
        model: Trained model
        X: Features
        y: True targets
        split_name: Name of split for logging

    Returns:
        Dictionary of metrics and predictions
    """
    # Make predictions
    y_pred = model.predict(X)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\n{split_name.upper()} Set Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")

    return {
        'predictions': y_pred,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
        }
    }


def train_model_with_mlflow(model_type='xgboost',
                            params=None,
                            register_model=False,
                            run_description=None):
    """
    Complete training pipeline with MLflow tracking

    Args:
        model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'baseline')
        params: Model hyperparameters
        register_model: Whether to register model in MLflow Model Registry
        run_description: Optional description for this run

    Returns:
        Trained model and MLflow run ID
    """
    print("=" * 80)
    print(f"TRAINING NBA PRA PREDICTION MODEL - {model_type.upper()}")
    print("=" * 80)

    # Initialize MLflow
    mlflow_config = MLflowConfig()
    experiment_id = mlflow_config.get_or_create_experiment(model_type)

    # Generate run name
    run_name = mlflow_config.generate_run_name(model_type, run_description)

    # Load data
    train_df, val_df, test_df = load_train_val_test()

    # Prepare features and target
    X_train, y_train, meta_train = prepare_features_target(train_df)
    X_val, y_val, meta_val = prepare_features_target(val_df)
    X_test, y_test, meta_test = prepare_features_target(test_df)

    # Start MLflow run
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:

        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"Experiment ID: {experiment_id}")

        # Initialize logger
        logger = MLflowLogger(
            experiment_id=experiment_id,
            run_name=run_name,
            tags={
                'model_type': model_type,
                'description': run_description or f'{model_type} model',
                'training_date': datetime.now().isoformat(),
            }
        )

        # Log data split information
        split_config = {
            'train_end': str(train_df['game_date'].max()),
            'val_end': str(val_df['game_date'].max()),
        }
        logger.log_data_split_info(train_df, val_df, test_df, split_config)

        # Log feature information
        logger.log_feature_info(X_train)

        # Get framework versions
        framework_versions = get_framework_versions()

        # Train model
        start_time = time.time()

        if model_type == 'xgboost':
            model, feature_importance = train_xgboost(X_train, y_train, X_val, y_val, params)
        elif model_type == 'lightgbm':
            model, feature_importance = train_lightgbm(X_train, y_train, X_val, y_val, params)
        elif model_type == 'random_forest':
            model, feature_importance = train_random_forest(X_train, y_train, X_val, y_val, params)
        elif model_type == 'baseline':
            model, feature_importance = train_baseline(X_train, y_train, X_val, y_val, params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        training_time = time.time() - start_time

        # Log training info
        model_params = params if params else {}
        logger.log_training_info(training_time, model_params, framework_versions)

        # Log feature importance
        logger.log_feature_importance(feature_importance)

        # Evaluate on all splits
        print("\n" + "=" * 80)
        print("EVALUATING MODEL PERFORMANCE")
        print("=" * 80)

        # Train set
        train_results = evaluate_model(model, X_train, y_train, 'train')
        logger.log_model_performance(
            y_train,
            train_results['predictions'],
            split_name='train',
            save_predictions=False  # Don't save train predictions to save space
        )

        # Validation set
        val_results = evaluate_model(model, X_val, y_val, 'val')
        logger.log_model_performance(
            y_val,
            val_results['predictions'],
            split_name='val',
            save_predictions=True
        )

        # Test set
        test_results = evaluate_model(model, X_test, y_test, 'test')
        logger.log_model_performance(
            y_test,
            test_results['predictions'],
            split_name='test',
            save_predictions=True
        )

        # Create model signature
        signature = create_model_signature(X_train, train_results['predictions'])

        # Log model
        registered_model_name = None
        if register_model:
            registered_model_name = f"nba_pra_{model_type}"

        logger.log_model(
            model,
            model_type=model_type,
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=registered_model_name
        )

        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Model Type: {model_type}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Training Time: {training_time:.2f} seconds")
        print("\nPerformance Summary:")
        print(f"  Train RMSE: {train_results['metrics']['rmse']:.4f}")
        print(f"  Val RMSE: {val_results['metrics']['rmse']:.4f}")
        print(f"  Test RMSE: {test_results['metrics']['rmse']:.4f}")
        print(f"  Test R²: {test_results['metrics']['r2']:.4f}")
        print("\nView results in MLflow UI:")
        print(f"  mlflow ui --port 5000")
        print(f"  Then navigate to: http://localhost:5000")

        return model, run.info.run_id


def hyperparameter_tuning_with_mlflow(model_type='xgboost',
                                     param_grid=None,
                                     n_trials=10,
                                     parent_run_name=None):
    """
    Hyperparameter tuning with nested MLflow runs

    Args:
        model_type: Type of model
        param_grid: Grid of parameters to try
        n_trials: Number of random trials
        parent_run_name: Name for parent run

    Returns:
        Best parameters and run ID
    """
    print("=" * 80)
    print(f"HYPERPARAMETER TUNING - {model_type.upper()}")
    print("=" * 80)

    # Initialize MLflow
    mlflow_config = MLflowConfig()
    experiment_id = mlflow_config.get_or_create_experiment(model_type)

    if parent_run_name is None:
        parent_run_name = f"{model_type}_hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Default parameter grids
    if param_grid is None:
        if model_type == 'xgboost':
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
            }
        elif model_type == 'lightgbm':
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'num_leaves': [31, 63, 127],
            }

    # Start parent run
    with mlflow.start_run(experiment_id=experiment_id, run_name=parent_run_name) as parent_run:

        mlflow.set_tag('run_type', 'hyperparameter_search')
        mlflow.log_param('n_trials', n_trials)

        best_rmse = float('inf')
        best_params = None
        best_run_id = None

        # Random search
        import random
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")

            # Sample random parameters
            trial_params = {
                param: random.choice(values)
                for param, values in param_grid.items()
            }

            # Start child run
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"{model_type}_trial_{trial+1}",
                nested=True
            ) as child_run:

                mlflow.set_tag('run_type', 'trial')
                mlflow.set_tag('parent_run_id', parent_run.info.run_id)

                # Train model with these parameters
                try:
                    model, run_id = train_model_with_mlflow(
                        model_type=model_type,
                        params=trial_params,
                        register_model=False,
                        run_description=f"trial_{trial+1}"
                    )

                    # Get validation RMSE
                    val_rmse = mlflow.get_run(child_run.info.run_id).data.metrics.get('val_rmse')

                    # Track best
                    if val_rmse < best_rmse:
                        best_rmse = val_rmse
                        best_params = trial_params
                        best_run_id = child_run.info.run_id

                        mlflow.set_tag('best_trial', trial + 1, run_id=parent_run.info.run_id)
                        mlflow.log_metric('best_val_rmse', best_rmse, run_id=parent_run.info.run_id)

                except Exception as e:
                    print(f"Trial {trial+1} failed: {e}")
                    mlflow.set_tag('status', 'failed')
                    continue

        # Log best parameters to parent run
        if best_params:
            for param, value in best_params.items():
                mlflow.log_param(f'best_{param}', value, run_id=parent_run.info.run_id)

            print("\n" + "=" * 80)
            print("HYPERPARAMETER SEARCH COMPLETE")
            print("=" * 80)
            print(f"Best Validation RMSE: {best_rmse:.4f}")
            print(f"Best Parameters: {best_params}")
            print(f"Best Run ID: {best_run_id}")

        return best_params, best_run_id


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Train NBA PRA prediction model with MLflow')

    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'random_forest', 'baseline'],
                       help='Type of model to train')

    parser.add_argument('--register', action='store_true',
                       help='Register model in MLflow Model Registry')

    parser.add_argument('--description', type=str, default=None,
                       help='Description for this run')

    parser.add_argument('--hyperparameter-search', action='store_true',
                       help='Perform hyperparameter search')

    parser.add_argument('--n-trials', type=int, default=10,
                       help='Number of trials for hyperparameter search')

    args = parser.parse_args()

    if args.hyperparameter_search:
        # Hyperparameter tuning
        best_params, best_run_id = hyperparameter_tuning_with_mlflow(
            model_type=args.model_type,
            n_trials=args.n_trials
        )

        # Train final model with best parameters
        if best_params:
            print("\n" + "=" * 80)
            print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
            print("=" * 80)

            model, run_id = train_model_with_mlflow(
                model_type=args.model_type,
                params=best_params,
                register_model=args.register,
                run_description='best_from_search'
            )
    else:
        # Single training run
        model, run_id = train_model_with_mlflow(
            model_type=args.model_type,
            register_model=args.register,
            run_description=args.description
        )


if __name__ == "__main__":
    main()
