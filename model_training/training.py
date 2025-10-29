"""
Model Training
Main training orchestration for NBA PRA prediction models
"""

import mlflow
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from model_training.config import (
    MODEL_DIR,
    LOGS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TAGS,
    EXPECTED_R2_TARGET,
    EXPECTED_RMSE_TARGET,
)
from model_training.utils import (
    setup_logger,
    load_split_data,
    prepare_features_target,
    calculate_regression_metrics,
    validate_temporal_order,
    check_missing_values,
    format_metrics_table,
    load_cv_fold,
    get_available_cv_folds,
    calculate_cv_summary_statistics,
)
from model_training.models import XGBoostModel, LightGBMModel, BaseModel, EnsembleModel


def train_model(
    model_type: str = "xgboost",
    params: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
    mlflow_tracking: bool = True,
    log_file: Optional[Path] = None
) -> Tuple[BaseModel, Dict[str, float]]:
    """
    Train a single model with specified configuration

    Args:
        model_type: One of ['xgboost', 'lightgbm']
        params: Optional custom hyperparameters (uses config defaults if None)
        save_model: Whether to save trained model to disk
        mlflow_tracking: Whether to log to MLflow
        log_file: Optional log file path

    Returns:
        Tuple of (trained_model, all_metrics)
            - trained_model: Trained model instance
            - all_metrics: Dictionary with train/val/test metrics

    Raises:
        ValueError: If model_type is invalid
        FileNotFoundError: If split files don't exist
    """
    # Setup logging
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOGS_DIR / f"training_{model_type}_{timestamp}.log"

    logger = setup_logger("training", log_file=log_file)

    logger.info("="*60)
    logger.info(f"Starting {model_type.upper()} Model Training")
    logger.info("="*60)

    # ========================================================================
    # 1. Load Data
    # ========================================================================

    logger.info("\n[1/8] Loading train/val/test splits")

    try:
        train_df = load_split_data("train")
        val_df = load_split_data("val")
        test_df = load_split_data("test")
    except FileNotFoundError as e:
        logger.error(str(e))
        raise

    logger.info(f"  Train: {len(train_df):,} samples")
    logger.info(f"  Val:   {len(val_df):,} samples")
    logger.info(f"  Test:  {len(test_df):,} samples")

    # ========================================================================
    # 2. Prepare Features and Target
    # ========================================================================

    logger.info("\n[2/8] Preparing features and target")

    X_train, y_train, _ = prepare_features_target(train_df)
    X_val, y_val, _ = prepare_features_target(val_df)
    X_test, y_test, _ = prepare_features_target(test_df)

    logger.info(f"  Features: {len(X_train.columns)} columns")
    logger.info(f"  Target: {y_train.name}")

    # Check for missing values
    missing_train = check_missing_values(X_train)
    if missing_train:
        logger.warning(f"  Missing values in train set: {len(missing_train)} columns")
        logger.warning("  Assuming train_split.py handled imputation")

    # ========================================================================
    # 3. Validate Temporal Order
    # ========================================================================

    logger.info("\n[3/8] Validating temporal order")

    try:
        validate_temporal_order(train_df)
        validate_temporal_order(val_df)
        validate_temporal_order(test_df)
        logger.info("  ✓ All splits in chronological order")
    except ValueError as e:
        logger.error(f"  ✗ Temporal validation failed: {e}")
        raise

    # ========================================================================
    # 4. Initialize Model
    # ========================================================================

    logger.info("\n[4/8] Initializing model")

    if model_type.lower() == "xgboost":
        model = XGBoostModel(params)
    elif model_type.lower() == "lightgbm":
        model = LightGBMModel(params)
    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            "Must be one of ['xgboost', 'lightgbm']"
        )

    logger.info(f"  Model: {model.name}")
    logger.info(f"  Parameters: {len(model.params)} configured")

    # ========================================================================
    # 5. Train Model
    # ========================================================================

    logger.info("\n[5/8] Training model with early stopping")
    logger.info(f"  Early stopping: Monitoring validation RMSE")

    training_info = model.train(X_train, y_train, X_val, y_val)

    logger.info(f"  ✓ Training complete")
    logger.info(f"  Best iteration: {training_info['best_iteration']}")
    logger.info(f"  Estimators used: {training_info['n_estimators_used']}")

    # ========================================================================
    # 6. Calculate Metrics on All Splits
    # ========================================================================

    logger.info("\n[6/8] Calculating metrics on all splits")

    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = calculate_regression_metrics(y_train, y_train_pred, prefix="train_")
    val_metrics = calculate_regression_metrics(y_val, y_val_pred, prefix="val_")
    test_metrics = calculate_regression_metrics(y_test, y_test_pred, prefix="test_")

    # Combine all metrics
    all_metrics = {**training_info, **train_metrics, **val_metrics, **test_metrics}

    # Log metrics table
    logger.info("\n" + format_metrics_table(all_metrics, title="Model Performance"))

    # Check performance vs targets
    if all_metrics['test_r2'] >= EXPECTED_R2_TARGET:
        logger.info(f"✓ Test R² ({all_metrics['test_r2']:.4f}) meets target ({EXPECTED_R2_TARGET:.4f})")
    else:
        logger.warning(f"⚠ Test R² ({all_metrics['test_r2']:.4f}) below target ({EXPECTED_R2_TARGET:.4f})")

    if all_metrics['test_rmse'] <= EXPECTED_RMSE_TARGET:
        logger.info(f"✓ Test RMSE ({all_metrics['test_rmse']:.4f}) meets target ({EXPECTED_RMSE_TARGET:.4f})")
    else:
        logger.warning(f"⚠ Test RMSE ({all_metrics['test_rmse']:.4f}) above target ({EXPECTED_RMSE_TARGET:.4f})")

    # ========================================================================
    # 7. MLflow Logging
    # ========================================================================

    if mlflow_tracking:
        logger.info("\n[7/8] Logging to MLflow")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_type}_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log tags
            mlflow.set_tags(MLFLOW_TAGS)
            mlflow.set_tag("model_type", model_type)

            # Log parameters
            mlflow.log_params(model.params)
            mlflow.log_param("n_features", len(X_train.columns))
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))

            # Log metrics
            for metric_name, metric_value in all_metrics.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(metric_name, metric_value)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log feature importance
            feature_importance = model.get_feature_importance()
            importance_path = MODEL_DIR / "temp_feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            importance_path.unlink()  # Delete temp file

            logger.info(f"  ✓ Logged to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
            logger.info(f"  Run name: {run_name}")

    else:
        logger.info("\n[7/8] Skipping MLflow logging (disabled)")

    # ========================================================================
    # 8. Save Model
    # ========================================================================

    if save_model:
        logger.info("\n[8/8] Saving model to disk")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"{model_type}_model_{timestamp}.pkl"
        model_path = MODEL_DIR / model_filename

        model.save(model_path)
        logger.info(f"  ✓ Model saved: {model_path}")

        # Save feature importance
        feature_importance = model.get_feature_importance()
        importance_path = MODEL_DIR / f"{model_type}_feature_importance_{timestamp}.csv"
        feature_importance.to_csv(importance_path, index=False)
        logger.info(f"  ✓ Feature importance saved: {importance_path}")

    else:
        logger.info("\n[8/8] Skipping model save (disabled)")

    # ========================================================================
    # Summary
    # ========================================================================

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model: {model.name}")
    logger.info(f"Validation RMSE: {all_metrics['val_rmse']:.4f}")
    logger.info(f"Validation R²: {all_metrics['val_r2']:.4f}")
    logger.info(f"Test RMSE: {all_metrics['test_rmse']:.4f}")
    logger.info(f"Test R²: {all_metrics['test_r2']:.4f}")
    logger.info("="*60)

    return model, all_metrics


def train_with_cv(
    model_type: str = "xgboost",
    params: Optional[Dict[str, Any]] = None,
    save_fold_models: bool = False,
    save_ensemble: bool = True,
    mlflow_tracking: bool = True,
    log_file: Optional[Path] = None
) -> Tuple[EnsembleModel, Dict[str, Any]]:
    """
    Train model using time-series cross-validation with 3-year rolling windows

    Args:
        model_type: One of ['xgboost', 'lightgbm']
        params: Optional custom hyperparameters (uses config defaults if None)
        save_fold_models: Whether to save individual fold models
        save_ensemble: Whether to save ensemble model
        mlflow_tracking: Whether to log to MLflow
        log_file: Optional log file path

    Returns:
        Tuple of (ensemble_model, cv_summary_metrics)
            - ensemble_model: Ensemble of all fold models
            - cv_summary_metrics: Aggregated CV metrics (mean ± std across folds)

    Raises:
        ValueError: If model_type is invalid or no CV folds found
        FileNotFoundError: If CV fold directories don't exist
    """
    # Setup logging
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = LOGS_DIR / f"training_cv_{model_type}_{timestamp}.log"

    logger = setup_logger("training_cv", log_file=log_file)

    logger.info("="*60)
    logger.info(f"Starting {model_type.upper()} Model Training with TIME-SERIES CV")
    logger.info("="*60)

    # ========================================================================
    # 1. Detect Available CV Folds
    # ========================================================================

    logger.info("\n[1/6] Detecting available CV folds")

    fold_ids = get_available_cv_folds()

    if not fold_ids:
        raise FileNotFoundError(
            "No CV folds found! Please run:\n"
            "  uv run model_training/train_split.py --cv-mode"
        )

    logger.info(f"  Found {len(fold_ids)} CV folds: {fold_ids}")

    # ========================================================================
    # 2. Train Model on Each Fold
    # ========================================================================

    logger.info("\n[2/6] Training model on each fold")

    fold_models = []
    fold_metrics_list = []

    for fold_id in fold_ids:
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_id}/{len(fold_ids)-1}")
        logger.info(f"{'='*60}")

        # Load fold data
        logger.info(f"\n  Loading fold {fold_id} data...")
        try:
            fold_data = load_cv_fold(fold_id)
        except FileNotFoundError as e:
            logger.error(f"  ✗ Failed to load fold {fold_id}: {e}")
            raise

        train_df = fold_data['train']
        val_df = fold_data['val']
        test_df = fold_data['test']

        logger.info(f"    Train: {len(train_df):,} games")
        logger.info(f"    Val:   {len(val_df):,} games")
        logger.info(f"    Test:  {len(test_df):,} games")

        # Prepare features and target
        X_train, y_train, _ = prepare_features_target(train_df)
        X_val, y_val, _ = prepare_features_target(val_df)
        X_test, y_test, _ = prepare_features_target(test_df)

        logger.info(f"    Features: {len(X_train.columns)} columns")

        # Initialize model
        if model_type.lower() == "xgboost":
            fold_model = XGBoostModel(params)
        elif model_type.lower() == "lightgbm":
            fold_model = LightGBMModel(params)
        else:
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                "Must be one of ['xgboost', 'lightgbm']"
            )

        logger.info(f"\n  Training fold {fold_id} model...")

        # Train fold model
        training_info = fold_model.train(X_train, y_train, X_val, y_val)

        logger.info(f"    ✓ Training complete")
        logger.info(f"    Best iteration: {training_info['best_iteration']}")

        # Calculate metrics on test set
        y_test_pred = fold_model.predict(X_test)
        test_metrics = calculate_regression_metrics(y_test, y_test_pred, prefix="test_")

        # Store fold results
        fold_models.append(fold_model)
        fold_metrics_list.append(test_metrics)

        logger.info(f"\n  Fold {fold_id} Test Metrics:")
        logger.info(f"    MAE:  {test_metrics['test_mae']:.4f}")
        logger.info(f"    RMSE: {test_metrics['test_rmse']:.4f}")
        logger.info(f"    R²:   {test_metrics['test_r2']:.4f}")

        # Save fold model if requested
        if save_fold_models:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fold_model_path = MODEL_DIR / f"{model_type}_fold_{fold_id}_{timestamp}.pkl"
            fold_model.save(fold_model_path)
            logger.info(f"    ✓ Saved fold model: {fold_model_path}")

    # ========================================================================
    # 3. Aggregate CV Metrics
    # ========================================================================

    logger.info(f"\n{'='*60}")
    logger.info("[3/6] Aggregating CV metrics across folds")
    logger.info(f"{'='*60}")

    cv_summary = calculate_cv_summary_statistics(fold_metrics_list)

    logger.info("\n  CV Summary Statistics:")
    logger.info(f"  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    logger.info("  " + "-"*60)
    for _, row in cv_summary.iterrows():
        logger.info(
            f"  {row['metric_name']:<20} "
            f"{row['mean']:>10.4f} "
            f"{row['std']:>10.4f} "
            f"{row['min']:>10.4f} "
            f"{row['max']:>10.4f}"
        )

    # Extract key metrics for summary
    mae_row = cv_summary[cv_summary['metric_name'] == 'test_mae'].iloc[0]
    rmse_row = cv_summary[cv_summary['metric_name'] == 'test_rmse'].iloc[0]
    r2_row = cv_summary[cv_summary['metric_name'] == 'test_r2'].iloc[0]

    logger.info(f"\n  Cross-Validation Results:")
    logger.info(f"    MAE:  {mae_row['mean']:.4f} ± {mae_row['std']:.4f}")
    logger.info(f"    RMSE: {rmse_row['mean']:.4f} ± {rmse_row['std']:.4f}")
    logger.info(f"    R²:   {r2_row['mean']:.4f} ± {r2_row['std']:.4f}")

    # ========================================================================
    # 4. Create Ensemble Model
    # ========================================================================

    logger.info(f"\n{'='*60}")
    logger.info("[4/6] Creating ensemble model")
    logger.info(f"{'='*60}")

    ensemble = EnsembleModel(fold_models)

    logger.info(f"  ✓ Ensemble created: {ensemble.name}")
    logger.info(f"  Number of fold models: {len(fold_models)}")

    # Evaluate ensemble on each fold's test set and aggregate
    logger.info("\n  Evaluating ensemble on all test sets...")

    ensemble_metrics_list = []
    for fold_id in fold_ids:
        fold_data = load_cv_fold(fold_id)
        test_df = fold_data['test']
        X_test, y_test, _ = prepare_features_target(test_df)

        y_test_pred = ensemble.predict(X_test)
        ensemble_test_metrics = calculate_regression_metrics(y_test, y_test_pred, prefix="ensemble_test_")
        ensemble_metrics_list.append(ensemble_test_metrics)

    # Aggregate ensemble metrics
    ensemble_summary = calculate_cv_summary_statistics(ensemble_metrics_list)

    logger.info("\n  Ensemble Performance (averaged across folds):")
    for _, row in ensemble_summary.iterrows():
        logger.info(
            f"    {row['metric_name']:<25} "
            f"{row['mean']:>10.4f} ± {row['std']:>10.4f}"
        )

    # ========================================================================
    # 5. MLflow Logging
    # ========================================================================

    if mlflow_tracking:
        logger.info(f"\n{'='*60}")
        logger.info("[5/6] Logging to MLflow")
        logger.info(f"{'='*60}")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_type}_cv_ensemble_{timestamp}"

        with mlflow.start_run(run_name=run_name):
            # Log tags
            mlflow.set_tags(MLFLOW_TAGS)
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("training_method", "time_series_cv")
            mlflow.set_tag("n_folds", len(fold_ids))

            # Log parameters
            mlflow.log_params(ensemble.params)
            mlflow.log_param("n_features", len(ensemble.feature_names))
            mlflow.log_param("n_folds", len(fold_ids))
            mlflow.log_param("ensemble_method", "averaging")

            # Log CV summary metrics (mean values)
            for _, row in cv_summary.iterrows():
                metric_name = row['metric_name'].replace('test_', 'cv_')
                mlflow.log_metric(f"{metric_name}_mean", row['mean'])
                mlflow.log_metric(f"{metric_name}_std", row['std'])

            # Log ensemble metrics
            for _, row in ensemble_summary.iterrows():
                metric_name = row['metric_name']
                mlflow.log_metric(f"{metric_name}_mean", row['mean'])
                mlflow.log_metric(f"{metric_name}_std", row['std'])

            # Log ensemble model
            mlflow.sklearn.log_model(ensemble, "ensemble_model")

            # Log ensemble feature importance
            ensemble_importance = ensemble.get_feature_importance()
            importance_path = MODEL_DIR / "temp_ensemble_feature_importance.csv"
            ensemble_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            importance_path.unlink()  # Delete temp file

            logger.info(f"  ✓ Logged to MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
            logger.info(f"  Run name: {run_name}")

    else:
        logger.info(f"\n[5/6] Skipping MLflow logging (disabled)")

    # ========================================================================
    # 6. Save Ensemble Model
    # ========================================================================

    if save_ensemble:
        logger.info(f"\n{'='*60}")
        logger.info("[6/6] Saving ensemble model")
        logger.info(f"{'='*60}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ensemble_filename = f"{model_type}_ensemble_{len(fold_ids)}folds_{timestamp}.pkl"
        ensemble_path = MODEL_DIR / ensemble_filename

        ensemble.save(ensemble_path)
        logger.info(f"  ✓ Ensemble saved: {ensemble_path}")

        # Save ensemble feature importance
        ensemble_importance = ensemble.get_feature_importance()
        importance_path = MODEL_DIR / f"{model_type}_ensemble_feature_importance_{timestamp}.csv"
        ensemble_importance.to_csv(importance_path, index=False)
        logger.info(f"  ✓ Feature importance saved: {importance_path}")

        # Save CV summary
        cv_summary_path = LOGS_DIR / f"{model_type}_cv_summary_{timestamp}.csv"
        cv_summary.to_csv(cv_summary_path, index=False)
        logger.info(f"  ✓ CV summary saved: {cv_summary_path}")

    else:
        logger.info(f"\n[6/6] Skipping ensemble save (disabled)")

    # ========================================================================
    # Summary
    # ========================================================================

    logger.info("\n" + "="*60)
    logger.info("TIME-SERIES CV TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model: {ensemble.name}")
    logger.info(f"Folds trained: {len(fold_ids)}")
    logger.info(f"CV MAE:  {mae_row['mean']:.4f} ± {mae_row['std']:.4f}")
    logger.info(f"CV RMSE: {rmse_row['mean']:.4f} ± {rmse_row['std']:.4f}")
    logger.info(f"CV R²:   {r2_row['mean']:.4f} ± {r2_row['std']:.4f}")
    logger.info("="*60)

    # Prepare summary metrics dict
    cv_summary_dict = cv_summary.set_index('metric_name').to_dict('index')

    return ensemble, cv_summary_dict


def train_all_models(
    mlflow_tracking: bool = True,
    save_models: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train all model types with default configs (convenience function)

    Args:
        mlflow_tracking: Whether to log to MLflow
        save_models: Whether to save trained models

    Returns:
        Dictionary mapping model_name -> {'model': model, 'metrics': metrics}
    """
    logger = setup_logger("training")

    logger.info("="*60)
    logger.info("TRAINING ALL MODELS")
    logger.info("="*60)

    results = {}
    model_types = ["xgboost", "lightgbm"]

    for model_type in model_types:
        logger.info(f"\n\n{'='*60}")
        logger.info(f"Training {model_type.upper()}")
        logger.info(f"{'='*60}\n")

        try:
            model, metrics = train_model(
                model_type=model_type,
                mlflow_tracking=mlflow_tracking,
                save_model=save_models
            )

            results[model_type] = {
                'model': model,
                'metrics': metrics
            }

        except Exception as e:
            logger.error(f"Failed to train {model_type}: {e}")
            results[model_type] = {
                'model': None,
                'metrics': None,
                'error': str(e)
            }

    # ========================================================================
    # Comparison Table
    # ========================================================================

    logger.info("\n\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)

    comparison_data = []
    for model_name, result in results.items():
        if result['metrics']:
            comparison_data.append({
                'Model': model_name.upper(),
                'Val RMSE': result['metrics']['val_rmse'],
                'Val R²': result['metrics']['val_r2'],
                'Test RMSE': result['metrics']['test_rmse'],
                'Test R²': result['metrics']['test_r2'],
            })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test RMSE')  # Best model first

        logger.info("\n" + comparison_df.to_string(index=False))

        # Identify best model
        best_model_name = comparison_df.iloc[0]['Model'].lower()
        best_rmse = comparison_df.iloc[0]['Test RMSE']
        best_r2 = comparison_df.iloc[0]['Test R²']

        logger.info(f"\n✓ Best Model: {best_model_name.upper()}")
        logger.info(f"  Test RMSE: {best_rmse:.4f}")
        logger.info(f"  Test R²: {best_r2:.4f}")

    logger.info("="*60)

    return results


def main():
    """
    Main entry point for training
    """
    import argparse

    parser = argparse.ArgumentParser(description="Train NBA PRA prediction model")
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "all"],
        help="Model type to train (default: xgboost)"
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Use time-series cross-validation (3-year rolling windows)"
    )
    parser.add_argument(
        "--save-fold-models",
        action="store_true",
        help="Save individual fold models (only with --cv)"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save trained model to disk"
    )

    args = parser.parse_args()

    # Train model(s)
    if args.cv:
        # CV MODE: Train with time-series cross-validation
        if args.model_type == "all":
            print("ERROR: --cv mode does not support 'all' model types")
            print("Please specify a specific model type: --model-type xgboost or --model-type lightgbm")
            return

        train_with_cv(
            model_type=args.model_type,
            save_fold_models=args.save_fold_models,
            save_ensemble=not args.no_save,
            mlflow_tracking=not args.no_mlflow
        )
    else:
        # SINGLE SPLIT MODE: Train on single train/val/test split
        if args.model_type == "all":
            train_all_models(
                mlflow_tracking=not args.no_mlflow,
                save_models=not args.no_save
            )
        else:
            train_model(
                model_type=args.model_type,
                mlflow_tracking=not args.no_mlflow,
                save_model=not args.no_save
            )


if __name__ == "__main__":
    main()
