"""
Example MLflow Usage for NBA PRA Prediction

This script demonstrates common MLflow workflows:
1. Basic training with tracking
2. Hyperparameter comparison
3. Model registration
4. Loading models from registry

Run this script to familiarize yourself with MLflow integration.
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from pathlib import Path

# Import our MLflow utilities
from mlflow_config import (
    MLflowConfig,
    MLflowLogger,
    get_framework_versions,
    register_best_model,
    transition_model_stage
)


def example_1_basic_tracking():
    """
    Example 1: Basic MLflow tracking

    Demonstrates:
    - Creating an experiment
    - Starting a run
    - Logging parameters and metrics
    - Viewing in MLflow UI
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic MLflow Tracking")
    print("=" * 80)

    # Initialize MLflow
    config = MLflowConfig()
    experiment_id = config.get_or_create_experiment('xgboost')

    # Start a run
    with mlflow.start_run(experiment_id=experiment_id, run_name='example_basic_tracking') as run:

        # Log parameters
        mlflow.log_params({
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
        })

        # Simulate training and log metrics
        mlflow.log_metrics({
            'train_rmse': 3.2,
            'val_rmse': 3.5,
            'test_rmse': 3.6,
            'test_r2': 0.84,
        })

        # Log tags
        mlflow.set_tags({
            'model_type': 'xgboost',
            'description': 'Example run for demonstration',
            'status': 'completed',
        })

        print(f"\nRun ID: {run.info.run_id}")
        print(f"Experiment ID: {experiment_id}")
        print("\nView this run in MLflow UI:")
        print("  mlflow ui --port 5000")
        print("  http://localhost:5000")


def example_2_comparing_runs():
    """
    Example 2: Comparing multiple runs

    Demonstrates:
    - Running multiple experiments with different parameters
    - Querying and comparing runs
    - Finding best model
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Comparing Multiple Runs")
    print("=" * 80)

    config = MLflowConfig()
    experiment_id = config.get_or_create_experiment('xgboost')

    # Simulate training with different hyperparameters
    param_sets = [
        {'max_depth': 4, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 200},
    ]

    run_ids = []

    for i, params in enumerate(param_sets):
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f'example_comparison_{i+1}'
        ) as run:

            # Log parameters
            mlflow.log_params(params)

            # Simulate different performance (deeper trees overfit a bit)
            base_rmse = 3.5
            depth_penalty = (params['max_depth'] - 6) * 0.1
            test_rmse = base_rmse + depth_penalty + np.random.normal(0, 0.05)

            mlflow.log_metrics({
                'test_rmse': test_rmse,
                'test_r2': 1 - (test_rmse / 15) ** 2,  # Rough R² estimate
            })

            run_ids.append(run.info.run_id)
            print(f"Run {i+1}: max_depth={params['max_depth']}, RMSE={test_rmse:.4f}")

    # Query and compare runs
    print("\n" + "-" * 80)
    print("Querying and Comparing Runs")
    print("-" * 80)

    client = MlflowClient()

    # Get all runs from this experiment, sorted by RMSE
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.description = 'Example run for demonstration' OR run_id IN ('" + "','".join(run_ids) + "')",
        order_by=['metrics.test_rmse ASC'],
        max_results=10
    )

    # Create comparison table
    comparison = []
    for run in runs:
        if run.info.run_id in run_ids:  # Only our example runs
            comparison.append({
                'run_id': run.info.run_id[:8] + '...',
                'max_depth': run.data.params.get('max_depth'),
                'learning_rate': run.data.params.get('learning_rate'),
                'n_estimators': run.data.params.get('n_estimators'),
                'test_rmse': run.data.metrics.get('test_rmse'),
                'test_r2': run.data.metrics.get('test_r2'),
            })

    comparison_df = pd.DataFrame(comparison)
    print("\nComparison Table (sorted by RMSE):")
    print(comparison_df.to_string(index=False))

    best_run = runs[0] if runs else None
    if best_run and best_run.info.run_id in run_ids:
        print(f"\nBest Run: {best_run.info.run_id[:8]}...")
        print(f"  Best RMSE: {best_run.data.metrics.get('test_rmse'):.4f}")
        print(f"  Parameters: max_depth={best_run.data.params.get('max_depth')}, "
              f"lr={best_run.data.params.get('learning_rate')}")


def example_3_model_registry():
    """
    Example 3: Using Model Registry

    Demonstrates:
    - Registering a model
    - Transitioning stages
    - Loading models from registry
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Model Registry")
    print("=" * 80)

    config = MLflowConfig()
    experiment_id = config.get_or_create_experiment('xgboost')

    # Create a simple "model" (in reality, this would be a trained model)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='example_for_registry'
    ) as run:

        # Log parameters and metrics
        mlflow.log_params({'max_depth': 6, 'learning_rate': 0.1})
        mlflow.log_metrics({'test_rmse': 3.45, 'test_r2': 0.85})

        # Create a dummy model (sklearn)
        from sklearn.linear_model import Ridge
        dummy_model = Ridge()

        # For demonstration, create dummy data
        X_dummy = pd.DataFrame(np.random.randn(100, 10))
        y_dummy = np.random.randn(100)
        dummy_model.fit(X_dummy, y_dummy)

        # Log model with registry
        mlflow.sklearn.log_model(
            dummy_model,
            artifact_path='model',
            registered_model_name='nba_pra_xgboost_demo'
        )

        run_id = run.info.run_id
        print(f"\nRegistered model from run: {run_id[:8]}...")

    # Transition stages
    print("\n" + "-" * 80)
    print("Model Stage Transitions")
    print("-" * 80)

    client = MlflowClient()

    # Get latest version
    latest_versions = client.get_latest_versions('nba_pra_xgboost_demo')
    if latest_versions:
        version = latest_versions[0].version

        print(f"\nLatest version: {version}")

        # Transition to Staging
        transition_model_stage(
            model_name='nba_pra_xgboost_demo',
            version=version,
            stage='Staging',
            archive_existing=False
        )

        # Simulate testing in staging...
        print("\n[Simulating testing in Staging environment...]")

        # Transition to Production
        transition_model_stage(
            model_name='nba_pra_xgboost_demo',
            version=version,
            stage='Production',
            archive_existing=True
        )

        print("\n" + "-" * 80)
        print("Loading Model from Registry")
        print("-" * 80)

        # Load production model
        try:
            model_uri = f"models:/nba_pra_xgboost_demo/Production"
            production_model = mlflow.sklearn.load_model(model_uri)

            print(f"\nLoaded production model: {model_uri}")
            print(f"Model type: {type(production_model).__name__}")

            # Make a dummy prediction
            test_input = pd.DataFrame(np.random.randn(5, 10))
            predictions = production_model.predict(test_input)

            print(f"\nSample predictions (dummy data): {predictions[:3]}")

        except Exception as e:
            print(f"Note: {e}")
            print("This is expected in a demo - model might not have proper signature")


def example_4_framework_versions():
    """
    Example 4: Logging framework versions for reproducibility

    Demonstrates:
    - Tracking environment information
    - Ensuring reproducibility
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Framework Versions and Reproducibility")
    print("=" * 80)

    # Get framework versions
    versions = get_framework_versions()

    print("\nCurrent Environment:")
    for framework, version in versions.items():
        print(f"  {framework}: {version}")

    # Log to MLflow
    config = MLflowConfig()
    experiment_id = config.get_or_create_experiment('baseline')

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='example_reproducibility'
    ) as run:

        # Log versions as tags
        for framework, version in versions.items():
            mlflow.set_tag(f'env_{framework}', version)

        # Also log as a JSON artifact
        import json
        from pathlib import Path

        env_file = Path('environment_info.json')
        with open(env_file, 'w') as f:
            json.dump(versions, f, indent=2)

        mlflow.log_artifact(str(env_file), artifact_path='environment')
        env_file.unlink()  # Clean up

        print(f"\nLogged environment info to run: {run.info.run_id[:8]}...")


def example_5_nested_runs():
    """
    Example 5: Nested runs for hyperparameter tuning

    Demonstrates:
    - Parent-child run relationships
    - Organizing hyperparameter searches
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Nested Runs (Hyperparameter Search)")
    print("=" * 80)

    config = MLflowConfig()
    experiment_id = config.get_or_create_experiment('xgboost')

    # Parent run for the search
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name='example_hyperparameter_search'
    ) as parent_run:

        mlflow.set_tag('run_type', 'hyperparameter_search')
        mlflow.log_param('n_trials', 3)

        best_rmse = float('inf')
        best_params = None

        # Child runs for each trial
        for trial in range(3):
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f'example_trial_{trial+1}',
                nested=True
            ) as child_run:

                # Random parameters
                params = {
                    'max_depth': np.random.choice([4, 6, 8]),
                    'learning_rate': np.random.choice([0.05, 0.1, 0.15]),
                }

                mlflow.log_params(params)
                mlflow.set_tag('parent_run_id', parent_run.info.run_id)
                mlflow.set_tag('trial_number', trial + 1)

                # Simulate training
                rmse = 3.5 + np.random.normal(0, 0.2)
                mlflow.log_metric('val_rmse', rmse)

                print(f"Trial {trial+1}: max_depth={params['max_depth']}, "
                      f"lr={params['learning_rate']}, RMSE={rmse:.4f}")

                # Track best
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params

        # Log best to parent
        if best_params:
            for param, value in best_params.items():
                mlflow.log_param(f'best_{param}', value, run_id=parent_run.info.run_id)

            mlflow.log_metric('best_val_rmse', best_rmse, run_id=parent_run.info.run_id)

        print(f"\nBest parameters found: {best_params}")
        print(f"Best RMSE: {best_rmse:.4f}")
        print(f"\nParent run ID: {parent_run.info.run_id[:8]}...")


def main():
    """Run all examples"""
    print("=" * 80)
    print("NBA PRA PREDICTION - MLflow Examples")
    print("=" * 80)
    print("\nThese examples demonstrate MLflow integration for:")
    print("  1. Basic tracking")
    print("  2. Comparing runs")
    print("  3. Model registry")
    print("  4. Reproducibility")
    print("  5. Nested runs (hyperparameter tuning)")
    print("\nAfter running, view results with:")
    print("  mlflow ui --port 5000")
    print("  http://localhost:5000")

    # Run examples
    example_1_basic_tracking()
    example_2_comparing_runs()
    example_3_model_registry()
    example_4_framework_versions()
    example_5_nested_runs()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. View results: mlflow ui --port 5000")
    print("  2. Explore experiments and runs in the UI")
    print("  3. Try the real training script: python model_training/train_model.py")
    print("  4. Read the full guide: MLFLOW_GUIDE.md")


if __name__ == "__main__":
    main()
