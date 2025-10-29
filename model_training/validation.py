"""
Model Validation
Comprehensive validation beyond training metrics - temporal checks, residual analysis, feature importance
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

from model_training.config import (
    MODEL_DIR,
    LOGS_DIR,
    TOP_N_FEATURES,
    RESIDUAL_PLOT_BINS,
    RESIDUAL_OUTLIER_THRESHOLD,
    EXPECTED_R2_MIN,
    EXPECTED_RMSE_MAX,
    EXPECTED_MAE_MAX,
)
from model_training.utils import (
    setup_logger,
    load_split_data,
    prepare_features_target,
    calculate_regression_metrics,
    get_feature_groups,
    format_metrics_table,
)
from model_training.models import BaseModel


def validate_model_on_test(
    model: BaseModel,
    generate_plots: bool = True,
    save_predictions: bool = True,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Comprehensive validation on held-out test set

    Args:
        model: Trained model instance
        generate_plots: Whether to create diagnostic plots
        save_predictions: Whether to save predictions to disk
        output_dir: Output directory (default: models/validation_TIMESTAMP/)

    Returns:
        Dictionary with test metrics, validation checks, and plot paths
    """
    # Setup output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = MODEL_DIR / f"validation_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logger("validation", log_file=output_dir / "validation.log")

    logger.info("="*60)
    logger.info("MODEL VALIDATION ON TEST SET")
    logger.info("="*60)
    logger.info(f"Model: {model.name}")
    logger.info(f"Output directory: {output_dir}")

    # ========================================================================
    # 1. Load Test Data
    # ========================================================================

    logger.info("\n[1/6] Loading test split")

    test_df = load_split_data("test")
    X_test, y_test = prepare_features_target(test_df)

    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Features: {len(X_test.columns)}")

    # ========================================================================
    # 2. Generate Predictions
    # ========================================================================

    logger.info("\n[2/6] Generating predictions")

    y_pred = model.predict(X_test)

    # ========================================================================
    # 3. Calculate Test Metrics
    # ========================================================================

    logger.info("\n[3/6] Calculating test metrics")

    test_metrics = calculate_regression_metrics(y_test, y_pred, prefix="test_")

    logger.info("\n" + format_metrics_table(test_metrics, title="Test Set Metrics"))

    # ========================================================================
    # 4. Validation Checks
    # ========================================================================

    logger.info("\n[4/6] Running validation checks")

    validation_checks = {}

    # Check 1: Performance thresholds
    threshold_checks = check_performance_thresholds(test_metrics)
    validation_checks['thresholds'] = threshold_checks

    for check_name, passed in threshold_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {check_name}")

    # Check 2: Temporal consistency
    temporal_checks = check_temporal_consistency(
        y_true=y_test,
        y_pred=y_pred,
        dates=test_df['game_date']
    )
    validation_checks['temporal'] = temporal_checks

    for check_name, passed in temporal_checks.items():
        status = "✓ PASS" if passed else "⚠ WARNING"
        logger.info(f"  {status}: {check_name}")

    # ========================================================================
    # 5. Residual Analysis
    # ========================================================================

    logger.info("\n[5/6] Analyzing residuals")

    residual_stats = analyze_residuals(
        y_true=y_test.values,
        y_pred=y_pred,
        output_dir=output_dir if generate_plots else None
    )

    validation_checks['residuals'] = residual_stats

    logger.info(f"  Mean: {residual_stats['mean']:.4f}")
    logger.info(f"  Std: {residual_stats['std']:.4f}")
    logger.info(f"  Outliers (>{RESIDUAL_OUTLIER_THRESHOLD}σ): {residual_stats['n_outliers']} ({residual_stats['pct_outliers']:.2f}%)")

    # ========================================================================
    # 6. Feature Importance Analysis
    # ========================================================================

    logger.info("\n[6/6] Analyzing feature importance")

    feature_importance = analyze_feature_importance(
        model=model,
        top_n=TOP_N_FEATURES,
        output_dir=output_dir if generate_plots else None
    )

    logger.info(f"  Top 10 features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"    {row['feature']}: {row['importance_gain']:.2f}")

    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"  ✓ Saved feature importance: {importance_path}")

    # ========================================================================
    # Save Predictions
    # ========================================================================

    if save_predictions:
        predictions_df = test_df[['player_id', 'game_id', 'game_date', 'player_name']].copy()
        predictions_df['actual_pra'] = y_test.values
        predictions_df['predicted_pra'] = y_pred
        predictions_df['residual'] = y_test.values - y_pred
        predictions_df['abs_error'] = np.abs(predictions_df['residual'])

        predictions_path = output_dir / "test_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"\n✓ Saved predictions: {predictions_path}")

    # ========================================================================
    # Generate Plots
    # ========================================================================

    if generate_plots:
        logger.info("\nGenerating diagnostic plots")

        plot_paths = create_diagnostic_plots(
            y_true=y_test.values,
            y_pred=y_pred,
            output_dir=output_dir
        )

        for plot_name, plot_path in plot_paths.items():
            logger.info(f"  ✓ {plot_name}: {plot_path.name}")

    # ========================================================================
    # Summary
    # ========================================================================

    logger.info("\n" + "="*60)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*60)

    all_checks_passed = all(threshold_checks.values())
    if all_checks_passed:
        logger.info("✓ All performance thresholds met!")
    else:
        failed_checks = [k for k, v in threshold_checks.items() if not v]
        logger.warning(f"⚠ Failed checks: {failed_checks}")

    logger.info("="*60)

    return {
        'metrics': test_metrics,
        'validation_checks': validation_checks,
        'feature_importance': feature_importance,
        'output_dir': output_dir
    }


def check_performance_thresholds(metrics: Dict[str, float]) -> Dict[str, bool]:
    """
    Validate model meets expected performance thresholds

    Args:
        metrics: Dictionary of test metrics

    Returns:
        Dictionary with pass/fail for each threshold
    """
    checks = {}

    # R² threshold
    test_r2 = metrics.get('test_r2', 0)
    checks['R² >= minimum'] = test_r2 >= EXPECTED_R2_MIN

    # RMSE threshold
    test_rmse = metrics.get('test_rmse', float('inf'))
    checks['RMSE <= maximum'] = test_rmse <= EXPECTED_RMSE_MAX

    # MAE threshold
    test_mae = metrics.get('test_mae', float('inf'))
    checks['MAE <= maximum'] = test_mae <= EXPECTED_MAE_MAX

    # MAE/RMSE ratio (should be < 1, ideally ~0.7-0.8)
    mae_rmse_ratio = test_mae / test_rmse if test_rmse > 0 else float('inf')
    checks['MAE/RMSE ratio reasonable'] = 0.5 <= mae_rmse_ratio <= 0.9

    return checks


def check_temporal_consistency(
    y_true: pd.Series,
    y_pred: np.ndarray,
    dates: pd.Series
) -> Dict[str, bool]:
    """
    Validate model doesn't rely on future information
    Checks for temporal patterns that would indicate data leakage

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Game dates

    Returns:
        Dictionary with temporal consistency checks
    """
    checks = {}

    # Create DataFrame for analysis
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'actual': y_true.values if isinstance(y_true, pd.Series) else y_true,
        'predicted': y_pred,
        'residual': y_true.values - y_pred if isinstance(y_true, pd.Series) else y_true - y_pred
    })

    df = df.sort_values('date')

    # Split into early and late test periods
    midpoint = df['date'].quantile(0.5)
    early = df[df['date'] < midpoint]
    late = df[df['date'] >= midpoint]

    # Calculate RMSE for each period
    rmse_early = np.sqrt(np.mean(early['residual']**2))
    rmse_late = np.sqrt(np.mean(late['residual']**2))

    # Performance should be similar (within 20%) - if late is much better, potential leakage
    rmse_ratio = rmse_late / rmse_early if rmse_early > 0 else 1.0
    checks['Performance stable over time'] = 0.8 <= rmse_ratio <= 1.2

    # Residuals should have consistent variance over time (homoscedasticity)
    var_early = np.var(early['residual'])
    var_late = np.var(late['residual'])
    var_ratio = var_late / var_early if var_early > 0 else 1.0
    checks['Residual variance consistent'] = 0.7 <= var_ratio <= 1.4

    # Mean residual should be close to 0 in both periods (no systematic bias)
    mean_resid_early = early['residual'].mean()
    mean_resid_late = late['residual'].mean()
    checks['No systematic bias early'] = abs(mean_resid_early) < 1.0
    checks['No systematic bias late'] = abs(mean_resid_late) < 1.0

    return checks


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Detailed residual analysis

    Args:
        y_true: Actual values
        y_pred: Predicted values
        output_dir: Optional directory to save plots

    Returns:
        Dictionary with residual statistics
    """
    residuals = y_true - y_pred

    # Calculate statistics
    stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'median': float(np.median(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'q25': float(np.percentile(residuals, 25)),
        'q75': float(np.percentile(residuals, 75)),
    }

    # Identify outliers
    standardized_residuals = (residuals - stats['mean']) / stats['std']
    outliers = np.abs(standardized_residuals) > RESIDUAL_OUTLIER_THRESHOLD
    stats['n_outliers'] = int(np.sum(outliers))
    stats['pct_outliers'] = float(100 * np.mean(outliers))

    return stats


def analyze_feature_importance(
    model: BaseModel,
    top_n: int = TOP_N_FEATURES,
    output_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Analyze and optionally visualize feature importance

    Args:
        model: Trained model
        top_n: Number of top features to plot
        output_dir: Optional directory to save plot

    Returns:
        DataFrame with feature importance rankings
    """
    feature_importance = model.get_feature_importance()

    # Add feature groups
    feature_groups = get_feature_groups(feature_importance['feature'].tolist())

    def get_group(feature):
        for group, features in feature_groups.items():
            if feature in features:
                return group
        return 'other'

    feature_importance['group'] = feature_importance['feature'].apply(get_group)

    # Save plot if output directory specified
    if output_dir:
        plot_feature_importance(feature_importance, top_n, output_dir)

    return feature_importance


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int,
    output_dir: Path
) -> Path:
    """
    Create feature importance bar plot

    Args:
        feature_importance: DataFrame with feature importance
        top_n: Number of top features to plot
        output_dir: Output directory

    Returns:
        Path to saved plot
    """
    plt.figure(figsize=(10, 8))

    top_features = feature_importance.head(top_n)

    # Create color map by group
    unique_groups = top_features['group'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_groups)))
    color_map = dict(zip(unique_groups, colors))
    bar_colors = [color_map[g] for g in top_features['group']]

    plt.barh(range(len(top_features)), top_features['importance_gain'], color=bar_colors)
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
    plt.xlabel('Importance (Gain)', fontsize=10)
    plt.title(f'Top {top_n} Feature Importances', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()  # Most important at top

    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=color_map[g]) for g in unique_groups]
    plt.legend(handles, unique_groups, loc='lower right', fontsize=8)

    plt.tight_layout()

    plot_path = output_dir / "feature_importance_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def create_diagnostic_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create comprehensive diagnostic plots

    Args:
        y_true: Actual values
        y_pred: Predicted values
        output_dir: Output directory

    Returns:
        Dictionary mapping plot_name -> plot_path
    """
    residuals = y_true - y_pred

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Diagnostic Plots', fontsize=16, fontweight='bold')

    # 1. Predicted vs Actual
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual PRA')
    axes[0, 0].set_ylabel('Predicted PRA')
    axes[0, 0].set_title('Predicted vs Actual')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Residuals vs Predicted
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted PRA')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residual Distribution
    axes[0, 2].hist(residuals, bins=RESIDUAL_PLOT_BINS, edgecolor='black', alpha=0.7)
    axes[0, 2].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Residual Distribution')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Absolute Residuals vs Predicted
    axes[1, 1].scatter(y_pred, np.abs(residuals), alpha=0.5, s=10)
    axes[1, 1].set_xlabel('Predicted PRA')
    axes[1, 1].set_ylabel('Absolute Residuals')
    axes[1, 1].set_title('Absolute Residuals vs Predicted')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Error Distribution by Range
    error_by_range = pd.DataFrame({
        'predicted': y_pred,
        'abs_error': np.abs(residuals)
    })
    error_by_range['pra_range'] = pd.cut(error_by_range['predicted'], bins=5)
    range_errors = error_by_range.groupby('pra_range')['abs_error'].mean()

    axes[1, 2].bar(range(len(range_errors)), range_errors.values, alpha=0.7)
    axes[1, 2].set_xticks(range(len(range_errors)))
    axes[1, 2].set_xticklabels([str(x) for x in range_errors.index], rotation=45, ha='right')
    axes[1, 2].set_xlabel('Predicted PRA Range')
    axes[1, 2].set_ylabel('Mean Absolute Error')
    axes[1, 2].set_title('Error by Prediction Range')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = output_dir / "diagnostic_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return {'diagnostic_plots': plot_path}


def main():
    """
    Main entry point for validation
    """
    import argparse

    parser = argparse.ArgumentParser(description="Validate trained NBA PRA model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.pkl file)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation"
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Skip saving predictions"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: models/validation_TIMESTAMP/)"
    )

    args = parser.parse_args()

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return

    print(f"Loading model from {model_path}")

    # Determine model type from path
    if "xgboost" in str(model_path).lower():
        from model_training.models import XGBoostModel
        model = XGBoostModel.load(model_path)
    elif "lightgbm" in str(model_path).lower():
        from model_training.models import LightGBMModel
        model = LightGBMModel.load(model_path)
    else:
        print("Error: Could not determine model type from path")
        return

    # Run validation
    output_dir = Path(args.output_dir) if args.output_dir else None

    results = validate_model_on_test(
        model=model,
        generate_plots=not args.no_plots,
        save_predictions=not args.no_save_predictions,
        output_dir=output_dir
    )

    print(f"\nValidation complete! Results saved to: {results['output_dir']}")


if __name__ == "__main__":
    main()
