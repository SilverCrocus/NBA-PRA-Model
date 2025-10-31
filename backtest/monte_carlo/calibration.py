"""
Calibration Module for Monte Carlo Predictions

This module implements conformal prediction and calibration validation
to ensure predicted probabilities match empirical frequencies.

Key Concepts:
- Calibration: P(PRA > x) should match actual frequency of PRA > x
- PIT (Probability Integral Transform): CDF values should be uniform
- Expected Calibration Error (ECE): Average deviation from perfect calibration
- Conformal Prediction: Distribution-free coverage guarantees

Calibration is CRITICAL for betting - miscalibrated probabilities lead
to incorrect bet sizing and poor Kelly criterion performance.
"""

import numpy as np
from scipy import stats
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConformalCalibrator:
    """
    Conformal prediction calibrator for distribution-free coverage guarantees.

    Adjusts predicted distributions to ensure empirical coverage matches
    nominal confidence level. This is the gold standard for calibration.

    Usage:
        calibrator = ConformalCalibrator(alpha=0.05)  # 95% coverage
        calibrator.fit(y_val, mean_pred_val, var_pred_val)
        alpha_cal, beta_cal = calibrator.apply(alpha, beta)
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize conformal calibrator.

        Args:
            alpha: Miscoverage rate (e.g., 0.05 for 95% coverage)
                   The calibrator ensures (1-α) empirical coverage
        """
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be in (0, 1), got {alpha}")

        self.alpha = alpha
        self.coverage_target = 1 - alpha
        self.quantile = None
        self.is_fitted = False

        logger.info(f"Initialized ConformalCalibrator with {self.coverage_target:.1%} target coverage")

    def fit(
        self,
        y_true: np.ndarray,
        mean_pred: np.ndarray,
        var_pred: np.ndarray
    ) -> 'ConformalCalibrator':
        """
        Fit calibration correction on validation data.

        Calculates the quantile of absolute residuals needed to achieve
        desired coverage.

        Args:
            y_true: True PRA values, shape (n_samples,)
            mean_pred: Predicted means, shape (n_samples,)
            var_pred: Predicted variances, shape (n_samples,)

        Returns:
            self (fitted calibrator)
        """
        logger.info(f"Fitting conformal calibrator on {len(y_true)} samples")

        # Calculate non-conformity scores (absolute residuals)
        residuals = np.abs(y_true - mean_pred)

        # Find (1-α) quantile of residuals
        # This is the threshold that captures (1-α) of the data
        self.quantile = np.quantile(residuals, self.coverage_target)
        self.is_fitted = True

        logger.info(f"Conformal quantile ({self.coverage_target:.1%}): {self.quantile:.3f}")

        # Diagnostic: Check coverage on calibration set
        coverage = (residuals <= self.quantile).mean()
        logger.info(f"Empirical coverage on calibration set: {coverage:.1%}")

        # Additional diagnostics
        logger.info(f"Mean absolute residual: {residuals.mean():.3f}")
        logger.info(f"Std of residuals: {residuals.std():.3f}")

        return self

    def apply(
        self,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply conformal correction to Gamma parameters.

        Inflates variance by adding conformal quantile squared to ensure
        prediction intervals capture the desired coverage.

        Args:
            alpha: Original shape parameters, shape (n_predictions,)
            beta: Original rate parameters, shape (n_predictions,)

        Returns:
            alpha_cal: Calibrated shape parameters
            beta_cal: Calibrated rate parameters
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before applying. Call fit() first.")

        # Current mean and variance
        mean = alpha / beta
        var = alpha / (beta ** 2)

        # Inflate variance by adding conformal quantile squared
        # New variance = old variance + quantile²
        # This ensures prediction interval width captures (1-α) of data
        var_calibrated = var + (self.quantile ** 2)

        # Convert back to Gamma parameters
        # β = μ/σ², α = μ*β
        beta_cal = mean / var_calibrated
        alpha_cal = mean * beta_cal

        # Ensure valid parameters (α, β > 0)
        alpha_cal = np.maximum(alpha_cal, 0.1)
        beta_cal = np.maximum(beta_cal, 0.01)

        # Log diagnostics
        if len(alpha) > 100:
            var_increase = (var_calibrated - var).mean()
            logger.debug(f"Average variance increase: {var_increase:.3f}")

        return alpha_cal, beta_cal


def evaluate_calibration(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Evaluate calibration by comparing predicted probabilities to empirical frequencies.

    This is the primary metric for assessing whether your Monte Carlo system
    is trustworthy. A well-calibrated model has predicted probabilities that
    match actual outcomes.

    Args:
        y_true: True PRA values, shape (n_samples,)
        alpha: Shape parameters, shape (n_samples,)
        beta: Rate parameters, shape (n_samples,)
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics:
        - ece: Expected Calibration Error
        - calibration_data: Detailed bin-level data
        - ks_pvalue: Uniformity test p-value (PIT)

    Example:
        >>> metrics = evaluate_calibration(y_test, alpha, beta)
        >>> print(f"ECE: {metrics['ece']:.4f}")
        >>> # ECE < 0.05 is well-calibrated
    """
    logger.info("Evaluating calibration...")

    # Test multiple thresholds (deciles of observed distribution)
    thresholds = np.percentile(y_true, [10, 25, 50, 75, 90])

    calibration_data = []
    total_samples = len(y_true)

    for threshold in thresholds:
        # Predicted probabilities P(PRA > threshold)
        prob_over = 1 - stats.gamma.cdf(threshold, a=alpha, scale=1/beta)

        # Bin predictions into n_bins groups
        bin_edges = np.linspace(0, 1, n_bins + 1)

        for i in range(n_bins):
            bin_mask = (prob_over >= bin_edges[i]) & (prob_over < bin_edges[i+1])

            if bin_mask.sum() >= 5:  # Need at least 5 samples per bin
                predicted_prob = prob_over[bin_mask].mean()
                actual_freq = (y_true[bin_mask] > threshold).mean()
                n_samples = bin_mask.sum()

                calibration_data.append({
                    'threshold': threshold,
                    'bin': i,
                    'predicted_prob': predicted_prob,
                    'actual_freq': actual_freq,
                    'n_samples': n_samples,
                    'error': abs(predicted_prob - actual_freq)
                })

    # Calculate Expected Calibration Error (ECE)
    # ECE = Σ |predicted_prob - actual_freq| * (n_samples / total_samples)
    ece = sum(d['error'] * d['n_samples'] / total_samples for d in calibration_data)

    # PIT (Probability Integral Transform) test
    # Calculate CDF at observed values - should be uniformly distributed
    pit_values = stats.gamma.cdf(y_true, a=alpha, scale=1/beta)

    # Test uniformity using Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = stats.kstest(pit_values, 'uniform')

    # Aggregate results
    results = {
        'ece': ece,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'is_calibrated': (ece < 0.05) and (ks_pvalue > 0.05),
        'calibration_data': calibration_data,
        'pit_values': pit_values,
        'n_samples': total_samples
    }

    # Log summary
    logger.info(f"Calibration Results:")
    logger.info(f"  Expected Calibration Error (ECE): {ece:.4f} {'✓ PASS' if ece < 0.05 else '✗ FAIL'}")
    logger.info(f"  KS test p-value: {ks_pvalue:.4f} {'✓ PASS' if ks_pvalue > 0.05 else '✗ FAIL'}")
    logger.info(f"  Overall: {'✓ WELL-CALIBRATED' if results['is_calibrated'] else '✗ NEEDS CALIBRATION'}")

    return results


def calculate_coverage(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    confidence_levels: list = [0.68, 0.80, 0.90, 0.95]
) -> Dict[float, Dict[str, float]]:
    """
    Calculate empirical coverage of prediction intervals.

    For a well-calibrated model, X% prediction intervals should contain
    X% of observations.

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters
        confidence_levels: List of confidence levels to test

    Returns:
        Dictionary mapping confidence level to coverage metrics

    Example:
        >>> coverage = calculate_coverage(y_test, alpha, beta)
        >>> print(f"95% interval coverage: {coverage[0.95]['actual']:.1%}")
        >>> # Should be close to 95%
    """
    results = {}

    for conf_level in confidence_levels:
        # Calculate tail probabilities
        tail_prob = (1 - conf_level) / 2

        # Calculate prediction intervals
        lower = stats.gamma.ppf(tail_prob, a=alpha, scale=1/beta)
        upper = stats.gamma.ppf(1 - tail_prob, a=alpha, scale=1/beta)

        # Check actual coverage
        covered = (y_true >= lower) & (y_true <= upper)
        actual_coverage = covered.mean()

        # Calculate error
        error = abs(actual_coverage - conf_level)

        results[conf_level] = {
            'expected': conf_level,
            'actual': actual_coverage,
            'error': error,
            'is_valid': error < 0.03,  # Within ±3%
            'n_samples': len(y_true),
            'mean_interval_width': (upper - lower).mean()
        }

        logger.info(f"{conf_level:.0%} interval: "
                   f"Expected {conf_level:.1%}, "
                   f"Actual {actual_coverage:.1%}, "
                   f"Error {error:.1%} "
                   f"{'✓' if error < 0.03 else '✗'}")

    return results


def calculate_sharpness(
    alpha: np.ndarray,
    beta: np.ndarray,
    confidence_levels: list = [0.68, 0.80, 0.90, 0.95]
) -> Dict[float, float]:
    """
    Calculate sharpness (average width of prediction intervals).

    Sharpness measures how precise predictions are. Lower is better,
    but ONLY if the model is also calibrated. An overconfident model
    can have narrow intervals (sharp) but poor coverage (miscalibrated).

    Args:
        alpha: Shape parameters
        beta: Rate parameters
        confidence_levels: List of confidence levels

    Returns:
        Dictionary mapping confidence level to average interval width
    """
    sharpness = {}

    for conf_level in confidence_levels:
        tail_prob = (1 - conf_level) / 2

        lower = stats.gamma.ppf(tail_prob, a=alpha, scale=1/beta)
        upper = stats.gamma.ppf(1 - tail_prob, a=alpha, scale=1/beta)

        interval_width = upper - lower
        avg_width = interval_width.mean()

        sharpness[conf_level] = avg_width

        logger.debug(f"{conf_level:.0%} interval width: {avg_width:.2f} PRA")

    return sharpness


def diagnose_calibration_issues(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray
) -> Dict[str, Any]:
    """
    Diagnose calibration issues and suggest corrections.

    Analyzes the PIT histogram shape to identify:
    - Overconfident predictions (U-shaped PIT)
    - Underconfident predictions (inverted-U PIT)
    - Biased predictions (skewed PIT)

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters

    Returns:
        Dictionary with diagnosis and recommended fixes
    """
    logger.info("Diagnosing calibration issues...")

    # Calculate PIT values
    pit_values = stats.gamma.cdf(y_true, a=alpha, scale=1/beta)

    # Create histogram
    hist, bin_edges = np.histogram(pit_values, bins=10, range=(0, 1))
    hist_normalized = hist / hist.sum()

    # Check for U-shape (overconfident)
    edges_count = hist_normalized[0] + hist_normalized[-1]
    middle_count = hist_normalized[4] + hist_normalized[5]

    diagnosis = {
        'pit_mean': pit_values.mean(),
        'pit_std': pit_values.std(),
        'histogram': hist_normalized.tolist(),
        'issue': None,
        'recommendation': None
    }

    if edges_count > 1.5 * middle_count:
        diagnosis['issue'] = 'OVERCONFIDENT'
        diagnosis['recommendation'] = 'Intervals too narrow. Increase variance by 20-30%.'
        logger.warning("⚠ OVERCONFIDENT: Prediction intervals too narrow")

    elif middle_count > 1.5 * edges_count:
        diagnosis['issue'] = 'UNDERCONFIDENT'
        diagnosis['recommendation'] = 'Intervals too wide. Decrease variance by 20-30%.'
        logger.warning("⚠ UNDERCONFIDENT: Prediction intervals too wide")

    elif pit_values.mean() < 0.4:
        diagnosis['issue'] = 'NEGATIVE_BIAS'
        diagnosis['recommendation'] = 'Predictions too high. Recalibrate mean predictions downward.'
        logger.warning("⚠ NEGATIVE BIAS: Predictions consistently too high")

    elif pit_values.mean() > 0.6:
        diagnosis['issue'] = 'POSITIVE_BIAS'
        diagnosis['recommendation'] = 'Predictions too low. Recalibrate mean predictions upward.'
        logger.warning("⚠ POSITIVE BIAS: Predictions consistently too low")

    else:
        diagnosis['issue'] = None
        diagnosis['recommendation'] = 'Model appears well-calibrated. No adjustments needed.'
        logger.info("✓ Model appears well-calibrated")

    return diagnosis


def calculate_brier_score(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    thresholds: np.ndarray
) -> float:
    """
    Calculate Brier score for probabilistic predictions.

    Brier score = mean((predicted_prob - actual_outcome)²)
    Lower is better. Perfect predictions have Brier score = 0.

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters
        thresholds: Betting lines to evaluate

    Returns:
        Average Brier score across all thresholds
    """
    brier_scores = []

    for threshold in thresholds:
        # Predicted probability
        prob_over = 1 - stats.gamma.cdf(threshold, a=alpha, scale=1/beta)

        # Actual outcome (0 or 1)
        actual_over = (y_true > threshold).astype(float)

        # Brier score for this threshold
        brier = np.mean((prob_over - actual_over) ** 2)
        brier_scores.append(brier)

    avg_brier = np.mean(brier_scores)

    logger.info(f"Brier score: {avg_brier:.4f}")

    return avg_brier
