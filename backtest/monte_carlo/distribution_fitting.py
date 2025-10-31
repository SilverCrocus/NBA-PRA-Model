"""
Distribution Fitting for Monte Carlo Simulation

This module handles fitting Gamma distributions to PRA predictions and
calculating probabilities for betting decisions.

Key Functions:
- fit_gamma_parameters: Convert (mean, variance) to Gamma(α, β)
- calculate_probability_over_line: P(PRA > betting_line) analytically
- get_prediction_interval: Confidence intervals for predictions
- sample_from_gamma: Generate Monte Carlo samples (when needed)

Gamma Distribution Properties:
- Non-negative support (PRA ≥ 0)
- Right-skewed (matches empirical PRA distributions)
- Two parameters: shape (α) and rate (β)
- E[X] = α/β, Var[X] = α/β²
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def fit_gamma_parameters(
    mean: np.ndarray,
    variance: np.ndarray,
    min_alpha: float = 0.1,
    min_beta: float = 0.01
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert mean and variance to Gamma distribution parameters.

    Uses method of moments:
        E[X] = α/β = μ
        Var[X] = α/β² = σ²

    Solving for α and β:
        β = μ/σ²
        α = μ²/σ²

    Args:
        mean: Predicted means (μ), shape (n_predictions,)
        variance: Predicted variances (σ²), shape (n_predictions,)
        min_alpha: Minimum shape parameter (prevents invalid distributions)
        min_beta: Minimum rate parameter (prevents invalid distributions)

    Returns:
        alpha: Shape parameters, shape (n_predictions,)
        beta: Rate parameters, shape (n_predictions,)

    Example:
        >>> mean = np.array([25.0, 18.5, 30.2])
        >>> variance = np.array([16.0, 9.0, 20.0])
        >>> alpha, beta = fit_gamma_parameters(mean, variance)
        >>> # alpha/beta should equal mean
        >>> np.allclose(alpha/beta, mean)
        True
    """
    # Ensure variance is positive and non-zero
    variance = np.maximum(variance, 0.01)

    # Calculate Gamma parameters via method of moments
    beta = mean / variance
    alpha = mean * beta

    # Ensure valid parameters (α, β > 0)
    alpha = np.maximum(alpha, min_alpha)
    beta = np.maximum(beta, min_beta)

    # Log diagnostics for large prediction sets
    if len(alpha) > 100:
        logger.debug(f"Gamma parameters - Alpha: {alpha.mean():.2f} ± {alpha.std():.2f}")
        logger.debug(f"Gamma parameters - Beta: {beta.mean():.2f} ± {beta.std():.2f}")

        # Check for extreme parameters (may indicate fitting issues)
        extreme_alpha = ((alpha > 1000) | (alpha < 0.01)).sum()
        extreme_beta = ((beta > 1000) | (beta < 0.01)).sum()

        if extreme_alpha > 0:
            logger.warning(f"{extreme_alpha} extreme alpha parameters detected")
        if extreme_beta > 0:
            logger.warning(f"{extreme_beta} extreme beta parameters detected")

    return alpha, beta


def calculate_probability_over_line(
    alpha: np.ndarray,
    beta: np.ndarray,
    betting_line: np.ndarray,
    method: str = 'analytical'
) -> np.ndarray:
    """
    Calculate P(PRA > betting_line) using Gamma distribution.

    This is the core function for betting decisions. The probability
    represents the model's confidence that the player will exceed the
    betting line.

    Args:
        alpha: Shape parameters, shape (n_predictions,)
        beta: Rate parameters, shape (n_predictions,)
        betting_line: Betting thresholds, shape (n_predictions,) or scalar
        method: 'analytical' (fast, exact) or 'monte_carlo' (slower, approximate)

    Returns:
        probabilities: P(PRA > line), shape (n_predictions,)

    Example:
        >>> alpha = np.array([39.06, 28.86])  # Gamma parameters
        >>> beta = np.array([1.56, 1.56])
        >>> line = np.array([25.5, 18.5])
        >>> prob = calculate_probability_over_line(alpha, beta, line)
        >>> # prob[0] is probability first player exceeds 25.5 PRA
    """
    # Ensure betting_line is an array
    if np.isscalar(betting_line):
        betting_line = np.full_like(alpha, betting_line)

    if method == 'analytical':
        # Analytical solution using Gamma CDF (FAST)
        # P(X > line) = 1 - P(X ≤ line) = 1 - CDF(line)
        # scipy.stats.gamma uses scale = 1/beta
        cdf_values = stats.gamma.cdf(betting_line, a=alpha, scale=1/beta)
        probabilities = 1 - cdf_values

    elif method == 'monte_carlo':
        # Monte Carlo approximation (SLOWER, use only for debugging)
        n_samples = 10000
        probabilities = np.zeros(len(alpha))

        for i in range(len(alpha)):
            samples = stats.gamma.rvs(a=alpha[i], scale=1/beta[i], size=n_samples)
            probabilities[i] = (samples > betting_line[i]).mean()

        logger.debug("Using Monte Carlo method for probability calculation")

    else:
        raise ValueError(f"Unknown method: {method}. Use 'analytical' or 'monte_carlo'")

    # Ensure probabilities are in [0, 1]
    probabilities = np.clip(probabilities, 0.0, 1.0)

    return probabilities


def get_prediction_interval(
    alpha: np.ndarray,
    beta: np.ndarray,
    confidence: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction interval [lower, upper] for given confidence level.

    Args:
        alpha: Shape parameters, shape (n_predictions,)
        beta: Rate parameters, shape (n_predictions,)
        confidence: Confidence level (e.g., 0.95 for 95% interval)

    Returns:
        lower: Lower bounds, shape (n_predictions,)
        upper: Upper bounds, shape (n_predictions,)

    Example:
        >>> alpha, beta = fit_gamma_parameters(mean, variance)
        >>> lower, upper = get_prediction_interval(alpha, beta, confidence=0.95)
        >>> # 95% of outcomes should fall in [lower, upper]
    """
    if not 0 < confidence < 1:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    # Calculate tail probabilities
    tail_prob = (1 - confidence) / 2

    # Calculate quantiles using inverse CDF (PPF)
    lower = stats.gamma.ppf(tail_prob, a=alpha, scale=1/beta)
    upper = stats.gamma.ppf(1 - tail_prob, a=alpha, scale=1/beta)

    return lower, upper


def sample_from_gamma(
    alpha: np.ndarray,
    beta: np.ndarray,
    n_samples: int = 10000,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate Monte Carlo samples from Gamma distribution.

    This function is useful when you need the full distribution of outcomes,
    not just probabilities. For example:
    - Visualizing player performance distributions
    - Calculating complex statistics (e.g., P(player1 > player2))
    - Simulating entire games with correlated players

    Args:
        alpha: Shape parameters, shape (n_predictions,)
        beta: Rate parameters, shape (n_predictions,)
        n_samples: Number of samples per prediction
        random_state: Random seed for reproducibility

    Returns:
        samples: Array of shape (n_predictions, n_samples)

    Example:
        >>> alpha = np.array([39.06, 28.86])
        >>> beta = np.array([1.56, 1.56])
        >>> samples = sample_from_gamma(alpha, beta, n_samples=10000)
        >>> samples.shape
        (2, 10000)
        >>> # Mean of samples should approximate alpha/beta
        >>> np.allclose(samples.mean(axis=1), alpha/beta, atol=0.5)
        True
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_predictions = len(alpha)
    samples = np.zeros((n_predictions, n_samples))

    for i in range(n_predictions):
        # scipy.stats.gamma uses scale = 1/beta
        samples[i] = stats.gamma.rvs(
            a=alpha[i],
            scale=1/beta[i],
            size=n_samples,
            random_state=random_state
        )

    logger.debug(f"Generated {n_predictions} x {n_samples} Monte Carlo samples")

    return samples


def calculate_percentiles(
    alpha: np.ndarray,
    beta: np.ndarray,
    percentiles: list = [5, 25, 50, 75, 95]
) -> dict:
    """
    Calculate percentiles of Gamma distribution.

    Args:
        alpha: Shape parameters
        beta: Rate parameters
        percentiles: List of percentiles to calculate (0-100)

    Returns:
        Dictionary mapping percentile to array of values

    Example:
        >>> percentile_dict = calculate_percentiles(alpha, beta, [10, 50, 90])
        >>> # percentile_dict[50] is the median (50th percentile)
    """
    percentile_dict = {}

    for p in percentiles:
        if not 0 < p < 100:
            raise ValueError(f"Percentile must be in (0, 100), got {p}")

        # Convert percentile to probability
        prob = p / 100

        # Calculate quantile
        values = stats.gamma.ppf(prob, a=alpha, scale=1/beta)
        percentile_dict[p] = values

    return percentile_dict


def calculate_expected_value(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Calculate expected value of Gamma distribution.

    For Gamma(α, β): E[X] = α/β

    Args:
        alpha: Shape parameters
        beta: Rate parameters

    Returns:
        Expected values
    """
    return alpha / beta


def calculate_std_dev(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Calculate standard deviation of Gamma distribution.

    For Gamma(α, β): Std[X] = √(α/β²)

    Args:
        alpha: Shape parameters
        beta: Rate parameters

    Returns:
        Standard deviations
    """
    variance = alpha / (beta ** 2)
    return np.sqrt(variance)


def validate_distribution_fit(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray
) -> dict:
    """
    Validate that Gamma distribution fits the observed data.

    Uses Kolmogorov-Smirnov test to assess goodness-of-fit.

    Args:
        y_true: Observed PRA values
        alpha: Fitted shape parameters
        beta: Fitted rate parameters

    Returns:
        Dictionary with validation metrics
    """
    results = []

    for i, (a, b, y) in enumerate(zip(alpha, beta, y_true)):
        # Create Gamma distribution
        dist = stats.gamma(a=a, scale=1/b)

        # Calculate CDF at observed value
        cdf_value = dist.cdf(y)

        # Calculate z-score (how many std devs from mean)
        mean = a / b
        std = np.sqrt(a / (b ** 2))
        z_score = (y - mean) / std

        results.append({
            'observed': y,
            'expected': mean,
            'std': std,
            'z_score': z_score,
            'cdf_value': cdf_value
        })

    # Aggregate statistics
    z_scores = np.array([r['z_score'] for r in results])
    cdf_values = np.array([r['cdf_value'] for r in results])

    # Test if CDF values are uniformly distributed (should be for well-fitted model)
    from scipy.stats import kstest
    ks_stat, ks_pvalue = kstest(cdf_values, 'uniform')

    validation = {
        'mean_abs_z_score': np.abs(z_scores).mean(),
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'is_well_fitted': ks_pvalue > 0.05,  # p > 0.05 indicates good fit
        'details': results
    }

    logger.info(f"Distribution fit validation:")
    logger.info(f"  Mean |z-score|: {validation['mean_abs_z_score']:.2f}")
    logger.info(f"  KS p-value: {ks_pvalue:.4f} ({'PASS' if ks_pvalue > 0.05 else 'FAIL'})")

    return validation
