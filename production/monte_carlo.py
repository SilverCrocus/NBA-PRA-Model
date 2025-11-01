"""
Monte Carlo Probabilistic Prediction Utilities

Self-contained module for Gamma distribution fitting and probability calculations.
Extracted from backtest/monte_carlo/ to remove cross-module dependencies.

Author: NBA PRA Prediction System
Date: 2025-11-01
"""
import numpy as np
from scipy import stats
from typing import Tuple, Union


def fit_gamma_parameters(mean: float, variance: float) -> Tuple[float, float]:
    """
    Fit Gamma distribution parameters (alpha, beta) from mean and variance.

    Gamma distribution:
    - PDF: f(x; α, β) = (β^α / Γ(α)) * x^(α-1) * e^(-βx)
    - Mean: E[X] = α / β
    - Variance: Var[X] = α / β²

    Solving for parameters:
    - β = mean / variance
    - α = mean * β

    Args:
        mean: Expected value (must be positive)
        variance: Variance (must be positive)

    Returns:
        Tuple of (alpha, beta) parameters

    Raises:
        ValueError: If mean or variance <= 0
    """
    if mean <= 0:
        raise ValueError(f"Mean must be positive, got {mean}")
    if variance <= 0:
        raise ValueError(f"Variance must be positive, got {variance}")

    beta = mean / variance
    alpha = mean * beta

    return alpha, beta


def calculate_probability_over_line(
    mean: float,
    std_dev: float,
    line: float,
    method: str = 'analytical'
) -> float:
    """
    Calculate P(PRA > line) using Gamma distribution.

    Args:
        mean: Mean prediction
        std_dev: Standard deviation
        line: Betting line
        method: 'analytical' (fast) or 'monte_carlo' (slow but accurate)

    Returns:
        Probability that actual PRA exceeds line (0-1)
    """
    if std_dev <= 0:
        # No uncertainty: deterministic prediction
        return 1.0 if mean > line else 0.0

    variance = std_dev ** 2

    try:
        alpha, beta = fit_gamma_parameters(mean, variance)
    except ValueError:
        # Fallback to normal distribution if Gamma fails
        return 1 - stats.norm.cdf(line, loc=mean, scale=std_dev)

    if method == 'analytical':
        # Use survival function: P(X > line) = 1 - CDF(line)
        prob_over = 1 - stats.gamma.cdf(line, a=alpha, scale=1/beta)

    elif method == 'monte_carlo':
        # Monte Carlo simulation (slower but more flexible)
        n_samples = 10000
        samples = np.random.gamma(alpha, scale=1/beta, size=n_samples)
        prob_over = np.mean(samples > line)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure probability is in valid range
    return np.clip(prob_over, 0.0, 1.0)


def american_odds_to_probability(odds: int) -> float:
    """
    Convert American odds to breakeven probability.

    American odds:
    - Negative (e.g., -110): Risk |odds| to win 100
    - Positive (e.g., +150): Risk 100 to win odds

    Breakeven probability:
    - Negative: |odds| / (|odds| + 100)
    - Positive: 100 / (odds + 100)

    Args:
        odds: American odds (e.g., -110, +150)

    Returns:
        Breakeven probability (0-1)

    Examples:
        >>> american_odds_to_probability(-110)
        0.5238  # Need 52.38% to break even

        >>> american_odds_to_probability(100)
        0.5000  # Even money

        >>> american_odds_to_probability(-200)
        0.6667  # Heavy favorite
    """
    if odds == 0:
        raise ValueError("Odds cannot be zero")

    if odds < 0:
        # Negative odds: favorite
        breakeven = abs(odds) / (abs(odds) + 100)
    else:
        # Positive odds: underdog
        breakeven = 100 / (odds + 100)

    return breakeven


def calculate_bet_edge(prob_win: float, odds: int) -> float:
    """
    Calculate edge over breakeven probability.

    Edge = P(win) - P(breakeven)

    Positive edge = profitable bet
    Negative edge = unprofitable bet

    Args:
        prob_win: Probability of winning (0-1)
        odds: American odds

    Returns:
        Edge over breakeven (-1 to 1)

    Examples:
        >>> calculate_bet_edge(0.65, -110)
        0.1262  # 12.62% edge (65% - 52.38%)

        >>> calculate_bet_edge(0.45, -110)
        -0.0738  # -7.38% edge (unprofitable)
    """
    breakeven = american_odds_to_probability(odds)
    edge = prob_win - breakeven

    return edge


def calculate_kelly_fraction(
    prob_win: float,
    odds: int,
    kelly_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly criterion bet size.

    Kelly formula:
    - f* = (p * (b + 1) - 1) / b

    Where:
    - p = probability of winning
    - b = decimal odds - 1

    We use fractional Kelly (e.g., 0.25 = quarter Kelly) for risk management.

    Args:
        prob_win: Probability of winning (0-1)
        odds: American odds
        kelly_fraction: Fraction of Kelly to bet (default 0.25)

    Returns:
        Recommended bet size as fraction of bankroll (0-1)

    Examples:
        >>> calculate_kelly_fraction(0.65, -110, kelly_fraction=0.25)
        0.0315  # Bet 3.15% of bankroll
    """
    # Convert American odds to decimal
    if odds < 0:
        decimal_odds = 1 + (100 / abs(odds))
    else:
        decimal_odds = 1 + (odds / 100)

    b = decimal_odds - 1

    # Kelly formula
    kelly = (prob_win * (b + 1) - 1) / b

    # Apply fractional Kelly
    fractional_kelly = kelly * kelly_fraction

    # Ensure non-negative
    return max(0.0, fractional_kelly)
