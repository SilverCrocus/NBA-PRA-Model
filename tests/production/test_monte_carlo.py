"""
Tests for Monte Carlo probabilistic prediction utilities.
"""
import pytest
import numpy as np
from production.monte_carlo import (
    fit_gamma_parameters,
    calculate_probability_over_line,
    calculate_bet_edge,
    american_odds_to_probability,
    calculate_kelly_fraction
)


def test_fit_gamma_parameters_valid_inputs():
    """Test Gamma parameter fitting with valid mean and variance"""
    mean = 25.0
    variance = 16.0

    alpha, beta = fit_gamma_parameters(mean, variance)

    # Check parameters are positive
    assert alpha > 0
    assert beta > 0

    # Check fitted distribution matches input moments
    fitted_mean = alpha / beta
    fitted_variance = alpha / (beta ** 2)

    assert abs(fitted_mean - mean) < 0.01
    assert abs(fitted_variance - variance) < 0.01


def test_fit_gamma_parameters_edge_cases():
    """Test Gamma fitting with edge cases"""
    # Very small variance
    alpha, beta = fit_gamma_parameters(20.0, 0.1)
    assert alpha > 0 and beta > 0

    # Very large variance
    alpha, beta = fit_gamma_parameters(30.0, 100.0)
    assert alpha > 0 and beta > 0

    # Zero variance should raise or handle gracefully
    with pytest.raises((ValueError, ZeroDivisionError)):
        fit_gamma_parameters(25.0, 0.0)


def test_calculate_probability_over_line():
    """Test probability calculation for betting lines"""
    mean = 25.0
    std_dev = 4.0
    variance = std_dev ** 2
    line = 23.5

    # Convert to Gamma parameters
    alpha, beta = fit_gamma_parameters(mean, variance)
    prob = calculate_probability_over_line(alpha, beta, line)

    # Probability should be between 0 and 1
    assert 0 <= prob <= 1

    # When line < mean, probability should be > 0.5
    assert prob > 0.5

    # When line = mean, probability should be ~ 0.5
    alpha_eq, beta_eq = fit_gamma_parameters(mean, variance)
    prob_equal = calculate_probability_over_line(alpha_eq, beta_eq, mean)
    assert abs(prob_equal - 0.5) < 0.05

    # When line > mean, probability should be < 0.5
    alpha_over, beta_over = fit_gamma_parameters(mean, variance)
    prob_over = calculate_probability_over_line(alpha_over, beta_over, mean + 5)
    assert prob_over < 0.5


def test_american_odds_to_probability():
    """Test American odds conversion to breakeven probability"""
    # -110 odds (standard line)
    prob = american_odds_to_probability(-110)
    assert abs(prob - 0.5238) < 0.01  # 52.38% breakeven

    # +100 odds (even money)
    prob = american_odds_to_probability(100)
    assert abs(prob - 0.5) < 0.01

    # -200 odds (heavy favorite)
    prob = american_odds_to_probability(-200)
    assert prob > 0.65

    # +200 odds (underdog)
    prob = american_odds_to_probability(200)
    assert prob < 0.35


def test_calculate_bet_edge():
    """Test edge calculation"""
    prob_win = 0.65
    odds = -110

    edge = calculate_bet_edge(prob_win, odds)

    # Edge should be positive when prob_win > breakeven
    assert edge > 0

    # Edge should be ~ 0.13 (65% - 52.38%)
    assert abs(edge - 0.1262) < 0.01

    # Negative edge when prob_win < breakeven
    edge_negative = calculate_bet_edge(0.45, -110)
    assert edge_negative < 0


def test_calculate_kelly_fraction():
    """Test Kelly criterion bet sizing"""
    # Standard bet with positive edge
    prob_win = 0.65
    odds = -110
    kelly_frac = 0.25  # Quarter Kelly

    bet_size = calculate_kelly_fraction(prob_win, odds, kelly_frac)

    # Should return positive bet size
    assert bet_size > 0

    # Should be reasonable fraction of bankroll (< 10%)
    assert bet_size < 0.1

    # Should be approximately 6.63% for these inputs (quarter Kelly)
    assert abs(bet_size - 0.0663) < 0.005

    # Test with negative edge (should return 0)
    bet_size_negative = calculate_kelly_fraction(0.45, -110, kelly_frac)
    assert bet_size_negative == 0.0

    # Test with positive odds (underdog)
    bet_size_underdog = calculate_kelly_fraction(0.55, 200, kelly_frac)
    assert bet_size_underdog > 0

    # Test with full Kelly (higher bet size)
    bet_size_full = calculate_kelly_fraction(prob_win, odds, kelly_fraction=1.0)
    assert bet_size_full > bet_size
    assert bet_size_full == pytest.approx(bet_size * 4, rel=0.01)
