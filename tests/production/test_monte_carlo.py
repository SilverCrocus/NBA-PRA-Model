"""
Tests for Monte Carlo probabilistic prediction utilities.
"""
import pytest
import numpy as np
from production.monte_carlo import (
    fit_gamma_parameters,
    calculate_probability_over_line,
    calculate_std_dev,
    calculate_bet_edge,
    american_odds_to_probability
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
    line = 23.5

    prob = calculate_probability_over_line(mean, std_dev, line)

    # Probability should be between 0 and 1
    assert 0 <= prob <= 1

    # When line < mean, probability should be > 0.5
    assert prob > 0.5

    # When line = mean, probability should be ~ 0.5
    prob_equal = calculate_probability_over_line(mean, std_dev, mean)
    assert abs(prob_equal - 0.5) < 0.05

    # When line > mean, probability should be < 0.5
    prob_over = calculate_probability_over_line(mean, std_dev, mean + 5)
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
