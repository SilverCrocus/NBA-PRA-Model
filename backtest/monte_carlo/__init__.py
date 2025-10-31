"""
Monte Carlo Simulation Module for NBA PRA Predictions

This module provides probabilistic forecasting capabilities for NBA player
performance predictions, enabling confidence-based betting decisions and
Kelly criterion bet sizing.

Core Components:
- variance_model: Train player-specific variance models
- distribution_fitting: Fit Gamma distributions to predictions
- calibration: Conformal prediction calibration
- betting_calculator: Calculate P(PRA > line) for betting
- visualization: Calibration diagnostics and plots

Usage:
    from backtest.monte_carlo import VarianceModel, fit_gamma_parameters

    # Train variance model
    variance_model = VarianceModel()
    variance_model.fit(X, y, mean_predictions)

    # Generate distributions
    mean_pred = mean_model.predict(X_new)
    var_pred = variance_model.predict(X_new)
    alpha, beta = fit_gamma_parameters(mean_pred, var_pred)

    # Calculate betting probabilities
    prob_over = calculate_probability_over_line(alpha, beta, betting_line)
"""

from backtest.monte_carlo.variance_model import VarianceModel
from backtest.monte_carlo.distribution_fitting import (
    fit_gamma_parameters,
    calculate_probability_over_line,
    get_prediction_interval,
    sample_from_gamma
)
from backtest.monte_carlo.calibration import (
    ConformalCalibrator,
    evaluate_calibration
)
from backtest.monte_carlo.betting_calculator import (
    calculate_kelly_size,
    filter_by_confidence,
    calculate_bet_decisions
)

__all__ = [
    'VarianceModel',
    'fit_gamma_parameters',
    'calculate_probability_over_line',
    'get_prediction_interval',
    'sample_from_gamma',
    'ConformalCalibrator',
    'evaluate_calibration',
    'calculate_kelly_size',
    'filter_by_confidence',
    'calculate_bet_decisions',
]

__version__ = '1.0.0'
