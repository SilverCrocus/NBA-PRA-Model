"""
Tests for production predictor module.
"""
import pytest
import pandas as pd
import numpy as np
from production.predictor import ProductionPredictor


def test_predictor_initialization(mock_ensemble_data):
    """Test predictor initialization with ensemble data"""
    predictor = ProductionPredictor(mock_ensemble_data)

    assert predictor.n_folds == 3
    assert len(predictor.mean_models) == 3
    assert len(predictor.feature_names) == 10


def test_predictor_ensemble_prediction(mock_ensemble_data, sample_features_df):
    """Test ensemble prediction averaging"""
    predictor = ProductionPredictor(mock_ensemble_data)

    # Get features that match the model's expected features
    # Use the first 10 features from sample_features_df
    feature_cols = [
        'pra_mean_last5', 'pra_mean_last10', 'pra_ewma_3', 'pra_ewma_7',
        'minutes_mean_last5', 'rest_days', 'games_in_last_7',
        'opp_def_rating', 'opp_pace', 'usage_rate'
    ]
    X = sample_features_df[feature_cols].copy()

    # Predict using ensemble
    mean_pred, variance_pred = predictor._predict_ensemble(X)

    # Check outputs
    assert len(mean_pred) == len(X)
    assert len(variance_pred) == len(X)
    assert all(mean_pred > 0)  # PRA should be positive
    assert all(variance_pred >= 0)  # Variance non-negative


def test_predictor_filter_players(sample_features_df):
    """Test player filtering based on minimum games"""
    from production.predictor import ProductionPredictor
    from production.config import MIN_CAREER_GAMES, MIN_RECENT_GAMES

    # Create a mock ensemble for initialization
    mock_ensemble = {
        'mean_models': [],
        'variance_models': None,
        'feature_names': [],
        'n_folds': 0
    }

    predictor = ProductionPredictor(mock_ensemble)

    # Add career_games column to test filtering
    df = sample_features_df.copy()
    df['career_games'] = np.random.randint(5, 500, len(df))
    df['games_last30'] = np.random.randint(0, 30, len(df))

    # Set some players below threshold
    df.loc[:10, 'career_games'] = MIN_CAREER_GAMES - 1
    df.loc[:5, 'games_last30'] = MIN_RECENT_GAMES - 1

    initial_count = len(df)
    filtered_df = predictor.filter_players(df)

    # Check that filtering occurred
    assert len(filtered_df) < initial_count

    # Check that all remaining players meet criteria
    if len(filtered_df) > 0:
        assert all(filtered_df['career_games'] >= MIN_CAREER_GAMES)
        assert all(filtered_df['games_last30'] >= MIN_RECENT_GAMES)


def test_predictor_probability_calculation(mock_ensemble_data):
    """Test probability calculation for betting lines"""
    predictor = ProductionPredictor(mock_ensemble_data)

    mean = 25.0
    std_dev = 4.0
    variance = std_dev ** 2
    line = 23.5

    # Calculate probability using internal method
    from production.monte_carlo import calculate_probability_over_line, fit_gamma_parameters
    alpha, beta = fit_gamma_parameters(mean, variance)
    prob_over = calculate_probability_over_line(alpha, beta, line)

    assert 0 <= prob_over <= 1
    assert prob_over > 0.5  # Line below mean, so prob should be > 0.5


def test_predictor_with_missing_features(mock_ensemble_data, sample_features_df):
    """Test predictor handles missing features gracefully"""
    predictor = ProductionPredictor(mock_ensemble_data)

    # Drop a required feature
    feature_cols = [
        'pra_mean_last5', 'pra_mean_last10', 'pra_ewma_3', 'pra_ewma_7',
        'minutes_mean_last5', 'rest_days', 'games_in_last_7',
        'opp_def_rating', 'opp_pace', 'usage_rate'
    ]
    incomplete_df = sample_features_df[feature_cols].drop(columns=['pra_mean_last5'])

    # Should raise error about missing features
    with pytest.raises((KeyError, ValueError)):
        predictor._predict_ensemble(incomplete_df)


def test_predictor_probability_over_line():
    """Test probability calculation for various scenarios"""
    from production.monte_carlo import calculate_probability_over_line, fit_gamma_parameters

    # Scenario 1: Line below mean (should have high prob)
    alpha1, beta1 = fit_gamma_parameters(30.0, 5.0 ** 2)
    prob1 = calculate_probability_over_line(alpha1, beta1, 25.0)
    assert prob1 > 0.6

    # Scenario 2: Line above mean (should have low prob)
    alpha2, beta2 = fit_gamma_parameters(30.0, 5.0 ** 2)
    prob2 = calculate_probability_over_line(alpha2, beta2, 35.0)
    assert prob2 < 0.4

    # Scenario 3: Line equal to mean (should be ~0.5)
    alpha3, beta3 = fit_gamma_parameters(30.0, 5.0 ** 2)
    prob3 = calculate_probability_over_line(alpha3, beta3, 30.0)
    assert 0.45 <= prob3 <= 0.55


def test_predictor_zero_variance():
    """Test prediction with zero/very low variance"""
    from production.monte_carlo import calculate_probability_over_line, fit_gamma_parameters

    # Zero variance case
    mean = 25.0
    std_dev = 0.01  # Very small
    variance = std_dev ** 2

    # Line below mean
    alpha, beta = fit_gamma_parameters(mean, variance)
    prob_over = calculate_probability_over_line(alpha, beta, 24.0)
    assert prob_over > 0.99  # Should be near certainty

    # Line above mean
    alpha2, beta2 = fit_gamma_parameters(mean, variance)
    prob_under = calculate_probability_over_line(alpha2, beta2, 26.0)
    assert prob_under < 0.01  # Should be near zero


def test_predictor_large_variance():
    """Test prediction with large variance"""
    from production.monte_carlo import calculate_probability_over_line, fit_gamma_parameters

    mean = 25.0
    std_dev = 10.0  # Large uncertainty
    variance = std_dev ** 2
    line = 25.0

    alpha, beta = fit_gamma_parameters(mean, variance)
    prob_over = calculate_probability_over_line(alpha, beta, line)

    # With large variance at mean, probability should still be near 0.5
    assert 0.4 <= prob_over <= 0.6


def test_predictor_generate_predictions(mock_ensemble_data, sample_features_df):
    """Test complete prediction generation workflow"""
    predictor = ProductionPredictor(mock_ensemble_data)

    # Prepare features
    feature_cols = [
        'pra_mean_last5', 'pra_mean_last10', 'pra_ewma_3', 'pra_ewma_7',
        'minutes_mean_last5', 'rest_days', 'games_in_last_7',
        'opp_def_rating', 'opp_pace', 'usage_rate'
    ]
    df = sample_features_df[feature_cols + ['player_name', 'game_date']].copy()

    # Generate predictions
    mean_pred, variance_pred = predictor._predict_ensemble(df[feature_cols])

    # Create predictions DataFrame
    predictions_df = df[['player_name', 'game_date']].copy()
    predictions_df['mean_pred'] = mean_pred
    predictions_df['variance_pred'] = variance_pred
    predictions_df['std_dev'] = np.sqrt(variance_pred)

    # Verify predictions structure
    assert len(predictions_df) == len(df)
    assert 'mean_pred' in predictions_df.columns
    assert 'std_dev' in predictions_df.columns
    assert all(predictions_df['std_dev'] >= 0)


def test_predictor_edge_cases():
    """Test edge cases for probability calculations"""
    from production.monte_carlo import calculate_probability_over_line, fit_gamma_parameters

    # Very high mean vs low line
    alpha_high, beta_high = fit_gamma_parameters(50.0, 3.0 ** 2)
    prob_high = calculate_probability_over_line(alpha_high, beta_high, 20.0)
    assert prob_high > 0.999

    # Very low mean vs high line
    alpha_low, beta_low = fit_gamma_parameters(10.0, 2.0 ** 2)
    prob_low = calculate_probability_over_line(alpha_low, beta_low, 30.0)
    assert prob_low < 0.001
