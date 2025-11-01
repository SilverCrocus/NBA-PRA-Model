"""
Tests to validate fixtures are working correctly.
"""
import pytest
import pandas as pd


def test_temp_production_dir(temp_production_dir):
    """Test temp production directory structure is created"""
    assert temp_production_dir.exists()
    assert (temp_production_dir / "models").exists()
    assert (temp_production_dir / "outputs" / "predictions").exists()
    assert (temp_production_dir / "outputs" / "bets").exists()
    assert (temp_production_dir / "outputs" / "ledger").exists()


def test_sample_features_df(sample_features_df):
    """Test sample features DataFrame is created correctly"""
    assert isinstance(sample_features_df, pd.DataFrame)
    assert len(sample_features_df) == 100
    assert 'player_id' in sample_features_df.columns
    assert 'player_name' in sample_features_df.columns
    assert 'pra_mean_last5' in sample_features_df.columns
    assert 'pra_ewma_3' in sample_features_df.columns
    assert sample_features_df['player_id'].nunique() == 10


def test_sample_predictions_df(sample_predictions_df):
    """Test sample predictions DataFrame is created correctly"""
    assert isinstance(sample_predictions_df, pd.DataFrame)
    assert len(sample_predictions_df) == 20
    assert 'player_name' in sample_predictions_df.columns
    assert 'mean_pred' in sample_predictions_df.columns
    assert 'prob_over' in sample_predictions_df.columns
    assert 'prob_under' in sample_predictions_df.columns

    # Verify probabilities sum to 1
    assert all(abs(sample_predictions_df['prob_over'] + sample_predictions_df['prob_under'] - 1.0) < 0.01)


def test_sample_odds_response(sample_odds_response):
    """Test sample odds API response structure"""
    assert isinstance(sample_odds_response, dict)
    assert sample_odds_response['id'] == 'event123'
    assert sample_odds_response['sport_key'] == 'basketball_nba'
    assert 'bookmakers' in sample_odds_response
    assert len(sample_odds_response['bookmakers']) > 0

    bookmaker = sample_odds_response['bookmakers'][0]
    assert 'markets' in bookmaker

    market_keys = [m['key'] for m in bookmaker['markets']]
    assert 'player_points' in market_keys
    assert 'player_rebounds' in market_keys
    assert 'player_assists' in market_keys


def test_mock_ensemble_data(mock_ensemble_data):
    """Test mock ensemble data structure"""
    assert 'mean_models' in mock_ensemble_data
    assert 'variance_models' in mock_ensemble_data
    assert 'feature_names' in mock_ensemble_data
    assert 'n_folds' in mock_ensemble_data
    assert 'training_metrics' in mock_ensemble_data

    assert len(mock_ensemble_data['mean_models']) == 3
    assert len(mock_ensemble_data['variance_models']) == 3
    assert len(mock_ensemble_data['feature_names']) == 10
    assert mock_ensemble_data['n_folds'] == 3

    # Verify models are trained
    for model in mock_ensemble_data['mean_models']:
        assert hasattr(model, 'predict')
