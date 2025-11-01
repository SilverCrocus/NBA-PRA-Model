"""
Pytest configuration and shared fixtures for production tests.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil


@pytest.fixture
def temp_production_dir(tmp_path):
    """Create temporary production directory structure"""
    prod_dir = tmp_path / "production"
    (prod_dir / "models").mkdir(parents=True)
    (prod_dir / "outputs" / "predictions").mkdir(parents=True)
    (prod_dir / "outputs" / "bets").mkdir(parents=True)
    (prod_dir / "outputs" / "ledger").mkdir(parents=True)
    return prod_dir


@pytest.fixture
def sample_features_df():
    """Create sample feature DataFrame for testing"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'player_id': np.repeat(range(1, 11), 10),  # 10 players, 10 games each
        'player_name': np.repeat([f'Player_{i}' for i in range(1, 11)], 10),
        'game_id': range(100),
        'game_date': dates,
        'team_abbreviation': np.random.choice(['LAL', 'BOS', 'GSW'], 100),
        'opponent': np.random.choice(['MIA', 'CHI', 'PHX'], 100),
        'home_game': np.random.choice([0, 1], 100),
        # Rolling features
        'pra_mean_last5': np.random.uniform(15, 35, 100),
        'pra_mean_last10': np.random.uniform(15, 35, 100),
        'pra_ewma_3': np.random.uniform(15, 35, 100),
        'pra_ewma_7': np.random.uniform(15, 35, 100),
        'minutes_mean_last5': np.random.uniform(20, 35, 100),
        # Contextual features
        'rest_days': np.random.randint(0, 5, 100),
        'games_in_last_7': np.random.randint(2, 5, 100),
        # Matchup features
        'opp_def_rating': np.random.uniform(100, 115, 100),
        'opp_pace': np.random.uniform(95, 105, 100),
        # Advanced metrics
        'usage_rate': np.random.uniform(15, 30, 100),
        'assist_rate': np.random.uniform(10, 40, 100),
        # Position features
        'position_z_score': np.random.uniform(-2, 2, 100),
        'position_percentile': np.random.uniform(0, 1, 100),
        # Injury features
        'dnp_last30': np.random.randint(0, 3, 100),
        'minutes_restriction_flag': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })

    return df


@pytest.fixture
def sample_predictions_df():
    """Create sample predictions DataFrame"""
    np.random.seed(42)

    df = pd.DataFrame({
        'player_id': range(1, 21),
        'player_name': [f'Player_{i}' for i in range(1, 21)],
        'game_date': '2024-11-01',
        'team_abbreviation': np.random.choice(['LAL', 'BOS', 'GSW'], 20),
        'opponent': np.random.choice(['MIA', 'CHI', 'PHX'], 20),
        'home_game': np.random.choice([0, 1], 20),
        'mean_pred': np.random.uniform(15, 40, 20),
        'std_dev': np.random.uniform(3, 8, 20),
        'betting_line': np.random.uniform(15, 40, 20),
        'pra_odds': -110,
        'breakeven_prob': 0.524,
        'prob_over': np.random.uniform(0.4, 0.8, 20),
        'prob_under': np.random.uniform(0.2, 0.6, 20),
        'edge_over': np.random.uniform(-0.1, 0.15, 20),
        'edge_under': np.random.uniform(-0.1, 0.15, 20),
        'confidence_score': np.random.uniform(0.5, 0.9, 20),
        'cv': np.random.uniform(0.15, 0.4, 20)
    })

    # Ensure prob_over + prob_under = 1
    df['prob_under'] = 1 - df['prob_over']

    return df


@pytest.fixture
def sample_odds_response():
    """Sample API response from TheOddsAPI"""
    return {
        "id": "event123",
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "commence_time": "2024-11-01T19:00:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "player_points",
                        "outcomes": [
                            {
                                "name": "LeBron James",
                                "description": "Over",
                                "price": -110,
                                "point": 25.5
                            }
                        ]
                    },
                    {
                        "key": "player_rebounds",
                        "outcomes": [
                            {
                                "name": "LeBron James",
                                "description": "Over",
                                "price": -110,
                                "point": 7.5
                            }
                        ]
                    },
                    {
                        "key": "player_assists",
                        "outcomes": [
                            {
                                "name": "LeBron James",
                                "description": "Over",
                                "price": -110,
                                "point": 6.5
                            }
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_ensemble_data():
    """Mock trained ensemble data"""
    from sklearn.ensemble import RandomForestRegressor

    # Create simple mock models
    mean_models = [RandomForestRegressor(n_estimators=10, random_state=i) for i in range(3)]
    variance_models = [RandomForestRegressor(n_estimators=10, random_state=i+100) for i in range(3)]

    # Train on dummy data
    X_dummy = np.random.rand(100, 10)
    y_dummy = np.random.rand(100)

    for model in mean_models + variance_models:
        model.fit(X_dummy, y_dummy)

    return {
        'mean_models': mean_models,
        'variance_models': variance_models,
        'feature_names': [f'feature_{i}' for i in range(10)],
        'n_folds': 3,
        'training_metrics': {
            'mae': 2.5,
            'rmse': 3.5,
            'r2': 0.85
        }
    }
