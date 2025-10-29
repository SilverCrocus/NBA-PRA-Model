"""
Pytest fixtures for feature engineering tests
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_player_data():
    """Sample player game data for testing (5 games, 2 players)"""
    return pd.DataFrame({
        'player_id': [1, 1, 1, 2, 2],
        'game_id': [100, 101, 102, 103, 104],
        'game_date': pd.to_datetime([
            '2024-01-01', '2024-01-03', '2024-01-05',
            '2024-01-02', '2024-01-04'
        ]),
        'pra': [25, 30, 28, 20, 22],
        'points': [15, 18, 16, 12, 14],
        'rebounds': [7, 8, 9, 5, 6],
        'assists': [3, 4, 3, 3, 2],
        'minutes': [32, 35, 30, 28, 25],
        'season': ['2023-24', '2023-24', '2023-24', '2023-24', '2023-24'],
        'player_name': ['Player A', 'Player A', 'Player A', 'Player B', 'Player B'],
        'opponent_team': ['LAL', 'BOS', 'GSW', 'MIA', 'PHX']
    })


@pytest.fixture
def empty_dataframe():
    """Empty DataFrame with correct columns"""
    return pd.DataFrame(columns=['player_id', 'game_id', 'game_date', 'pra'])


@pytest.fixture
def sequential_data():
    """Sequential data for temporal ordering tests"""
    return pd.DataFrame({
        'player_id': [1] * 10,
        'game_id': range(100, 110),
        'game_date': pd.date_range('2024-01-01', periods=10, freq='2D'),
        'pra': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'points': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'rebounds': [3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        'assists': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        'minutes': [25, 30, 35, 40, 35, 30, 25, 30, 35, 40],
        'season': ['2023-24'] * 10,
        'player_name': ['Player A'] * 10,
        'opponent_team': ['LAL', 'BOS', 'GSW', 'MIA', 'PHX', 'DAL', 'NYK', 'CHI', 'ATL', 'ORL']
    })


@pytest.fixture
def multi_player_data():
    """Multi-player data for cross-player validation"""
    dates = pd.date_range('2024-01-01', periods=5, freq='2D')
    data = []

    for player_id in range(1, 4):  # 3 players
        for i, date in enumerate(dates):
            data.append({
                'player_id': player_id,
                'game_id': 100 + player_id * 10 + i,
                'game_date': date,
                'pra': 20 + player_id * 5 + i * 2,
                'points': 10 + player_id * 2 + i,
                'rebounds': 5 + player_id + i,
                'assists': 5 + i,
                'minutes': 30 + i * 2,
                'season': '2023-24',
                'player_name': f'Player {player_id}',
                'opponent_team': ['LAL', 'BOS', 'GSW', 'MIA', 'PHX'][i],
                'position': ['Guard', 'Forward', 'Center'][player_id - 1]
            })

    return pd.DataFrame(data)


@pytest.fixture
def data_with_dnp():
    """Data with DNP (Did Not Play) scenarios for injury testing"""
    return pd.DataFrame({
        'player_id': [1] * 15,
        'game_id': range(100, 115),
        'game_date': pd.date_range('2024-01-01', periods=15, freq='2D'),
        'pra': [25, 30, 0, 0, 28, 32, 30, 0, 29, 31, 33, 30, 28, 0, 27],  # 0 = DNP
        'points': [15, 18, 0, 0, 16, 19, 17, 0, 16, 18, 20, 17, 15, 0, 14],
        'rebounds': [7, 8, 0, 0, 9, 8, 9, 0, 9, 9, 10, 9, 9, 0, 9],
        'assists': [3, 4, 0, 0, 3, 5, 4, 0, 4, 4, 3, 4, 4, 0, 4],
        'minutes': [32, 35, 0, 0, 30, 36, 33, 0, 31, 34, 35, 32, 30, 0, 29],
        'season': ['2023-24'] * 15,
        'player_name': ['Player A'] * 15,
        'opponent_team': ['LAL', 'BOS', 'GSW', 'MIA', 'PHX', 'DAL', 'NYK', 'CHI',
                          'ATL', 'ORL', 'DEN', 'MEM', 'SAC', 'POR', 'UTA']
    })
