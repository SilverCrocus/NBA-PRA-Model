"""
Reusable test data fixtures for production tests.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_mock_player_games(n_players=10, games_per_player=50):
    """
    Create mock player game data for testing.

    Args:
        n_players: Number of unique players
        games_per_player: Games per player

    Returns:
        DataFrame with player game data
    """
    np.random.seed(42)
    n_games = n_players * games_per_player

    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(games_per_player)]

    data = {
        'player_id': np.repeat(range(1, n_players + 1), games_per_player),
        'player_name': np.repeat([f'Player_{i}' for i in range(1, n_players + 1)], games_per_player),
        'game_id': range(n_games),
        'game_date': dates * n_players,
        'team_abbreviation': np.random.choice(['LAL', 'BOS', 'GSW', 'MIA'], n_games),
        'pra': np.random.uniform(10, 45, n_games),
        'minutes': np.random.uniform(15, 38, n_games),
        'pts': np.random.uniform(5, 30, n_games),
        'reb': np.random.uniform(2, 12, n_games),
        'ast': np.random.uniform(1, 10, n_games)
    }

    return pd.DataFrame(data)


def create_mock_betting_lines(n_players=20, date='2024-11-01'):
    """
    Create mock betting lines for testing.

    Args:
        n_players: Number of players
        date: Game date

    Returns:
        DataFrame with betting lines
    """
    np.random.seed(42)

    data = {
        'player_name': [f'Player_{i}' for i in range(1, n_players + 1)],
        'team_abbreviation': np.random.choice(['LAL', 'BOS', 'GSW'], n_players),
        'opponent': np.random.choice(['MIA', 'CHI', 'PHX'], n_players),
        'pra_line': np.random.uniform(15, 40, n_players),
        'points_line': np.random.uniform(10, 30, n_players),
        'rebounds_line': np.random.uniform(3, 12, n_players),
        'assists_line': np.random.uniform(2, 10, n_players),
        'bookmaker': np.random.choice(['draftkings', 'fanduel', 'betmgm'], n_players),
        'odds': -110,
        'game_date': date
    }

    return pd.DataFrame(data)
