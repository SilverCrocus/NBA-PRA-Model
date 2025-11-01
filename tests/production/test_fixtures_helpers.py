"""
Tests for fixtures.py helper functions.
"""
import pytest
import pandas as pd
from tests.production.fixtures import create_mock_player_games, create_mock_betting_lines


def test_create_mock_player_games_default():
    """Test creating mock player games with default parameters"""
    df = create_mock_player_games()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500  # 10 players * 50 games
    assert 'player_id' in df.columns
    assert 'player_name' in df.columns
    assert 'pra' in df.columns
    assert 'minutes' in df.columns
    assert df['player_id'].nunique() == 10


def test_create_mock_player_games_custom():
    """Test creating mock player games with custom parameters"""
    df = create_mock_player_games(n_players=5, games_per_player=20)

    assert len(df) == 100  # 5 players * 20 games
    assert df['player_id'].nunique() == 5


def test_create_mock_betting_lines_default():
    """Test creating mock betting lines with default parameters"""
    df = create_mock_betting_lines()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert 'player_name' in df.columns
    assert 'pra_line' in df.columns
    assert 'points_line' in df.columns
    assert 'rebounds_line' in df.columns
    assert 'assists_line' in df.columns
    assert 'game_date' in df.columns
    assert all(df['game_date'] == '2024-11-01')


def test_create_mock_betting_lines_custom():
    """Test creating mock betting lines with custom parameters"""
    df = create_mock_betting_lines(n_players=10, date='2025-01-15')

    assert len(df) == 10
    assert all(df['game_date'] == '2025-01-15')
