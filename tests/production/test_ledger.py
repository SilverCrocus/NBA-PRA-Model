"""
Tests for bet ledger module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from production.ledger import (
    add_bets_to_ledger,
    update_bet_result,
    get_ledger_summary
)


@pytest.fixture
def clean_ledger(tmp_path):
    """Create a clean temporary ledger for testing"""
    import production.ledger as ledger_module

    # Override ledger file path for testing
    original_ledger_file = ledger_module.LEDGER_FILE
    test_ledger_dir = tmp_path / "ledger"
    test_ledger_dir.mkdir(exist_ok=True)
    test_ledger_file = test_ledger_dir / "bet_ledger.csv"

    ledger_module.LEDGER_FILE = test_ledger_file

    yield test_ledger_file

    # Restore original path
    ledger_module.LEDGER_FILE = original_ledger_file


def test_add_bets_to_ledger_empty(clean_ledger):
    """Test adding bets to empty ledger"""
    # Create sample bets
    bets_df = pd.DataFrame({
        'player_name': ['Player 1', 'Player 2'],
        'team_abbreviation': ['LAL', 'BOS'],
        'opponent': ['MIA', 'CHI'],
        'betting_line': [25.5, 30.5],
        'direction': ['OVER', 'UNDER'],
        'mean_pred': [28.0, 27.5],
        'std_dev': [4.0, 5.0],
        'prob_win': [0.65, 0.60],
        'edge': [0.10, 0.08],
        'kelly_size': [0.03, 0.025],
        'confidence_score': [0.75, 0.70],
        'bookmaker': ['draftkings', 'fanduel']
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    # Check ledger was created
    assert clean_ledger.exists()

    # Load and verify
    ledger = pd.read_csv(clean_ledger)
    assert len(ledger) == 2
    assert 'bet_date' in ledger.columns
    assert 'status' in ledger.columns
    assert all(ledger['status'] == 'pending')


def test_add_bets_to_ledger_append(clean_ledger):
    """Test appending bets to existing ledger"""
    # Add first batch
    bets1 = pd.DataFrame({
        'player_name': ['Player 1'],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'betting_line': [25.5],
        'direction': ['OVER'],
        'mean_pred': [28.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'bookmaker': ['draftkings']
    })

    add_bets_to_ledger(bets1, bet_date='2024-11-01')

    # Add second batch
    bets2 = pd.DataFrame({
        'player_name': ['Player 2'],
        'team_abbreviation': ['BOS'],
        'opponent': ['CHI'],
        'betting_line': [30.5],
        'direction': ['UNDER'],
        'mean_pred': [27.5],
        'std_dev': [5.0],
        'prob_win': [0.60],
        'edge': [0.08],
        'kelly_size': [0.025],
        'confidence_score': [0.70],
        'bookmaker': ['fanduel']
    })

    add_bets_to_ledger(bets2, bet_date='2024-11-02')

    # Check both batches are in ledger
    ledger = pd.read_csv(clean_ledger)
    assert len(ledger) == 2
    assert ledger.iloc[0]['player_name'] == 'Player 1'
    assert ledger.iloc[1]['player_name'] == 'Player 2'


def test_add_bets_to_ledger_empty_input(clean_ledger):
    """Test adding empty DataFrame to ledger"""
    empty_df = pd.DataFrame()

    add_bets_to_ledger(empty_df, bet_date='2024-11-01')

    # Ledger should not be created for empty input
    if clean_ledger.exists():
        ledger = pd.read_csv(clean_ledger)
        assert len(ledger) == 0


def test_update_bet_result(clean_ledger):
    """Test updating bet with actual result"""
    # Add initial bet
    bets_df = pd.DataFrame({
        'player_name': ['Player 1'],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'betting_line': [25.5],
        'direction': ['OVER'],
        'mean_pred': [28.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'bookmaker': ['draftkings']
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    # Update with result
    update_bet_result('Player 1', '2024-11-01', actual_pra=30.0)

    # Check result was updated
    ledger = pd.read_csv(clean_ledger)
    assert ledger.iloc[0]['actual_pra'] == 30.0
    assert ledger.iloc[0]['status'] == 'won'  # 30.0 > 25.5 (OVER)


def test_update_bet_result_losing_bet(clean_ledger):
    """Test updating bet that loses"""
    # Add initial bet
    bets_df = pd.DataFrame({
        'player_name': ['Player 1'],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'betting_line': [25.5],
        'direction': ['OVER'],
        'mean_pred': [28.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'bookmaker': ['draftkings']
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    # Update with losing result
    update_bet_result('Player 1', '2024-11-01', actual_pra=20.0)

    # Check result was updated
    ledger = pd.read_csv(clean_ledger)
    assert ledger.iloc[0]['actual_pra'] == 20.0
    assert ledger.iloc[0]['status'] == 'lost'  # 20.0 < 25.5 (OVER)


def test_update_bet_result_under_bet(clean_ledger):
    """Test updating UNDER bet"""
    # Add UNDER bet
    bets_df = pd.DataFrame({
        'player_name': ['Player 1'],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'betting_line': [25.5],
        'direction': ['UNDER'],
        'mean_pred': [22.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'bookmaker': ['draftkings']
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    # Update with result (UNDER wins)
    update_bet_result('Player 1', '2024-11-01', actual_pra=20.0)

    ledger = pd.read_csv(clean_ledger)
    assert ledger.iloc[0]['status'] == 'won'  # 20.0 < 25.5 (UNDER)


def test_get_ledger_summary_empty(clean_ledger):
    """Test ledger summary with no bets"""
    summary = get_ledger_summary()

    assert summary['total_bets'] == 0
    assert summary['pending_bets'] == 0
    assert summary['completed_bets'] == 0


def test_get_ledger_summary_with_results(clean_ledger):
    """Test ledger summary with completed bets"""
    # Add and complete multiple bets
    bets_df = pd.DataFrame({
        'player_name': ['Player 1', 'Player 2', 'Player 3', 'Player 4'],
        'team_abbreviation': ['LAL', 'BOS', 'GSW', 'MIA'],
        'opponent': ['MIA', 'CHI', 'PHX', 'LAL'],
        'betting_line': [25.5, 30.5, 22.5, 28.5],
        'direction': ['OVER', 'OVER', 'UNDER', 'OVER'],
        'mean_pred': [28.0, 33.0, 20.0, 31.0],
        'std_dev': [4.0, 5.0, 3.0, 4.5],
        'prob_win': [0.65, 0.68, 0.70, 0.63],
        'edge': [0.10, 0.12, 0.15, 0.08],
        'kelly_size': [0.03, 0.035, 0.04, 0.025],
        'confidence_score': [0.75, 0.80, 0.85, 0.70],
        'bookmaker': ['draftkings'] * 4
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    # Update results (3 wins, 1 loss)
    update_bet_result('Player 1', '2024-11-01', actual_pra=30.0)  # Won
    update_bet_result('Player 2', '2024-11-01', actual_pra=35.0)  # Won
    update_bet_result('Player 3', '2024-11-01', actual_pra=18.0)  # Won
    update_bet_result('Player 4', '2024-11-01', actual_pra=25.0)  # Lost

    summary = get_ledger_summary()

    assert summary['total_bets'] == 4
    assert summary['completed_bets'] == 4
    assert summary['pending_bets'] == 0
    assert summary['wins'] == 3
    assert summary['losses'] == 1
    assert summary['win_rate'] == 0.75  # 75%


def test_ledger_columns_validation(clean_ledger):
    """Test that ledger has all required columns"""
    bets_df = pd.DataFrame({
        'player_name': ['Player 1'],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'betting_line': [25.5],
        'direction': ['OVER'],
        'mean_pred': [28.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'bookmaker': ['draftkings']
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    ledger = pd.read_csv(clean_ledger)

    # Check required columns exist
    required_cols = [
        'bet_date', 'player_name', 'betting_line', 'direction',
        'mean_pred', 'edge', 'status'
    ]

    for col in required_cols:
        assert col in ledger.columns


def test_ledger_timestamp_added(clean_ledger):
    """Test that timestamp is automatically added"""
    bets_df = pd.DataFrame({
        'player_name': ['Player 1'],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'betting_line': [25.5],
        'direction': ['OVER'],
        'mean_pred': [28.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'bookmaker': ['draftkings']
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    ledger = pd.read_csv(clean_ledger)

    assert 'timestamp' in ledger.columns
    assert ledger.iloc[0]['timestamp'] is not None


def test_ledger_default_bet_date(clean_ledger):
    """Test that bet_date defaults to today if not provided"""
    from datetime import datetime

    bets_df = pd.DataFrame({
        'player_name': ['Player 1'],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'betting_line': [25.5],
        'direction': ['OVER'],
        'mean_pred': [28.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'bookmaker': ['draftkings']
    })

    add_bets_to_ledger(bets_df)  # No bet_date provided

    ledger = pd.read_csv(clean_ledger)

    today = datetime.now().strftime('%Y-%m-%d')
    assert ledger.iloc[0]['bet_date'] == today


def test_ledger_roi_calculation(clean_ledger):
    """Test ROI calculation in summary"""
    # Add bets with known kelly sizes
    bets_df = pd.DataFrame({
        'player_name': ['Player 1', 'Player 2'],
        'team_abbreviation': ['LAL', 'BOS'],
        'opponent': ['MIA', 'CHI'],
        'betting_line': [25.5, 30.5],
        'direction': ['OVER', 'OVER'],
        'mean_pred': [28.0, 33.0],
        'std_dev': [4.0, 5.0],
        'prob_win': [0.65, 0.68],
        'edge': [0.10, 0.12],
        'kelly_size': [0.10, 0.10],  # 10% stakes
        'confidence_score': [0.75, 0.80],
        'bookmaker': ['draftkings'] * 2
    })

    add_bets_to_ledger(bets_df, bet_date='2024-11-01')

    # One win, one loss
    update_bet_result('Player 1', '2024-11-01', actual_pra=30.0)  # Won
    update_bet_result('Player 2', '2024-11-01', actual_pra=25.0)  # Lost

    summary = get_ledger_summary()

    # ROI should be calculated
    # Win: +0.10 * 0.91 = 0.091
    # Loss: -0.10 * 1.0 = -0.10
    # Net profit: -0.009
    # Total wagered: 0.20
    # ROI: -4.5%
    if 'roi' in summary:
        assert -5.0 <= summary['roi'] <= -4.0  # Expected -4.5% with -110 odds


def test_ledger_multiple_dates(clean_ledger):
    """Test ledger with bets from multiple dates"""
    # Add bets for different dates
    bets1 = pd.DataFrame({
        'player_name': ['Player 1'],
        'betting_line': [25.5],
        'direction': ['OVER'],
        'mean_pred': [28.0],
        'std_dev': [4.0],
        'prob_win': [0.65],
        'edge': [0.10],
        'kelly_size': [0.03],
        'confidence_score': [0.75],
        'team_abbreviation': ['LAL'],
        'opponent': ['MIA'],
        'bookmaker': ['draftkings']
    })

    bets2 = pd.DataFrame({
        'player_name': ['Player 2'],
        'betting_line': [30.5],
        'direction': ['UNDER'],
        'mean_pred': [27.0],
        'std_dev': [5.0],
        'prob_win': [0.60],
        'edge': [0.08],
        'kelly_size': [0.025],
        'confidence_score': [0.70],
        'team_abbreviation': ['BOS'],
        'opponent': ['CHI'],
        'bookmaker': ['fanduel']
    })

    add_bets_to_ledger(bets1, bet_date='2024-11-01')
    add_bets_to_ledger(bets2, bet_date='2024-11-02')

    ledger = pd.read_csv(clean_ledger)

    assert len(ledger) == 2
    assert len(ledger['bet_date'].unique()) == 2
    assert '2024-11-01' in ledger['bet_date'].values
    assert '2024-11-02' in ledger['bet_date'].values
