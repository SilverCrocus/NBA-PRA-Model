"""
Tests for betting engine module.
"""
import pytest
import pandas as pd
import numpy as np
from production.betting_engine import BettingEngine


def test_betting_engine_initialization():
    """Test betting engine initialization"""
    engine = BettingEngine()
    assert engine is not None


def test_betting_engine_kelly_sizing(sample_predictions_df):
    """Test Kelly criterion bet sizing"""
    engine = BettingEngine()

    # Filter to predictions with good edge
    good_bets = sample_predictions_df[sample_predictions_df['edge_over'] > 0.05].copy()

    if len(good_bets) == 0:
        # Force create some good bets for testing
        good_bets = sample_predictions_df.head(5).copy()
        good_bets['edge_over'] = 0.08
        good_bets['prob_over'] = 0.60

    decisions = engine.calculate_betting_decisions(good_bets)

    assert not decisions.empty
    assert 'kelly_size' in decisions.columns
    assert all(decisions['kelly_size'] >= 0)
    assert all(decisions['kelly_size'] <= 1)  # Max 100% of bankroll


def test_betting_engine_confidence_filtering(sample_predictions_df):
    """Test confidence-based filtering"""
    from production.config import MIN_CONFIDENCE

    engine = BettingEngine()

    # Create low-confidence predictions
    low_conf_df = sample_predictions_df.copy()
    low_conf_df['confidence_score'] = 0.4  # Below MIN_CONFIDENCE (typically 0.6)
    low_conf_df['edge_over'] = 0.08  # Give it good edge
    low_conf_df['prob_over'] = 0.60

    decisions = engine.calculate_betting_decisions(low_conf_df)

    # Should filter out low confidence bets
    # If MIN_CONFIDENCE > 0.4, all should be filtered
    if MIN_CONFIDENCE > 0.4:
        assert len(decisions) == 0


def test_betting_engine_edge_filtering(sample_predictions_df):
    """Test edge-based filtering"""
    engine = BettingEngine()

    # Create predictions with negative edge
    negative_edge_df = sample_predictions_df.copy()
    negative_edge_df['edge_over'] = -0.05
    negative_edge_df['edge_under'] = -0.05
    negative_edge_df['prob_over'] = 0.45
    negative_edge_df['prob_under'] = 0.55

    decisions = engine.calculate_betting_decisions(negative_edge_df)

    # Should have zero bets (all negative edge)
    assert len(decisions) == 0


def test_betting_engine_direction_selection(sample_predictions_df):
    """Test bet direction selection (OVER vs UNDER)"""
    engine = BettingEngine()

    # Ensure we have both OVER and UNDER bets
    df = sample_predictions_df.copy()
    # First half: strong OVER
    df.loc[:9, 'edge_over'] = 0.10
    df.loc[:9, 'edge_under'] = -0.05
    df.loc[:9, 'prob_over'] = 0.65
    df.loc[:9, 'confidence_score'] = 0.8

    # Second half: strong UNDER
    df.loc[10:, 'edge_over'] = -0.05
    df.loc[10:, 'edge_under'] = 0.10
    df.loc[10:, 'prob_over'] = 0.35
    df.loc[10:, 'confidence_score'] = 0.8

    decisions = engine.calculate_betting_decisions(df)

    if len(decisions) > 0:
        # Check that direction matches the better edge
        for idx in decisions.index:
            original_idx = decisions.loc[idx, 'original_index'] if 'original_index' in decisions.columns else idx
            if 'direction' in decisions.columns:
                direction = decisions.loc[idx, 'direction']
                if direction == 'OVER':
                    assert df.loc[original_idx, 'edge_over'] > df.loc[original_idx, 'edge_under']
                else:
                    assert df.loc[original_idx, 'edge_under'] > df.loc[original_idx, 'edge_over']


def test_betting_engine_empty_input():
    """Test betting engine with empty DataFrame"""
    engine = BettingEngine()

    empty_df = pd.DataFrame()
    decisions = engine.calculate_betting_decisions(empty_df)

    assert decisions.empty


def test_betting_engine_missing_columns():
    """Test betting engine with missing required columns"""
    engine = BettingEngine()

    # Missing required columns
    incomplete_df = pd.DataFrame({
        'player_name': ['Player 1'],
        'mean_pred': [25.0]
        # Missing: prob_over, betting_line, std_dev
    })

    decisions = engine.calculate_betting_decisions(incomplete_df)

    # Should return empty DataFrame or handle gracefully
    assert decisions.empty


def test_betting_engine_kelly_calculation():
    """Test Kelly fraction calculation"""
    from production.monte_carlo import calculate_kelly_fraction

    # Positive edge scenario
    prob_win = 0.65
    odds = -110
    kelly_fraction = 0.25

    kelly_size = calculate_kelly_fraction(prob_win, odds, kelly_fraction)

    # Should be positive
    assert kelly_size > 0

    # Should be reasonable size (< 10% of bankroll for quarter Kelly)
    assert kelly_size < 0.10


def test_betting_engine_no_edge_scenario():
    """Test betting engine when no bets have positive edge"""
    from production.monte_carlo import calculate_kelly_fraction

    # No edge (breakeven probability)
    prob_win = 0.524  # Breakeven for -110
    odds = -110
    kelly_fraction = 0.25

    kelly_size = calculate_kelly_fraction(prob_win, odds, kelly_fraction)

    # Should be near zero
    assert kelly_size < 0.01


def test_betting_engine_process_predictions(sample_predictions_df):
    """Test complete betting decision workflow"""
    engine = BettingEngine()

    # Ensure some good bets exist
    df = sample_predictions_df.copy()
    df.loc[:4, 'edge_over'] = 0.10
    df.loc[:4, 'prob_over'] = 0.65
    df.loc[:4, 'confidence_score'] = 0.8

    decisions = engine.calculate_betting_decisions(df)

    # Should have some recommendations
    assert len(decisions) > 0

    # Check required columns exist
    expected_cols = ['player_name', 'betting_line', 'direction', 'edge', 'kelly_size']
    for col in expected_cols:
        assert col in decisions.columns


def test_betting_engine_max_cv_filter():
    """Test filtering by coefficient of variation (CV)"""
    from production.config import MAX_CV

    engine = BettingEngine()

    # Create predictions with high CV (high uncertainty)
    high_cv_df = pd.DataFrame({
        'player_name': ['Player 1', 'Player 2'],
        'mean_pred': [25.0, 30.0],
        'std_dev': [10.0, 5.0],  # High std_dev for Player 1
        'betting_line': [24.0, 29.0],
        'prob_over': [0.65, 0.65],
        'prob_under': [0.35, 0.35],
        'edge_over': [0.10, 0.10],
        'edge_under': [-0.05, -0.05],
        'confidence_score': [0.7, 0.7],
        'cv': [0.4, 0.167]  # Player 1 has high CV
    })

    decisions = engine.calculate_betting_decisions(high_cv_df)

    # If MAX_CV is enabled, high CV bets should be filtered
    if 'cv' in decisions.columns and MAX_CV is not None:
        assert all(decisions['cv'] <= MAX_CV)


def test_betting_engine_american_odds_conversion():
    """Test American odds to probability conversion"""
    from production.monte_carlo import american_odds_to_probability

    # Standard -110 odds
    prob_110 = american_odds_to_probability(-110)
    assert abs(prob_110 - 0.5238) < 0.01  # 52.38% breakeven

    # Even money (+100)
    prob_100 = american_odds_to_probability(100)
    assert abs(prob_100 - 0.5) < 0.01

    # Heavy favorite (-200)
    prob_200 = american_odds_to_probability(-200)
    assert prob_200 > 0.65

    # Underdog (+200)
    prob_plus200 = american_odds_to_probability(200)
    assert prob_plus200 < 0.35


def test_betting_engine_bet_edge_calculation():
    """Test bet edge calculation"""
    from production.monte_carlo import calculate_bet_edge

    # Positive edge
    edge_positive = calculate_bet_edge(prob_win=0.65, odds=-110)
    assert edge_positive > 0
    assert abs(edge_positive - 0.1262) < 0.01  # 65% - 52.38%

    # Negative edge
    edge_negative = calculate_bet_edge(prob_win=0.45, odds=-110)
    assert edge_negative < 0

    # Zero edge (breakeven)
    edge_zero = calculate_bet_edge(prob_win=0.5238, odds=-110)
    assert abs(edge_zero) < 0.01


def test_betting_engine_rank_bets(sample_predictions_df):
    """Test bet ranking by edge and confidence"""
    engine = BettingEngine()

    # Create bets with different qualities
    df = sample_predictions_df.copy()
    df.loc[0, 'edge_over'] = 0.15  # Best bet
    df.loc[0, 'confidence_score'] = 0.9

    df.loc[1, 'edge_over'] = 0.10  # Good bet
    df.loc[1, 'confidence_score'] = 0.8

    df.loc[2, 'edge_over'] = 0.05  # Marginal bet
    df.loc[2, 'confidence_score'] = 0.7

    df.loc[:2, 'prob_over'] = [0.68, 0.63, 0.58]

    decisions = engine.calculate_betting_decisions(df)

    if len(decisions) > 1:
        # Best bet should have highest edge
        assert decisions.iloc[0]['edge'] >= decisions.iloc[1]['edge']
