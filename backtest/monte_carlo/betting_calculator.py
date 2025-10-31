"""
Betting Calculator for Monte Carlo Predictions

This module converts probabilistic predictions into betting decisions
using Kelly criterion, confidence filtering, and +EV calculations.

Key Functions:
- calculate_kelly_size: Optimal bet sizing based on edge and odds
- filter_by_confidence: Remove low-confidence bets
- calculate_bet_decisions: Convert probabilities to betting actions

Betting Strategy:
1. Calculate P(PRA > line) from Gamma distribution
2. Compare to breakeven probability from odds (52.4% for -110)
3. Only bet when edge > minimum threshold
4. Size bets using Kelly criterion (fraction of bankroll)
5. Filter out high-uncertainty predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_kelly_size(
    prob_win: np.ndarray,
    odds: float = -110,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03
) -> np.ndarray:
    """
    Calculate Kelly criterion bet sizing.

    The Kelly criterion maximizes long-term growth rate by sizing bets
    proportional to edge. We use fractional Kelly (default 25%) to
    reduce variance.

    Formula:
        f = (p * b - (1-p)) / b
        where p = win probability, b = payoff ratio

    For -110 odds: b = 100/110 = 0.909

    Args:
        prob_win: Win probabilities, shape (n_bets,)
        odds: American odds (e.g., -110)
        kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly)
        min_edge: Minimum edge required to bet (e.g., 0.03 = 3%)

    Returns:
        Bet sizes as fraction of bankroll, shape (n_bets,)
        Returns 0 for bets below min_edge threshold

    Example:
        >>> prob_win = np.array([0.60, 0.55, 0.52])
        >>> bet_sizes = calculate_kelly_size(prob_win, odds=-110)
        >>> # bet_sizes might be [0.25, 0.10, 0.00] (third bet too small)
    """
    # Convert American odds to decimal payoff
    if odds < 0:
        # Negative odds: -110 means bet 110 to win 100
        payoff = 100 / abs(odds)
    else:
        # Positive odds: +150 means bet 100 to win 150
        payoff = odds / 100

    # Calculate breakeven probability
    breakeven_prob = 1 / (1 + payoff)

    # Calculate edge
    edge = prob_win - breakeven_prob

    # Full Kelly sizing
    # f = (p * b - (1-p)) / b = (p * b - 1 + p) / b = p(1+b)/b - 1/b
    full_kelly = (prob_win * payoff - (1 - prob_win)) / payoff

    # Apply Kelly fraction (reduce for risk management)
    bet_sizes = full_kelly * kelly_fraction

    # Zero out bets with insufficient edge
    bet_sizes = np.where(edge >= min_edge, bet_sizes, 0)

    # Zero out negative bet sizes (shouldn't happen, but safety check)
    bet_sizes = np.maximum(bet_sizes, 0)

    # Cap at kelly_fraction (prevent over-betting)
    bet_sizes = np.minimum(bet_sizes, kelly_fraction)

    logger.debug(f"Kelly sizing: Mean bet size {bet_sizes.mean():.3f}, "
                f"{(bet_sizes > 0).sum()}/{len(bet_sizes)} bets passed threshold")

    return bet_sizes


def filter_by_confidence(
    prob_over: np.ndarray,
    std_dev: np.ndarray,
    mean_pred: np.ndarray,
    min_confidence: float = 0.7,
    max_cv: float = 0.30
) -> np.ndarray:
    """
    Filter bets by confidence level.

    Removes bets with high uncertainty even if they have positive edge.
    This prevents betting on high-variance role players where outcomes
    are essentially coin flips.

    Two criteria:
    1. Confidence score: How close probability is to 0 or 1
    2. Coefficient of variation: Relative uncertainty (std/mean)

    Args:
        prob_over: P(PRA > line), shape (n_bets,)
        std_dev: Standard deviation of predictions, shape (n_bets,)
        mean_pred: Mean predictions, shape (n_bets,)
        min_confidence: Minimum confidence score (0-1)
        max_cv: Maximum coefficient of variation

    Returns:
        Boolean mask, True for bets that pass filter

    Example:
        >>> mask = filter_by_confidence(prob_over, std_dev, mean_pred)
        >>> filtered_bets = bets[mask]
    """
    # Confidence score: How far probability is from 0.5 (higher = more confident)
    # Range: [0, 0.5] where 0.5 = 100% confident (prob = 0 or 1)
    confidence_score = np.abs(prob_over - 0.5)

    # Normalize to [0, 1] range
    confidence_score = confidence_score / 0.5

    # Coefficient of variation (relative uncertainty)
    cv = std_dev / (mean_pred + 1e-6)  # Add small constant to prevent division by zero

    # Apply filters
    confidence_mask = confidence_score >= min_confidence
    cv_mask = cv <= max_cv

    # Combined mask
    combined_mask = confidence_mask & cv_mask

    logger.debug(f"Confidence filtering: "
                f"{confidence_mask.sum()} passed confidence, "
                f"{cv_mask.sum()} passed CV, "
                f"{combined_mask.sum()} passed both")

    return combined_mask


def calculate_bet_decisions(
    prob_over: np.ndarray,
    betting_lines: np.ndarray,
    mean_pred: np.ndarray,
    std_dev: np.ndarray,
    odds: float = -110,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.03,
    min_confidence: float = 0.6,
    max_cv: float = 0.35
) -> pd.DataFrame:
    """
    Calculate betting decisions from probabilistic predictions.

    This is the main function that converts Monte Carlo predictions into
    actionable betting decisions with proper sizing and filtering.

    Args:
        prob_over: P(PRA > line), shape (n_predictions,)
        betting_lines: Betting thresholds, shape (n_predictions,)
        mean_pred: Mean PRA predictions, shape (n_predictions,)
        std_dev: Std dev of predictions, shape (n_predictions,)
        odds: American odds (default: -110)
        kelly_fraction: Fraction of full Kelly (default: 0.25)
        min_edge: Minimum edge to bet (default: 0.03 = 3%)
        min_confidence: Minimum confidence score (default: 0.6)
        max_cv: Maximum coefficient of variation (default: 0.35)

    Returns:
        DataFrame with columns:
        - prob_over: Probability of going over line
        - prob_under: Probability of going under line
        - edge: Edge over breakeven
        - kelly_size: Recommended bet size (fraction of bankroll)
        - confidence_score: Confidence metric
        - cv: Coefficient of variation
        - should_bet_over: Boolean, bet the over
        - should_bet_under: Boolean, bet the under
        - bet_size: Final bet size (0 if filtered out)

    Example:
        >>> decisions = calculate_bet_decisions(
        ...     prob_over, betting_lines, mean_pred, std_dev
        ... )
        >>> over_bets = decisions[decisions['should_bet_over']]
    """
    n_predictions = len(prob_over)

    # Calculate probabilities
    prob_under = 1 - prob_over

    # Calculate breakeven probability for odds
    if odds < 0:
        payoff = 100 / abs(odds)
    else:
        payoff = odds / 100
    breakeven_prob = 1 / (1 + payoff)

    # Calculate edges
    edge_over = prob_over - breakeven_prob
    edge_under = prob_under - breakeven_prob

    # Calculate Kelly sizes
    kelly_size_over = calculate_kelly_size(prob_over, odds, kelly_fraction, min_edge)
    kelly_size_under = calculate_kelly_size(prob_under, odds, kelly_fraction, min_edge)

    # Calculate confidence metrics
    confidence_score = np.abs(prob_over - 0.5) / 0.5
    cv = std_dev / (mean_pred + 1e-6)

    # Apply confidence filter
    confidence_mask = filter_by_confidence(prob_over, std_dev, mean_pred, min_confidence, max_cv)

    # Determine which bets to make
    should_bet_over = (edge_over >= min_edge) & confidence_mask
    should_bet_under = (edge_under >= min_edge) & confidence_mask

    # Final bet sizes (zero if filtered out)
    bet_size_over = np.where(should_bet_over, kelly_size_over, 0)
    bet_size_under = np.where(should_bet_under, kelly_size_under, 0)

    # Create DataFrame
    decisions = pd.DataFrame({
        'prob_over': prob_over,
        'prob_under': prob_under,
        'edge_over': edge_over,
        'edge_under': edge_under,
        'kelly_size_over': kelly_size_over,
        'kelly_size_under': kelly_size_under,
        'confidence_score': confidence_score,
        'cv': cv,
        'should_bet_over': should_bet_over,
        'should_bet_under': should_bet_under,
        'bet_size_over': bet_size_over,
        'bet_size_under': bet_size_under,
        'betting_line': betting_lines,
        'mean_pred': mean_pred,
        'std_dev': std_dev
    })

    # Log summary
    n_over_bets = should_bet_over.sum()
    n_under_bets = should_bet_under.sum()
    n_total_bets = n_over_bets + n_under_bets

    logger.info(f"Bet decisions: {n_over_bets} over, {n_under_bets} under, "
               f"{n_total_bets} total ({n_total_bets/n_predictions*100:.1f}% of predictions)")

    if n_total_bets > 0:
        avg_edge = np.concatenate([
            edge_over[should_bet_over],
            edge_under[should_bet_under]
        ]).mean()
        logger.info(f"Average edge on bets: {avg_edge:.1%}")

    return decisions


def calculate_expected_value(
    prob_over: np.ndarray,
    betting_lines: np.ndarray,
    mean_pred: np.ndarray,
    std_dev: np.ndarray,
    odds: float = -110
) -> Dict[str, np.ndarray]:
    """
    Calculate expected value (EV) for betting over and under.

    EV = (prob_win * payoff) - (prob_loss * 1)

    Positive EV indicates profitable bet in the long run.

    Args:
        prob_over: P(PRA > line)
        betting_lines: Betting thresholds
        mean_pred: Mean predictions
        std_dev: Standard deviations
        odds: American odds

    Returns:
        Dictionary with 'ev_over' and 'ev_under' arrays
    """
    # Convert odds to decimal payoff
    if odds < 0:
        payoff = 100 / abs(odds)
    else:
        payoff = odds / 100

    prob_under = 1 - prob_over

    # Calculate EV
    # EV = (prob_win * payoff) - (prob_loss * 1)
    ev_over = (prob_over * payoff) - prob_under
    ev_under = (prob_under * payoff) - prob_over

    return {
        'ev_over': ev_over,
        'ev_under': ev_under
    }


def analyze_bet_distribution(decisions: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the distribution of betting decisions.

    Provides statistics on bet sizing, edge distribution, and confidence.

    Args:
        decisions: DataFrame from calculate_bet_decisions()

    Returns:
        Dictionary with analysis metrics
    """
    # Filter to only actual bets
    bets = decisions[(decisions['should_bet_over']) | (decisions['should_bet_under'])].copy()

    if len(bets) == 0:
        logger.warning("No bets passed filters")
        return {'n_bets': 0}

    # Combine over/under bets
    bets['edge'] = np.where(bets['should_bet_over'], bets['edge_over'], bets['edge_under'])
    bets['bet_size'] = np.where(bets['should_bet_over'], bets['bet_size_over'], bets['bet_size_under'])
    bets['direction'] = np.where(bets['should_bet_over'], 'over', 'under')

    analysis = {
        'n_bets': len(bets),
        'n_over': bets['should_bet_over'].sum(),
        'n_under': bets['should_bet_under'].sum(),
        'avg_edge': bets['edge'].mean(),
        'median_edge': bets['edge'].median(),
        'min_edge': bets['edge'].min(),
        'max_edge': bets['edge'].max(),
        'avg_bet_size': bets['bet_size'].mean(),
        'median_bet_size': bets['bet_size'].median(),
        'avg_confidence': bets['confidence_score'].mean(),
        'avg_cv': bets['cv'].mean(),
        'edge_by_quintile': bets.groupby(pd.qcut(bets['edge'], 5, duplicates='drop'))['edge'].agg(['mean', 'count']).to_dict()
    }

    logger.info(f"Bet Analysis:")
    logger.info(f"  Total bets: {analysis['n_bets']} ({analysis['n_over']} over, {analysis['n_under']} under)")
    logger.info(f"  Avg edge: {analysis['avg_edge']:.1%}")
    logger.info(f"  Avg bet size: {analysis['avg_bet_size']:.3f}")
    logger.info(f"  Avg confidence: {analysis['avg_confidence']:.2f}")

    return analysis


def simulate_betting_performance(
    decisions: pd.DataFrame,
    actual_pra: np.ndarray,
    bankroll: float = 1000.0,
    odds: float = -110
) -> Dict[str, Any]:
    """
    Simulate betting performance given actual outcomes.

    Args:
        decisions: DataFrame from calculate_bet_decisions()
        actual_pra: Actual PRA values
        bankroll: Starting bankroll
        odds: American odds

    Returns:
        Dictionary with performance metrics
    """
    # Convert odds to payoff
    if odds < 0:
        payoff = 100 / abs(odds)
    else:
        payoff = odds / 100

    # Initialize tracking
    current_bankroll = bankroll
    bankroll_history = [bankroll]
    bet_history = []

    # Filter to actual bets
    bets_mask = (decisions['should_bet_over']) | (decisions['should_bet_under'])
    bets = decisions[bets_mask].copy()

    if len(bets) == 0:
        return {'n_bets': 0, 'final_bankroll': bankroll}

    # Add actual outcomes
    bets['actual_pra'] = actual_pra[bets_mask]

    for idx, row in bets.iterrows():
        # Determine bet details
        if row['should_bet_over']:
            direction = 'over'
            bet_size_fraction = row['bet_size_over']
            bet_correct = row['actual_pra'] > row['betting_line']
        else:
            direction = 'under'
            bet_size_fraction = row['bet_size_under']
            bet_correct = row['actual_pra'] <= row['betting_line']

        # Calculate bet amount
        bet_amount = current_bankroll * bet_size_fraction

        # Calculate profit/loss
        if bet_correct:
            profit = bet_amount * payoff
        else:
            profit = -bet_amount

        # Update bankroll
        current_bankroll += profit
        bankroll_history.append(current_bankroll)

        bet_history.append({
            'direction': direction,
            'bet_size': bet_size_fraction,
            'bet_amount': bet_amount,
            'profit': profit,
            'correct': bet_correct,
            'bankroll': current_bankroll
        })

    # Calculate metrics
    bet_history_df = pd.DataFrame(bet_history)

    performance = {
        'n_bets': len(bet_history),
        'starting_bankroll': bankroll,
        'final_bankroll': current_bankroll,
        'total_profit': current_bankroll - bankroll,
        'roi': ((current_bankroll - bankroll) / bankroll) * 100,
        'win_rate': bet_history_df['correct'].mean(),
        'avg_profit_per_bet': bet_history_df['profit'].mean(),
        'max_bankroll': max(bankroll_history),
        'min_bankroll': min(bankroll_history),
        'max_drawdown': bankroll - min(bankroll_history),
        'max_drawdown_pct': ((bankroll - min(bankroll_history)) / bankroll) * 100,
        'bankroll_history': bankroll_history,
        'bet_history': bet_history_df
    }

    logger.info(f"Simulated Performance:")
    logger.info(f"  Win rate: {performance['win_rate']:.1%}")
    logger.info(f"  ROI: {performance['roi']:.1f}%")
    logger.info(f"  Final bankroll: ${performance['final_bankroll']:.2f} (from ${bankroll:.2f})")

    return performance
