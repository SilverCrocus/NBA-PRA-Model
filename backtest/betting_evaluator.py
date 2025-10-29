"""
Betting Performance Evaluation

Calculates betting metrics: win rate, ROI, profit, edge analysis.

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.config import (
    WIN_PAYOUT, LOSS_COST, BREAK_EVEN_WIN_RATE,
    EDGE_THRESHOLD_SMALL, EDGE_THRESHOLD_MEDIUM, EDGE_THRESHOLD_LARGE
)

logger = logging.getLogger(__name__)


def calculate_bet_decisions(predictions: pd.Series,
                              betting_lines: pd.Series) -> pd.DataFrame:
    """
    Determine bet decisions (over/under) based on predictions vs betting lines

    Args:
        predictions: Predicted PRA values
        betting_lines: Betting lines from sportsbooks

    Returns:
        DataFrame with bet_decision (True=over, False=under), has_line (bool)
    """
    results = pd.DataFrame(index=predictions.index)

    # Only bet where we have a betting line
    results['has_line'] = ~betting_lines.isna()

    # Bet "over" if prediction > line, "under" if prediction < line
    results['bet_decision'] = predictions > betting_lines
    results['bet_over'] = (predictions > betting_lines).astype(int)

    # Edge size (how much model differs from market)
    results['edge_size'] = (predictions - betting_lines).abs()

    return results


def calculate_bet_outcomes(actual_pra: pd.Series,
                            betting_lines: pd.Series,
                            bet_decisions: pd.Series) -> pd.DataFrame:
    """
    Calculate betting outcomes (win/loss) based on actual results

    Args:
        actual_pra: Actual PRA values
        betting_lines: Betting lines
        bet_decisions: Bet decisions (True=over, False=under)

    Returns:
        DataFrame with bet_correct (bool), profit (float)
    """
    results = pd.DataFrame(index=actual_pra.index)

    # Actual outcome: did player go over or under the line?
    actual_over = actual_pra > betting_lines

    # Bet was correct if: (bet over AND went over) OR (bet under AND went under)
    results['bet_correct'] = (bet_decisions == actual_over)

    # Calculate profit
    # Win: +0.909 units (assuming -110 odds)
    # Loss: -1.0 units
    results['profit'] = np.where(
        results['bet_correct'],
        WIN_PAYOUT,  # Win
        LOSS_COST     # Loss
    )

    # For games without betting lines, profit = 0
    results.loc[betting_lines.isna(), 'profit'] = 0.0

    return results


def calculate_win_rate(bet_correct: pd.Series,
                       has_line: pd.Series) -> float:
    """
    Calculate win rate (% of bets that were correct)

    Args:
        bet_correct: Boolean series of whether bet was correct
        has_line: Boolean series of whether bet had a line

    Returns:
        Win rate as decimal (0.0 to 1.0)
    """
    bets_with_lines = bet_correct[has_line]

    if len(bets_with_lines) == 0:
        return np.nan

    win_rate = bets_with_lines.sum() / len(bets_with_lines)
    return win_rate


def calculate_roi(profits: pd.Series,
                  has_line: pd.Series) -> float:
    """
    Calculate ROI (return on investment) as percentage

    ROI = (Total Profit / Total Bets) * 100

    Args:
        profits: Series of profit/loss per bet
        has_line: Boolean series of whether bet had a line

    Returns:
        ROI as percentage
    """
    bets_with_lines = profits[has_line]

    if len(bets_with_lines) == 0:
        return np.nan

    total_profit = bets_with_lines.sum()
    total_bets = len(bets_with_lines)

    roi = (total_profit / total_bets) * 100  # As percentage

    return roi


def calculate_edge_metrics(predictions_df: pd.DataFrame,
                            edge_threshold: float = EDGE_THRESHOLD_MEDIUM) -> Dict[str, float]:
    """
    Calculate betting metrics for bets with significant edge

    "Edge" = when model prediction differs significantly from betting line

    Args:
        predictions_df: DataFrame with columns: prediction, betting_line, bet_correct, profit, edge_size
        edge_threshold: Minimum edge size to consider (PRA points)

    Returns:
        Dictionary with edge metrics: count, win_rate, roi, avg_profit
    """
    # Filter to bets with sufficient edge
    edge_bets = predictions_df[predictions_df['edge_size'] >= edge_threshold]

    if len(edge_bets) == 0:
        return {
            'edge_count': 0,
            'edge_win_rate': np.nan,
            'edge_roi': np.nan,
            'edge_avg_profit': np.nan
        }

    edge_win_rate = edge_bets['bet_correct'].mean()
    edge_roi = (edge_bets['profit'].sum() / len(edge_bets)) * 100
    edge_avg_profit = edge_bets['profit'].mean()

    return {
        'edge_count': len(edge_bets),
        'edge_win_rate': edge_win_rate,
        'edge_roi': edge_roi,
        'edge_avg_profit': edge_avg_profit
    }


def analyze_betting_performance(predictions_df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive betting performance analysis

    Args:
        predictions_df: DataFrame with columns:
            - prediction, actual_pra, betting_line
            - bet_decision, bet_correct, profit, edge_size

    Returns:
        Dictionary with all betting metrics
    """
    # Filter to games with betting lines
    has_line = ~predictions_df['betting_line'].isna()
    games_with_lines = predictions_df[has_line]

    if len(games_with_lines) == 0:
        logger.warning("No games with betting lines, cannot calculate betting metrics")
        return {}

    # Basic metrics
    total_bets = len(games_with_lines)
    wins = games_with_lines['bet_correct'].sum()
    losses = total_bets - wins

    win_rate = wins / total_bets
    total_profit = games_with_lines['profit'].sum()
    roi = (total_profit / total_bets) * 100

    # Profitability analysis
    is_profitable = win_rate > BREAK_EVEN_WIN_RATE
    win_rate_vs_breakeven = (win_rate - BREAK_EVEN_WIN_RATE) * 100  # Percentage points above/below

    # Edge analysis (different thresholds)
    edge_small = calculate_edge_metrics(games_with_lines, EDGE_THRESHOLD_SMALL)
    edge_medium = calculate_edge_metrics(games_with_lines, EDGE_THRESHOLD_MEDIUM)
    edge_large = calculate_edge_metrics(games_with_lines, EDGE_THRESHOLD_LARGE)

    # Cumulative profit (for plotting)
    cumulative_profit = games_with_lines['profit'].cumsum()

    metrics = {
        # Basic counts
        'total_bets': total_bets,
        'wins': int(wins),
        'losses': int(losses),

        # Win rate
        'win_rate': win_rate,
        'break_even_win_rate': BREAK_EVEN_WIN_RATE,
        'win_rate_vs_breakeven': win_rate_vs_breakeven,

        # Profitability
        'total_profit': total_profit,
        'roi': roi,
        'is_profitable': is_profitable,
        'avg_profit_per_bet': total_profit / total_bets,

        # Edge analysis
        'edge_small': edge_small,
        'edge_medium': edge_medium,
        'edge_large': edge_large,

        # Cumulative
        'final_cumulative_profit': cumulative_profit.iloc[-1] if len(cumulative_profit) > 0 else 0,
        'max_cumulative_profit': cumulative_profit.max() if len(cumulative_profit) > 0 else 0,
        'min_cumulative_profit': cumulative_profit.min() if len(cumulative_profit) > 0 else 0
    }

    return metrics


def analyze_by_player(predictions_df: pd.DataFrame, min_games: int = 10) -> pd.DataFrame:
    """
    Analyze betting performance by player

    Identifies which players are most/least predictable for betting purposes.

    Args:
        predictions_df: DataFrame with predictions and betting results
        min_games: Minimum games required to include player in analysis

    Returns:
        DataFrame with per-player metrics: games, win_rate, roi, total_profit
    """
    if 'player_name' not in predictions_df.columns:
        logger.warning("No player_name column, cannot analyze by player")
        return pd.DataFrame()

    # Filter to games with betting lines
    has_line = ~predictions_df['betting_line'].isna()
    games_with_lines = predictions_df[has_line]

    if len(games_with_lines) == 0:
        return pd.DataFrame()

    # Group by player
    player_stats = games_with_lines.groupby('player_name').agg({
        'bet_correct': ['sum', 'count', 'mean'],
        'profit': ['sum', 'mean'],
        'edge_size': 'mean'
    }).reset_index()

    # Flatten column names
    player_stats.columns = [
        'player_name',
        'wins', 'num_games', 'win_rate',
        'total_profit', 'avg_profit', 'avg_edge'
    ]

    # Calculate ROI
    player_stats['roi'] = (player_stats['total_profit'] / player_stats['num_games']) * 100

    # Filter to minimum games
    player_stats = player_stats[player_stats['num_games'] >= min_games]

    # Sort by ROI
    player_stats = player_stats.sort_values('roi', ascending=False)

    logger.info(f"Analyzed {len(player_stats)} players with >={min_games} games")

    return player_stats


def calculate_daily_betting_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate betting metrics per game day

    Args:
        predictions_df: DataFrame with game_date, predictions, and betting results

    Returns:
        DataFrame with daily betting metrics
    """
    if 'game_date' not in predictions_df.columns:
        logger.warning("No game_date column, cannot calculate daily metrics")
        return pd.DataFrame()

    # Filter to games with betting lines
    has_line = ~predictions_df['betting_line'].isna()
    games_with_lines = predictions_df[has_line]

    if len(games_with_lines) == 0:
        return pd.DataFrame()

    # Group by date
    daily_stats = games_with_lines.groupby('game_date').agg({
        'bet_correct': ['sum', 'count', 'mean'],
        'profit': ['sum', 'mean']
    }).reset_index()

    # Flatten column names
    daily_stats.columns = [
        'game_date',
        'wins', 'num_bets', 'win_rate',
        'daily_profit', 'avg_profit'
    ]

    # Calculate cumulative profit
    daily_stats['cumulative_profit'] = daily_stats['daily_profit'].cumsum()

    # Calculate ROI
    daily_stats['roi'] = (daily_stats['daily_profit'] / daily_stats['num_bets']) * 100

    logger.info(f"Calculated daily metrics for {len(daily_stats)} game days")

    return daily_stats


def evaluate_betting_performance(predictions_df: pd.DataFrame) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Full betting performance evaluation

    Args:
        predictions_df: DataFrame with all predictions and results

    Returns:
        Tuple of (overall_metrics, player_analysis, daily_metrics)
    """
    logger.info("Evaluating betting performance...")

    # Overall performance
    overall_metrics = analyze_betting_performance(predictions_df)

    # Player-level analysis
    player_analysis = analyze_by_player(predictions_df, min_games=10)

    # Daily metrics
    daily_metrics = calculate_daily_betting_metrics(predictions_df)

    # Log summary
    if overall_metrics:
        logger.info(f"\nBetting Performance Summary:")
        logger.info(f"  Total bets: {overall_metrics['total_bets']}")
        logger.info(f"  Win rate: {overall_metrics['win_rate']:.1%}")
        logger.info(f"  ROI: {overall_metrics['roi']:.2f}%")
        logger.info(f"  Total profit: {overall_metrics['total_profit']:.2f} units")
        logger.info(f"  Profitable: {overall_metrics['is_profitable']}")

    return overall_metrics, player_analysis, daily_metrics
