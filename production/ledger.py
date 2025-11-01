"""
Simple Bet Ledger

Tracks all bet recommendations over time for performance monitoring.

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.config import BETS_DIR
from production.logging_config import setup_production_logging

logger = setup_production_logging('ledger')

LEDGER_DIR = Path(__file__).parent / "outputs" / "ledger"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)
LEDGER_FILE = LEDGER_DIR / "bet_ledger.csv"


def add_bets_to_ledger(bets_df: pd.DataFrame, bet_date: str = None):
    """
    Append bets to the master ledger file.

    Args:
        bets_df: DataFrame with bet recommendations
        bet_date: Date of bets (defaults to today)
    """
    if bets_df.empty:
        logger.info("No bets to add to ledger")
        return

    # Add timestamp and bet_date if not present
    if 'bet_date' not in bets_df.columns:
        if bet_date is None:
            bet_date = datetime.now().strftime('%Y-%m-%d')
        bets_df['bet_date'] = bet_date

    if 'timestamp' not in bets_df.columns:
        bets_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Add status column for tracking results later
    if 'status' not in bets_df.columns:
        bets_df['status'] = 'pending'  # pending, won, lost

    if 'actual_pra' not in bets_df.columns:
        bets_df['actual_pra'] = None

    # Ledger columns (keep it simple)
    ledger_cols = [
        'bet_date',
        'timestamp',
        'player_name',
        'team_abbreviation',
        'opponent',
        'betting_line',
        'direction',
        'mean_pred',
        'std_dev',
        'prob_win',
        'edge',
        'kelly_size',
        'confidence_score',
        'bookmaker',
        'status',
        'actual_pra'
    ]

    # Filter to available columns
    available_cols = [col for col in ledger_cols if col in bets_df.columns]
    ledger_entry = bets_df[available_cols].copy()

    # Append to ledger
    if LEDGER_FILE.exists():
        # Load existing ledger
        existing_ledger = pd.read_csv(LEDGER_FILE)
        updated_ledger = pd.concat([existing_ledger, ledger_entry], ignore_index=True)
    else:
        updated_ledger = ledger_entry

    # Save ledger
    updated_ledger.to_csv(LEDGER_FILE, index=False)
    logger.info(f"Added {len(ledger_entry)} bets to ledger ({len(updated_ledger)} total bets tracked)")


def update_bet_result(player_name: str, bet_date: str, actual_pra: float):
    """
    Update a bet with actual result.

    Args:
        player_name: Player name
        bet_date: Date of bet
        actual_pra: Actual PRA scored
    """
    if not LEDGER_FILE.exists():
        logger.warning("Ledger file does not exist")
        return

    ledger = pd.read_csv(LEDGER_FILE)

    # Find the bet
    mask = (ledger['player_name'] == player_name) & (ledger['bet_date'] == bet_date)

    if not mask.any():
        logger.warning(f"Bet not found: {player_name} on {bet_date}")
        return

    # Update actual PRA
    ledger.loc[mask, 'actual_pra'] = actual_pra

    # Determine win/loss
    for idx in ledger[mask].index:
        direction = ledger.loc[idx, 'direction']
        line = ledger.loc[idx, 'betting_line']

        if direction == 'OVER':
            status = 'won' if actual_pra > line else 'lost'
        else:  # UNDER
            status = 'won' if actual_pra <= line else 'lost'

        ledger.loc[idx, 'status'] = status

    # Save updated ledger
    ledger.to_csv(LEDGER_FILE, index=False)
    logger.info(f"Updated result for {player_name} on {bet_date}: {actual_pra} PRA")


def get_ledger_summary():
    """
    Get summary statistics from ledger.

    Returns:
        Dictionary with summary stats
    """
    if not LEDGER_FILE.exists():
        logger.warning("Ledger file does not exist")
        return {}

    ledger = pd.read_csv(LEDGER_FILE)

    # Filter to completed bets
    completed = ledger[ledger['status'].isin(['won', 'lost'])]

    if completed.empty:
        return {
            'total_bets': len(ledger),
            'completed_bets': 0,
            'pending_bets': len(ledger),
            'win_rate': None,
            'roi': None
        }

    # Calculate stats
    wins = (completed['status'] == 'won').sum()
    losses = (completed['status'] == 'lost').sum()
    win_rate = wins / (wins + losses)

    # Simple ROI calculation (assuming -110 odds)
    # Win: +0.91 units, Loss: -1 unit
    total_profit = wins * 0.91 - losses * 1.0
    total_wagered = wins + losses
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

    summary = {
        'total_bets': len(ledger),
        'completed_bets': len(completed),
        'pending_bets': (ledger['status'] == 'pending').sum(),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'roi': roi,
        'total_profit': total_profit,
        'avg_edge': completed['edge'].mean(),
        'avg_kelly_size': completed['kelly_size'].mean()
    }

    return summary


def print_ledger_summary():
    """Print ledger summary to console."""
    summary = get_ledger_summary()

    if not summary:
        print("No ledger data available")
        return

    print("\n" + "="*60)
    print("BET LEDGER SUMMARY")
    print("="*60)
    print(f"Total bets tracked: {summary['total_bets']}")
    print(f"Completed bets: {summary['completed_bets']}")
    print(f"Pending bets: {summary['pending_bets']}")

    if summary['completed_bets'] > 0:
        print(f"\nPerformance:")
        print(f"  Wins: {summary['wins']}")
        print(f"  Losses: {summary['losses']}")
        print(f"  Win Rate: {summary['win_rate']:.1%}")
        print(f"  ROI: {summary['roi']:.1f}%")
        print(f"  Total Profit: {summary['total_profit']:+.2f} units")
        print(f"\nAverage Metrics:")
        print(f"  Avg Edge: {summary['avg_edge']:.1%}")
        print(f"  Avg Kelly Size: {summary['avg_kelly_size']:.3f}")

    print("="*60 + "\n")


if __name__ == "__main__":
    """View ledger summary"""
    print_ledger_summary()

    # Show recent bets
    if LEDGER_FILE.exists():
        ledger = pd.read_csv(LEDGER_FILE)
        print("Recent bets:")
        print(ledger.tail(10)[['bet_date', 'player_name', 'direction', 'betting_line', 'edge', 'status']])
