"""
Betting Engine Module

Converts probabilistic predictions into betting decisions using Kelly criterion
and confidence filtering.

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from production.config import (
    KELLY_FRACTION,
    MIN_EDGE_KELLY,
    MIN_EDGE_DISPLAY,
    MIN_CONFIDENCE,
    MAX_CV,
    BETTING_ODDS,
    BETS_DIR,
    setup_logging
)
from production.ledger import add_bets_to_ledger

# Import Monte Carlo betting calculator
from backtest.monte_carlo.betting_calculator import (
    calculate_bet_decisions as mc_calculate_bet_decisions,
    analyze_bet_distribution
)

logger = setup_logging('betting_engine')


class BettingEngine:
    """
    Convert predictions into actionable betting decisions

    Features:
    - Kelly criterion bet sizing
    - Confidence-based filtering
    - Edge calculation
    - Bet quality analysis
    """

    def __init__(self):
        """Initialize betting engine"""
        logger.info("BettingEngine initialized")

    def calculate_betting_decisions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate betting decisions from predictions

        Args:
            predictions: DataFrame with predictions and probabilities
                        Must have: prob_over, betting_line, mean_pred, std_dev

        Returns:
            DataFrame with betting decisions
        """
        if predictions.empty:
            logger.warning("No predictions provided")
            return pd.DataFrame()

        # Check required columns
        required_cols = ['prob_over', 'betting_line', 'mean_pred', 'std_dev']
        missing_cols = [col for col in required_cols if col not in predictions.columns]

        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()

        logger.info(f"Calculating betting decisions for {len(predictions)} predictions...")

        # Extract arrays
        prob_over = predictions['prob_over'].values
        betting_lines = predictions['betting_line'].values
        mean_pred = predictions['mean_pred'].values
        std_dev = predictions['std_dev'].values

        # Use Monte Carlo betting calculator
        decisions = mc_calculate_bet_decisions(
            prob_over=prob_over,
            betting_lines=betting_lines,
            mean_pred=mean_pred,
            std_dev=std_dev,
            odds=BETTING_ODDS,
            kelly_fraction=KELLY_FRACTION,
            min_edge=MIN_EDGE_KELLY,
            min_confidence=MIN_CONFIDENCE,
            max_cv=MAX_CV
        )

        # Merge with metadata
        for col in predictions.columns:
            if col not in decisions.columns:
                decisions[col] = predictions[col].values

        logger.info(f"Generated {len(decisions)} betting decision rows")

        return decisions

    def filter_to_bets(self, decisions: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to actual bets (passing all criteria)

        Args:
            decisions: Full betting decisions DataFrame

        Returns:
            DataFrame with only actionable bets
        """
        # Filter to bets that passed all criteria
        bets_mask = (decisions['should_bet_over']) | (decisions['should_bet_under'])
        bets = decisions[bets_mask].copy()

        logger.info(f"Filtered to {len(bets)} actionable bets")

        return bets

    def format_bet_output(self, bets: pd.DataFrame) -> pd.DataFrame:
        """
        Format bets for output

        Args:
            bets: Bets DataFrame from filter_to_bets

        Returns:
            Formatted DataFrame for export
        """
        if bets.empty:
            return pd.DataFrame()

        # Determine direction and select appropriate columns
        bets['direction'] = np.where(bets['should_bet_over'], 'OVER', 'UNDER')
        bets['prob_win'] = np.where(bets['should_bet_over'], bets['prob_over'], bets['prob_under'])
        bets['edge'] = np.where(bets['should_bet_over'], bets['edge_over'], bets['edge_under'])
        bets['kelly_size'] = np.where(bets['should_bet_over'], bets['kelly_size_over'], bets['kelly_size_under'])

        # Select relevant columns
        output_cols = [
            'player_name',
            'game_date',
            'team_abbreviation',
            'betting_line',
            'direction',
            'mean_pred',
            'std_dev',
            'prob_win',
            'edge',
            'kelly_size',
            'confidence_score',
            'cv'
        ]

        # Add optional columns if they exist
        optional_cols = ['bookmaker', 'opponent', 'home_team', 'away_team']
        for col in optional_cols:
            if col in bets.columns:
                output_cols.append(col)

        # Filter to existing columns
        output_cols = [col for col in output_cols if col in bets.columns]

        formatted = bets[output_cols].copy()

        # Sort by edge (best bets first)
        formatted = formatted.sort_values('edge', ascending=False)

        return formatted

    def analyze_bets(self, bets: pd.DataFrame):
        """
        Analyze bet quality and distribution

        Args:
            bets: Bets DataFrame
        """
        if bets.empty:
            logger.info("No bets to analyze")
            return

        n_over = (bets['direction'] == 'OVER').sum()
        n_under = (bets['direction'] == 'UNDER').sum()

        logger.info(f"\n{'='*60}")
        logger.info(f"BETTING ANALYSIS")
        logger.info(f"{'='*60}")
        logger.info(f"Total bets: {len(bets)}")
        logger.info(f"  Over bets: {n_over}")
        logger.info(f"  Under bets: {n_under}")
        logger.info(f"\nEdge distribution:")
        logger.info(f"  Mean: {bets['edge'].mean():.1%}")
        logger.info(f"  Median: {bets['edge'].median():.1%}")
        logger.info(f"  Min: {bets['edge'].min():.1%}")
        logger.info(f"  Max: {bets['edge'].max():.1%}")
        logger.info(f"\nKelly size distribution:")
        logger.info(f"  Mean: {bets['kelly_size'].mean():.3f}")
        logger.info(f"  Median: {bets['kelly_size'].median():.3f}")
        logger.info(f"\nConfidence:")
        logger.info(f"  Mean confidence score: {bets['confidence_score'].mean():.2f}")
        logger.info(f"  Mean CV: {bets['cv'].mean():.2%}")
        logger.info(f"\nProbabilities:")
        logger.info(f"  Mean win probability: {bets['prob_win'].mean():.1%}")
        logger.info(f"  Min win probability: {bets['prob_win'].min():.1%}")
        logger.info(f"{'='*60}\n")

    def export_bets(self, bets: pd.DataFrame, filename: str):
        """
        Export bets to CSV and add to ledger

        Args:
            bets: Formatted bets DataFrame
            filename: Output filename
        """
        if bets.empty:
            logger.warning("No bets to export")
            return

        # Full path
        output_path = BETS_DIR / filename

        bets.to_csv(output_path, index=False)
        logger.info(f"Exported {len(bets)} bets to {output_path}")

        # Add to ledger for tracking
        try:
            add_bets_to_ledger(bets)
            logger.info(f"Added {len(bets)} bets to ledger")
        except Exception as e:
            logger.warning(f"Failed to add bets to ledger: {e}")

    def process_predictions(self, predictions: pd.DataFrame,
                          export: bool = True,
                          filename: Optional[str] = None) -> pd.DataFrame:
        """
        Full pipeline: predictions → decisions → bets → analysis

        Args:
            predictions: Predictions DataFrame
            export: Whether to export bets
            filename: Export filename (defaults to bets_YYYY-MM-DD.csv)

        Returns:
            Formatted bets DataFrame
        """
        # Calculate decisions
        decisions = self.calculate_betting_decisions(predictions)

        if decisions.empty:
            logger.warning("No betting decisions generated")
            return pd.DataFrame()

        # Filter to actual bets
        bets = self.filter_to_bets(decisions)

        if bets.empty:
            logger.warning("No bets passed filtering criteria")
            return pd.DataFrame()

        # Format output
        formatted_bets = self.format_bet_output(bets)

        # Analyze
        self.analyze_bets(formatted_bets)

        # Export if requested
        if export:
            if filename is None:
                # Default filename with date
                if 'game_date' in formatted_bets.columns:
                    date_str = pd.to_datetime(formatted_bets['game_date'].iloc[0]).strftime('%Y-%m-%d')
                else:
                    date_str = pd.Timestamp.now().strftime('%Y-%m-%d')

                filename = f"bets_{date_str}.csv"

            self.export_bets(formatted_bets, filename)

        return formatted_bets


def generate_bets_from_predictions(predictions: pd.DataFrame,
                                  export: bool = True) -> pd.DataFrame:
    """
    Convenience function to generate bets from predictions

    Args:
        predictions: Predictions DataFrame
        export: Whether to export to CSV

    Returns:
        Formatted bets DataFrame
    """
    engine = BettingEngine()
    return engine.process_predictions(predictions, export=export)


if __name__ == "__main__":
    """Test the betting engine"""

    print("Testing BettingEngine...")
    print("-" * 60)

    # Create sample predictions
    np.random.seed(42)
    n = 20

    sample_predictions = pd.DataFrame({
        'player_name': [f'Player {i}' for i in range(n)],
        'game_date': pd.Timestamp.now() + pd.Timedelta(days=1),
        'team_abbreviation': ['LAL', 'BOS', 'GSW'] * (n // 3) + ['MIA'] * (n % 3),
        'betting_line': np.random.uniform(15, 35, n),
        'mean_pred': np.random.uniform(15, 35, n),
        'std_dev': np.random.uniform(3, 6, n),
        'prob_over': np.random.uniform(0.4, 0.7, n),
        'bookmaker': 'draftkings'
    })

    # Generate bets
    bets = generate_bets_from_predictions(sample_predictions, export=False)

    if not bets.empty:
        print(f"\nGenerated {len(bets)} bets from {len(sample_predictions)} predictions")
        print("\nSample bets:")
        print(bets.head())
    else:
        print("\nNo bets passed filtering (expected with random data)")
