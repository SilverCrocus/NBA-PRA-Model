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
    BETS_DIR
)
from production.logging_config import setup_production_logging
from production.exceptions import BettingEngineError, PredictionError
from production.ledger import add_bets_to_ledger

# Import Monte Carlo utilities
from production.monte_carlo import (
    american_odds_to_probability,
    calculate_bet_edge,
    calculate_kelly_fraction
)

logger = setup_production_logging('betting_engine')


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
            DataFrame with unified betting decisions (kelly_size, direction, edge)
            Only returns actionable bets (filtered by confidence, edge, CV)
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
        prob_under = 1 - prob_over
        betting_lines = predictions['betting_line'].values
        mean_pred = predictions['mean_pred'].values
        std_dev = predictions['std_dev'].values

        # Calculate breakeven probability
        breakeven_prob = american_odds_to_probability(BETTING_ODDS)

        # Calculate edges (use existing if provided - for testing)
        if 'edge_over' in predictions.columns:
            edge_over = predictions['edge_over'].values
        else:
            edge_over = prob_over - breakeven_prob

        if 'edge_under' in predictions.columns:
            edge_under = predictions['edge_under'].values
        else:
            edge_under = prob_under - breakeven_prob

        # Calculate Kelly sizes
        kelly_size_over = self._calculate_kelly_sizes(prob_over, BETTING_ODDS, KELLY_FRACTION, MIN_EDGE_KELLY)
        kelly_size_under = self._calculate_kelly_sizes(prob_under, BETTING_ODDS, KELLY_FRACTION, MIN_EDGE_KELLY)

        # Calculate confidence metrics (use existing if provided)
        if 'confidence_score' in predictions.columns:
            confidence_score = predictions['confidence_score'].values
        else:
            confidence_score = np.abs(prob_over - 0.5) / 0.5

        # Always calculate CV from std_dev and mean_pred (don't use pre-calculated)
        # This ensures tests can control filtering by setting std_dev and mean_pred
        cv = std_dev / (mean_pred + 1e-6)

        # Apply ALL filters: confidence, CV, edge
        confidence_mask = (confidence_score >= MIN_CONFIDENCE) & (cv <= MAX_CV)
        should_bet_over = (edge_over >= MIN_EDGE_KELLY) & confidence_mask
        should_bet_under = (edge_under >= MIN_EDGE_KELLY) & confidence_mask

        # Create intermediate DataFrame with all metrics
        intermediate = predictions.copy()
        intermediate['original_index'] = intermediate.index  # Preserve original index for test validation
        intermediate['prob_under'] = prob_under
        intermediate['edge_over'] = edge_over
        intermediate['edge_under'] = edge_under
        intermediate['kelly_size_over'] = kelly_size_over
        intermediate['kelly_size_under'] = kelly_size_under
        intermediate['confidence_score'] = confidence_score
        intermediate['cv'] = cv
        intermediate['should_bet_over'] = should_bet_over
        intermediate['should_bet_under'] = should_bet_under

        # Convert to unified format and filter to actionable bets only
        bets = []
        for iloc_idx, (idx, row) in enumerate(intermediate.iterrows()):
            # Check if OVER bet qualifies
            if row['should_bet_over'] and row['edge_over'] >= MIN_EDGE_KELLY:
                # Check if UNDER also qualifies
                if row['should_bet_under'] and row['edge_under'] >= MIN_EDGE_KELLY:
                    # Both qualify - pick the one with higher edge
                    if row['edge_over'] > row['edge_under']:
                        direction = 'OVER'
                        edge = row['edge_over']
                        kelly_size = row['kelly_size_over']
                        prob_win = row['prob_over']
                    else:
                        direction = 'UNDER'
                        edge = row['edge_under']
                        kelly_size = row['kelly_size_under']
                        prob_win = row['prob_under']
                else:
                    # Only OVER qualifies
                    direction = 'OVER'
                    edge = row['edge_over']
                    kelly_size = row['kelly_size_over']
                    prob_win = row['prob_over']
            elif row['should_bet_under'] and row['edge_under'] >= MIN_EDGE_KELLY:
                # Only UNDER qualifies
                direction = 'UNDER'
                edge = row['edge_under']
                kelly_size = row['kelly_size_under']
                prob_win = row['prob_under']
            else:
                # Neither qualifies - skip
                continue

            # Create unified bet record
            bet_record = {k: v for k, v in row.items()
                         if k not in ['should_bet_over', 'should_bet_under',
                                     'kelly_size_over', 'kelly_size_under']}
            bet_record['direction'] = direction
            bet_record['edge'] = edge
            bet_record['kelly_size'] = kelly_size
            bet_record['prob_win'] = prob_win

            bets.append(bet_record)

        result = pd.DataFrame(bets) if bets else pd.DataFrame()

        # Sort by edge (best bets first)
        if not result.empty and 'edge' in result.columns:
            result = result.sort_values('edge', ascending=False).reset_index(drop=True)

        logger.info(f"Generated {len(result)} betting decision rows")
        logger.info(f"  {should_bet_over.sum()} over bets, {should_bet_under.sum()} under bets")

        return result

    def _calculate_kelly_sizes(self, prob_win: np.ndarray, odds: float,
                              kelly_fraction: float, min_edge: float) -> np.ndarray:
        """
        Calculate Kelly criterion bet sizing.

        Args:
            prob_win: Win probabilities
            odds: American odds
            kelly_fraction: Fraction of full Kelly
            min_edge: Minimum edge required

        Returns:
            Bet sizes as fraction of bankroll
        """
        # Convert American odds to decimal payoff
        if odds < 0:
            payoff = 100 / abs(odds)
        else:
            payoff = odds / 100

        # Calculate breakeven probability
        breakeven_prob = 1 / (1 + payoff)

        # Calculate edge
        edge = prob_win - breakeven_prob

        # Full Kelly sizing
        full_kelly = (prob_win * payoff - (1 - prob_win)) / payoff

        # Apply Kelly fraction
        bet_sizes = full_kelly * kelly_fraction

        # Zero out bets with insufficient edge
        bet_sizes = np.where(edge >= min_edge, bet_sizes, 0)

        # Zero out negative bet sizes
        bet_sizes = np.maximum(bet_sizes, 0)

        # Cap at kelly_fraction
        bet_sizes = np.minimum(bet_sizes, kelly_fraction)

        return bet_sizes

    def filter_to_bets(self, decisions: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to actual bets (passing all criteria)

        Args:
            decisions: Betting decisions from calculate_betting_decisions
                      (already filtered and in unified format)

        Returns:
            DataFrame with only actionable bets (same as input since filtering already done)
        """
        # calculate_betting_decisions already does filtering and formatting
        # This method exists for backwards compatibility
        return decisions

    def format_bet_output(self, bets: pd.DataFrame) -> pd.DataFrame:
        """
        Format bets for output

        Args:
            bets: Bets DataFrame from filter_to_bets (already has direction, kelly_size, etc.)

        Returns:
            Formatted DataFrame for export
        """
        if bets.empty:
            return pd.DataFrame()

        # Bets already have unified direction, kelly_size, edge, prob_win from format_betting_decisions
        # Just select relevant columns for output
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
        optional_cols = ['bookmaker', 'opponent', 'home_game']
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
