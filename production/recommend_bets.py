"""
Bet Recommendation Script

⚠️ DEPRECATED: This script is deprecated as of v2.0.0 and will be removed in v3.0.0

Use the new unified CLI instead:
    OLD: PYTHONPATH=/path/to/NBA_PRA uv run python production/recommend_bets.py
    NEW: nba-pra recommend

Migration guide:
    python production/recommend_bets.py --min-edge 0.05
    →  nba-pra recommend --min-edge 0.05

    python production/recommend_bets.py --min-confidence 0.7
    →  nba-pra recommend --min-confidence 0.7

    python production/recommend_bets.py --top-n 5
    →  nba-pra recommend --top-n 5

For more information, see production/README.md or docs/production_architecture.md

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import pandas as pd
import warnings

# Show deprecation warning
warnings.warn(
    "\n\n"
    "=" * 80 + "\n"
    "⚠️  DEPRECATION WARNING\n"
    "=" * 80 + "\n"
    "This script (recommend_bets.py) is deprecated and will be removed in v3.0.0\n\n"
    "Please use the new unified CLI instead:\n"
    "  OLD: PYTHONPATH=/path uv run python production/recommend_bets.py\n"
    "  NEW: nba-pra recommend\n\n"
    "See production/README.md for migration guide.\n"
    "=" * 80 + "\n",
    DeprecationWarning,
    stacklevel=2
)
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# Paths
PREDICTIONS_DIR = Path(__file__).parent / "outputs" / "predictions"


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Recommend best bets from predictions')

    parser.add_argument(
        '--date',
        type=str,
        default=None,
        help='Date of predictions (YYYY-MM-DD), defaults to latest file'
    )

    parser.add_argument(
        '--min-edge',
        type=float,
        default=0.05,
        help='Minimum edge required (default: 0.05 = 5%%)'
    )

    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.70,
        help='Minimum confidence score (default: 0.70)'
    )

    parser.add_argument(
        '--max-cv',
        type=float,
        default=0.30,
        help='Maximum coefficient of variation (default: 0.30 = 30%%)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top bets to show (default: 10)'
    )

    return parser.parse_args()


def find_latest_predictions(date: Optional[str] = None) -> Path:
    """Find predictions file for given date or latest"""
    if date:
        filename = f"predictions_{date}.csv"
        filepath = PREDICTIONS_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Predictions not found for {date}")
        return filepath

    # Find latest file
    files = list(PREDICTIONS_DIR.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError("No prediction files found")

    latest = max(files, key=lambda p: p.stat().st_mtime)
    return latest


def load_predictions(filepath: Path) -> pd.DataFrame:
    """Load predictions CSV"""
    print(f"📂 Loading predictions from: {filepath.name}")
    df = pd.read_csv(filepath)
    print(f"   Found {len(df)} player predictions\n")
    return df


def analyze_bets(df: pd.DataFrame, min_edge: float, min_confidence: float, max_cv: float) -> pd.DataFrame:
    """
    Analyze and filter bets based on criteria

    Returns:
        DataFrame with recommended bets
    """
    # Determine direction and select best edge
    df['direction'] = df.apply(
        lambda row: 'OVER' if row['edge_over'] > row['edge_under'] else 'UNDER',
        axis=1
    )

    df['edge'] = df.apply(
        lambda row: row['edge_over'] if row['direction'] == 'OVER' else row['edge_under'],
        axis=1
    )

    df['win_prob'] = df.apply(
        lambda row: row['prob_over'] if row['direction'] == 'OVER' else row['prob_under'],
        axis=1
    )

    # Apply filters
    filtered = df[
        (df['edge'] >= min_edge) &
        (df['confidence_score'] >= min_confidence) &
        (df['cv'] <= max_cv)
    ].copy()

    # Sort by edge (best first)
    filtered = filtered.sort_values('edge', ascending=False)

    return filtered


def classify_bet_strength(edge: float, confidence: float) -> str:
    """Classify bet as STRONG, GOOD, or OK"""
    if edge >= 0.15 and confidence >= 0.85:
        return "🔥 STRONG"
    elif edge >= 0.10 and confidence >= 0.75:
        return "✅ GOOD"
    else:
        return "⚠️  OK"


def display_recommendations(bets: pd.DataFrame, top_n: int):
    """Display betting recommendations in formatted output"""

    if bets.empty:
        print("❌ No bets passed the filtering criteria")
        print("\nTry lowering thresholds:")
        print("  --min-edge 0.03")
        print("  --min-confidence 0.60")
        print("  --max-cv 0.35")
        return

    print(f"✅ Found {len(bets)} qualifying bets")
    print(f"📊 Showing top {min(top_n, len(bets))} recommendations\n")
    print("=" * 100)

    for idx, (_, bet) in enumerate(bets.head(top_n).iterrows(), 1):
        strength = classify_bet_strength(bet['edge'], bet['confidence_score'])

        print(f"\n{strength} #{idx}: {bet['player_name']} - {bet['direction']}")
        print("-" * 100)

        # Key metrics
        print(f"  📍 Line:        {bet['pra_line']:.1f} PRA @ {bet['pra_odds']:.0f} odds ({bet['bookmaker']})")
        print(f"  🎯 Prediction:  {bet['mean_pred']:.1f} ± {bet['std_dev']:.1f} PRA")
        print(f"  💰 Edge:        {bet['edge']:.1%} (need {bet['breakeven_prob']:.1%}, have {bet['win_prob']:.1%})")
        print(f"  📊 Confidence:  {bet['confidence_score']:.3f} (Volatility: {bet['cv']:.1%})")

        # Analysis
        diff = abs(bet['mean_pred'] - bet['pra_line'])
        print(f"\n  Analysis:")
        print(f"    • Prediction vs Line: {diff:.1f} points {'below' if bet['direction'] == 'UNDER' else 'above'}")
        print(f"    • Win Probability: {bet['win_prob']:.1%}")
        print(f"    • Breakeven: {bet['breakeven_prob']:.1%}")
        print(f"    • Edge: {bet['edge']:.1%}")

        # Recommendation
        if bet['edge'] >= 0.15:
            print(f"    ⚡ STRONG BET - Huge edge with high confidence")
        elif bet['edge'] >= 0.10:
            print(f"    ✅ SOLID BET - Good edge and reasonable confidence")
        else:
            print(f"    ⚠️  MARGINAL - Meets minimum thresholds")

        print()

    print("=" * 100)

    # Summary stats
    print(f"\n📈 Portfolio Summary:")
    print(f"  Total Bets:       {len(bets)}")
    print(f"  Over Bets:        {(bets['direction'] == 'OVER').sum()}")
    print(f"  Under Bets:       {(bets['direction'] == 'UNDER').sum()}")
    print(f"  Avg Edge:         {bets['edge'].mean():.1%}")
    print(f"  Median Edge:      {bets['edge'].median():.1%}")
    print(f"  Avg Confidence:   {bets['confidence_score'].mean():.3f}")
    print(f"  Avg Win Prob:     {bets['win_prob'].mean():.1%}")

    # Risk categories
    strong = bets[bets['edge'] >= 0.15]
    good = bets[(bets['edge'] >= 0.10) & (bets['edge'] < 0.15)]
    ok = bets[bets['edge'] < 0.10]

    print(f"\n🎯 By Strength:")
    print(f"  🔥 STRONG:        {len(strong)} bets (edge ≥ 15%)")
    print(f"  ✅ GOOD:          {len(good)} bets (edge 10-15%)")
    print(f"  ⚠️  OK:            {len(ok)} bets (edge 5-10%)")


def export_bets(bets: pd.DataFrame, output_path: Path):
    """Export recommended bets to CSV"""
    if bets.empty:
        return

    # Select relevant columns
    export_cols = [
        'player_name', 'direction', 'pra_line', 'pra_odds', 'bookmaker',
        'mean_pred', 'std_dev', 'edge', 'win_prob', 'confidence_score', 'cv'
    ]

    export_df = bets[export_cols].copy()
    export_df.to_csv(output_path, index=False)
    print(f"\n💾 Exported bets to: {output_path}")


def main():
    """Main execution"""
    args = parse_arguments()

    print("\n" + "=" * 100)
    print("🎰 NBA PRA BET RECOMMENDER")
    print("=" * 100)
    print()

    # Display filters
    print(f"⚙️  Filters:")
    print(f"  Minimum Edge:         {args.min_edge:.1%}")
    print(f"  Minimum Confidence:   {args.min_confidence:.2f}")
    print(f"  Maximum CV:           {args.max_cv:.1%}")
    print()

    try:
        # Load predictions
        filepath = find_latest_predictions(args.date)
        df = load_predictions(filepath)

        # Analyze bets
        bets = analyze_bets(df, args.min_edge, args.min_confidence, args.max_cv)

        # Display recommendations
        display_recommendations(bets, args.top_n)

        # Export
        if not bets.empty:
            date_str = filepath.stem.replace('predictions_', '')
            output_path = PREDICTIONS_DIR.parent / "bets" / f"recommended_bets_{date_str}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_bets(bets, output_path)

        print()
        print("=" * 100)
        print()

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
