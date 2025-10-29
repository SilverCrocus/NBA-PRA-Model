"""
Backtest Reporting and Visualization

Generates summary reports and analysis of backtest results.

Author: NBA PRA Prediction System
Date: 2025-10-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import logging

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backtest.config import (
    BACKTEST_REPORT_PATH, DAILY_PREDICTIONS_PATH,
    DAILY_METRICS_PATH, PLAYER_ANALYSIS_PATH, BETTING_PERFORMANCE_PATH,
    BREAK_EVEN_WIN_RATE, TARGET_WIN_RATE
)
from backtest.betting_evaluator import (
    evaluate_betting_performance
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def calculate_overall_metrics(predictions_df: pd.DataFrame) -> Dict:
    """
    Calculate overall prediction accuracy metrics

    Args:
        predictions_df: DataFrame with predictions and actuals

    Returns:
        Dictionary with metrics
    """
    mae = mean_absolute_error(predictions_df['actual_pra'], predictions_df['prediction'])
    rmse = np.sqrt(mean_squared_error(predictions_df['actual_pra'], predictions_df['prediction']))
    r2 = r2_score(predictions_df['actual_pra'], predictions_df['prediction'])

    # Calculate by has_ctg_data if available
    if 'has_ctg_data' in predictions_df.columns:
        with_ctg = predictions_df[predictions_df['has_ctg_data'] == 1]
        without_ctg = predictions_df[predictions_df['has_ctg_data'] == 0]

        mae_with_ctg = mean_absolute_error(with_ctg['actual_pra'], with_ctg['prediction']) if len(with_ctg) > 0 else np.nan
        mae_without_ctg = mean_absolute_error(without_ctg['actual_pra'], without_ctg['prediction']) if len(without_ctg) > 0 else np.nan
    else:
        mae_with_ctg, mae_without_ctg = np.nan, np.nan

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mae_with_ctg': mae_with_ctg,
        'mae_without_ctg': mae_without_ctg,
        'num_predictions': len(predictions_df)
    }

    return metrics


def generate_markdown_report(predictions_df: pd.DataFrame,
                            overall_metrics: Dict,
                            betting_metrics: Dict,
                            player_analysis: pd.DataFrame) -> str:
    """
    Generate markdown report summarizing backtest results

    Args:
        predictions_df: All predictions
        overall_metrics: Overall prediction metrics
        betting_metrics: Betting performance metrics
        player_analysis: Per-player analysis

    Returns:
        Markdown-formatted report string
    """
    report = []

    report.append("# NBA PRA Walk-Forward Backtest Report\n")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append("---\n\n")

    # Executive Summary
    report.append("## Executive Summary\n\n")

    if betting_metrics:
        is_profitable = betting_metrics.get('is_profitable', False)
        profit_status = "✅ **PROFITABLE**" if is_profitable else "❌ **NOT PROFITABLE**"

        report.append(f"**Status:** {profit_status}\n\n")

        report.append(f"- **Total Predictions:** {overall_metrics['num_predictions']:,}\n")
        report.append(f"- **Prediction MAE:** {overall_metrics['mae']:.3f} PRA points\n")
        report.append(f"- **Total Bets (with lines):** {betting_metrics['total_bets']:,}\n")
        report.append(f"- **Win Rate:** {betting_metrics['win_rate']:.1%} (Break-even: {BREAK_EVEN_WIN_RATE:.1%})\n")
        report.append(f"- **ROI:** {betting_metrics['roi']:.2f}%\n")
        report.append(f"- **Total Profit:** {betting_metrics['total_profit']:.2f} units\n\n")

    # Prediction Accuracy
    report.append("## Prediction Accuracy\n\n")
    report.append("### Overall Performance\n\n")
    report.append(f"| Metric | Value |\n")
    report.append(f"|--------|-------|\n")
    report.append(f"| **MAE** | {overall_metrics['mae']:.3f} |\n")
    report.append(f"| **RMSE** | {overall_metrics['rmse']:.3f} |\n")
    report.append(f"| **R²** | {overall_metrics['r2']:.4f} |\n")
    report.append(f"| **Predictions** | {overall_metrics['num_predictions']:,} |\n\n")

    if not np.isnan(overall_metrics['mae_with_ctg']):
        report.append("### Performance by CTG Data Availability\n\n")
        report.append(f"| Segment | MAE | Impact |\n")
        report.append(f"|---------|-----|--------|\n")
        report.append(f"| **With CTG** | {overall_metrics['mae_with_ctg']:.3f} | Baseline |\n")
        report.append(f"| **Without CTG** | {overall_metrics['mae_without_ctg']:.3f} | +{overall_metrics['mae_without_ctg'] - overall_metrics['mae_with_ctg']:.3f} |\n\n")

    # Betting Performance
    if betting_metrics:
        report.append("## Betting Performance\n\n")
        report.append("### Overall Results\n\n")
        report.append(f"| Metric | Value |\n")
        report.append(f"|--------|-------|\n")
        report.append(f"| **Total Bets** | {betting_metrics['total_bets']:,} |\n")
        report.append(f"| **Wins** | {betting_metrics['wins']:,} |\n")
        report.append(f"| **Losses** | {betting_metrics['losses']:,} |\n")
        report.append(f"| **Win Rate** | {betting_metrics['win_rate']:.2%} |\n")
        report.append(f"| **Break-Even Win Rate** | {BREAK_EVEN_WIN_RATE:.2%} |\n")
        report.append(f"| **Win Rate vs Break-Even** | {betting_metrics['win_rate_vs_breakeven']:+.2f}% |\n")
        report.append(f"| **ROI** | {betting_metrics['roi']:.2f}% |\n")
        report.append(f"| **Total Profit** | {betting_metrics['total_profit']:.2f} units |\n")
        report.append(f"| **Avg Profit per Bet** | {betting_metrics['avg_profit_per_bet']:.3f} units |\n\n")

        # Edge Analysis
        report.append("### Edge Analysis\n\n")
        report.append("Performance when model prediction differs significantly from betting line:\n\n")

        if betting_metrics.get('edge_medium'):
            edge = betting_metrics['edge_medium']
            report.append(f"**Medium Edge (2+ points difference):**\n")
            report.append(f"- Bets: {edge['edge_count']:,}\n")
            report.append(f"- Win Rate: {edge['edge_win_rate']:.2%}\n")
            report.append(f"- ROI: {edge['edge_roi']:.2f}%\n\n")

        if betting_metrics.get('edge_large'):
            edge = betting_metrics['edge_large']
            report.append(f"**Large Edge (3+ points difference):**\n")
            report.append(f"- Bets: {edge['edge_count']:,}\n")
            report.append(f"- Win Rate: {edge['edge_win_rate']:.2%}\n")
            report.append(f"- ROI: {edge['edge_roi']:.2f}%\n\n")

    # Player Analysis
    if not player_analysis.empty:
        report.append("## Player Analysis\n\n")

        report.append("### Top 10 Most Profitable Players\n\n")
        top_10 = player_analysis.nlargest(10, 'roi')
        report.append(f"| Player | Games | Win Rate | ROI | Total Profit |\n")
        report.append(f"|--------|-------|----------|-----|-------------|\n")
        for _, row in top_10.iterrows():
            report.append(f"| {row['player_name']} | {row['num_games']} | {row['win_rate']:.1%} | {row['roi']:.1f}% | {row['total_profit']:.2f} |\n")
        report.append("\n")

        report.append("### Bottom 10 Least Profitable Players\n\n")
        bottom_10 = player_analysis.nsmallest(10, 'roi')
        report.append(f"| Player | Games | Win Rate | ROI | Total Profit |\n")
        report.append(f"|--------|-------|----------|-----|-------------|\n")
        for _, row in bottom_10.iterrows():
            report.append(f"| {row['player_name']} | {row['num_games']} | {row['win_rate']:.1%} | {row['roi']:.1f}% | {row['total_profit']:.2f} |\n")
        report.append("\n")

    # Recommendations
    report.append("## Recommendations\n\n")

    if betting_metrics:
        if betting_metrics['is_profitable']:
            report.append("✅ **Model is profitable at standard -110 odds**\n\n")
            report.append("Recommendations:\n")
            report.append(f"- Continue with current strategy\n")
            report.append(f"- Consider increasing bet sizing on high-edge opportunities (3+ point edges)\n")
            report.append(f"- Monitor performance weekly to detect any degradation\n")
        else:
            report.append("❌ **Model is not yet profitable**\n\n")
            report.append("Recommendations:\n")
            report.append(f"- Only bet on large-edge opportunities (3+ point difference from line)\n")
            report.append(f"- Consider additional feature engineering to improve win rate\n")
            report.append(f"- Focus on top-performing players identified in analysis\n")
            report.append(f"- Need {((BREAK_EVEN_WIN_RATE - betting_metrics['win_rate']) * betting_metrics['total_bets']):.0f} more wins to reach break-even\n")

    report.append("\n---\n")
    report.append(f"\n*Report generated from {overall_metrics['num_predictions']:,} predictions across 2024-25 NBA season*\n")

    return "".join(report)


def save_summary_files(predictions_df: pd.DataFrame) -> None:
    """
    Save summary CSV files for further analysis

    Args:
        predictions_df: All predictions and results
    """
    logger.info("Generating summary files...")

    # Overall metrics
    overall_metrics = calculate_overall_metrics(predictions_df)

    # Betting performance
    betting_metrics, player_analysis, daily_metrics = evaluate_betting_performance(predictions_df)

    # Save player analysis
    if not player_analysis.empty:
        player_analysis.to_csv(PLAYER_ANALYSIS_PATH, index=False)
        logger.info(f"  Saved player analysis to {PLAYER_ANALYSIS_PATH}")

    # Save daily metrics
    if not daily_metrics.empty:
        daily_metrics.to_csv(DAILY_METRICS_PATH, index=False)
        logger.info(f"  Saved daily metrics to {DAILY_METRICS_PATH}")

    # Save betting performance summary
    if betting_metrics:
        betting_summary = pd.DataFrame([betting_metrics])
        betting_summary.to_csv(BETTING_PERFORMANCE_PATH, index=False)
        logger.info(f"  Saved betting summary to {BETTING_PERFORMANCE_PATH}")

    # Generate and save markdown report
    report_text = generate_markdown_report(
        predictions_df,
        overall_metrics,
        betting_metrics,
        player_analysis
    )

    with open(BACKTEST_REPORT_PATH, 'w') as f:
        f.write(report_text)

    logger.info(f"  Saved markdown report to {BACKTEST_REPORT_PATH}")

    logger.info("\n✓ All summary files generated")


def generate_backtest_report(predictions_df: pd.DataFrame) -> None:
    """
    Main entry point for report generation

    Args:
        predictions_df: All predictions and results from backtest
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING BACKTEST REPORT")
    logger.info("="*60 + "\n")

    save_summary_files(predictions_df)

    logger.info(f"\nReport saved to: {BACKTEST_REPORT_PATH}")
    logger.info("Open this file to view detailed analysis\n")
