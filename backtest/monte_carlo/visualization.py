"""
Visualization Module for Monte Carlo Calibration

This module provides visualization tools for assessing calibration quality:
- Calibration curves (reliability diagrams)
- PIT histograms (probability integral transform)
- Edge calibration plots (edge vs win rate)
- Prediction interval coverage plots

These visualizations are CRITICAL for validating that your Monte Carlo
system is trustworthy before using it for real betting decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_calibration_curve(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).

    A calibration curve shows predicted probabilities vs actual frequencies.
    A perfectly calibrated model lies on the diagonal line.

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters
        n_bins: Number of bins for grouping predictions
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Test multiple thresholds
    thresholds = np.percentile(y_true, [25, 50, 75])
    colors = ['blue', 'green', 'red']

    for threshold, color in zip(thresholds, colors):
        # Calculate predicted probabilities
        prob_over = 1 - stats.gamma.cdf(threshold, a=alpha, scale=1/beta)

        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_freqs = []

        for i in range(n_bins):
            mask = (prob_over >= bin_edges[i]) & (prob_over < bin_edges[i+1])

            if mask.sum() >= 5:  # Need at least 5 samples
                bin_centers.append(prob_over[mask].mean())
                bin_freqs.append((y_true[mask] > threshold).mean())

        # Plot
        ax.scatter(bin_centers, bin_freqs, s=100, alpha=0.6, color=color,
                  label=f'Threshold = {threshold:.1f}')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')

    # Formatting
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Frequency', fontsize=12)
    ax.set_title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Add calibration zones
    ax.fill_between([0, 1], [0, 1.05], [0, 0.95], alpha=0.1, color='green',
                    label='Well-calibrated (±5%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration curve saved to {save_path}")

    return fig


def plot_pit_histogram(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    n_bins: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot PIT (Probability Integral Transform) histogram.

    PIT values should be uniformly distributed for a well-calibrated model.
    - Flat histogram = well-calibrated
    - U-shaped = overconfident (intervals too narrow)
    - Inverted-U = underconfident (intervals too wide)
    - Skewed = biased predictions

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters
        n_bins: Number of histogram bins
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate PIT values
    pit_values = stats.gamma.cdf(y_true, a=alpha, scale=1/beta)

    # Plot histogram
    ax.hist(pit_values, bins=n_bins, range=(0, 1), density=True,
           alpha=0.7, color='steelblue', edgecolor='black')

    # Perfect calibration line (uniform distribution)
    ax.axhline(1.0, color='red', linestyle='--', lw=2, label='Perfect calibration (uniform)')

    # Formatting
    ax.set_xlabel('PIT Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('PIT Histogram - Calibration Diagnostic', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    mean_pit = pit_values.mean()
    if abs(mean_pit - 0.5) > 0.1:
        interpretation = f"⚠ Biased (mean={mean_pit:.2f}, expect 0.5)"
        color = 'red'
    else:
        interpretation = f"✓ Unbiased (mean={mean_pit:.2f})"
        color = 'green'

    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"PIT histogram saved to {save_path}")

    return fig


def plot_coverage_analysis(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot prediction interval coverage analysis.

    Shows whether X% prediction intervals actually contain X% of observations.

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Test confidence levels
    confidence_levels = np.arange(0.50, 1.0, 0.05)
    actual_coverage = []

    for conf_level in confidence_levels:
        # Calculate prediction intervals
        tail_prob = (1 - conf_level) / 2
        lower = stats.gamma.ppf(tail_prob, a=alpha, scale=1/beta)
        upper = stats.gamma.ppf(1 - tail_prob, a=alpha, scale=1/beta)

        # Check coverage
        covered = (y_true >= lower) & (y_true <= upper)
        actual_coverage.append(covered.mean())

    actual_coverage = np.array(actual_coverage)

    # Plot 1: Coverage vs Nominal
    ax1.plot(confidence_levels, actual_coverage, 'o-', markersize=8, linewidth=2,
            color='steelblue', label='Actual coverage')
    ax1.plot([0.5, 1.0], [0.5, 1.0], 'r--', linewidth=2, label='Perfect calibration')

    # Add error bands
    ax1.fill_between(confidence_levels,
                     confidence_levels - 0.03,
                     confidence_levels + 0.03,
                     alpha=0.2, color='green', label='±3% tolerance')

    ax1.set_xlabel('Nominal Coverage', fontsize=12)
    ax1.set_ylabel('Actual Coverage', fontsize=12)
    ax1.set_title('Prediction Interval Coverage', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coverage Error
    coverage_error = actual_coverage - confidence_levels
    ax2.plot(confidence_levels, coverage_error * 100, 'o-', markersize=8,
            linewidth=2, color='darkred')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(confidence_levels, -3, 3, alpha=0.2, color='green')

    ax2.set_xlabel('Nominal Coverage', fontsize=12)
    ax2.set_ylabel('Coverage Error (%)', fontsize=12)
    ax2.set_title('Coverage Calibration Error', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Coverage analysis saved to {save_path}")

    return fig


def plot_edge_calibration(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    betting_lines: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot edge calibration (edge vs actual win rate).

    This is CRITICAL for betting - verifies that larger predicted edges
    correspond to higher actual win rates. If not monotonic, the model
    is not trustworthy for bet sizing.

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters
        betting_lines: Betting thresholds
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate probabilities and edges
    prob_over = 1 - stats.gamma.cdf(betting_lines, a=alpha, scale=1/beta)
    edge = prob_over - 0.524  # vs -110 odds

    # Actual outcomes
    actual_over = y_true > betting_lines

    # Bin by edge size
    edge_bins = np.percentile(edge[edge > 0], [0, 20, 40, 60, 80, 100])
    edge_centers = []
    win_rates = []
    n_bets_per_bin = []

    for i in range(len(edge_bins) - 1):
        mask = (edge >= edge_bins[i]) & (edge < edge_bins[i+1])

        if mask.sum() >= 10:
            edge_centers.append(edge[mask].mean())
            win_rates.append(actual_over[mask].mean())
            n_bets_per_bin.append(mask.sum())

    # Plot actual win rate vs edge
    sizes = np.array(n_bets_per_bin) * 5  # Scale for visualization
    ax.scatter(edge_centers, win_rates, s=sizes, alpha=0.6, color='steelblue')

    # Add labels showing number of bets
    for x, y, n in zip(edge_centers, win_rates, n_bets_per_bin):
        ax.annotate(f'n={n}', (x, y), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9)

    # Expected win rate line
    edge_range = np.linspace(0, max(edge_centers), 100)
    expected_win_rate = 0.524 + edge_range
    ax.plot(edge_range, expected_win_rate, 'r--', linewidth=2,
           label='Expected (52.4% + edge)')

    # Breakeven line
    ax.axhline(0.524, color='black', linestyle=':', linewidth=1,
              label='Breakeven (52.4% for -110 odds)')

    # Formatting
    ax.set_xlabel('Model Edge (predicted prob - breakeven)', fontsize=12)
    ax.set_ylabel('Actual Win Rate', fontsize=12)
    ax.set_title('Edge Calibration - Model Edge vs Actual Win Rate', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Check monotonicity
    is_monotonic = all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1))
    status = "✓ MONOTONIC" if is_monotonic else "✗ NOT MONOTONIC"
    color = 'green' if is_monotonic else 'red'

    ax.text(0.98, 0.02, f"Calibration: {status}",
           transform=ax.transAxes, fontsize=12, verticalalignment='bottom',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Edge calibration plot saved to {save_path}")

    return fig


def plot_variance_analysis(
    y_true: np.ndarray,
    mean_pred: np.ndarray,
    var_pred: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Analyze variance predictions vs actual squared residuals.

    Validates that the variance model is capturing heteroskedasticity correctly.

    Args:
        y_true: True PRA values
        mean_pred: Mean predictions
        var_pred: Variance predictions
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Calculate actual variance (squared residuals)
    residuals = y_true - mean_pred
    actual_var = residuals ** 2

    # Plot 1: Predicted vs Actual Variance
    ax1.scatter(var_pred, actual_var, alpha=0.3, s=10)
    ax1.plot([0, max(var_pred)], [0, max(var_pred)], 'r--', linewidth=2)
    ax1.set_xlabel('Predicted Variance', fontsize=11)
    ax1.set_ylabel('Actual Squared Residual', fontsize=11)
    ax1.set_title('Variance Prediction Accuracy', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Correlation
    corr = np.corrcoef(var_pred, actual_var)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}',
            transform=ax1.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: Residuals vs Predicted Mean (check heteroskedasticity)
    ax2.scatter(mean_pred, np.abs(residuals), alpha=0.3, s=10)
    ax2.set_xlabel('Predicted Mean', fontsize=11)
    ax2.set_ylabel('Absolute Residual', fontsize=11)
    ax2.set_title('Heteroskedasticity Check', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Distribution of Standardized Residuals
    std_residuals = residuals / np.sqrt(var_pred)
    ax3.hist(std_residuals, bins=50, density=True, alpha=0.7, color='steelblue')

    # Overlay standard normal
    x = np.linspace(-4, 4, 100)
    ax3.plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='Standard Normal')
    ax3.set_xlabel('Standardized Residual', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Standardized Residuals (should be N(0,1))', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Q-Q Plot
    stats.probplot(std_residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Variance analysis saved to {save_path}")

    return fig


def create_calibration_report(
    y_true: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    betting_lines: np.ndarray,
    mean_pred: np.ndarray,
    var_pred: np.ndarray,
    output_dir: str = 'backtest/monte_carlo/visuals'
) -> Dict[str, Any]:
    """
    Create comprehensive calibration report with all visualizations.

    Generates and saves:
    - Calibration curve
    - PIT histogram
    - Coverage analysis
    - Edge calibration
    - Variance analysis

    Args:
        y_true: True PRA values
        alpha: Shape parameters
        beta: Rate parameters
        betting_lines: Betting thresholds
        mean_pred: Mean predictions
        var_pred: Variance predictions
        output_dir: Directory to save plots

    Returns:
        Dictionary with paths to saved figures and summary metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating calibration report in {output_dir}")

    # Generate all plots
    figures = {}

    # 1. Calibration curve
    fig1 = plot_calibration_curve(y_true, alpha, beta,
                                  save_path=str(output_path / 'calibration_curve.png'))
    figures['calibration_curve'] = fig1
    plt.close(fig1)

    # 2. PIT histogram
    fig2 = plot_pit_histogram(y_true, alpha, beta,
                              save_path=str(output_path / 'pit_histogram.png'))
    figures['pit_histogram'] = fig2
    plt.close(fig2)

    # 3. Coverage analysis
    fig3 = plot_coverage_analysis(y_true, alpha, beta,
                                  save_path=str(output_path / 'coverage_analysis.png'))
    figures['coverage_analysis'] = fig3
    plt.close(fig3)

    # 4. Edge calibration
    fig4 = plot_edge_calibration(y_true, alpha, beta, betting_lines,
                                 save_path=str(output_path / 'edge_calibration.png'))
    figures['edge_calibration'] = fig4
    plt.close(fig4)

    # 5. Variance analysis
    fig5 = plot_variance_analysis(y_true, mean_pred, var_pred,
                                  save_path=str(output_path / 'variance_analysis.png'))
    figures['variance_analysis'] = fig5
    plt.close(fig5)

    logger.info(f"✓ Calibration report complete. {len(figures)} plots saved to {output_dir}")

    return {
        'output_dir': str(output_path),
        'figures': list(figures.keys()),
        'n_samples': len(y_true)
    }
