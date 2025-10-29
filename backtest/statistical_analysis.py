"""
Comprehensive Statistical Analysis of NBA PRA Backtest Results

This script performs rigorous statistical testing to evaluate:
1. Statistical significance of win rate and ROI
2. Risk of overfitting and selection bias
3. Variance analysis and confidence intervals
4. Sustainability assessment
5. Edge analysis validation
6. Player-level distribution patterns
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import binom, norm, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options
pd.options.display.float_format = '{:.4f}'.format
sns.set_style("whitegrid")

# Load data
print("=" * 80)
print("NBA PRA BACKTEST - COMPREHENSIVE STATISTICAL ANALYSIS")
print("=" * 80)
print()

results_dir = Path("/Users/diyagamah/Documents/NBA_PRA/backtest/results")

# Load all datasets
betting_perf = pd.read_csv(results_dir / "betting_performance.csv")
player_analysis = pd.read_csv(results_dir / "player_analysis.csv")
daily_predictions = pd.read_csv(results_dir / "daily_predictions.csv")
daily_metrics = pd.read_csv(results_dir / "daily_metrics.csv")

# Extract key metrics
total_bets = betting_perf['total_bets'].iloc[0]
wins = betting_perf['wins'].iloc[0]
losses = betting_perf['losses'].iloc[0]
win_rate = betting_perf['win_rate'].iloc[0]
roi = betting_perf['roi'].iloc[0] / 100  # Convert to decimal
total_profit = betting_perf['total_profit'].iloc[0]
breakeven_rate = betting_perf['break_even_win_rate'].iloc[0]

print(f"Dataset Overview:")
print(f"  Total Bets: {total_bets:,}")
print(f"  Wins: {wins:,}")
print(f"  Losses: {losses:,}")
print(f"  Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
print(f"  Break-even Rate: {breakeven_rate:.4f} ({breakeven_rate*100:.2f}%)")
print(f"  ROI: {roi:.4f} ({roi*100:.2f}%)")
print(f"  Total Profit: {total_profit:.2f} units")
print()

# =============================================================================
# 1. STATISTICAL SIGNIFICANCE TESTING
# =============================================================================
print("=" * 80)
print("1. STATISTICAL SIGNIFICANCE TESTING")
print("=" * 80)
print()

# 1.1 Binomial Test for Win Rate
print("1.1 Binomial Test: Win Rate vs Break-Even")
print("-" * 60)

# Under null hypothesis: true win rate = break-even rate (0.524)
# Alternative hypothesis: true win rate > break-even rate
p_value_binomial = binom.sf(wins - 1, total_bets, breakeven_rate)

print(f"  H0: True win rate = {breakeven_rate:.4f} (break-even)")
print(f"  H1: True win rate > {breakeven_rate:.4f}")
print(f"  Observed wins: {wins} out of {total_bets}")
print(f"  Expected wins (under H0): {breakeven_rate * total_bets:.1f}")
print(f"  p-value: {p_value_binomial:.2e}")

if p_value_binomial < 0.001:
    print(f"  Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
    print(f"  The win rate is statistically different from break-even.")
elif p_value_binomial < 0.01:
    print(f"  Result: VERY SIGNIFICANT (p < 0.01) **")
elif p_value_binomial < 0.05:
    print(f"  Result: SIGNIFICANT (p < 0.05) *")
else:
    print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")

# Calculate z-score for win rate
z_score = (win_rate - breakeven_rate) / np.sqrt(breakeven_rate * (1 - breakeven_rate) / total_bets)
print(f"  Z-score: {z_score:.4f}")
print()

# 1.2 Confidence Intervals for Win Rate
print("1.2 Confidence Intervals for Win Rate")
print("-" * 60)

# Wilson score interval (more accurate for proportions)
def wilson_ci(successes, n, confidence=0.95):
    """Calculate Wilson score confidence interval for a proportion"""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n

    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator

    return center - margin, center + margin

ci_90 = wilson_ci(wins, total_bets, 0.90)
ci_95 = wilson_ci(wins, total_bets, 0.95)
ci_99 = wilson_ci(wins, total_bets, 0.99)

print(f"  Observed Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
print(f"  90% Confidence Interval: [{ci_90[0]:.4f}, {ci_90[1]:.4f}] ({ci_90[0]*100:.2f}%, {ci_90[1]*100:.2f}%)")
print(f"  95% Confidence Interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}] ({ci_95[0]*100:.2f}%, {ci_95[1]*100:.2f}%)")
print(f"  99% Confidence Interval: [{ci_99[0]:.4f}, {ci_99[1]:.4f}] ({ci_99[0]*100:.2f}%, {ci_99[1]*100:.2f}%)")
print(f"  Break-even Rate: {breakeven_rate:.4f} ({breakeven_rate*100:.2f}%)")

if ci_95[0] > breakeven_rate:
    print(f"  Result: Lower bound of 95% CI ({ci_95[0]*100:.2f}%) EXCEEDS break-even ({breakeven_rate*100:.2f}%)")
    print(f"  This suggests a robust and statistically significant edge.")
else:
    print(f"  Result: Lower bound of 95% CI ({ci_95[0]*100:.2f}%) does not exceed break-even")
    print(f"  Caution: Win rate may not be sustainably above break-even.")
print()

# 1.3 T-test for Profit per Bet
print("1.3 One-Sample T-Test: Profit per Bet")
print("-" * 60)

# Calculate profit per bet for each bet
daily_with_bets = daily_predictions[daily_predictions['has_line'] == True].copy()
profit_per_bet = daily_with_bets['profit'].values

mean_profit = np.mean(profit_per_bet)
std_profit = np.std(profit_per_bet, ddof=1)
se_profit = std_profit / np.sqrt(len(profit_per_bet))

# One-sample t-test: H0: mean profit = 0
t_stat, p_value_ttest = stats.ttest_1samp(profit_per_bet, 0)

print(f"  H0: Mean profit per bet = 0 (no edge)")
print(f"  H1: Mean profit per bet ≠ 0")
print(f"  Sample size: {len(profit_per_bet):,}")
print(f"  Mean profit per bet: {mean_profit:.4f} units")
print(f"  Std dev: {std_profit:.4f} units")
print(f"  Standard error: {se_profit:.4f} units")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value_ttest:.2e}")

if p_value_ttest < 0.001:
    print(f"  Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
elif p_value_ttest < 0.01:
    print(f"  Result: VERY SIGNIFICANT (p < 0.01) **")
elif p_value_ttest < 0.05:
    print(f"  Result: SIGNIFICANT (p < 0.05) *")
else:
    print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")

# 95% CI for mean profit
ci_profit = stats.t.interval(0.95, len(profit_per_bet)-1, loc=mean_profit, scale=se_profit)
print(f"  95% CI for mean profit: [{ci_profit[0]:.4f}, {ci_profit[1]:.4f}] units")
print()

# =============================================================================
# 2. OVERFITTING AND SELECTION BIAS ANALYSIS
# =============================================================================
print("=" * 80)
print("2. OVERFITTING AND SELECTION BIAS ANALYSIS")
print("=" * 80)
print()

# 2.1 Temporal Stability Analysis
print("2.1 Temporal Stability: Win Rate Over Time")
print("-" * 60)

daily_metrics_sorted = daily_metrics.sort_values('game_date')

# Split into early, middle, late periods
n_days = len(daily_metrics_sorted)
period_size = n_days // 3

early = daily_metrics_sorted.iloc[:period_size]
middle = daily_metrics_sorted.iloc[period_size:2*period_size]
late = daily_metrics_sorted.iloc[2*period_size:]

def calc_period_stats(period_df):
    total_bets = period_df['num_bets'].sum()
    total_wins = period_df['wins'].sum()
    win_rate = total_wins / total_bets if total_bets > 0 else 0
    total_profit = period_df['daily_profit'].sum()
    roi = (total_profit / total_bets) if total_bets > 0 else 0
    return total_bets, total_wins, win_rate, roi, total_profit

early_stats = calc_period_stats(early)
middle_stats = calc_period_stats(middle)
late_stats = calc_period_stats(late)

print(f"  Early Period (Days 1-{period_size}):")
print(f"    Bets: {early_stats[0]}, Wins: {early_stats[1]}, Win Rate: {early_stats[2]:.4f}, ROI: {early_stats[3]:.4f}")
print(f"  Middle Period (Days {period_size+1}-{2*period_size}):")
print(f"    Bets: {middle_stats[0]}, Wins: {middle_stats[1]}, Win Rate: {middle_stats[2]:.4f}, ROI: {middle_stats[3]:.4f}")
print(f"  Late Period (Days {2*period_size+1}-{n_days}):")
print(f"    Bets: {late_stats[0]}, Wins: {late_stats[1]}, Win Rate: {late_stats[2]:.4f}, ROI: {late_stats[3]:.4f}")
print()

# Chi-square test for homogeneity of win rates across periods
contingency = np.array([
    [early_stats[1], early_stats[0] - early_stats[1]],  # wins, losses
    [middle_stats[1], middle_stats[0] - middle_stats[1]],
    [late_stats[1], late_stats[0] - late_stats[1]]
])
chi2, p_chi2, dof, expected = chi2_contingency(contingency)

print(f"  Chi-square test for temporal homogeneity:")
print(f"    H0: Win rates are equal across all periods")
print(f"    Chi-square statistic: {chi2:.4f}")
print(f"    p-value: {p_chi2:.4f}")

if p_chi2 < 0.05:
    print(f"    Result: SIGNIFICANT difference between periods (p < 0.05)")
    print(f"    WARNING: Win rate is not stable over time - possible overfitting or regime change")
else:
    print(f"    Result: No significant difference between periods (p >= 0.05)")
    print(f"    This supports temporal stability of the model.")

# Calculate win rate trend
win_rate_trend = late_stats[2] - early_stats[2]
print(f"  Win Rate Trend: {win_rate_trend:+.4f} ({win_rate_trend*100:+.2f}%)")
if abs(win_rate_trend) < 0.05:
    print(f"    Interpretation: Stable (< 5% change)")
elif win_rate_trend > 0:
    print(f"    Interpretation: Improving over time")
else:
    print(f"    Interpretation: Degrading over time - CAUTION")
print()

# 2.2 Rolling Window Analysis
print("2.2 Rolling Window Analysis (30-day windows)")
print("-" * 60)

daily_metrics_sorted['cumulative_bets'] = daily_metrics_sorted['num_bets'].cumsum()
daily_metrics_sorted['cumulative_wins'] = daily_metrics_sorted['wins'].cumsum()
daily_metrics_sorted['cumulative_wr'] = daily_metrics_sorted['cumulative_wins'] / daily_metrics_sorted['cumulative_bets']

# Calculate 30-day rolling win rate
window_size = 30
if len(daily_metrics_sorted) >= window_size:
    rolling_bets = daily_metrics_sorted['num_bets'].rolling(window=window_size).sum()
    rolling_wins = daily_metrics_sorted['wins'].rolling(window=window_size).sum()
    rolling_wr = rolling_wins / rolling_bets

    rolling_wr_valid = rolling_wr.dropna()

    print(f"  30-day rolling win rate statistics:")
    print(f"    Mean: {rolling_wr_valid.mean():.4f}")
    print(f"    Std: {rolling_wr_valid.std():.4f}")
    print(f"    Min: {rolling_wr_valid.min():.4f}")
    print(f"    25th percentile: {rolling_wr_valid.quantile(0.25):.4f}")
    print(f"    Median: {rolling_wr_valid.median():.4f}")
    print(f"    75th percentile: {rolling_wr_valid.quantile(0.75):.4f}")
    print(f"    Max: {rolling_wr_valid.max():.4f}")

    # Check how often rolling win rate is above break-even
    pct_above_breakeven = (rolling_wr_valid > breakeven_rate).mean()
    print(f"  Percentage of 30-day windows above break-even: {pct_above_breakeven:.2%}")

    if pct_above_breakeven > 0.90:
        print(f"    Interpretation: EXCELLENT consistency (>90% of windows profitable)")
    elif pct_above_breakeven > 0.75:
        print(f"    Interpretation: Good consistency (>75% of windows profitable)")
    elif pct_above_breakeven > 0.60:
        print(f"    Interpretation: Moderate consistency (>60% of windows profitable)")
    else:
        print(f"    Interpretation: POOR consistency - high variance or overfitting")
else:
    print(f"  Not enough data for 30-day rolling analysis")
print()

# 2.3 Player Sample Size Analysis
print("2.3 Player-Level Sample Size Distribution")
print("-" * 60)

player_sample_sizes = player_analysis['num_games'].values

print(f"  Total unique players: {len(player_analysis)}")
print(f"  Games per player:")
print(f"    Mean: {player_sample_sizes.mean():.1f}")
print(f"    Median: {np.median(player_sample_sizes):.1f}")
print(f"    Min: {player_sample_sizes.min()}")
print(f"    Max: {player_sample_sizes.max()}")
print(f"  Players with < 10 bets: {(player_sample_sizes < 10).sum()} ({(player_sample_sizes < 10).mean():.1%})")
print(f"  Players with >= 20 bets: {(player_sample_sizes >= 20).sum()} ({(player_sample_sizes >= 20).mean():.1%})")

# Small sample bias check
small_sample = player_analysis[player_analysis['num_games'] < 15]
large_sample = player_analysis[player_analysis['num_games'] >= 15]

if len(small_sample) > 0 and len(large_sample) > 0:
    small_wr = (small_sample['wins'] / small_sample['num_games']).mean()
    large_wr = (large_sample['wins'] / large_sample['num_games']).mean()

    print(f"  Average win rate for players with < 15 bets: {small_wr:.4f}")
    print(f"  Average win rate for players with >= 15 bets: {large_wr:.4f}")

    if small_wr > large_wr + 0.05:
        print(f"  WARNING: Small sample players have higher win rate - possible selection bias")
    elif large_wr > small_wr + 0.05:
        print(f"  Note: Large sample players have higher win rate - model may improve with more data")
    else:
        print(f"  No significant sample size bias detected")
print()

# =============================================================================
# 3. VARIANCE ANALYSIS AND KELLY CRITERION
# =============================================================================
print("=" * 80)
print("3. VARIANCE ANALYSIS AND RISK ASSESSMENT")
print("=" * 80)
print()

# 3.1 Profit Distribution Analysis
print("3.1 Profit Distribution Analysis")
print("-" * 60)

# Assuming -110 odds: win = +0.909 units, loss = -1 unit
profit_win = 0.909
profit_loss = -1.0

print(f"  Profit per bet distribution:")
print(f"    Mean: {mean_profit:.4f} units")
print(f"    Std: {std_profit:.4f} units")
print(f"    Skewness: {stats.skew(profit_per_bet):.4f}")
print(f"    Kurtosis: {stats.kurtosis(profit_per_bet):.4f}")
print(f"    Min: {profit_per_bet.min():.4f} units")
print(f"    Max: {profit_per_bet.max():.4f} units")

# Sharpe ratio (assuming 0 risk-free rate)
sharpe_ratio = mean_profit / std_profit if std_profit > 0 else 0
print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")

if sharpe_ratio > 1.0:
    print(f"    Interpretation: EXCELLENT risk-adjusted returns")
elif sharpe_ratio > 0.5:
    print(f"    Interpretation: Good risk-adjusted returns")
elif sharpe_ratio > 0:
    print(f"    Interpretation: Positive but modest risk-adjusted returns")
else:
    print(f"    Interpretation: Poor risk-adjusted returns")
print()

# 3.2 Drawdown Analysis
print("3.2 Drawdown Analysis")
print("-" * 60)

# Calculate cumulative profit over time
daily_metrics_sorted['cumulative_profit'] = daily_metrics_sorted['daily_profit'].cumsum()
cumulative_profit = daily_metrics_sorted['cumulative_profit'].values

# Calculate running maximum and drawdown
running_max = np.maximum.accumulate(cumulative_profit)
drawdown = cumulative_profit - running_max

max_drawdown = drawdown.min()
max_drawdown_pct = (max_drawdown / running_max[np.argmin(drawdown)]) * 100 if running_max[np.argmin(drawdown)] > 0 else 0

print(f"  Maximum Drawdown: {max_drawdown:.2f} units ({max_drawdown_pct:.2f}%)")
print(f"  Final Profit: {total_profit:.2f} units")
print(f"  Profit-to-Drawdown Ratio: {abs(total_profit / max_drawdown):.2f}" if max_drawdown != 0 else "  Profit-to-Drawdown Ratio: N/A")

if abs(total_profit / max_drawdown) > 5 if max_drawdown != 0 else False:
    print(f"    Interpretation: EXCELLENT - profit significantly exceeds max drawdown")
elif abs(total_profit / max_drawdown) > 3 if max_drawdown != 0 else False:
    print(f"    Interpretation: Good - profit well exceeds max drawdown")
elif abs(total_profit / max_drawdown) > 1 if max_drawdown != 0 else False:
    print(f"    Interpretation: Moderate - profit exceeds max drawdown")
else:
    print(f"    Interpretation: CONCERNING - drawdown comparable to total profit")

# Time in drawdown
in_drawdown = drawdown < 0
pct_time_in_drawdown = in_drawdown.mean()
print(f"  Percentage of time in drawdown: {pct_time_in_drawdown:.1%}")
print()

# 3.3 Kelly Criterion
print("3.3 Kelly Criterion (Optimal Bet Sizing)")
print("-" * 60)

# Kelly formula: f = (bp - q) / b
# where b = odds received (0.909 for -110), p = win probability, q = loss probability
b = profit_win  # 0.909
p = win_rate
q = 1 - win_rate

kelly_fraction = (b * p - q) / b

print(f"  Win probability: {p:.4f}")
print(f"  Odds received (b): {b:.4f}")
print(f"  Kelly fraction: {kelly_fraction:.4f} ({kelly_fraction*100:.2f}% of bankroll)")

if kelly_fraction > 0.10:
    print(f"  Recommendation: Consider FRACTIONAL KELLY (1/4 or 1/2 Kelly) = {kelly_fraction/4:.4f} or {kelly_fraction/2:.4f}")
    print(f"    Full Kelly is aggressive and may lead to large drawdowns")
elif kelly_fraction > 0.05:
    print(f"  Recommendation: Kelly sizing is reasonable, consider 1/2 Kelly = {kelly_fraction/2:.4f} for safety")
elif kelly_fraction > 0:
    print(f"  Recommendation: Kelly sizing is conservative, full Kelly may be appropriate")
else:
    print(f"  WARNING: Negative Kelly - system has no edge at current odds")

# Expected growth rate
expected_growth_rate = p * np.log(1 + b * kelly_fraction) + q * np.log(1 - kelly_fraction)
print(f"  Expected growth rate per bet: {expected_growth_rate:.4f} ({expected_growth_rate*100:.2f}%)")
print()

# =============================================================================
# 4. EDGE ANALYSIS VALIDATION
# =============================================================================
print("=" * 80)
print("4. EDGE ANALYSIS VALIDATION")
print("=" * 80)
print()

# Parse edge data from betting_performance
import json
import re

def parse_edge_dict(edge_str):
    """Parse edge dictionary string with np.float64 values"""
    # Replace np.float64(...) with just the number
    cleaned = re.sub(r"np\.float64\(([0-9.]+)\)", r"\1", edge_str)
    # Replace single quotes with double quotes for JSON parsing
    cleaned = cleaned.replace("'", '"')
    return json.loads(cleaned)

edge_small_str = betting_perf['edge_small'].iloc[0]
edge_medium_str = betting_perf['edge_medium'].iloc[0]
edge_large_str = betting_perf['edge_large'].iloc[0]

# Parse the dictionary strings
edge_small = parse_edge_dict(edge_small_str)
edge_medium = parse_edge_dict(edge_medium_str)
edge_large = parse_edge_dict(edge_large_str)

edges = {
    'Small (1+ pts)': edge_small,
    'Medium (2+ pts)': edge_medium,
    'Large (3+ pts)': edge_large
}

print("4.1 Edge Stratification Analysis")
print("-" * 60)
print()

for edge_name, edge_data in edges.items():
    edge_count = edge_data['edge_count']
    edge_wr = float(edge_data['edge_win_rate'])
    edge_roi = float(edge_data['edge_roi']) / 100

    print(f"  {edge_name}:")
    print(f"    Bets: {edge_count}")
    print(f"    Win Rate: {edge_wr:.4f} ({edge_wr*100:.2f}%)")
    print(f"    ROI: {edge_roi:.4f} ({edge_roi*100:.2f}%)")

    # Statistical test: is this win rate significantly > break-even?
    p_val = binom.sf(int(edge_wr * edge_count) - 1, edge_count, breakeven_rate)
    print(f"    p-value vs break-even: {p_val:.2e}")

    if p_val < 0.001:
        print(f"    Result: HIGHLY SIGNIFICANT ***")
    elif p_val < 0.01:
        print(f"    Result: VERY SIGNIFICANT **")
    elif p_val < 0.05:
        print(f"    Result: SIGNIFICANT *")
    else:
        print(f"    Result: NOT SIGNIFICANT")

    # Wilson CI for this edge category
    ci_edge = wilson_ci(int(edge_wr * edge_count), edge_count, 0.95)
    print(f"    95% CI: [{ci_edge[0]:.4f}, {ci_edge[1]:.4f}]")
    print()

# Test for monotonicity: does win rate increase with edge size?
print("4.2 Edge Monotonicity Test")
print("-" * 60)

edge_sizes = ['Small (1+ pts)', 'Medium (2+ pts)', 'Large (3+ pts)']
win_rates = [
    float(edges['Small (1+ pts)']['edge_win_rate']),
    float(edges['Medium (2+ pts)']['edge_win_rate']),
    float(edges['Large (3+ pts)']['edge_win_rate'])
]

is_monotonic = all(win_rates[i] <= win_rates[i+1] for i in range(len(win_rates)-1))

print(f"  Win rates by edge size:")
for size, wr in zip(edge_sizes, win_rates):
    print(f"    {size}: {wr:.4f}")

if is_monotonic:
    print(f"  Result: WIN RATES ARE MONOTONICALLY INCREASING")
    print(f"  Interpretation: Edge calibration is correct - larger edges = higher win rates")
else:
    print(f"  Result: WIN RATES ARE NOT MONOTONIC")
    print(f"  WARNING: Edge calibration may be incorrect or noisy")

# Calculate correlation between edge size and profit
daily_with_edge = daily_predictions[daily_predictions['has_line'] == True].copy()
if 'edge_size' in daily_with_edge.columns and daily_with_edge['edge_size'].notna().sum() > 0:
    corr = daily_with_edge['edge_size'].corr(daily_with_edge['profit'])
    print(f"  Correlation between edge size and profit: {corr:.4f}")
    if corr > 0.1:
        print(f"    Interpretation: Positive correlation - larger edges tend to be more profitable")
    else:
        print(f"    Interpretation: Weak correlation - edge size is not a strong predictor")
print()

# =============================================================================
# 5. PLAYER-LEVEL DISTRIBUTION ANALYSIS
# =============================================================================
print("=" * 80)
print("5. PLAYER-LEVEL DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

print("5.1 Player Win Rate Distribution")
print("-" * 60)

player_win_rates = player_analysis['win_rate'].values

print(f"  Win Rate Distribution:")
print(f"    Mean: {player_win_rates.mean():.4f}")
print(f"    Median: {np.median(player_win_rates):.4f}")
print(f"    Std: {player_win_rates.std():.4f}")
print(f"    Min: {player_win_rates.min():.4f}")
print(f"    Max: {player_win_rates.max():.4f}")
print(f"  Players with 100% win rate: {(player_win_rates == 1.0).sum()}")
print(f"  Players with win rate > 90%: {(player_win_rates > 0.90).sum()}")
print(f"  Players with win rate > break-even: {(player_win_rates > breakeven_rate).sum()} ({(player_win_rates > breakeven_rate).mean():.1%})")
print(f"  Players with win rate < break-even: {(player_win_rates < breakeven_rate).sum()} ({(player_win_rates < breakeven_rate).mean():.1%})")
print()

# Identify concerning perfect records with small samples
perfect_players = player_analysis[player_analysis['win_rate'] == 1.0]
perfect_small_sample = perfect_players[perfect_players['num_games'] < 15]

print(f"5.2 Small Sample Bias Detection")
print("-" * 60)
print(f"  Players with 100% win rate and < 15 bets: {len(perfect_small_sample)}")
if len(perfect_small_sample) > 0:
    total_bets_perfect = perfect_small_sample['num_games'].sum()
    pct_of_total = total_bets_perfect / total_bets * 100
    print(f"  Total bets from these players: {total_bets_perfect} ({pct_of_total:.1%} of all bets)")
    if pct_of_total > 10:
        print(f"  WARNING: >10% of bets from small-sample perfect records - results may be inflated")
    else:
        print(f"  These players represent a small fraction - limited impact on overall results")
else:
    print(f"  No players with perfect records on small samples")
print()

# Weighted vs unweighted player-level win rate
weighted_wr = (player_analysis['wins'] * player_analysis['num_games']).sum() / (player_analysis['num_games'] ** 2).sum()
unweighted_wr = player_win_rates.mean()

print(f"5.3 Player-Level Aggregation")
print("-" * 60)
print(f"  Unweighted mean player win rate: {unweighted_wr:.4f}")
print(f"  Bet-weighted win rate: {win_rate:.4f}")
print(f"  Difference: {win_rate - unweighted_wr:.4f}")

if unweighted_wr > win_rate + 0.05:
    print(f"  WARNING: Unweighted mean is much higher - results driven by low-volume players")
elif win_rate > unweighted_wr + 0.05:
    print(f"  Note: Weighted mean is higher - results driven by high-volume players")
else:
    print(f"  Results are consistent across player volumes - no strong bias detected")
print()

# =============================================================================
# 6. RED FLAGS AND CONCERNS
# =============================================================================
print("=" * 80)
print("6. RED FLAGS AND SUSTAINABILITY CONCERNS")
print("=" * 80)
print()

red_flags = []
warnings = []
good_signs = []

# Check 1: Win rate too high
if win_rate > 0.85:
    red_flags.append("Win rate exceeds 85% - unusually high, may not be sustainable")
elif win_rate > 0.75:
    warnings.append("Win rate exceeds 75% - monitor for regression to mean")
else:
    good_signs.append("Win rate is high but within reasonable range")

# Check 2: Sample size
if total_bets < 1000:
    red_flags.append("Sample size < 1000 bets - insufficient for robust conclusions")
elif total_bets < 3000:
    warnings.append("Sample size < 3000 bets - results may have high variance")
else:
    good_signs.append(f"Sample size ({total_bets:,} bets) is adequate for statistical testing")

# Check 3: Temporal stability
if abs(win_rate_trend) > 0.10:
    red_flags.append(f"Win rate changed by {abs(win_rate_trend)*100:.1f}% over time - lacks temporal stability")
elif abs(win_rate_trend) > 0.05:
    warnings.append(f"Win rate changed by {abs(win_rate_trend)*100:.1f}% over time - monitor for regime change")
else:
    good_signs.append("Win rate is temporally stable")

# Check 4: Small sample bias
if len(perfect_small_sample) > 0:
    pct_perfect = len(perfect_small_sample) / len(player_analysis) * 100
    if pct_perfect > 20:
        red_flags.append(f"{pct_perfect:.1f}% of players have perfect records with small samples - severe selection bias risk")
    elif pct_perfect > 10:
        warnings.append(f"{pct_perfect:.1f}% of players have perfect records with small samples - moderate selection bias risk")

# Check 5: Lower bound of CI vs break-even
if ci_95[0] < breakeven_rate:
    red_flags.append(f"95% CI lower bound ({ci_95[0]*100:.2f}%) is below break-even - edge may not be robust")
elif ci_90[0] < breakeven_rate:
    warnings.append(f"90% CI lower bound is below break-even - edge has some uncertainty")
else:
    good_signs.append("95% CI lower bound exceeds break-even - robust statistical edge")

# Check 6: Edge monotonicity
if not is_monotonic:
    warnings.append("Edge categories are not monotonic - edge calibration may be noisy")
else:
    good_signs.append("Edge categories show monotonic win rate increase - well-calibrated")

# Check 7: Consistency across time
if len(daily_metrics_sorted) >= window_size:
    if pct_above_breakeven < 0.60:
        red_flags.append(f"Only {pct_above_breakeven:.1%} of 30-day windows are profitable - high variance")
    elif pct_above_breakeven < 0.75:
        warnings.append(f"{pct_above_breakeven:.1%} of 30-day windows are profitable - moderate consistency")
    else:
        good_signs.append(f"{pct_above_breakeven:.1%} of 30-day windows are profitable - high consistency")

# Check 8: Drawdown severity
if abs(max_drawdown / total_profit) > 0.5 if total_profit > 0 else False:
    warnings.append(f"Max drawdown is {abs(max_drawdown/total_profit)*100:.1f}% of total profit - significant volatility")
else:
    good_signs.append("Drawdown is well-controlled relative to total profit")

# Print results
if red_flags:
    print("RED FLAGS (Critical Concerns):")
    for i, flag in enumerate(red_flags, 1):
        print(f"  {i}. {flag}")
    print()

if warnings:
    print("WARNINGS (Monitor Closely):")
    for i, warning in enumerate(warnings, 1):
        print(f"  {i}. {warning}")
    print()

if good_signs:
    print("POSITIVE INDICATORS:")
    for i, sign in enumerate(good_signs, 1):
        print(f"  {i}. {sign}")
    print()

# =============================================================================
# 7. FINAL ASSESSMENT AND RECOMMENDATIONS
# =============================================================================
print("=" * 80)
print("7. FINAL ASSESSMENT AND RECOMMENDATIONS")
print("=" * 80)
print()

print("7.1 Overall Sustainability Assessment")
print("-" * 60)

# Calculate sustainability score (0-100)
score = 0

# Factor 1: Statistical significance (20 points)
if p_value_binomial < 0.001:
    score += 20
elif p_value_binomial < 0.01:
    score += 15
elif p_value_binomial < 0.05:
    score += 10
else:
    score += 0

# Factor 2: Sample size (20 points)
if total_bets >= 5000:
    score += 20
elif total_bets >= 3000:
    score += 15
elif total_bets >= 1000:
    score += 10
else:
    score += 5

# Factor 3: Temporal stability (20 points)
if abs(win_rate_trend) < 0.03:
    score += 20
elif abs(win_rate_trend) < 0.05:
    score += 15
elif abs(win_rate_trend) < 0.10:
    score += 10
else:
    score += 5

# Factor 4: 95% CI robustness (20 points)
if ci_95[0] > breakeven_rate + 0.05:
    score += 20
elif ci_95[0] > breakeven_rate:
    score += 15
elif ci_90[0] > breakeven_rate:
    score += 10
else:
    score += 5

# Factor 5: Consistency (20 points)
if len(daily_metrics_sorted) >= window_size:
    if pct_above_breakeven > 0.85:
        score += 20
    elif pct_above_breakeven > 0.75:
        score += 15
    elif pct_above_breakeven > 0.60:
        score += 10
    else:
        score += 5
else:
    score += 10  # Neutral if not enough data

print(f"  Sustainability Score: {score}/100")
print()

if score >= 85:
    print(f"  Rating: EXCELLENT - Highly sustainable edge with robust statistical support")
    print(f"  Confidence: HIGH")
elif score >= 70:
    print(f"  Rating: GOOD - Sustainable edge with good statistical support")
    print(f"  Confidence: MODERATE-HIGH")
elif score >= 55:
    print(f"  Rating: FAIR - Some evidence of edge but with concerns")
    print(f"  Confidence: MODERATE")
elif score >= 40:
    print(f"  Rating: QUESTIONABLE - Weak evidence of sustainable edge")
    print(f"  Confidence: LOW")
else:
    print(f"  Rating: POOR - Insufficient evidence of sustainable edge")
    print(f"  Confidence: VERY LOW")

print()
print("7.2 Key Recommendations")
print("-" * 60)

print("  1. Statistical Validation:")
print(f"     - Continue tracking performance to increase sample size beyond {total_bets:,} bets")
print(f"     - Target: 5,000+ bets for high confidence in edge estimation")
print(f"     - Monitor rolling 30-day win rate and stop if consistently below {breakeven_rate*100:.1f}%")
print()

print("  2. Risk Management:")
print(f"     - Use fractional Kelly sizing: recommend 1/4 Kelly = {kelly_fraction/4:.4f} of bankroll")
print(f"     - Set stop-loss at {abs(max_drawdown)*1.5:.0f} units (1.5x historical max drawdown)")
print(f"     - Track daily/weekly performance and investigate if win rate drops below {ci_95[0]*100:.1f}%")
print()

print("  3. Edge Optimization:")
print(f"     - Focus on large edge opportunities (3+ points): {float(edges['Large (3+ pts)']['edge_win_rate'])*100:.1f}% win rate")
print(f"     - Consider filtering out small edges if ROI improves")
print(f"     - Validate edge calibration: does predicted edge correlate with actual results?")
print()

print("  4. Monitoring Protocol:")
print(f"     - Weekly: Calculate 7-day win rate and compare to {breakeven_rate*100:.1f}% break-even")
print(f"     - Monthly: Recalculate 95% CI and verify lower bound > {breakeven_rate*100:.1f}%")
print(f"     - Quarterly: Perform temporal stability analysis (compare latest 3 months to previous)")
print(f"     - Annual: Re-train model and validate on out-of-sample data")
print()

print("  5. Red Flags to Watch:")
print(f"     - Win rate drops below {ci_95[0]*100:.1f}% for 2+ consecutive weeks")
print(f"     - 30-day rolling win rate drops below {breakeven_rate*100:.1f}%")
print(f"     - Drawdown exceeds {abs(max_drawdown)*2:.0f} units (2x historical max)")
print(f"     - Sharp changes in win rate across time periods (>10% change)")
print()

print("7.3 Final Verdict")
print("-" * 60)

if score >= 70 and ci_95[0] > breakeven_rate and total_bets >= 3000:
    print("  VERDICT: The backtest results are STATISTICALLY SIGNIFICANT and show")
    print("  evidence of a sustainable edge. The win rate is robust, temporally stable,")
    print("  and well above break-even with high confidence.")
    print()
    print("  However, continue to monitor performance closely as:")
    print(f"  - The win rate of {win_rate*100:.1f}% is exceptionally high")
    print("  - Real-world conditions may differ from backtest assumptions")
    print("  - Regression to mean is expected over longer time periods")
elif score >= 55:
    print("  VERDICT: The backtest results show MODERATE evidence of an edge.")
    print("  While statistically significant, there are some concerns about")
    print("  sustainability and robustness.")
    print()
    print("  Proceed with CAUTION and monitor performance closely.")
else:
    print("  VERDICT: The backtest results are INCONCLUSIVE or show WEAK evidence")
    print("  of a sustainable edge. Additional data and validation are needed before")
    print("  deploying this system with real capital.")
    print()
    print("  RECOMMENDATION: Gather more data or improve model before live trading.")

print()
print("=" * 80)
print("END OF STATISTICAL ANALYSIS")
print("=" * 80)
print()

# Save summary to file
summary_path = Path("/Users/diyagamah/Documents/NBA_PRA/backtest/results/statistical_analysis_summary.txt")
with open(summary_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NBA PRA BACKTEST - STATISTICAL ANALYSIS SUMMARY\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"Overall Results:\n")
    f.write(f"  Total Bets: {total_bets:,}\n")
    f.write(f"  Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)\n")
    f.write(f"  ROI: {roi:.4f} ({roi*100:.2f}%)\n")
    f.write(f"  Total Profit: {total_profit:.2f} units\n\n")

    f.write(f"Statistical Significance:\n")
    f.write(f"  Binomial test p-value: {p_value_binomial:.2e}\n")
    f.write(f"  95% CI: [{ci_95[0]*100:.2f}%, {ci_95[1]*100:.2f}%]\n")
    f.write(f"  Lower bound vs break-even: {ci_95[0]*100:.2f}% vs {breakeven_rate*100:.2f}%\n\n")

    f.write(f"Sustainability Score: {score}/100\n\n")

    if red_flags:
        f.write("Red Flags:\n")
        for flag in red_flags:
            f.write(f"  - {flag}\n")
        f.write("\n")

    if warnings:
        f.write("Warnings:\n")
        for warning in warnings:
            f.write(f"  - {warning}\n")
        f.write("\n")

print(f"Summary saved to: {summary_path}")
