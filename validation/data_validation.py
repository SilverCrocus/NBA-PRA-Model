"""
Comprehensive Data Validation Script for NBA PRA Pipeline
Validates data quality, coverage, and feature integrity

This script checks:
1. Data coverage and completeness
2. Data quality issues
3. Feature distributions
4. CTG data integration
5. Temporal consistency
6. Edge cases
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NBA_API_DIR = DATA_DIR / "nba_api"
CTG_DATA_DIR = DATA_DIR / "ctg_data_organized"
VALIDATION_OUTPUT = PROJECT_ROOT / "validation" / "reports"
VALIDATION_OUTPUT.mkdir(parents=True, exist_ok=True)


class DataValidator:
    """
    Comprehensive data validation for NBA PRA pipeline
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.warnings = []
        self.errors = []
        self.stats = {}

    def log(self, message, level='INFO'):
        """Log validation messages"""
        prefix = {
            'INFO': '   ',
            'WARN': '⚠️  ',
            'ERROR': '❌ ',
            'SUCCESS': '✓  '
        }.get(level, '   ')

        if self.verbose:
            print(f"{prefix}{message}")

        if level == 'WARN':
            self.warnings.append(message)
        elif level == 'ERROR':
            self.errors.append(message)

    def validate_data_coverage(self, df):
        """
        Validate data coverage from NBA API

        Checks:
        - Season coverage
        - Games per season
        - Players per season
        - Known data gaps (lockout, COVID)
        """
        self.log("\n" + "="*80)
        self.log("1. DATA COVERAGE ANALYSIS")
        self.log("="*80)

        # Season coverage
        seasons = sorted(df['season'].unique())
        self.log(f"\nSeasons covered: {len(seasons)}")
        self.log(f"Range: {seasons[0]} to {seasons[-1]}")
        self.log(f"Seasons: {', '.join(seasons)}")

        self.stats['seasons_covered'] = len(seasons)
        self.stats['season_range'] = (seasons[0], seasons[-1])

        # Check for known problematic seasons
        problematic_seasons = {
            '2011-12': 'Lockout season (66 games)',
            '2019-20': 'COVID-shortened season (bubble)',
            '2020-21': 'COVID season (72 games)'
        }

        for season, reason in problematic_seasons.items():
            if season in seasons:
                self.log(f"Found {season}: {reason}", level='WARN')

        # Games and players per season
        self.log("\nGames and Players per Season:")
        self.log("-" * 80)

        season_stats = df.groupby('season').agg({
            'game_id': 'nunique',
            'player_id': 'nunique',
            'game_date': ['min', 'max']
        }).round(0)

        for season in seasons:
            season_data = df[df['season'] == season]
            games = season_data['game_id'].nunique()
            players = season_data['player_id'].nunique()
            date_range = f"{season_data['game_date'].min().date()} to {season_data['game_date'].max().date()}"

            self.log(f"{season}: {games:,} games, {players:,} players ({date_range})")

            # Validate expected game count
            expected_games = self._expected_games_per_season(season)
            if expected_games and games < expected_games * 0.8:
                self.log(f"Low game count for {season}: {games} vs expected ~{expected_games}", level='WARN')

        # Total statistics
        self.log(f"\nTotal Statistics:")
        self.log(f"  Total games: {df['game_id'].nunique():,}")
        self.log(f"  Total player-games: {len(df):,}")
        self.log(f"  Unique players: {df['player_id'].nunique():,}")
        self.log(f"  Date range: {df['game_date'].min().date()} to {df['game_date'].max().date()}")

        self.stats['total_games'] = df['game_id'].nunique()
        self.stats['total_player_games'] = len(df)
        self.stats['unique_players'] = df['player_id'].nunique()

        # Games per player distribution
        games_per_player = df.groupby('player_id').size()
        self.log(f"\nGames per Player Distribution:")
        self.log(f"  Mean: {games_per_player.mean():.1f}")
        self.log(f"  Median: {games_per_player.median():.1f}")
        self.log(f"  Min: {games_per_player.min()}")
        self.log(f"  Max: {games_per_player.max()}")
        self.log(f"  Players with <10 games: {(games_per_player < 10).sum():,}")
        self.log(f"  Players with <5 games: {(games_per_player < 5).sum():,}")

        return season_stats

    def _expected_games_per_season(self, season):
        """Get expected number of NBA games per season"""
        lockout_seasons = {'2011-12': 66 * 30 / 2}  # 66 game season
        covid_seasons = {'2019-20': 65 * 30 / 2, '2020-21': 72 * 30 / 2}

        if season in lockout_seasons:
            return lockout_seasons[season]
        elif season in covid_seasons:
            return covid_seasons[season]
        else:
            return 82 * 30 / 2  # Normal season: 82 games × 30 teams ÷ 2

    def validate_data_quality(self, df):
        """
        Check for common data quality issues

        Checks:
        - Missing values
        - Invalid values (negative stats, impossible percentages)
        - Outliers
        - Duplicate records
        """
        self.log("\n" + "="*80)
        self.log("2. DATA QUALITY CHECKS")
        self.log("="*80)

        # Missing values
        self.log("\nMissing Values Analysis:")
        self.log("-" * 80)

        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        missing_df = pd.DataFrame({
            'Missing Count': missing[missing > 0],
            'Missing %': missing_pct[missing > 0]
        }).sort_values('Missing %', ascending=False)

        if len(missing_df) > 0:
            for col, row in missing_df.iterrows():
                level = 'ERROR' if row['Missing %'] > 50 else 'WARN' if row['Missing %'] > 10 else 'INFO'
                self.log(f"{col}: {row['Missing Count']:,.0f} ({row['Missing %']:.2f}%)", level=level)
        else:
            self.log("No missing values found", level='SUCCESS')

        # Check for zero minutes played (DNP)
        if 'minutes' in df.columns:
            df['minutes_float'] = df['minutes'].apply(self._convert_minutes)
            zero_minutes = (df['minutes_float'] == 0).sum()
            self.log(f"\nGames with 0 minutes played (DNP): {zero_minutes:,} ({zero_minutes/len(df)*100:.2f}%)")

            if zero_minutes > len(df) * 0.05:
                self.log("High number of DNP games - consider filtering", level='WARN')

        # Check for duplicate records
        duplicates = df.duplicated(subset=['player_id', 'game_id']).sum()
        if duplicates > 0:
            self.log(f"\nDuplicate player-game records: {duplicates}", level='ERROR')
        else:
            self.log("\nDuplicate check: PASSED", level='SUCCESS')

        # Check for invalid stat values
        self.log("\nInvalid Value Checks:")
        self.log("-" * 80)

        stat_cols = ['points', 'rebounds', 'assists', 'pra', 'steals', 'blocks', 'turnovers']
        for col in stat_cols:
            if col in df.columns:
                negative = (df[col] < 0).sum()
                if negative > 0:
                    self.log(f"{col}: {negative} negative values", level='ERROR')

                # Check for extreme outliers
                if col == 'points':
                    extreme = (df[col] > 100).sum()
                    if extreme > 0:
                        self.log(f"{col}: {extreme} games with >100 points (impossible)", level='ERROR')
                elif col == 'pra':
                    extreme = (df[col] > 150).sum()
                    if extreme > 0:
                        self.log(f"{col}: {extreme} games with >150 PRA (very rare)", level='WARN')

        # Shooting percentage validation
        pct_cols = ['fg_pct', 'fg3_pct', 'ft_pct']
        for col in pct_cols:
            if col in df.columns:
                invalid = ((df[col] < 0) | (df[col] > 1)).sum()
                if invalid > 0:
                    self.log(f"{col}: {invalid} invalid percentages (not 0-1)", level='ERROR')

        # PRA calculation validation
        if all(col in df.columns for col in ['points', 'rebounds', 'assists', 'pra']):
            calculated_pra = df['points'] + df['rebounds'] + df['assists']
            pra_mismatch = (calculated_pra != df['pra']).sum()

            if pra_mismatch > 0:
                self.log(f"\nPRA calculation mismatch: {pra_mismatch} records", level='ERROR')
                # Show sample
                mismatch_sample = df[calculated_pra != df['pra']].head(3)
                for idx, row in mismatch_sample.iterrows():
                    self.log(f"  Game {row['game_id']}: P={row['points']} R={row['rebounds']} A={row['assists']} "
                           f"Calculated={row['points']+row['rebounds']+row['assists']} vs PRA={row['pra']}")
            else:
                self.log("\nPRA calculation: CORRECT", level='SUCCESS')

    def validate_target_distribution(self, df):
        """
        Analyze target variable (PRA) distribution

        Checks:
        - Distribution statistics
        - Skewness
        - Outliers
        - Consistency across seasons
        """
        self.log("\n" + "="*80)
        self.log("3. TARGET VARIABLE (PRA) DISTRIBUTION")
        self.log("="*80)

        if 'pra' not in df.columns:
            self.log("PRA column not found!", level='ERROR')
            return

        # Overall distribution
        self.log("\nOverall PRA Statistics:")
        self.log("-" * 80)
        self.log(f"Mean:     {df['pra'].mean():.2f}")
        self.log(f"Median:   {df['pra'].median():.2f}")
        self.log(f"Std Dev:  {df['pra'].std():.2f}")
        self.log(f"Min:      {df['pra'].min():.2f}")
        self.log(f"Max:      {df['pra'].max():.2f}")
        self.log(f"Skewness: {df['pra'].skew():.2f}")

        # Percentiles
        percentiles = df['pra'].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        self.log(f"\nPercentiles:")
        for pct, val in percentiles.items():
            self.log(f"  {int(pct*100)}th: {val:.2f}")

        # Check for extreme skewness
        if abs(df['pra'].skew()) > 1:
            self.log(f"\nHigh skewness detected ({df['pra'].skew():.2f}) - consider transformation", level='WARN')

        # Distribution by season
        self.log("\nPRA Distribution by Season:")
        self.log("-" * 80)

        season_pra = df.groupby('season')['pra'].agg(['mean', 'median', 'std', 'count'])
        for season, row in season_pra.iterrows():
            self.log(f"{season}: Mean={row['mean']:.2f}, Median={row['median']:.2f}, "
                   f"Std={row['std']:.2f}, N={row['count']:,}")

        # Check for temporal drift
        mean_drift = season_pra['mean'].max() - season_pra['mean'].min()
        if mean_drift > 5:
            self.log(f"\nSignificant mean PRA drift across seasons: {mean_drift:.2f}", level='WARN')

        self.stats['pra_mean'] = df['pra'].mean()
        self.stats['pra_std'] = df['pra'].std()
        self.stats['pra_skew'] = df['pra'].skew()

    def validate_player_consistency(self, df):
        """
        Check for player-specific data quality issues

        Checks:
        - Players with very few games
        - Mid-season team changes
        - Unusual stat patterns
        """
        self.log("\n" + "="*80)
        self.log("4. PLAYER-LEVEL VALIDATION")
        self.log("="*80)

        # Players with few games
        games_per_player = df.groupby('player_id').agg({
            'game_id': 'nunique',
            'player_name': 'first'
        })

        few_games = games_per_player[games_per_player['game_id'] < 5]
        self.log(f"\nPlayers with <5 games: {len(few_games):,}")

        if len(few_games) > 0:
            self.log("Sample players with few games:")
            for pid, row in few_games.head(10).iterrows():
                self.log(f"  {row['player_name']}: {row['game_id']} games")

        # Players who changed teams mid-season
        self.log("\nChecking for mid-season trades...")

        # Count unique teams per player per season
        teams_per_player_season = df.groupby(['player_id', 'season', 'player_name']).apply(
            lambda x: self._count_unique_teams(x['matchup'])
        ).reset_index()
        teams_per_player_season.columns = ['player_id', 'season', 'player_name', 'num_teams']

        traded_players = teams_per_player_season[teams_per_player_season['num_teams'] > 1]

        self.log(f"Player-seasons with trades: {len(traded_players):,}")

        if len(traded_players) > 0:
            self.log("Sample traded players:")
            for _, row in traded_players.head(10).iterrows():
                self.log(f"  {row['player_name']} ({row['season']}): {row['num_teams']} teams")
            self.log("\nConsider creating pre/post-trade features", level='WARN')

        self.stats['traded_player_seasons'] = len(traded_players)

        # Players with unusual variance
        self.log("\nChecking for unusual performance variance...")

        player_variance = df.groupby('player_id').agg({
            'pra': ['mean', 'std', 'count'],
            'player_name': 'first'
        })
        player_variance.columns = ['pra_mean', 'pra_std', 'games', 'player_name']

        # Calculate coefficient of variation (CV)
        player_variance['cv'] = player_variance['pra_std'] / player_variance['pra_mean']

        # Find players with very high variance (CV > 1)
        high_variance = player_variance[
            (player_variance['cv'] > 1) & (player_variance['games'] >= 20)
        ].sort_values('cv', ascending=False)

        if len(high_variance) > 0:
            self.log(f"Players with high performance variance (CV>1): {len(high_variance)}")
            self.log("Top 5 most inconsistent players:")
            for pid, row in high_variance.head(5).iterrows():
                self.log(f"  {row['player_name']}: Mean={row['pra_mean']:.1f}, "
                       f"Std={row['pra_std']:.1f}, CV={row['cv']:.2f}")

    def validate_ctg_integration(self, df):
        """
        Validate CTG data integration

        Checks:
        - CTG data availability
        - Player name matching
        - Merge success rate
        """
        self.log("\n" + "="*80)
        self.log("5. CTG DATA INTEGRATION VALIDATION")
        self.log("="*80)

        # Check if CTG directory exists
        if not CTG_DATA_DIR.exists():
            self.log("CTG data directory not found!", level='ERROR')
            return

        # Load CTG data for available seasons
        ctg_seasons = [d.name for d in (CTG_DATA_DIR / 'players').iterdir() if d.is_dir()]
        self.log(f"\nCTG seasons available: {len(ctg_seasons)}")
        self.log(f"Seasons: {', '.join(sorted(ctg_seasons))}")

        # Check overlap with NBA API data
        nba_seasons = set(df['season'].unique())
        ctg_seasons_set = set(ctg_seasons)

        overlap = nba_seasons & ctg_seasons_set
        missing_ctg = nba_seasons - ctg_seasons_set

        self.log(f"\nSeasons with CTG data: {len(overlap)}/{len(nba_seasons)}")
        if missing_ctg:
            self.log(f"Seasons missing CTG data: {', '.join(sorted(missing_ctg))}", level='WARN')

        # Sample a season and check player matching
        if overlap:
            sample_season = sorted(overlap)[-1]  # Most recent season with CTG
            self.log(f"\nTesting player name matching for {sample_season}...")

            ctg_file = CTG_DATA_DIR / 'players' / sample_season / 'regular_season' / 'offensive_overview' / 'offensive_overview.csv'

            if ctg_file.exists():
                ctg_df = pd.read_csv(ctg_file)
                ctg_df['Player'] = ctg_df['Player'].str.strip()

                nba_players = set(df[df['season'] == sample_season]['player_name'].str.strip().unique())
                ctg_players = set(ctg_df['Player'].unique())

                # Account for players who changed teams (appear multiple times in CTG)
                ctg_players_unique = set([p.split(',')[0].strip() for p in ctg_players])

                matched = nba_players & ctg_players_unique
                nba_only = nba_players - ctg_players_unique
                ctg_only = ctg_players_unique - nba_players

                match_rate = len(matched) / len(nba_players) * 100 if nba_players else 0

                self.log(f"NBA API players: {len(nba_players)}")
                self.log(f"CTG players: {len(ctg_players_unique)}")
                self.log(f"Matched players: {len(matched)} ({match_rate:.1f}%)")

                if match_rate < 80:
                    self.log(f"Low match rate ({match_rate:.1f}%) - player name standardization needed", level='WARN')

                # Show sample mismatches
                if nba_only:
                    self.log(f"\nSample NBA players not in CTG:")
                    for player in sorted(nba_only)[:10]:
                        self.log(f"  {player}")

                    if len(nba_only) > 100:
                        self.log(f"\nMany NBA players not in CTG - likely due to minimum minutes filter", level='INFO')

                self.stats['ctg_match_rate'] = match_rate
            else:
                self.log(f"CTG file not found: {ctg_file}", level='WARN')

    def validate_edge_cases(self, df):
        """
        Identify and count edge cases that need special handling

        Edge cases:
        - First game of season (no history)
        - Players returning from injury (large gaps)
        - Rookies
        - 10-day contracts (very few games)
        - Back-to-back games
        """
        self.log("\n" + "="*80)
        self.log("6. EDGE CASE IDENTIFICATION")
        self.log("="*80)

        df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

        # First game of season per player
        first_games = df.groupby(['player_id', 'season']).head(1)
        self.log(f"\nFirst game of season: {len(first_games):,} player-seasons")
        self.log("  -> No historical features available", level='WARN')

        # Calculate days between games
        df['days_since_last'] = df.groupby('player_id')['game_date'].diff().dt.days

        # Long gaps (potential injury returns)
        long_gaps = df[df['days_since_last'] > 30]
        self.log(f"\nGames after 30+ day gap (injury returns): {len(long_gaps):,}")

        if len(long_gaps) > 0:
            self.log(f"  Mean PRA after long gap: {long_gaps['pra'].mean():.2f}")
            self.log(f"  Overall mean PRA: {df['pra'].mean():.2f}")
            self.log("  -> Consider 'games_since_injury' feature", level='WARN')

        # Back-to-back games
        back_to_back = df[df['days_since_last'] == 1]
        self.log(f"\nBack-to-back games: {len(back_to_back):,} ({len(back_to_back)/len(df)*100:.1f}%)")

        if len(back_to_back) > 0:
            self.log(f"  Mean PRA in B2B: {back_to_back['pra'].mean():.2f}")
            self.log(f"  Mean PRA overall: {df['pra'].mean():.2f}")
            diff = back_to_back['pra'].mean() - df['pra'].mean()
            if abs(diff) > 1:
                self.log(f"  B2B impact: {diff:+.2f} PRA", level='WARN')

        # Players with very few total games (10-day contracts, two-way players)
        total_games_per_player = df.groupby('player_id').size()
        few_game_players = (total_games_per_player < 10).sum()
        self.log(f"\nPlayers with <10 total games: {few_game_players:,}")
        self.log("  -> Likely 10-day contracts or two-way players", level='INFO')

        # Rookies (players appearing in first season of dataset)
        player_first_season = df.groupby('player_id')['season'].min()
        rookies_by_season = player_first_season.value_counts().sort_index()

        self.log(f"\nNew players by season (potential rookies):")
        for season, count in rookies_by_season.items():
            self.log(f"  {season}: {count:,} new players")

        # DNP games (if minutes available)
        if 'minutes' in df.columns:
            df['minutes_float'] = df['minutes'].apply(self._convert_minutes)
            dnp_games = (df['minutes_float'] == 0).sum()
            self.log(f"\nDNP (Did Not Play) games: {dnp_games:,} ({dnp_games/len(df)*100:.2f}%)")

            if dnp_games > 0:
                self.log("  -> Consider filtering DNP games or handling separately", level='WARN')

    def validate_temporal_consistency(self, df):
        """
        Check for temporal consistency in data

        Checks:
        - Date ordering
        - Season boundaries
        - Game frequency
        """
        self.log("\n" + "="*80)
        self.log("7. TEMPORAL CONSISTENCY CHECKS")
        self.log("="*80)

        # Check date ordering within player
        df_sorted = df.sort_values(['player_id', 'game_date'])

        # Verify no future games before past games
        date_order_issues = 0
        for player_id in df['player_id'].unique()[:100]:  # Sample check
            player_games = df_sorted[df_sorted['player_id'] == player_id]
            if not player_games['game_date'].is_monotonic_increasing:
                date_order_issues += 1

        if date_order_issues > 0:
            self.log(f"Players with date ordering issues: {date_order_issues}", level='ERROR')
        else:
            self.log("Date ordering: CORRECT", level='SUCCESS')

        # Season boundary validation
        self.log("\nSeason date ranges:")
        season_dates = df.groupby('season')['game_date'].agg(['min', 'max'])

        for season, row in season_dates.iterrows():
            self.log(f"  {season}: {row['min'].date()} to {row['max'].date()}")

            # Check for date overlap with next season
            # NBA seasons run Oct-April typically
            if row['max'].month > 6:
                self.log(f"    Warning: {season} extends past June (playoffs?)", level='WARN')

        # Average games per day (should be ~15 for NBA)
        games_per_date = df.groupby('game_date')['game_id'].nunique()
        self.log(f"\nAverage NBA games per day: {games_per_date.mean():.1f}")

        if games_per_date.mean() < 5:
            self.log("  Low games per day - data may be incomplete", level='WARN')

    def _convert_minutes(self, minutes):
        """Convert minutes to float"""
        if pd.isna(minutes):
            return 0.0
        if isinstance(minutes, (int, float)):
            return float(minutes)
        if isinstance(minutes, str) and ':' in minutes:
            try:
                parts = minutes.split(':')
                return float(parts[0]) + float(parts[1]) / 60
            except:
                return 0.0
        try:
            return float(minutes)
        except:
            return 0.0

    def _count_unique_teams(self, matchup_series):
        """Extract unique teams from matchup strings"""
        teams = set()
        for matchup in matchup_series:
            if isinstance(matchup, str):
                team = matchup.split()[0]
                teams.add(team)
        return len(teams)

    def generate_summary_report(self):
        """Generate summary validation report"""
        self.log("\n" + "="*80)
        self.log("VALIDATION SUMMARY REPORT")
        self.log("="*80)

        self.log(f"\nTotal Warnings: {len(self.warnings)}")
        self.log(f"Total Errors: {len(self.errors)}")

        if self.errors:
            self.log("\nCRITICAL ERRORS:", level='ERROR')
            for error in self.errors:
                self.log(f"  - {error}", level='ERROR')

        if self.warnings:
            self.log("\nWARNINGS:", level='WARN')
            for warning in self.warnings[:20]:  # Show first 20
                self.log(f"  - {warning}", level='WARN')

        if not self.errors and not self.warnings:
            self.log("\nAll validation checks PASSED!", level='SUCCESS')

        # Key statistics
        self.log("\nKEY STATISTICS:")
        for key, value in self.stats.items():
            self.log(f"  {key}: {value}")

        return {
            'warnings': self.warnings,
            'errors': self.errors,
            'stats': self.stats
        }


def run_full_validation(data_path=None):
    """
    Run complete validation suite

    Args:
        data_path: Path to player_games.parquet (optional)
    """
    print("="*80)
    print("NBA PRA DATA VALIDATION")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Load data
    if data_path is None:
        data_path = NBA_API_DIR / "player_games.parquet"

    print(f"Loading data from: {data_path}")

    if not Path(data_path).exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("\nPlease run data collection first:")
        print("  python feature_engineering/data_loader.py")
        return None

    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records\n")

    # Initialize validator
    validator = DataValidator(verbose=True)

    # Run all validations
    try:
        validator.validate_data_coverage(df)
        validator.validate_data_quality(df)
        validator.validate_target_distribution(df)
        validator.validate_player_consistency(df)
        validator.validate_ctg_integration(df)
        validator.validate_edge_cases(df)
        validator.validate_temporal_consistency(df)

        # Generate summary
        summary = validator.generate_summary_report()

        # Save summary to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = VALIDATION_OUTPUT / f"validation_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("NBA PRA DATA VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"Total Records: {len(df):,}\n")
            f.write(f"Warnings: {len(summary['warnings'])}\n")
            f.write(f"Errors: {len(summary['errors'])}\n\n")

            if summary['errors']:
                f.write("ERRORS:\n")
                for error in summary['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")

            if summary['warnings']:
                f.write("WARNINGS:\n")
                for warning in summary['warnings']:
                    f.write(f"  - {warning}\n")
                f.write("\n")

            f.write("KEY STATISTICS:\n")
            for key, value in summary['stats'].items():
                f.write(f"  {key}: {value}\n")

        print(f"\nValidation report saved to: {report_file}")

        return summary

    except Exception as e:
        print(f"\nValidation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    run_full_validation()
