"""
CSV-Based Data Loader for NBA PRA Model
Loads game logs from existing CSV files instead of API collection
Saves 10-14 hours compared to NBA API fetching

Usage:
    # Load all seasons
    df = load_from_csv_files(DATA_DIR / "game_logs")

    # Load specific seasons
    df = load_from_csv_files(
        DATA_DIR / "game_logs",
        seasons=['2015-16', '2016-17', '2017-18', ..., '2024_25']
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_DIR = DATA_DIR / "game_logs"
NBA_API_DIR = DATA_DIR / "nba_api"
NBA_API_DIR.mkdir(exist_ok=True)


def infer_season_from_date(date):
    """
    Infer NBA season from game date
    NBA season runs from October (year Y) to June (year Y+1)

    Args:
        date: datetime object or string

    Returns:
        Season string (e.g., "2023-24")
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    year = date.year
    month = date.month

    # If game is Oct-Dec, season is year to year+1
    if month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    # If game is Jan-Jun, season is year-1 to year
    else:
        return f"{year - 1}-{str(year)[2:]}"


def load_from_csv_files(data_dir, seasons=None, verbose=True):
    """
    Load NBA game logs from existing CSV files

    Args:
        data_dir: Path to directory containing game_logs_*.csv files
        seasons: Optional list of seasons to load (e.g., ['2023-24', '2024_25'])
                If None, loads all available seasons
        verbose: Print loading progress

    Returns:
        DataFrame with all game logs in standardized format
    """
    data_path = Path(data_dir)
    all_games = []

    # Get all CSV files
    csv_files = sorted(data_path.glob('game_logs_*.csv'))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    if verbose:
        print(f"Found {len(csv_files)} CSV files in {data_path}")

    for csv_file in csv_files:
        # Extract season from filename
        season = csv_file.stem.replace('game_logs_', '')

        # Filter by season if specified
        if seasons and season not in seasons:
            continue

        if verbose:
            print(f"Loading {season}...", end=" ")

        df = pd.read_csv(csv_file)

        # Handle 2024_25 duplicate column issue
        # This file has both 'Player_ID' (col 2) and 'PLAYER_ID' (col 28)
        # Strategy: Drop the mixed-case versions, keep the uppercase versions
        columns_to_drop = []
        columns_upper_map = {}

        for col in df.columns:
            col_upper = col.upper() if isinstance(col, str) else col
            if col_upper in columns_upper_map and col != col_upper:
                # We already have this column in uppercase, drop the mixed-case version
                columns_to_drop.append(col)
            else:
                columns_upper_map[col_upper] = col

        if columns_to_drop:
            if verbose:
                print(f"  Dropping duplicate columns: {columns_to_drop}", end=" ")
            df = df.drop(columns=columns_to_drop)

        # Now standardize remaining column names (handle any remaining mixed-case)
        column_renames = {
            'Player_ID': 'PLAYER_ID',
            'Game_ID': 'GAME_ID'
        }
        df = df.rename(columns=column_renames)

        # Add SEASON column if not present (for 2024_25 file)
        if 'SEASON' not in df.columns:
            if 'SEASON_ID' in df.columns:
                # Convert SEASON_ID (22024) to SEASON (2024-25)
                df['SEASON'] = df['SEASON_ID'].apply(
                    lambda x: f"{str(x)[1:]}-{str(int(str(x)[1:]) + 1)[2:]}" if pd.notna(x) else season
                )
            elif 'GAME_DATE' in df.columns:
                # Fallback: derive from GAME_DATE
                df['SEASON'] = pd.to_datetime(df['GAME_DATE']).apply(infer_season_from_date)
            else:
                # Last resort: use filename
                df['SEASON'] = season

        # Add SEASON_TYPE if missing (assume Regular Season)
        if 'SEASON_TYPE' not in df.columns:
            df['SEASON_TYPE'] = 'Regular Season'

        # Add TEAM_ABBREVIATION if missing (extract from MATCHUP)
        if 'TEAM_ABBREVIATION' not in df.columns and 'MATCHUP' in df.columns:
            # Extract from MATCHUP (e.g., "LAL vs. BOS" → "LAL")
            df['TEAM_ABBREVIATION'] = df['MATCHUP'].str.split(' ').str[0]

        # Add TEAM_ID and TEAM_NAME as NaN if missing (not critical for features)
        if 'TEAM_ID' not in df.columns:
            df['TEAM_ID'] = np.nan

        if 'TEAM_NAME' not in df.columns:
            df['TEAM_NAME'] = np.nan

        # Calculate PRA if not present (should already exist in CSVs)
        if 'PRA' not in df.columns:
            df['PRA'] = df['PTS'] + df['REB'] + df['AST']

        # Add derived columns that might be missing in 2024_25
        if 'FANTASY_PTS' not in df.columns:
            # Standard fantasy scoring: PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TOV
            df['FANTASY_PTS'] = (
                df['PTS'] +
                1.2 * df['REB'] +
                1.5 * df['AST'] +
                3 * df['STL'] +
                3 * df['BLK'] -
                df['TOV']
            )

        if 'DOUBLE_DOUBLE' not in df.columns:
            # Count stats >= 10 in PTS, REB, AST, STL, BLK
            stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK']
            double_double_count = sum((df[col] >= 10).astype(int) for col in stat_cols if col in df.columns)
            df['DOUBLE_DOUBLE'] = (double_double_count >= 2)

        if 'TRIPLE_DOUBLE' not in df.columns:
            stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK']
            triple_double_count = sum((df[col] >= 10).astype(int) for col in stat_cols if col in df.columns)
            df['TRIPLE_DOUBLE'] = (triple_double_count >= 3)

        # Add DK_POINTS and FD_POINTS if missing (use standard formulas)
        if 'DK_POINTS' not in df.columns:
            # DraftKings: PTS + 1.25*REB + 1.5*AST + 2*STL + 2*BLK - 0.5*TOV + bonuses
            df['DK_POINTS'] = (
                df['PTS'] +
                1.25 * df['REB'] +
                1.5 * df['AST'] +
                2 * df['STL'] +
                2 * df['BLK'] -
                0.5 * df['TOV']
            )
            # Add bonuses for double-double (+1.5) and triple-double (+3)
            df.loc[df['DOUBLE_DOUBLE'], 'DK_POINTS'] += 1.5
            df.loc[df['TRIPLE_DOUBLE'], 'DK_POINTS'] += 3

        if 'FD_POINTS' not in df.columns:
            # FanDuel: PTS + 1.2*REB + 1.5*AST + 3*STL + 3*BLK - TOV
            df['FD_POINTS'] = (
                df['PTS'] +
                1.2 * df['REB'] +
                1.5 * df['AST'] +
                3 * df['STL'] +
                3 * df['BLK'] -
                df['TOV']
            )

        # Convert GAME_DATE to datetime BEFORE concat (critical for proper alignment)
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

        # Final check: ensure no duplicate column names before appending
        if df.columns.duplicated().any():
            if verbose:
                print(f"  Warning: Removing duplicate columns in {season}")
            df = df.loc[:, ~df.columns.duplicated()]

        all_games.append(df)

        if verbose:
            print(f"✓ {len(df):,} rows")

    # Combine all seasons
    combined = pd.concat(all_games, ignore_index=True)

    if verbose:
        print(f"\nCombined data from {len(all_games)} seasons")

    # Clean and standardize the combined data
    combined = clean_csv_data(combined, verbose=verbose)

    return combined


def clean_csv_data(df, verbose=True):
    """
    Clean and standardize CSV game log data
    Handles multiple CSV formats and ensures pipeline compatibility

    Args:
        df: Raw combined DataFrame from CSV files
        verbose: Print cleaning steps

    Returns:
        Cleaned DataFrame with standardized columns matching pipeline expectations
    """
    if verbose:
        print("\nCleaning and standardizing data...")

    # Convert game date to datetime
    # Handle different date formats in CSV files
    # - Standard format: "2023-10-24" or "Oct 24, 2023"
    # - 2024_25 format: "Apr 13, 2025" (sometimes without year)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')

    # For any remaining NaT values, try to infer from season
    if df['GAME_DATE'].isna().any():
        # This shouldn't happen often, but log it if it does
        null_count = df['GAME_DATE'].isna().sum()
        if verbose:
            print(f"  ⚠ Warning: {null_count} rows with unparseable dates (will remain as NaT)")

    # Extract home/away from MATCHUP (e.g., "LAL vs. BOS" or "LAL @ BOS")
    if 'IS_HOME' not in df.columns:
        df['IS_HOME'] = df['MATCHUP'].str.contains('vs.', na=False).astype(int)

    # Extract opponent team
    if 'OPPONENT' not in df.columns:
        # Match pattern: "vs. XXX" or "@ XXX"
        opponent_extract = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s+([A-Z]{3})')
        df['OPPONENT'] = opponent_extract[0]

    # Ensure all percentage columns are numeric
    pct_cols = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
    for col in pct_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure all counting stats are numeric
    numeric_cols = [
        'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
        'PTS', 'PLUS_MINUS', 'PRA'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Select and rename key columns to match pipeline expectations
    columns_to_keep = {
        'PLAYER_ID': 'player_id',
        'PLAYER_NAME': 'player_name',
        'GAME_ID': 'game_id',
        'GAME_DATE': 'game_date',
        'SEASON': 'season',
        'MATCHUP': 'matchup',
        'IS_HOME': 'is_home',
        'OPPONENT': 'opponent_team',
        'WL': 'win_loss',
        'MIN': 'minutes',
        'PTS': 'points',
        'REB': 'rebounds',
        'AST': 'assists',
        'PRA': 'pra',
        'FGM': 'fg_made',
        'FGA': 'fg_attempted',
        'FG_PCT': 'fg_pct',
        'FG3M': 'fg3_made',
        'FG3A': 'fg3_attempted',
        'FG3_PCT': 'fg3_pct',
        'FTM': 'ft_made',
        'FTA': 'ft_attempted',
        'FT_PCT': 'ft_pct',
        'OREB': 'offensive_rebounds',
        'DREB': 'defensive_rebounds',
        'STL': 'steals',
        'BLK': 'blocks',
        'TOV': 'turnovers',
        'PF': 'personal_fouls',
        'PLUS_MINUS': 'plus_minus'
    }

    # Rename columns
    df = df.rename(columns=columns_to_keep)

    # Keep only columns that exist
    final_columns = [col for col in columns_to_keep.values() if col in df.columns]
    df = df[final_columns]

    # Sort by player and date for chronological processing
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    # Remove any duplicate rows (by player_id and game_id)
    duplicates_before = df.duplicated(subset=['player_id', 'game_id']).sum()
    if duplicates_before > 0:
        print(f"Warning: Removing {duplicates_before} duplicate player-game records")
        df = df.drop_duplicates(subset=['player_id', 'game_id'], keep='first')

    # Basic validation
    if verbose:
        print(f"\n✓ Data cleaning complete")
        print(f"  Total games: {len(df):,}")
        print(f"  Unique players: {df['player_id'].nunique():,}")
        print(f"  Unique games: {df['game_id'].nunique():,}")
        print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        print(f"  Seasons: {sorted(df['season'].unique())}")

        # Check for missing values in critical columns
        critical_cols = ['player_id', 'game_id', 'game_date', 'points', 'rebounds', 'assists', 'pra']
        missing_summary = df[critical_cols].isnull().sum()
        if missing_summary.sum() > 0:
            print(f"\n⚠ Missing values in critical columns:")
            print(missing_summary[missing_summary > 0])
        else:
            print(f"\n✓ No missing values in critical columns")

    return df


def save_player_gamelogs(df, filename='player_games.parquet'):
    """Save player game logs to parquet"""
    output_path = NBA_API_DIR / filename
    df.to_parquet(output_path, index=False)
    print(f"\n✓ Saved {len(df):,} game logs to {output_path}")
    return output_path


def load_player_gamelogs(filename='player_games.parquet'):
    """Load player game logs from parquet"""
    input_path = NBA_API_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Player game logs not found at {input_path}\n"
            f"Run csv_data_loader.py to create this file from CSV sources."
        )

    return pd.read_parquet(input_path)


def main():
    """
    Main function to load and save NBA data from CSV files
    Replaces the 10-14 hour API collection with instant CSV loading
    """
    print("=" * 80)
    print("NBA PRA Data Loader - CSV Mode")
    print("=" * 80)

    # Define seasons to load (adjust as needed)
    # Option 1: Load all seasons (22 seasons: 2003-04 to 2024-25)
    seasons = None  # None = load all

    # Option 2: Load specific seasons (10 seasons for model training)
    # seasons = [
    #     '2015-16', '2016-17', '2017-18', '2018-19', '2019-20',
    #     '2020-21', '2021-22', '2022-23', '2023-24', '2024_25'
    # ]

    print(f"\nLoading NBA game logs from CSV files...")
    if seasons:
        print(f"Target seasons: {len(seasons)} seasons")
    else:
        print(f"Target seasons: All available seasons")

    # Load data from CSVs
    gamelogs = load_from_csv_files(CSV_DIR, seasons=seasons)

    # Save to parquet in standard format
    output_path = save_player_gamelogs(gamelogs)

    print("\n" + "=" * 80)
    print("Data Loading Complete!")
    print("=" * 80)
    print(f"\n✓ Data saved to: {output_path}")
    print(f"✓ Ready for feature engineering pipeline")
    print(f"\nNext steps:")
    print(f"  1. Run feature engineering: uv run feature_engineering/run_pipeline.py")
    print(f"  2. Or run individual feature modules as needed")


if __name__ == "__main__":
    main()
