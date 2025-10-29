"""
Data Loader for NBA PRA Model
Fetches NBA API box scores and consolidates CTG data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players, teams
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CTG_DATA_DIR = DATA_DIR / "ctg_data_organized"
NBA_API_DIR = DATA_DIR / "nba_api"
NBA_API_DIR.mkdir(exist_ok=True)


def get_all_players():
    """Get list of all NBA players"""
    all_players = players.get_players()
    return pd.DataFrame(all_players)


def fetch_player_game_logs(player_id, seasons, delay=2.0):
    """
    Fetch game logs for a player across multiple seasons

    Args:
        player_id: NBA player ID
        seasons: List of season strings (e.g., ['2023-24', '2022-23'])
        delay: Seconds to wait between API calls (default 2.0)

    Returns:
        DataFrame with player game logs
    """
    all_games = []

    for season in seasons:
        try:
            time.sleep(delay)  # Rate limiting

            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season'
            )

            df = gamelog.get_data_frames()[0]
            if not df.empty:
                df['SEASON'] = season
                all_games.append(df)

        except Exception as e:
            print(f"Error fetching {season} for player {player_id}: {str(e)}")
            continue

    if all_games:
        return pd.concat(all_games, ignore_index=True)
    return pd.DataFrame()


def calculate_pra(df):
    """Calculate PRA (Points + Rebounds + Assists)"""
    df['PRA'] = df['PTS'] + df['REB'] + df['AST']
    return df


def fetch_all_player_gamelogs(seasons, sample_size=None):
    """
    Fetch game logs for all players across seasons

    Args:
        seasons: List of season strings (e.g., ['2023-24', '2022-23'])
        sample_size: If set, only fetch this many players (for testing)

    Returns:
        DataFrame with all player game logs
    """
    all_players_df = get_all_players()

    if sample_size:
        all_players_df = all_players_df.head(sample_size)

    all_gamelogs = []

    print(f"Fetching game logs for {len(all_players_df)} players across {len(seasons)} seasons...")

    for _, player in tqdm(all_players_df.iterrows(), total=len(all_players_df)):
        player_id = player['id']
        player_name = player['full_name']

        gamelogs = fetch_player_game_logs(player_id, seasons)

        if not gamelogs.empty:
            gamelogs['PLAYER_NAME'] = player_name
            gamelogs['PLAYER_ID'] = player_id
            all_gamelogs.append(gamelogs)

    if all_gamelogs:
        combined = pd.concat(all_gamelogs, ignore_index=True)
        combined = calculate_pra(combined)
        return combined

    return pd.DataFrame()


def clean_nba_api_data(df):
    """
    Clean and standardize NBA API game log data

    Args:
        df: Raw game log DataFrame from NBA API

    Returns:
        Cleaned DataFrame with standardized columns
    """
    # Convert game date to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Extract home/away from MATCHUP (e.g., "LAL vs. BOS" or "LAL @ BOS")
    df['IS_HOME'] = df['MATCHUP'].str.contains('vs.').astype(int)

    # Extract opponent team
    df['OPPONENT'] = df['MATCHUP'].str.extract(r'(vs\.|@)\s+([A-Z]{3})')[1]

    # Select and rename key columns
    columns_to_keep = {
        'PLAYER_ID': 'player_id',
        'PLAYER_NAME': 'player_name',
        'Game_ID': 'game_id',
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

    df = df.rename(columns=columns_to_keep)
    df = df[[col for col in columns_to_keep.values() if col in df.columns]]

    # Sort by player and date
    df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)

    return df


def load_ctg_player_data(season='2023-24', season_type='regular_season'):
    """
    Load CTG player data for a specific season

    Args:
        season: Season string (e.g., '2023-24')
        season_type: 'regular_season' or 'playoffs'

    Returns:
        Dictionary of DataFrames, one per CTG stat category
    """
    season_path = CTG_DATA_DIR / 'players' / season / season_type

    if not season_path.exists():
        print(f"Warning: CTG data not found at {season_path}")
        return {}

    ctg_data = {}

    # Load each CTG stat file
    stat_files = [
        'offensive_overview.csv',
        'shooting_accuracy.csv',
        'shooting_frequency.csv',
        'shooting_overall.csv',
        'defense_rebounding.csv',
        'foul_drawing.csv'
    ]

    for stat_file in stat_files:
        stat_name = stat_file.replace('.csv', '')
        file_path = season_path / stat_name / stat_file
        if file_path.exists():
            ctg_data[stat_name] = pd.read_csv(file_path)

    # Load on/off data
    onoff_path = season_path / 'on_off'
    if onoff_path.exists():
        for onoff_file in onoff_path.glob('*.csv'):
            stat_name = f"onoff_{onoff_file.stem}"
            ctg_data[stat_name] = pd.read_csv(onoff_file)

    return ctg_data


def consolidate_ctg_data_all_seasons():
    """
    Consolidate all CTG player data across seasons

    Returns:
        DataFrame with all CTG offensive overview data (player-season grain)
    """
    all_seasons = []

    players_dir = CTG_DATA_DIR / 'players'

    for season_dir in sorted(players_dir.glob('*')):
        if season_dir.is_dir():
            season = season_dir.name

            # Load regular season offensive overview (main stats)
            reg_season_path = season_dir / 'regular_season' / 'offensive_overview' / 'offensive_overview.csv'

            if reg_season_path.exists():
                df = pd.read_csv(reg_season_path)
                df['season'] = season
                df['season_type'] = 'regular_season'
                all_seasons.append(df)

    if all_seasons:
        combined = pd.concat(all_seasons, ignore_index=True)

        # Standardize column names
        combined.columns = combined.columns.str.lower().str.replace(' ', '_')

        # Convert percentage columns to numeric (remove '%' and divide by 100)
        percentage_cols = [col for col in combined.columns if '%' in col]
        for col in percentage_cols:
            combined[col] = combined[col].astype(str).str.replace('%', '').astype(float) / 100

        # Convert 'usage' column which also has percentage values but % not in column name
        if 'usage' in combined.columns:
            combined['usage'] = combined['usage'].astype(str).str.replace('%', '').astype(float) / 100

        # Convert other numeric columns
        numeric_cols = ['psa', 'ast:usg', 'age', 'min']
        for col in numeric_cols:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors='coerce')

        return combined

    return pd.DataFrame()


def save_player_gamelogs(df, filename='player_games.parquet'):
    """Save player game logs to parquet"""
    output_path = NBA_API_DIR / filename
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} game logs to {output_path}")


def load_player_gamelogs(filename='player_games.parquet'):
    """Load player game logs from parquet"""
    input_path = NBA_API_DIR / filename

    if not input_path.exists():
        raise FileNotFoundError(f"Player game logs not found at {input_path}")

    return pd.read_parquet(input_path)


def main():
    """
    Main function to fetch and save NBA API data
    Run this to collect the base dataset
    """
    # Define seasons to fetch (adjust as needed)
    # 10 seasons: 2024-25 (current) + 9 historical seasons back to 2015-16
    seasons = [
        '2024-25', '2023-24', '2022-23', '2021-22', '2020-21',
        '2019-20', '2018-19', '2017-18', '2016-17', '2015-16'
    ]

    print("Starting NBA API data collection...")
    print("This will take several hours due to rate limiting (2 sec per request)")
    print(f"Fetching seasons: {', '.join(seasons)}")

    # For testing, set sample_size to a small number (e.g., 10)
    # For production, set to None to fetch all players
    SAMPLE_SIZE = None  # Testing with 10 players

    # Fetch data
    gamelogs = fetch_all_player_gamelogs(seasons, sample_size=SAMPLE_SIZE)

    if not gamelogs.empty:
        # Clean and standardize
        gamelogs = clean_nba_api_data(gamelogs)

        # Save to parquet
        save_player_gamelogs(gamelogs)

        print(f"\nData collection complete!")
        print(f"Total games: {len(gamelogs)}")
        print(f"Total players: {gamelogs['player_id'].nunique()}")
        print(f"Date range: {gamelogs['game_date'].min()} to {gamelogs['game_date'].max()}")
    else:
        print("No data collected!")


if __name__ == "__main__":
    main()
