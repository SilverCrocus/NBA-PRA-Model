"""
Simple Upcoming Games Fetcher

Fetches upcoming NBA games and prepares features for prediction.

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
from pathlib import Path
from typing import List, Dict, Optional

from nba_api.stats.endpoints import scoreboardv2, commonteamroster
from nba_api.stats.static import teams

from production.config import MASTER_FEATURES_PATH
from production.logging_config import setup_production_logging
from production.exceptions import FeatureDataError

logger = setup_production_logging('upcoming_games_fetcher')


def get_upcoming_games(target_date: str) -> pd.DataFrame:
    """
    Fetch upcoming games from NBA API

    Args:
        target_date: Date in YYYY-MM-DD or MM/DD/YYYY format

    Returns:
        DataFrame with upcoming games (game_id, home_team, away_team, game_date)
    """
    logger.info(f"Fetching upcoming games for {target_date}")

    try:
        # Convert to MM/DD/YYYY format for NBA API
        dt = pd.to_datetime(target_date)
        nba_date = dt.strftime('%m/%d/%Y')

        # Fetch scoreboard
        scoreboard = scoreboardv2.ScoreboardV2(game_date=nba_date)
        games_df = scoreboard.get_data_frames()[0]

        if games_df.empty:
            logger.warning(f"No games found for {target_date}")
            return pd.DataFrame()

        # Extract relevant columns
        games = games_df[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']].copy()
        games['game_date'] = dt

        logger.info(f"Found {len(games)} upcoming games")

        return games

    except Exception as e:
        logger.error(f"Error fetching upcoming games: {e}")
        return pd.DataFrame()


def get_team_roster(team_id: int) -> List[int]:
    """
    Get current roster for a team

    Args:
        team_id: NBA team ID

    Returns:
        List of player IDs
    """
    try:
        time.sleep(0.6)  # Rate limiting
        roster = commonteamroster.CommonTeamRoster(team_id=team_id)
        roster_df = roster.get_data_frames()[0]

        if not roster_df.empty:
            return roster_df['PLAYER_ID'].tolist()
        return []

    except Exception as e:
        logger.error(f"Error fetching roster for team {team_id}: {e}")
        return []


def get_players_for_upcoming_games(games_df: pd.DataFrame) -> List[int]:
    """
    Get all players expected to play in upcoming games

    Args:
        games_df: DataFrame with upcoming games

    Returns:
        List of unique player IDs
    """
    logger.info("Fetching player rosters for upcoming games...")

    all_players = []

    # Get unique teams
    home_teams = games_df['HOME_TEAM_ID'].unique()
    away_teams = games_df['VISITOR_TEAM_ID'].unique()
    unique_teams = list(set(list(home_teams) + list(away_teams)))

    logger.info(f"Fetching rosters for {len(unique_teams)} teams...")

    for team_id in unique_teams:
        roster = get_team_roster(team_id)
        all_players.extend(roster)
        logger.info(f"  Team {team_id}: {len(roster)} players")

    unique_players = list(set(all_players))
    logger.info(f"Total unique players: {len(unique_players)}")

    return unique_players


def load_features_for_players(player_ids: List[int], target_date: str) -> pd.DataFrame:
    """
    Load most recent features for players from historical data

    Args:
        player_ids: List of player IDs
        target_date: Target game date

    Returns:
        DataFrame with features for each player
    """
    logger.info("Loading historical features for players...")

    # Load master features
    df = pd.read_parquet(MASTER_FEATURES_PATH)
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Filter to relevant players
    df = df[df['player_id'].isin(player_ids)].copy()

    # Get most recent game for each player (before target date)
    target_dt = pd.to_datetime(target_date)
    df = df[df['game_date'] < target_dt].copy()

    # Sort and get most recent
    df = df.sort_values('game_date', ascending=False)
    latest_features = df.groupby('player_id').first().reset_index()

    logger.info(f"Loaded features for {len(latest_features)} players")

    # Update game_date to target date (we're predicting for this date)
    latest_features['game_date'] = target_dt

    return latest_features


def get_upcoming_game_features(target_date: str) -> pd.DataFrame:
    """
    Main function: Get features for players in upcoming games

    Args:
        target_date: Date to predict (YYYY-MM-DD)

    Returns:
        DataFrame with features ready for prediction
    """
    logger.info(f"Preparing features for upcoming games on {target_date}")

    # 1. Get upcoming games
    games = get_upcoming_games(target_date)

    if games.empty:
        logger.warning("No upcoming games found")
        return pd.DataFrame()

    # 2. Get player rosters
    player_ids = get_players_for_upcoming_games(games)

    if not player_ids:
        logger.warning("No players found for upcoming games")
        return pd.DataFrame()

    # 3. Load features using historical data as context
    features = load_features_for_players(player_ids, target_date)

    if features.empty:
        logger.warning("No features loaded for players")
        return pd.DataFrame()

    logger.info(f"✓ Prepared {len(features)} player-games for prediction")

    return features


if __name__ == "__main__":
    """Test the fetcher"""

    print("Testing Upcoming Games Fetcher...")
    print("-" * 60)

    # Test with Oct 31
    target = "2025-10-31"

    features = get_upcoming_game_features(target)

    if not features.empty:
        print(f"\n✓ Got features for {len(features)} players")
        print(f"✓ Features shape: {features.shape}")
        print("\nSample columns:")
        print(features.columns[:10].tolist())
        print("\nSample data:")
        display_cols = [col for col in ['player_id', 'player_name', 'game_date', 'team_abbreviation'] if col in features.columns]
        if display_cols:
            print(features[display_cols].head(10))
    else:
        print("\n✗ No features generated")
