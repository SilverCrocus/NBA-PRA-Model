"""
Odds Fetcher Module

Integrates with TheOddsAPI to fetch NBA player prop betting lines.

Author: NBA PRA Prediction System
Date: 2025-10-31
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging

from production.config import (
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    ODDS_SPORT,
    ODDS_REGIONS,
    ODDS_MARKETS,
    ODDS_FORMAT,
    ODDS_API_RATE_LIMIT_DELAY,
    setup_logging
)

logger = setup_logging('odds_fetcher')


class OddsFetcher:
    """
    Fetches NBA player prop betting lines from TheOddsAPI

    Features:
    - Fetch upcoming NBA games
    - Fetch player props (points, rebounds, assists)
    - Calculate PRA lines from individual markets
    - Rate limiting for API quota management
    - Error handling and retries
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize odds fetcher

        Args:
            api_key: TheOddsAPI key (defaults to config)
        """
        self.api_key = api_key or ODDS_API_KEY
        self.base_url = ODDS_API_BASE_URL
        self.sport = ODDS_SPORT
        self.regions = ODDS_REGIONS
        self.markets = ODDS_MARKETS
        self.odds_format = ODDS_FORMAT
        self.rate_limit_delay = ODDS_API_RATE_LIMIT_DELAY

        logger.info(f"OddsFetcher initialized with API key: {self.api_key[:8]}...")

    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make API request with error handling

        Args:
            endpoint: API endpoint (e.g., '/sports/basketball_nba/events')
            params: Query parameters

        Returns:
            JSON response or None if error
        """
        url = f"{self.base_url}{endpoint}"
        params['apiKey'] = self.api_key

        try:
            logger.debug(f"Making request to: {endpoint}")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            # Check remaining quota
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                logger.info(f"API requests remaining: {remaining}")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            if response.status_code == 401:
                logger.error("Invalid API key")
            elif response.status_code == 429:
                logger.error("Rate limit exceeded")
            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def get_upcoming_games(self, date_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Get upcoming NBA games

        Args:
            date_filter: Date to filter games (YYYY-MM-DD), defaults to tomorrow

        Returns:
            DataFrame with columns: event_id, home_team, away_team, commence_time
        """
        endpoint = f"/sports/{self.sport}/events"
        params = {
            'dateFormat': 'iso',
            'oddsFormat': self.odds_format
        }

        logger.info("Fetching upcoming NBA games...")
        data = self._make_request(endpoint, params)

        if not data:
            logger.warning("No games data received")
            return pd.DataFrame()

        # Parse response
        games = []
        for event in data:
            game = {
                'event_id': event['id'],
                'home_team': event['home_team'],
                'away_team': event['away_team'],
                'commence_time': pd.to_datetime(event['commence_time']),
                'completed': event.get('completed', False)
            }
            games.append(game)

        df = pd.DataFrame(games)

        # Filter for upcoming games only
        now = pd.Timestamp.now(tz='UTC')
        df = df[df['commence_time'] > now].copy()

        # Optional date filter (convert UTC to EST for comparison)
        if date_filter:
            # Convert UTC times to EST for date comparison
            df['commence_time_est'] = df['commence_time'].dt.tz_convert('US/Eastern')
            target_date = pd.to_datetime(date_filter).date()
            df = df[df['commence_time_est'].dt.date == target_date].copy()
            df = df.drop(columns=['commence_time_est'])  # Clean up temp column

        logger.info(f"Found {len(df)} upcoming games")
        return df

    def get_player_props(self, event_id: str) -> pd.DataFrame:
        """
        Get player prop lines for a specific game

        Args:
            event_id: TheOddsAPI event ID

        Returns:
            DataFrame with player props
        """
        endpoint = f"/sports/{self.sport}/events/{event_id}/odds"
        params = {
            'regions': self.regions,
            'markets': self.markets,
            'oddsFormat': self.odds_format,
            'dateFormat': 'iso'
        }

        logger.debug(f"Fetching props for event: {event_id}")
        data = self._make_request(endpoint, params)

        if not data:
            logger.warning(f"No props data for event: {event_id}")
            return pd.DataFrame()

        # Parse response
        props = []

        try:
            # Get bookmakers
            bookmakers = data.get('bookmakers', [])

            for bookmaker in bookmakers:
                bookmaker_name = bookmaker['key']
                markets = bookmaker.get('markets', [])

                for market in markets:
                    market_key = market['key']  # e.g., 'player_points'

                    for outcome in market.get('outcomes', []):
                        # Extract player name from description field
                        # TheOddsAPI structure: outcome['description'] = player name, outcome['name'] = 'Over'/'Under'
                        player_name = outcome.get('description')

                        if not player_name:
                            # Fallback: if no description, skip this outcome
                            logger.warning(f"No player name found in outcome: {outcome}")
                            continue

                        # Extract player name and prop details
                        prop = {
                            'event_id': event_id,
                            'bookmaker': bookmaker_name,
                            'market': market_key,
                            'player_name': player_name,
                            'over_under': outcome.get('name'),  # 'Over' or 'Under'
                            'line': outcome.get('point'),
                            'odds': outcome.get('price')
                        }
                        props.append(prop)

            df = pd.DataFrame(props)
            logger.debug(f"Found {len(df)} prop outcomes for event {event_id}")

            # DEBUG: Log sample player names
            if not df.empty and 'player_name' in df.columns:
                sample_names = df['player_name'].unique()[:3]
                logger.debug(f"Sample player names from API: {list(sample_names)}")

            return df

        except (KeyError, TypeError) as e:
            logger.error(f"Error parsing props data: {e}")
            return pd.DataFrame()

    def calculate_pra_lines(self, props_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate PRA lines from individual markets (points, rebounds, assists)

        Args:
            props_df: DataFrame with individual prop markets

        Returns:
            DataFrame with PRA lines and odds
        """
        if props_df.empty:
            return pd.DataFrame()

        # Pivot to get one row per player-bookmaker with separate columns for each stat
        # Filter to just the 'Over' lines (we only need the line value)
        over_props = props_df[props_df['over_under'] == 'Over'].copy()

        # Pivot: rows = player-bookmaker, columns = market, values = line
        try:
            pra_pivot = over_props.pivot_table(
                index=['event_id', 'player_name', 'bookmaker'],
                columns='market',
                values='line',
                aggfunc='first'
            ).reset_index()

            # Also pivot to get odds (use points odds as proxy for PRA)
            odds_pivot = over_props.pivot_table(
                index=['event_id', 'player_name', 'bookmaker'],
                columns='market',
                values='odds',
                aggfunc='first'
            ).reset_index()

            # Calculate PRA if all three markets exist
            required_markets = ['player_points', 'player_rebounds', 'player_assists']

            if all(market in pra_pivot.columns for market in required_markets):
                pra_pivot['pra_line'] = (
                    pra_pivot['player_points'].fillna(0) +
                    pra_pivot['player_rebounds'].fillna(0) +
                    pra_pivot['player_assists'].fillna(0)
                )

                # Add odds (use points odds as proxy)
                if 'player_points' in odds_pivot.columns:
                    pra_pivot['pra_odds'] = odds_pivot['player_points']
                else:
                    pra_pivot['pra_odds'] = -110  # Fallback to standard odds

                # Keep only complete PRA lines
                pra_pivot = pra_pivot[pra_pivot[required_markets].notna().all(axis=1)].copy()

                logger.info(f"Calculated PRA lines for {len(pra_pivot)} player-bookmaker pairs")

                # DEBUG: Log sample player names
                if not pra_pivot.empty and 'player_name' in pra_pivot.columns:
                    sample_names = pra_pivot['player_name'].unique()[:5]
                    logger.info(f"Sample players with PRA lines: {list(sample_names)}")
            else:
                logger.warning("Not all required markets (points, rebounds, assists) available")
                pra_pivot['pra_line'] = np.nan
                pra_pivot['pra_odds'] = -110

            return pra_pivot

        except Exception as e:
            logger.error(f"Error calculating PRA lines: {e}")
            return pd.DataFrame()

    def get_all_pra_lines(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get PRA lines for all upcoming games on target date

        Args:
            target_date: Date to fetch lines for (YYYY-MM-DD), defaults to tomorrow

        Returns:
            DataFrame with PRA lines for all players
        """
        # Default to tomorrow
        if target_date is None:
            tomorrow = datetime.now() + timedelta(days=1)
            target_date = tomorrow.strftime('%Y-%m-%d')

        logger.info(f"Fetching PRA lines for {target_date}")

        # Get upcoming games
        games = self.get_upcoming_games(date_filter=target_date)

        if games.empty:
            logger.warning(f"No games found for {target_date}")
            return pd.DataFrame()

        logger.info(f"Found {len(games)} games on {target_date}")

        # Fetch props for each game
        all_pra_lines = []

        for _, game in games.iterrows():
            event_id = game['event_id']
            logger.info(f"Fetching props for {game['away_team']} @ {game['home_team']}")

            # Get player props
            props = self.get_player_props(event_id)

            if props.empty:
                logger.warning(f"No props for event {event_id}")
                continue

            # Calculate PRA lines
            pra_lines = self.calculate_pra_lines(props)

            if not pra_lines.empty:
                # Add game info
                pra_lines['home_team'] = game['home_team']
                pra_lines['away_team'] = game['away_team']
                pra_lines['commence_time'] = game['commence_time']
                pra_lines['game_date'] = game['commence_time'].date()

                all_pra_lines.append(pra_lines)

        # Combine all games
        if all_pra_lines:
            result = pd.concat(all_pra_lines, ignore_index=True)
            logger.info(f"Total PRA lines fetched: {len(result)}")
            return result
        else:
            logger.warning("No PRA lines found")
            return pd.DataFrame()

    def export_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Export PRA lines to CSV

        Args:
            df: DataFrame with PRA lines
            filename: Output filename
        """
        if df.empty:
            logger.warning("No data to export")
            return

        df.to_csv(filename, index=False)
        logger.info(f"Exported {len(df)} lines to {filename}")


def fetch_tomorrow_odds(output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to fetch tomorrow's PRA betting lines

    Args:
        output_path: Optional CSV path to save results

    Returns:
        DataFrame with PRA lines
    """
    fetcher = OddsFetcher()

    # Get tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)
    target_date = tomorrow.strftime('%Y-%m-%d')

    # Fetch lines
    pra_lines = fetcher.get_all_pra_lines(target_date=target_date)

    # Export if requested
    if output_path and not pra_lines.empty:
        fetcher.export_to_csv(pra_lines, output_path)

    return pra_lines


if __name__ == "__main__":
    """Test the odds fetcher"""

    print("Testing OddsFetcher...")
    print("-" * 60)

    # Test fetching tomorrow's odds
    pra_lines = fetch_tomorrow_odds()

    if not pra_lines.empty:
        print(f"\nFetched {len(pra_lines)} PRA lines")
        print("\nSample:")
        print(pra_lines.head(10))

        print("\nBookmakers:")
        print(pra_lines['bookmaker'].value_counts())

        print("\nPRA line distribution:")
        print(pra_lines['pra_line'].describe())
    else:
        print("\nNo PRA lines found (possibly no games tomorrow or API issue)")
