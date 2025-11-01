"""
Odds Fetcher Module

High-level wrapper around odds provider abstraction.
Integrates with TheOddsAPI to fetch NBA player prop betting lines.

Author: NBA PRA Prediction System
Date: 2025-11-01 (Refactored with provider abstraction)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
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
from production.odds_providers import TheOddsAPIProvider

logger = setup_logging('odds_fetcher')


def get_odds_provider(api_key: Optional[str] = None) -> TheOddsAPIProvider:
    """
    Get configured odds provider.

    Args:
        api_key: Optional API key override (defaults to config)

    Returns:
        Configured TheOddsAPIProvider instance
    """
    return TheOddsAPIProvider(
        api_key=api_key or ODDS_API_KEY,
        base_url=ODDS_API_BASE_URL,
        sport=ODDS_SPORT,
        regions=ODDS_REGIONS,
        markets=ODDS_MARKETS,
        odds_format=ODDS_FORMAT,
        rate_limit_delay=ODDS_API_RATE_LIMIT_DELAY
    )


class OddsFetcher:
    """
    Fetches NBA player prop betting lines using provider abstraction.

    DEPRECATED: This class is maintained for backward compatibility.
    New code should use get_odds_provider() directly.

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
        # Use the provider abstraction
        self.provider = get_odds_provider(api_key)

        # Keep these for backward compatibility
        self.api_key = self.provider.api_key
        self.base_url = self.provider.base_url
        self.sport = self.provider.sport
        self.regions = self.provider.regions
        self.markets = self.provider.markets
        self.odds_format = self.provider.odds_format
        self.rate_limit_delay = self.provider.rate_limit_delay

        logger.info(f"OddsFetcher initialized with API key: {self.api_key[:8]}...")


    def get_upcoming_games(self, date_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Get upcoming NBA games

        Args:
            date_filter: Date to filter games (YYYY-MM-DD), defaults to tomorrow

        Returns:
            DataFrame with columns: event_id, home_team, away_team, commence_time
        """
        logger.info("Fetching upcoming NBA games...")

        # Delegate to provider
        games = self.provider.fetch_upcoming_games(target_date=date_filter)

        if not games:
            logger.warning("No games data received")
            return pd.DataFrame()

        # Convert to DataFrame format expected by existing code
        games_list = []
        for event in games:
            game = {
                'event_id': event['id'],
                'home_team': event['home_team'],
                'away_team': event['away_team'],
                'commence_time': pd.to_datetime(event['commence_time']),
                'completed': event.get('completed', False)
            }
            games_list.append(game)

        df = pd.DataFrame(games_list)

        # Filter for upcoming games only
        now = pd.Timestamp.now(tz='UTC')
        df = df[df['commence_time'] > now].copy()

        logger.info(f"Found {len(df)} upcoming games")
        return df

    def get_player_props(self, event_id: str) -> pd.DataFrame:
        """
        Get player prop lines for a specific game

        Args:
            event_id: TheOddsAPI event ID

        Returns:
            DataFrame with player props (raw format for backward compatibility)
        """
        logger.debug(f"Fetching props for event: {event_id}")

        # Delegate to provider
        try:
            data = self.provider.fetch_player_props(event_id)
        except Exception as e:
            logger.warning(f"No props data for event: {event_id}: {e}")
            return pd.DataFrame()

        if not data:
            logger.warning(f"No props data for event: {event_id}")
            return pd.DataFrame()

        # Parse response to old format for backward compatibility
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
