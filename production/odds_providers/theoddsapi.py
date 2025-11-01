"""
TheOddsAPI Provider

Concrete implementation of OddsProvider for TheOddsAPI.com
"""
import requests
import pandas as pd
import time
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from production.odds_providers.base import OddsProvider, OddsProviderError

logger = logging.getLogger(__name__)


class TheOddsAPIProvider(OddsProvider):
    """
    TheOddsAPI.com provider implementation.

    Free tier: 500 requests/month
    Docs: https://the-odds-api.com/liveapi/guides/v4/
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.the-odds-api.com/v4",
        sport: str = "basketball_nba",
        regions: str = "us",
        markets: str = "player_points,player_rebounds,player_assists",
        odds_format: str = "american",
        rate_limit_delay: float = 1.0
    ):
        """
        Initialize TheOddsAPI provider.

        Args:
            api_key: TheOddsAPI key
            base_url: API base URL
            sport: Sport key (default: basketball_nba)
            regions: Bookmaker regions (default: us)
            markets: Prop markets to fetch
            odds_format: Odds format (american, decimal)
            rate_limit_delay: Delay between API calls (seconds)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.sport = sport
        self.regions = regions
        self.markets = markets
        self.odds_format = odds_format
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0

        logger.info(f"Initialized TheOddsAPI provider (sport={sport}, regions={regions})")

    def _apply_rate_limit(self):
        """Apply rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make authenticated API request.

        Args:
            endpoint: API endpoint (relative to base_url)
            params: Query parameters

        Returns:
            JSON response

        Raises:
            OddsProviderError: If request fails
        """
        self._apply_rate_limit()

        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params['apiKey'] = self.api_key

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise OddsProviderError("Invalid API key") from e
            elif response.status_code == 429:
                raise OddsProviderError("Rate limit exceeded") from e
            else:
                raise OddsProviderError(f"HTTP error {response.status_code}: {e}") from e

        except requests.exceptions.RequestException as e:
            raise OddsProviderError(f"Request failed: {e}") from e

    def fetch_upcoming_games(self, target_date: Optional[str] = None) -> List[Dict]:
        """Fetch upcoming NBA games"""
        logger.info(f"Fetching upcoming games for {target_date or 'today'}")

        endpoint = f"sports/{self.sport}/events"
        params = {
            'regions': self.regions,
            'oddsFormat': self.odds_format
        }

        games = self._make_request(endpoint, params)

        # Filter by target date if specified
        if target_date:
            target_dt = datetime.strptime(target_date, '%Y-%m-%d')
            games = [
                g for g in games
                if datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00')).date() == target_dt.date()
            ]

        logger.info(f"Found {len(games)} games")
        return games

    def fetch_player_props(self, event_id: str) -> Dict:
        """Fetch player props for event"""
        logger.info(f"Fetching player props for event {event_id}")

        endpoint = f"sports/{self.sport}/events/{event_id}/odds"
        params = {
            'regions': self.regions,
            'markets': self.markets,
            'oddsFormat': self.odds_format
        }

        return self._make_request(endpoint, params)

    def parse_pra_lines(self, event_data: Dict) -> pd.DataFrame:
        """
        Parse PRA lines from event data.

        Combines player_points, player_rebounds, player_assists into PRA total.
        """
        if not event_data or 'bookmakers' not in event_data:
            return pd.DataFrame()

        rows = []

        for bookmaker in event_data['bookmakers']:
            bookmaker_name = bookmaker['key']

            # Extract markets
            player_stats = {'points': {}, 'rebounds': {}, 'assists': {}}

            for market in bookmaker.get('markets', []):
                market_key = market['key']

                if market_key == 'player_points':
                    for outcome in market['outcomes']:
                        player_name = outcome['name']
                        player_stats['points'][player_name] = outcome['point']

                elif market_key == 'player_rebounds':
                    for outcome in market['outcomes']:
                        player_name = outcome['name']
                        player_stats['rebounds'][player_name] = outcome['point']

                elif market_key == 'player_assists':
                    for outcome in market['outcomes']:
                        player_name = outcome['name']
                        player_stats['assists'][player_name] = outcome['point']

            # Calculate PRA for players with all three stats
            all_players = set(player_stats['points'].keys()) & \
                         set(player_stats['rebounds'].keys()) & \
                         set(player_stats['assists'].keys())

            for player_name in all_players:
                pra_line = (
                    player_stats['points'][player_name] +
                    player_stats['rebounds'][player_name] +
                    player_stats['assists'][player_name]
                )

                rows.append({
                    'player_name': player_name,
                    'pra_line': pra_line,
                    'points_line': player_stats['points'][player_name],
                    'rebounds_line': player_stats['rebounds'][player_name],
                    'assists_line': player_stats['assists'][player_name],
                    'bookmaker': bookmaker_name,
                    'odds': -110,  # Standard odds
                    'game_date': event_data.get('commence_time', '')[:10]
                })

        return pd.DataFrame(rows)
