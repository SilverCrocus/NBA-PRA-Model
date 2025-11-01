"""
Base Odds Provider Interface

Defines abstract interface for betting odds providers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import pandas as pd


class OddsProviderError(Exception):
    """Exception raised for odds provider errors"""
    pass


class OddsProvider(ABC):
    """
    Abstract base class for betting odds providers.

    Concrete implementations must provide:
    - fetch_upcoming_games(): Get scheduled games
    - fetch_player_props(event_id): Get player prop markets
    - parse_pra_lines(response): Parse PRA lines from response
    """

    @abstractmethod
    def fetch_upcoming_games(self, target_date: Optional[str] = None) -> List[Dict]:
        """
        Fetch upcoming NBA games.

        Args:
            target_date: Target date (YYYY-MM-DD), defaults to today

        Returns:
            List of game dictionaries with event_id, home_team, away_team, commence_time

        Raises:
            OddsProviderError: If API call fails
        """
        pass

    @abstractmethod
    def fetch_player_props(self, event_id: str) -> Dict:
        """
        Fetch player prop markets for a specific event.

        Args:
            event_id: Event identifier from provider

        Returns:
            Dictionary with player prop data (points, rebounds, assists)

        Raises:
            OddsProviderError: If API call fails
        """
        pass

    @abstractmethod
    def parse_pra_lines(self, event_data: Dict) -> pd.DataFrame:
        """
        Parse PRA betting lines from event data.

        Args:
            event_data: Raw event data from provider

        Returns:
            DataFrame with columns: player_name, pra_line, points_line, rebounds_line,
                                   assists_line, bookmaker, odds, game_date
        """
        pass

    def get_all_pra_lines(self, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch and parse all PRA lines for target date.

        This is the main entry point for clients.

        Args:
            target_date: Target date (YYYY-MM-DD)

        Returns:
            DataFrame with all PRA lines
        """
        games = self.fetch_upcoming_games(target_date)

        all_lines = []
        for game in games:
            try:
                event_data = self.fetch_player_props(game['id'])
                lines = self.parse_pra_lines(event_data)
                all_lines.append(lines)
            except OddsProviderError as e:
                # Log error but continue with other games
                import logging
                logging.warning(f"Failed to fetch props for {game['id']}: {e}")
                continue

        if not all_lines:
            return pd.DataFrame()

        return pd.concat(all_lines, ignore_index=True)
