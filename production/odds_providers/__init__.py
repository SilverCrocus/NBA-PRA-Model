"""
Odds Provider Abstraction Layer

Provides interface for fetching betting odds from multiple providers.
"""
from production.odds_providers.base import OddsProvider, OddsProviderError
from production.odds_providers.theoddsapi import TheOddsAPIProvider

__all__ = ['OddsProvider', 'OddsProviderError', 'TheOddsAPIProvider']
