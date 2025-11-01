"""
Tests for odds provider abstraction layer.
"""
import pytest
import pandas as pd
from datetime import datetime
from production.odds_providers.base import OddsProvider, OddsProviderError
from production.odds_providers.theoddsapi import TheOddsAPIProvider


def test_odds_provider_interface():
    """Test that OddsProvider defines required interface"""
    # OddsProvider should be abstract and cannot be instantiated
    with pytest.raises(TypeError):
        OddsProvider()


def test_theoddsapi_provider_initialization(monkeypatch):
    """Test TheOddsAPI provider initialization"""
    # Mock API key
    monkeypatch.setenv('ODDS_API_FREE_KEY', 'test_key_123')

    provider = TheOddsAPIProvider(api_key='test_key_123')

    assert provider.api_key == 'test_key_123'
    assert provider.base_url == "https://api.the-odds-api.com/v4"


def test_theoddsapi_provider_fetch_games(monkeypatch, sample_odds_response):
    """Test fetching upcoming games"""
    from unittest.mock import Mock, patch

    provider = TheOddsAPIProvider(api_key='test_key')

    # Mock response object
    mock_response = Mock()
    mock_response.json.return_value = [sample_odds_response]
    mock_response.raise_for_status = Mock()

    with patch('requests.get', return_value=mock_response):
        games = provider.fetch_upcoming_games()

        assert len(games) == 1
        assert games[0]['home_team'] == 'Los Angeles Lakers'
        assert games[0]['away_team'] == 'Boston Celtics'


def test_theoddsapi_provider_fetch_player_props(monkeypatch, sample_odds_response):
    """Test fetching player props for event"""
    from unittest.mock import Mock, patch

    provider = TheOddsAPIProvider(api_key='test_key')

    # Mock response object
    mock_response = Mock()
    mock_response.json.return_value = sample_odds_response
    mock_response.raise_for_status = Mock()

    with patch('requests.get', return_value=mock_response):
        props = provider.fetch_player_props('event123')

        # The function returns the raw dict, check that bookmakers contain markets
        assert 'bookmakers' in props
        markets = props['bookmakers'][0]['markets']
        market_keys = [m['key'] for m in markets]
        assert 'player_points' in market_keys
        assert 'player_rebounds' in market_keys
        assert 'player_assists' in market_keys


def test_theoddsapi_provider_parse_pra_lines(sample_odds_response):
    """Test parsing PRA lines from API response"""
    provider = TheOddsAPIProvider(api_key='test_key')

    pra_lines = provider.parse_pra_lines(sample_odds_response)

    assert not pra_lines.empty
    assert 'player_name' in pra_lines.columns
    assert 'pra_line' in pra_lines.columns
    assert pra_lines.iloc[0]['player_name'] == 'LeBron James'
    assert pra_lines.iloc[0]['pra_line'] == 39.5  # 25.5 + 7.5 + 6.5


def test_theoddsapi_provider_rate_limiting():
    """Test rate limiting between API calls"""
    import time

    provider = TheOddsAPIProvider(api_key='test_key', rate_limit_delay=0.1)

    start = time.time()
    provider._apply_rate_limit()
    provider._apply_rate_limit()
    elapsed = time.time() - start

    # Should have waited at least 0.1 seconds
    assert elapsed >= 0.1


def test_odds_provider_error_handling():
    """Test error handling for failed API calls"""
    from unittest.mock import Mock, patch
    import requests

    provider = TheOddsAPIProvider(api_key='invalid_key')

    # Mock response with 401 error
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()

    with patch('requests.get', return_value=mock_response):
        with pytest.raises(OddsProviderError):
            provider.fetch_upcoming_games()
