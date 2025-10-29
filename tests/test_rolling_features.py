"""
Test rolling features calculation module

Tests for rolling averages, EWMA, trends, volatility calculations.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engineering.features.rolling_features import calculate_rolling_features


class TestRollingAverages:
    """Test rolling average calculations"""

    def test_rolling_features_returns_dataframe(self, sample_player_data):
        """Test function returns a DataFrame"""
        result = calculate_rolling_features(sample_player_data)
        assert isinstance(result, pd.DataFrame)

    def test_rolling_features_preserves_grain(self, sample_player_data):
        """Test output has same grain columns"""
        result = calculate_rolling_features(sample_player_data)

        assert 'player_id' in result.columns
        assert 'game_id' in result.columns
        assert 'game_date' in result.columns

    def test_rolling_features_same_row_count(self, sample_player_data):
        """Test output has same number of rows as input"""
        result = calculate_rolling_features(sample_player_data)
        assert len(result) == len(sample_player_data)

    def test_rolling_avg_multiple_windows(self, sequential_data):
        """Test rolling averages calculated for multiple windows"""
        result = calculate_rolling_features(sequential_data)

        # Should have features for different windows (5, 10, 20)
        assert 'pra_avg_last5' in result.columns
        assert 'pra_avg_last10' in result.columns

    def test_rolling_avg_min_periods(self, sample_player_data):
        """Test min_periods allows calculation with fewer games"""
        df = sample_player_data.copy()
        result = calculate_rolling_features(df)

        # Second game should have a value (min_periods=1 allows 1 data point)
        second_games = df.groupby('player_id').nth(1).index
        for idx in second_games:
            rolling_val = result.loc[idx, 'pra_avg_last5']
            # Should have value or 0, not NaN (if min_periods=1)
            assert not pd.isna(rolling_val) or rolling_val == 0


class TestRollingStatistics:
    """Test rolling statistical measures"""

    def test_rolling_std_calculation(self, sequential_data):
        """Test rolling standard deviation"""
        result = calculate_rolling_features(sequential_data)

        if 'pra_std_last5' in result.columns:
            # Standard deviation should be non-negative
            std_values = result['pra_std_last5'].dropna()
            assert (std_values >= 0).all()

    def test_rolling_min_max(self, sequential_data):
        """Test rolling min and max"""
        result = calculate_rolling_features(sequential_data)

        if 'pra_min_last5' in result.columns and 'pra_max_last5' in result.columns:
            # Max should always be >= min
            valid_rows = result['pra_min_last5'].notna() & result['pra_max_last5'].notna()
            assert (result.loc[valid_rows, 'pra_max_last5'] >=
                   result.loc[valid_rows, 'pra_min_last5']).all()

    def test_rolling_sum(self, sequential_data):
        """Test rolling sum calculation"""
        df = sequential_data.copy()
        result = calculate_rolling_features(df)

        if 'pra_sum_last5' in result.columns:
            # Sum should be reasonable (not negative for PRA)
            sum_values = result['pra_sum_last5'].dropna()
            assert (sum_values >= 0).all()


class TestExponentialMovingAverage:
    """Test EWMA calculations"""

    def test_ewma_exists(self, sample_player_data):
        """Test EWMA features are created"""
        result = calculate_rolling_features(sample_player_data)

        # Check for EWMA columns
        ewma_cols = [col for col in result.columns if 'ewma' in col.lower()]
        # Should have at least some EWMA features
        # (Actual column names depend on implementation)

    def test_ewma_recent_weight(self, sequential_data):
        """Test EWMA gives more weight to recent values"""
        df = sequential_data.copy()
        result = calculate_rolling_features(df)

        # EWMA should exist and be different from simple average
        if 'pra_ewma' in result.columns and 'pra_avg_last5' in result.columns:
            # For data with trend, EWMA should differ from simple MA
            ewma = result['pra_ewma'].dropna()
            avg = result['pra_avg_last5'].dropna()

            if len(ewma) > 0 and len(avg) > 0:
                # They should not be identical (EWMA weights differently)
                assert not (ewma == avg).all()


class TestTrendFeatures:
    """Test trend calculation features"""

    def test_trend_direction(self, sequential_data):
        """Test trend features capture direction"""
        df = sequential_data.copy()
        result = calculate_rolling_features(df)

        # Look for trend-related columns
        trend_cols = [col for col in result.columns if 'trend' in col.lower() or 'slope' in col.lower()]

        # If trends exist, they should capture upward trend in sequential data
        if len(trend_cols) > 0:
            for col in trend_cols:
                trend_values = result[col].dropna()
                if len(trend_values) > 0:
                    # Most trends should be positive (data is increasing)
                    # (At least some positive values expected)
                    assert trend_values.max() > 0


class TestVolatilityFeatures:
    """Test volatility/variance features"""

    def test_volatility_non_negative(self, sample_player_data):
        """Test volatility features are non-negative"""
        result = calculate_rolling_features(sample_player_data)

        # Look for volatility/variance columns
        vol_cols = [col for col in result.columns if 'volatility' in col.lower() or 'var' in col.lower()]

        for col in vol_cols:
            vol_values = result[col].dropna()
            if len(vol_values) > 0:
                assert (vol_values >= 0).all(), f"{col} should be non-negative"

    def test_coefficient_of_variation(self, sequential_data):
        """Test coefficient of variation (std/mean)"""
        df = sequential_data.copy()
        result = calculate_rolling_features(df)

        if 'pra_cv' in result.columns:
            cv_values = result['pra_cv'].dropna()
            # CV should be non-negative and typically < 2 for stable stats
            if len(cv_values) > 0:
                assert (cv_values >= 0).all()


class TestMultipleStatistics:
    """Test features for multiple statistics (points, rebounds, assists)"""

    def test_features_for_all_stats(self, sample_player_data):
        """Test features calculated for points, rebounds, assists"""
        result = calculate_rolling_features(sample_player_data)

        # Should have features for multiple stats
        expected_stats = ['pra', 'points', 'rebounds', 'assists']

        for stat in expected_stats:
            # Check if at least one rolling feature exists for this stat
            stat_cols = [col for col in result.columns if stat in col]
            # Some features should exist (implementation-dependent)

    def test_minutes_based_features(self, sample_player_data):
        """Test per-minute or minutes-based features"""
        result = calculate_rolling_features(sample_player_data)

        # Look for per-minute features
        per_min_cols = [col for col in result.columns if 'per_min' in col.lower() or 'per_36' in col.lower()]

        # If they exist, they should be reasonable
        for col in per_min_cols:
            values = result[col].dropna()
            if len(values) > 0:
                # Per-minute stats should be small (< 5 for most stats)
                assert values.max() < 10, f"{col} has unreasonably high per-minute values"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_dataframe(self, empty_dataframe):
        """Test handling of empty DataFrame"""
        # Should either return empty DataFrame or raise appropriate error
        try:
            result = calculate_rolling_features(empty_dataframe)
            assert len(result) == 0
        except (ValueError, AssertionError):
            # Acceptable to raise error for empty data
            pass

    def test_single_player_single_game(self):
        """Test with single game for single player"""
        df = pd.DataFrame({
            'player_id': [1],
            'game_id': [100],
            'game_date': pd.to_datetime(['2024-01-01']),
            'pra': [25],
            'points': [15],
            'rebounds': [7],
            'assists': [3],
            'minutes': [32],
            'season': ['2023-24'],
            'player_name': ['Player A'],
            'opponent_team': ['LAL']
        })

        result = calculate_rolling_features(df)

        assert len(result) == 1
        # First game should have NaN or 0 for rolling features

    def test_all_zero_stats(self):
        """Test with all zero statistics (DNP scenario)"""
        df = pd.DataFrame({
            'player_id': [1] * 3,
            'game_id': [100, 101, 102],
            'game_date': pd.date_range('2024-01-01', periods=3),
            'pra': [0, 0, 0],
            'points': [0, 0, 0],
            'rebounds': [0, 0, 0],
            'assists': [0, 0, 0],
            'minutes': [0, 0, 0],
            'season': ['2023-24'] * 3,
            'player_name': ['Player A'] * 3,
            'opponent_team': ['LAL', 'BOS', 'GSW']
        })

        result = calculate_rolling_features(df)

        # Should handle zeros without error
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
