"""
Test shared utility functions

Tests for data transformation utilities, safe operations, and helper functions.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_minutes_to_float(minutes):
    """Convert minutes from MM:SS format or numeric to float"""
    if pd.isna(minutes):
        return 0.0

    if isinstance(minutes, str):
        try:
            if ':' in minutes:
                parts = minutes.split(':')
                return float(parts[0]) + float(parts[1]) / 60.0
            else:
                return float(minutes)
        except (ValueError, IndexError):
            return 0.0

    return float(minutes)


def safe_divide(numerator, denominator, fill_value=0.0):
    """Safely divide two series/arrays, handling division by zero"""
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result


def calculate_z_score(values, mean, std):
    """Calculate z-score: (value - mean) / std"""
    return (values - mean) / std.replace(0, 1)  # Avoid division by zero


class TestMinutesConversion:
    """Test minutes format conversion"""

    def test_convert_minutes_colon_format(self):
        """Test conversion from MM:SS format"""
        assert convert_minutes_to_float("32:30") == 32.5
        assert convert_minutes_to_float("25:00") == 25.0
        assert convert_minutes_to_float("48:00") == 48.0
        assert convert_minutes_to_float("0:30") == 0.5

    def test_convert_minutes_numeric_string(self):
        """Test conversion from numeric string"""
        assert convert_minutes_to_float("32.5") == 32.5
        assert convert_minutes_to_float("25") == 25.0

    def test_convert_minutes_numeric_types(self):
        """Test conversion from numeric types"""
        assert convert_minutes_to_float(32.5) == 32.5
        assert convert_minutes_to_float(25) == 25.0
        assert convert_minutes_to_float(25.0) == 25.0

    def test_convert_minutes_nan(self):
        """Test conversion handles NaN"""
        assert convert_minutes_to_float(np.nan) == 0.0
        assert convert_minutes_to_float(None) == 0.0

    def test_convert_minutes_invalid(self):
        """Test conversion handles invalid input"""
        assert convert_minutes_to_float("invalid") == 0.0
        assert convert_minutes_to_float("25:") == 0.0
        assert convert_minutes_to_float(":30") == 0.0


class TestSafeDivision:
    """Test safe division operations"""

    def test_safe_divide_normal(self):
        """Test safe division with normal values"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 4, 5])

        result = safe_divide(numerator, denominator)

        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_divide_zero_denominator(self):
        """Test safe division handles division by zero"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 5])

        result = safe_divide(numerator, denominator, fill_value=0.0)

        assert result[0] == 5.0
        assert result[1] == 0.0  # Division by zero filled with 0
        assert result[2] == 6.0

    def test_safe_divide_custom_fill(self):
        """Test safe division with custom fill value"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0, 5])

        result = safe_divide(numerator, denominator, fill_value=-1.0)

        assert result[1] == -1.0  # Custom fill value

    def test_safe_divide_nan_handling(self):
        """Test safe division handles NaN values"""
        numerator = pd.Series([10, np.nan, 30])
        denominator = pd.Series([2, 4, 5])

        result = safe_divide(numerator, denominator, fill_value=0.0)

        assert result[0] == 5.0
        assert result[1] == 0.0  # NaN filled with 0
        assert result[2] == 6.0

    def test_safe_divide_infinity(self):
        """Test safe division handles infinity"""
        numerator = pd.Series([10, 20, 30])
        denominator = pd.Series([2, 0.0, 5])

        result = safe_divide(numerator, denominator, fill_value=0.0)

        # Division by 0.0 creates inf, should be filled
        assert not np.isinf(result).any()


class TestZScoreCalculation:
    """Test z-score calculations"""

    def test_calculate_z_score_normal(self):
        """Test z-score calculation with normal values"""
        values = pd.Series([20, 25, 30])
        mean = pd.Series([22, 22, 22])
        std = pd.Series([5, 5, 5])

        result = calculate_z_score(values, mean, std)

        assert abs(result[0] - (-0.4)) < 0.01
        assert abs(result[1] - 0.6) < 0.01
        assert abs(result[2] - 1.6) < 0.01

    def test_calculate_z_score_zero_std(self):
        """Test z-score handles zero standard deviation"""
        values = pd.Series([20, 25, 30])
        mean = pd.Series([22, 22, 22])
        std = pd.Series([0, 5, 0])  # Zero std should be handled

        result = calculate_z_score(values, mean, std)

        # When std=0, z-score should use 1 to avoid division by zero
        assert not np.isinf(result[0])
        assert not np.isinf(result[2])

    def test_calculate_z_score_same_value_as_mean(self):
        """Test z-score when value equals mean"""
        values = pd.Series([22, 22, 22])
        mean = pd.Series([22, 22, 22])
        std = pd.Series([5, 5, 5])

        result = calculate_z_score(values, mean, std)

        # z-score should be 0 when value = mean
        assert (result == 0).all()


class TestDataTransformations:
    """Test common data transformations"""

    def test_rolling_mean_calculation(self):
        """Test rolling mean calculation"""
        data = pd.Series([10, 20, 30, 40, 50])

        rolling_mean = data.rolling(window=3, min_periods=1).mean()

        assert rolling_mean[0] == 10.0  # Only one value
        assert rolling_mean[1] == 15.0  # (10+20)/2
        assert rolling_mean[2] == 20.0  # (10+20+30)/3
        assert rolling_mean[3] == 30.0  # (20+30+40)/3

    def test_ewma_calculation(self):
        """Test exponentially weighted moving average"""
        data = pd.Series([10, 20, 30, 40, 50])

        ewma = data.ewm(span=3, adjust=False).mean()

        # EWMA gives more weight to recent values
        assert ewma[0] == 10.0
        assert ewma[1] > 10.0 and ewma[1] < 20.0
        assert ewma[4] > 40.0  # Should be weighted toward recent values

    def test_rank_calculation(self):
        """Test ranking/percentile calculation"""
        data = pd.Series([10, 20, 30, 40, 50])

        ranks = data.rank(pct=True)

        assert ranks[0] == 0.2  # Lowest = 20th percentile
        assert ranks[4] == 1.0  # Highest = 100th percentile

    def test_lagged_values(self):
        """Test shift/lag operations"""
        data = pd.Series([10, 20, 30, 40, 50])

        lagged = data.shift(1)

        assert pd.isna(lagged[0])  # First value is NaN
        assert lagged[1] == 10.0
        assert lagged[2] == 20.0


class TestStringOperations:
    """Test string manipulation utilities"""

    def test_strip_whitespace(self):
        """Test whitespace removal"""
        names = pd.Series(['  LeBron James  ', 'Kevin Durant', '  '])

        cleaned = names.str.strip()

        assert cleaned[0] == 'LeBron James'
        assert cleaned[1] == 'Kevin Durant'
        assert cleaned[2] == ''

    def test_string_matching(self):
        """Test string matching operations"""
        teams = pd.Series(['Los Angeles Lakers', 'Boston Celtics', 'LA Clippers'])

        la_teams = teams.str.contains('LA', case=False)

        assert la_teams[0] == True
        assert la_teams[1] == False
        assert la_teams[2] == True

    def test_name_standardization(self):
        """Test name standardization"""
        names = pd.Series(['LEBRON JAMES', 'kevin durant', 'LuKa DoNcIc'])

        standardized = names.str.title()

        assert standardized[0] == 'Lebron James'
        assert standardized[1] == 'Kevin Durant'
        assert standardized[2] == 'Luka Doncic'


class TestDateOperations:
    """Test date/time utilities"""

    def test_date_difference_calculation(self):
        """Test calculation of days between dates"""
        dates = pd.Series(pd.to_datetime(['2024-01-01', '2024-01-05', '2024-01-10']))

        date_diff = dates.diff().dt.days

        assert pd.isna(date_diff[0])  # First diff is NaN
        assert date_diff[1] == 4
        assert date_diff[2] == 5

    def test_season_extraction(self):
        """Test extracting season from date"""
        dates = pd.Series(pd.to_datetime(['2023-10-01', '2024-02-15', '2024-06-01']))

        # NBA season spans two years (Oct-Jun)
        # 2023-10 -> 2023-24 season
        # 2024-02 -> 2023-24 season
        # 2024-06 -> 2023-24 season

        years = dates.dt.year
        months = dates.dt.month

        # Simple season logic
        seasons = years.astype(str) + '-' + (years + 1).astype(str).str[-2:]

        # Adjust for months before October (part of previous season)
        seasons = seasons.where(months >= 10, (years - 1).astype(str) + '-' + years.astype(str).str[-2:])

        assert seasons[0] == '2023-24'
        assert seasons[1] == '2023-24'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
