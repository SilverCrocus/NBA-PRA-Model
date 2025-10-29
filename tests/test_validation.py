"""
Test validation functions

Tests for data quality checks, grain validation, and utility validators.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def validate_required_columns(df, required_cols, function_name):
    """Test version of column validator"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"{function_name}: Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def validate_not_empty(df, function_name):
    """Test version of empty validator"""
    if len(df) == 0:
        raise ValueError(f"{function_name}: DataFrame is empty")


def validate_grain_uniqueness(df, grain_cols=['player_id', 'game_id', 'game_date']):
    """Test version of grain validator"""
    if not all(col in df.columns for col in grain_cols):
        missing = [col for col in grain_cols if col not in df.columns]
        raise ValueError(f"Missing grain columns: {missing}")

    duplicates = df.duplicated(subset=grain_cols, keep=False)
    if duplicates.any():
        dup_rows = df[duplicates]
        raise AssertionError(
            f"Found {duplicates.sum()} duplicate rows on grain {grain_cols}:\n"
            f"{dup_rows[grain_cols].head()}"
        )


class TestColumnValidation:
    """Test column validation functions"""

    def test_validate_required_columns_success(self, sample_player_data):
        """Test validation passes with all required columns"""
        # Should not raise exception
        validate_required_columns(
            sample_player_data,
            ['player_id', 'game_id', 'game_date'],
            'test_function'
        )

    def test_validate_required_columns_missing(self):
        """Test validation fails with missing columns"""
        df = pd.DataFrame({'player_id': [1, 2]})

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, ['player_id', 'game_id'], 'test_function')

    def test_validate_required_columns_empty_list(self, sample_player_data):
        """Test validation passes with empty requirements"""
        validate_required_columns(sample_player_data, [], 'test_function')


class TestEmptyDataValidation:
    """Test empty DataFrame validation"""

    def test_validate_not_empty_success(self, sample_player_data):
        """Test validation passes with non-empty DataFrame"""
        validate_not_empty(sample_player_data, 'test_function')

    def test_validate_not_empty_fails(self, empty_dataframe):
        """Test validation fails with empty DataFrame"""
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_not_empty(empty_dataframe, 'test_function')

    def test_validate_not_empty_single_row(self):
        """Test validation passes with single row"""
        df = pd.DataFrame({'player_id': [1], 'game_id': [100]})
        validate_not_empty(df, 'test_function')


class TestGrainValidation:
    """Test grain uniqueness validation"""

    def test_validate_grain_uniqueness_success(self, sample_player_data):
        """Test grain validation passes with unique rows"""
        validate_grain_uniqueness(sample_player_data)

    def test_validate_grain_uniqueness_fails(self):
        """Test grain validation fails with duplicates"""
        df = pd.DataFrame({
            'player_id': [1, 1],  # Duplicate!
            'game_id': [100, 100],
            'game_date': pd.to_datetime(['2024-01-01', '2024-01-01']),
            'pra': [25, 25]
        })

        with pytest.raises(AssertionError, match="duplicate rows"):
            validate_grain_uniqueness(df)

    def test_validate_grain_uniqueness_different_dates(self):
        """Test grain allows same player/game with different dates"""
        df = pd.DataFrame({
            'player_id': [1, 1],
            'game_id': [100, 100],
            'game_date': pd.to_datetime(['2024-01-01', '2024-01-02']),  # Different dates
            'pra': [25, 30]
        })

        # Should pass - different dates make rows unique
        validate_grain_uniqueness(df)

    def test_validate_grain_missing_columns(self):
        """Test grain validation fails with missing grain columns"""
        df = pd.DataFrame({
            'player_id': [1, 2],
            'game_id': [100, 101]
            # Missing game_date
        })

        with pytest.raises(ValueError, match="Missing grain columns"):
            validate_grain_uniqueness(df)


class TestDataQualityChecks:
    """Test data quality validation functions"""

    def test_no_infinite_values(self, sample_player_data):
        """Test detection of infinite values"""
        df = sample_player_data.copy()
        df.loc[0, 'pra'] = np.inf

        has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
        assert has_inf, "Should detect infinite values"

    def test_no_negative_stats(self, sample_player_data):
        """Test detection of negative stats (impossible in basketball)"""
        df = sample_player_data.copy()

        stat_cols = ['pra', 'points', 'rebounds', 'assists', 'minutes']
        for col in stat_cols:
            assert (df[col] >= 0).all(), f"{col} should not have negative values"

    def test_pra_calculation_correct(self, sample_player_data):
        """Test PRA = Points + Rebounds + Assists"""
        df = sample_player_data.copy()

        calculated_pra = df['points'] + df['rebounds'] + df['assists']
        assert (df['pra'] == calculated_pra).all(), "PRA calculation incorrect"

    def test_reasonable_value_ranges(self, sample_player_data):
        """Test stats are within reasonable ranges"""
        df = sample_player_data.copy()

        # NBA game constraints
        assert (df['minutes'] <= 48).all(), "Minutes should not exceed 48 (overtime possible but rare > 48)"
        assert (df['points'] <= 100).all(), "Points per game should be < 100"
        assert (df['rebounds'] <= 50).all(), "Rebounds per game should be < 50"
        assert (df['assists'] <= 40).all(), "Assists per game should be < 40"


class TestMissingValueHandling:
    """Test missing value validation"""

    def test_detect_missing_values(self):
        """Test detection of missing values"""
        df = pd.DataFrame({
            'player_id': [1, 2, 3],
            'game_id': [100, 101, 102],
            'pra': [25, np.nan, 30]
        })

        has_missing = df.isna().any().any()
        assert has_missing, "Should detect missing values"

    def test_missing_value_percentage(self, sample_player_data):
        """Test calculation of missing value percentage"""
        df = sample_player_data.copy()
        df.loc[0, 'pra'] = np.nan
        df.loc[1, 'pra'] = np.nan

        missing_pct = (df['pra'].isna().sum() / len(df)) * 100
        assert abs(missing_pct - 40.0) < 0.1, "Missing percentage calculation incorrect"

    def test_required_columns_not_null(self, sample_player_data):
        """Test that grain columns should never be null"""
        df = sample_player_data.copy()

        grain_cols = ['player_id', 'game_id', 'game_date']
        for col in grain_cols:
            assert df[col].notna().all(), f"Grain column {col} should not have nulls"


class TestFeatureTableValidation:
    """Test validation specific to feature tables"""

    def test_feature_table_has_grain(self, sample_player_data):
        """Test feature tables include grain columns"""
        # Simulate a feature table
        features = sample_player_data[['player_id', 'game_id', 'game_date']].copy()
        features['feature_1'] = 1.0

        validate_required_columns(
            features,
            ['player_id', 'game_id', 'game_date'],
            'feature_table'
        )

    def test_feature_table_merge_validity(self, sample_player_data):
        """Test feature tables can merge with base data on grain"""
        base = sample_player_data[['player_id', 'game_id', 'game_date', 'pra']].copy()
        features = sample_player_data[['player_id', 'game_id', 'game_date']].copy()
        features['new_feature'] = 42.0

        merged = base.merge(
            features,
            on=['player_id', 'game_id', 'game_date'],
            how='left',
            validate='1:1'
        )

        assert len(merged) == len(base), "Merge should preserve row count"
        assert 'new_feature' in merged.columns, "Merge should add feature column"

    def test_feature_table_no_target_leakage(self, sample_player_data):
        """Test feature tables don't include target variable"""
        # Simulate a feature table
        features = sample_player_data[['player_id', 'game_id', 'game_date']].copy()

        # Should NOT include 'pra' (target variable)
        assert 'pra' not in features.columns, "Feature table should not include target variable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
