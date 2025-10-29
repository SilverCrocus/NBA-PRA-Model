# NBA PRA Feature Engineering Tests

Comprehensive test suite for the NBA PRA feature engineering pipeline with focus on data leakage prevention and validation.

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures (sample data)
├── test_leakage_prevention.py     # CRITICAL: Data leakage tests (12+ tests)
├── test_validation.py             # Data quality and grain validation (18 tests)
├── test_utils.py                  # Utility function tests (22 tests)
├── test_rolling_features.py       # Rolling feature calculations (15 tests)
└── README.md                      # This file
```

**Total: 67+ tests**

## Running Tests

### Run All Tests
```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=feature_engineering --cov-report=html

# Run with detailed output
uv run pytest tests/ -vv -s
```

### Run Specific Test Files
```bash
# Run only leakage prevention tests (MOST IMPORTANT)
uv run pytest tests/test_leakage_prevention.py -v

# Run only validation tests
uv run pytest tests/test_validation.py -v

# Run only utility tests
uv run pytest tests/test_utils.py -v

# Run only rolling feature tests
uv run pytest tests/test_rolling_features.py -v
```

### Run Specific Test Classes or Functions
```bash
# Run specific test class
uv run pytest tests/test_leakage_prevention.py::TestRollingFeatureLeakage -v

# Run specific test
uv run pytest tests/test_leakage_prevention.py::TestRollingFeatureLeakage::test_rolling_features_exclude_current_game -v
```

### Run with Options
```bash
# Stop on first failure
uv run pytest tests/ -x

# Show local variables on failure
uv run pytest tests/ -l

# Run in parallel (requires pytest-xdist)
uv run pytest tests/ -n auto

# Run only failed tests from last run
uv run pytest tests/ --lf
```

## Test Categories

### 1. Data Leakage Prevention Tests (CRITICAL) 🔒

**File:** `test_leakage_prevention.py` (12+ tests)

**Why This Matters:** Data leakage is the #1 risk in time-series ML. These tests ensure features never use future information.

**Test Classes:**
- `TestRollingFeatureLeakage` - Rolling features exclude current game
- `TestContextualFeatureLeakage` - Rest days, back-to-back use past only
- `TestCrossSectionalLeakage` - Features isolated by player
- `TestEdgeCaseLeakage` - Single games, chronological sorting
- `TestTemporalGapHandling` - DNP and off-season gaps

**Key Tests:**
- ✅ First game has NaN (no history)
- ✅ Rolling averages exclude current game
- ✅ No future information in features
- ✅ Features isolated by player (no cross-contamination)
- ✅ Shift(1) applied before rolling operations

### 2. Validation Tests 🛡️

**File:** `test_validation.py` (18 tests)

**Test Classes:**
- `TestColumnValidation` - Required columns present
- `TestEmptyDataValidation` - Non-empty DataFrames
- `TestGrainValidation` - Unique [player_id, game_id, game_date]
- `TestDataQualityChecks` - No infinite values, reasonable ranges
- `TestMissingValueHandling` - Missing value detection
- `TestFeatureTableValidation` - Feature table structure

### 3. Utility Function Tests 🔧

**File:** `test_utils.py` (22 tests)

**Test Classes:**
- `TestMinutesConversion` - MM:SS to float conversion
- `TestSafeDivision` - Division by zero handling
- `TestZScoreCalculation` - Z-score calculations
- `TestDataTransformations` - Rolling, EWMA, ranking
- `TestStringOperations` - Name standardization
- `TestDateOperations` - Date differences, season extraction

### 4. Rolling Features Tests 📊

**File:** `test_rolling_features.py` (15 tests)

**Test Classes:**
- `TestRollingAverages` - Rolling mean calculations
- `TestRollingStatistics` - Std, min, max, sum
- `TestExponentialMovingAverage` - EWMA features
- `TestTrendFeatures` - Trend direction capture
- `TestVolatilityFeatures` - Volatility/variance
- `TestMultipleStatistics` - Points, rebounds, assists
- `TestEdgeCases` - Empty data, single games, all zeros

## Test Fixtures

Defined in `conftest.py`:

### `sample_player_data`
5 games across 2 players with basic stats (PRA, points, rebounds, assists, minutes).

### `empty_dataframe`
Empty DataFrame with correct columns for edge case testing.

### `sequential_data`
10 sequential games for one player with increasing stats (useful for trend testing).

### `multi_player_data`
15 games across 3 players (5 games each) for cross-player validation.

### `data_with_dnp`
15 games with DNP (Did Not Play) scenarios for injury testing.

## Test Philosophy

### 1. Focus on Critical Functionality
Data leakage prevention tests are the most important. These tests verify the core assumption of time-series modeling.

### 2. Use Sample Data
Tests use small fixtures (5-15 games) to run quickly. No need to load the full 587k game dataset.

### 3. Test Edge Cases
- First games (no history)
- Single game players
- DNP scenarios
- Empty data
- Duplicates

### 4. Clear Assertions
All assertions include descriptive error messages explaining what went wrong.

## Expected Test Results

### All Tests Pass ✅
```
======================== 67 passed in < 5 seconds ==========================
```

### Test Coverage
- Data leakage prevention: **100%** (CRITICAL)
- Grain validation: **100%** (CRITICAL)
- Utility functions: **90%+**
- Feature calculations: **80%+**

## Common Test Failures

### Import Errors
```
ModuleNotFoundError: No module named 'feature_engineering'
```
**Fix:** Ensure `feature_engineering/__init__.py` exists and you're running from project root.

### Data Leakage Detected
```
AssertionError: Rolling average includes current game! Expected ~15, got 20
```
**Fix:** Verify `.shift(1)` is applied before `.rolling()` in feature calculations.

### Grain Violation
```
AssertionError: Found 5 duplicate rows on grain ['player_id', 'game_id', 'game_date']
```
**Fix:** Check for duplicate data in feature table merges.

## Writing New Tests

### Template for New Test
```python
def test_my_new_feature(sample_player_data):
    """Test description"""
    # Arrange
    df = sample_player_data.copy()

    # Act
    result = calculate_my_feature(df)

    # Assert
    assert len(result) == len(df), "Row count should match"
    assert result['my_feature'].notna().any(), "Feature should have values"
```

### Best Practices
1. **One assertion per test** (when possible)
2. **Descriptive test names** - `test_rolling_avg_excludes_current_game` not `test_rolling`
3. **Use fixtures** - Reuse sample data from `conftest.py`
4. **Test edge cases** - Empty data, nulls, single values
5. **Fast tests** - Use small datasets, avoid loading full data

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run tests
  run: |
    uv run pytest tests/ -v --cov=feature_engineering --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hook
```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: pytest
      name: pytest
      entry: uv run pytest tests/
      language: system
      pass_filenames: false
```

## Test Maintenance

### When to Update Tests

1. **Adding new features** - Add corresponding tests
2. **Changing feature logic** - Update expected values
3. **Bug fixes** - Add regression test
4. **New edge cases discovered** - Add test coverage

### Quarterly Review Checklist
- [ ] All tests still pass
- [ ] Test coverage > 80%
- [ ] No flaky tests (intermittent failures)
- [ ] Test execution time < 10 seconds
- [ ] New features have tests

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Pandas Testing Utilities](https://pandas.pydata.org/docs/reference/api/pandas.testing.assert_frame_equal.html)

## Questions?

See `.claude/CLAUDE.md` for project documentation or the feature engineering README for pipeline details.
