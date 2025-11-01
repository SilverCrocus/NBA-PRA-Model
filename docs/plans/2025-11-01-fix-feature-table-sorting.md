# Fix Feature Table Sorting Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all feature engineering modules to sort output by game_date before saving, ensuring validation passes and preventing temporal leakage.

**Architecture:** Add `.sort_values(['player_id', 'game_date'])` before every `.to_parquet()` call in all feature modules. This ensures chronological ordering per player, which is critical for time-series validation and prevents data leakage.

**Tech Stack:** Python, Pandas, existing feature engineering modules

---

## Problem Analysis

**Root Cause:** All 6 feature modules save DataFrames without sorting by game_date, causing `validate_features.py` to fail with "Data not sorted by game_date" errors.

**Validation Check (validate_features.py:46):**
```python
if not df['game_date'].is_monotonic_increasing:
    issues.append("Data not sorted by game_date")
```

**Affected Files:**
- `feature_engineering/features/rolling_features.py`
- `feature_engineering/features/matchup_features.py`
- `feature_engineering/features/contextual_features.py`
- `feature_engineering/features/advanced_metrics.py`
- `feature_engineering/features/position_features.py`
- `feature_engineering/features/injury_features.py`

**Why This Matters:**
1. Temporal leakage prevention requires chronological ordering
2. Rolling calculations assume sorted data (`.shift(1)` only works correctly on sorted data)
3. Validation pipeline enforces this as a critical check
4. Model training relies on temporal ordering for train/test splits

---

## Task 1: Fix rolling_features.py

**Files:**
- Modify: `feature_engineering/features/rolling_features.py`

**Step 1: Locate the save operation**

Read the file and find the `.to_parquet()` call:

```bash
grep -n "to_parquet" feature_engineering/features/rolling_features.py
```

Expected: Line with `features.to_parquet(output_path)`

**Step 2: Add sorting before save**

Find the section that looks like:
```python
features.to_parquet(output_path)
```

Replace with:
```python
# Sort by player and game_date for temporal validation
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Step 3: Verify the fix**

Run the feature module:
```bash
uv run feature_engineering/features/rolling_features.py
```

Expected: Script completes without errors

**Step 4: Verify sorting**

Check the output is sorted:
```python
import pandas as pd
df = pd.read_parquet('data/feature_tables/rolling_features.parquet')
print(f"Is sorted: {df['game_date'].is_monotonic_increasing}")
```

Expected: `Is sorted: True`

**Step 5: Commit**

```bash
git add feature_engineering/features/rolling_features.py
git commit -m "fix: sort rolling_features by game_date before save"
```

---

## Task 2: Fix matchup_features.py

**Files:**
- Modify: `feature_engineering/features/matchup_features.py`

**Step 1: Locate the save operation**

```bash
grep -n "to_parquet" feature_engineering/features/matchup_features.py
```

**Step 2: Add sorting before save**

Find:
```python
features.to_parquet(output_path)
```

Replace with:
```python
# Sort by player and game_date for temporal validation
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Step 3: Verify the fix**

```bash
uv run feature_engineering/features/matchup_features.py
```

Expected: Script completes without errors

**Step 4: Verify sorting**

```python
import pandas as pd
df = pd.read_parquet('data/feature_tables/matchup_features.parquet')
print(f"Is sorted: {df['game_date'].is_monotonic_increasing}")
```

Expected: `Is sorted: True`

**Step 5: Commit**

```bash
git add feature_engineering/features/matchup_features.py
git commit -m "fix: sort matchup_features by game_date before save"
```

---

## Task 3: Fix contextual_features.py

**Files:**
- Modify: `feature_engineering/features/contextual_features.py`

**Step 1: Locate the save operation**

```bash
grep -n "to_parquet" feature_engineering/features/contextual_features.py
```

**Step 2: Add sorting before save**

Find:
```python
features.to_parquet(output_path)
```

Replace with:
```python
# Sort by player and game_date for temporal validation
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Step 3: Verify the fix**

```bash
uv run feature_engineering/features/contextual_features.py
```

Expected: Script completes without errors

**Step 4: Verify sorting**

```python
import pandas as pd
df = pd.read_parquet('data/feature_tables/contextual_features.parquet')
print(f"Is sorted: {df['game_date'].is_monotonic_increasing}")
```

Expected: `Is sorted: True`

**Step 5: Commit**

```bash
git add feature_engineering/features/contextual_features.py
git commit -m "fix: sort contextual_features by game_date before save"
```

---

## Task 4: Fix advanced_metrics.py

**Files:**
- Modify: `feature_engineering/features/advanced_metrics.py`

**Step 1: Locate the save operation**

```bash
grep -n "to_parquet" feature_engineering/features/advanced_metrics.py
```

**Step 2: Add sorting before save**

Find:
```python
features.to_parquet(output_path)
```

Replace with:
```python
# Sort by player and game_date for temporal validation
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Step 3: Verify the fix**

```bash
uv run feature_engineering/features/advanced_metrics.py
```

Expected: Script completes without errors

**Step 4: Verify sorting**

```python
import pandas as pd
df = pd.read_parquet('data/feature_tables/advanced_metrics.parquet')
print(f"Is sorted: {df['game_date'].is_monotonic_increasing}")
```

Expected: `Is sorted: True`

**Step 5: Commit**

```bash
git add feature_engineering/features/advanced_metrics.py
git commit -m "fix: sort advanced_metrics by game_date before save"
```

---

## Task 5: Fix position_features.py

**Files:**
- Modify: `feature_engineering/features/position_features.py`

**Step 1: Locate the save operation**

```bash
grep -n "to_parquet" feature_engineering/features/position_features.py
```

**Step 2: Add sorting before save**

Find:
```python
features.to_parquet(output_path)
```

Replace with:
```python
# Sort by player and game_date for temporal validation
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Step 3: Verify the fix**

```bash
uv run feature_engineering/features/position_features.py
```

Expected: Script completes without errors

**Step 4: Verify sorting**

```python
import pandas as pd
df = pd.read_parquet('data/feature_tables/position_features.parquet')
print(f"Is sorted: {df['game_date'].is_monotonic_increasing}")
```

Expected: `Is sorted: True`

**Step 5: Commit**

```bash
git add feature_engineering/features/position_features.py
git commit -m "fix: sort position_features by game_date before save"
```

---

## Task 6: Fix injury_features.py

**Files:**
- Modify: `feature_engineering/features/injury_features.py`

**Step 1: Locate the save operation**

```bash
grep -n "to_parquet" feature_engineering/features/injury_features.py
```

**Step 2: Add sorting before save**

Find:
```python
features.to_parquet(output_path)
```

Replace with:
```python
# Sort by player and game_date for temporal validation
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Step 3: Verify the fix**

```bash
uv run feature_engineering/features/injury_features.py
```

Expected: Script completes without errors

**Step 4: Verify sorting**

```python
import pandas as pd
df = pd.read_parquet('data/feature_tables/injury_features.parquet')
print(f"Is sorted: {df['game_date'].is_monotonic_increasing}")
```

Expected: `Is sorted: True`

**Step 5: Commit**

```bash
git add feature_engineering/features/injury_features.py
git commit -m "fix: sort injury_features by game_date before save"
```

---

## Task 7: Run Full Pipeline Validation

**Files:**
- None (verification only)

**Step 1: Run the complete feature pipeline**

```bash
uv run feature_engineering/run_pipeline.py
```

Expected: All 9 stages complete successfully including validation

**Step 2: Verify validation passes**

Check logs for:
```
✅ Stage validate_features completed
```

Expected: No "Data not sorted by game_date" errors

**Step 3: Verify master features**

```python
import pandas as pd
df = pd.read_parquet('data/feature_tables/master_features.parquet')
print(f"Master features sorted: {df['game_date'].is_monotonic_increasing}")
print(f"Shape: {df.shape}")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
```

Expected:
- `Master features sorted: True`
- Shape matches previous run (~587k rows, ~165 columns)
- Date range: 2003-10-28 to 2025-10-31

**Step 4: Run validation script directly**

```bash
uv run feature_engineering/validate_features.py
```

Expected: All checks pass with "🎉 All validations passed!"

---

## Task 8: Update Documentation

**Files:**
- Modify: `.claude/CLAUDE.md`

**Step 1: Add sorting requirement to architecture docs**

Find the section "Code Quality Standards" and add after the function example:

```markdown
### Feature Module Save Pattern

All feature modules MUST sort by grain before saving:

```python
# Save features (REQUIRED: sort first for temporal validation)
features = features.sort_values(['player_id', 'game_date'])
features.to_parquet(output_path)
```

**Why:** Validation enforces monotonic game_date ordering to prevent temporal leakage.
```

**Step 2: Verify documentation renders correctly**

```bash
cat .claude/CLAUDE.md | grep -A 5 "Feature Module Save Pattern"
```

Expected: Shows the new section

**Step 3: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: add feature sorting requirement to standards"
```

---

## Task 9: Add Utils Helper Function (Optional but Recommended)

**Files:**
- Modify: `feature_engineering/utils.py`

**Step 1: Add sort_and_save helper function**

Add to `utils.py` after the existing utility functions:

```python
def sort_and_save_features(
    df: pd.DataFrame,
    output_path: Path,
    grain_cols: List[str] = ['player_id', 'game_date']
) -> None:
    """
    Sort DataFrame by grain and save to parquet

    Args:
        df: Features DataFrame
        output_path: Path to save parquet file
        grain_cols: Columns to sort by (default: player_id, game_date)

    Returns:
        None

    Raises:
        ValueError: If grain columns missing from DataFrame
    """
    # Validate grain columns exist
    missing_cols = [col for col in grain_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing grain columns: {missing_cols}")

    # Sort by grain (critical for temporal validation)
    df_sorted = df.sort_values(grain_cols)

    # Save to parquet
    ensure_directory_exists(output_path.parent)
    df_sorted.to_parquet(output_path)

    logger.info(f"Saved {len(df_sorted):,} rows to {output_path}")
```

**Step 2: Update feature modules to use helper (rolling_features.py example)**

Replace the manual sort + save with:
```python
from utils import sort_and_save_features

# Old way:
# features = features.sort_values(['player_id', 'game_date'])
# features.to_parquet(output_path)

# New way:
sort_and_save_features(features, output_path)
```

**Step 3: Write test for helper function**

Create `tests/test_utils.py` if it doesn't exist, add:

```python
def test_sort_and_save_features(tmp_path):
    """Test sort_and_save_features sorts correctly"""
    import pandas as pd
    from utils import sort_and_save_features

    # Create unsorted data
    df = pd.DataFrame({
        'player_id': [1, 1, 2, 2],
        'game_date': pd.to_datetime(['2024-01-15', '2024-01-10', '2024-01-20', '2024-01-05']),
        'feature': [10, 20, 30, 40]
    })

    output_path = tmp_path / "test_features.parquet"
    sort_and_save_features(df, output_path)

    # Load and verify sorted
    result = pd.read_parquet(output_path)
    assert result['game_date'].is_monotonic_increasing
    assert len(result) == 4
```

**Step 4: Run test**

```bash
pytest tests/test_utils.py::test_sort_and_save_features -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add feature_engineering/utils.py tests/test_utils.py
git commit -m "feat: add sort_and_save_features utility function"
```

---

## Task 10: Run End-to-End CLI Test

**Files:**
- None (verification only)

**Step 1: Run the CLI pipeline command**

```bash
uv run nba-pra pipeline --skip-data-update
```

Expected: Complete successfully with no validation errors

**Step 2: Check CLI logs**

```bash
tail -50 logs/cli_*.log
```

Expected: Shows "✅ Stage validate_features completed"

**Step 3: Verify production artifacts**

```bash
ls -lh data/feature_tables/*.parquet
```

Expected: All 7 parquet files present (6 feature tables + master_features)

**Step 4: Verify ready for training**

```bash
uv run model_training/train_split.py --cv-mode
```

Expected: Creates CV splits without errors

---

## Verification Checklist

After completing all tasks, verify:

- [ ] All 6 feature modules sort before saving
- [ ] `validate_features.py` passes all checks
- [ ] Pipeline runs end-to-end without errors
- [ ] Master features are sorted chronologically
- [ ] Documentation updated with sorting requirement
- [ ] Optional: Utils helper function added and tested
- [ ] CLI command works: `uv run nba-pra pipeline --skip-data-update`
- [ ] Model training can proceed: `uv run model_training/train_split.py --cv-mode`

---

## Success Criteria

1. **Pipeline validation passes**: `uv run feature_engineering/run_pipeline.py` completes with "🎉 All validations passed!"
2. **All feature tables sorted**: Each `.parquet` file has `game_date.is_monotonic_increasing == True`
3. **CLI works**: `uv run nba-pra pipeline --skip-data-update` completes successfully
4. **No temporal leakage warnings**: Validation logs show no "Data not sorted" errors
5. **Ready for training**: Can proceed to model training without issues

---

## Notes

**Why sort by ['player_id', 'game_date'] not just ['game_date']?**
- Sorting by player first ensures each player's games are chronologically ordered
- This is critical for `.groupby('player_id').shift(1)` operations
- Prevents cross-player data mixing in rolling calculations

**Performance impact:**
- Sorting adds ~1-2 seconds per module (negligible)
- Already sorted data has O(n) verification complexity

**Alternative approaches considered:**
1. ❌ Sort in `build_features.py` only → Doesn't catch issues early
2. ❌ Sort in validation only → Fixes symptom, not root cause
3. ✅ Sort in each module → Correct by construction (chosen approach)
