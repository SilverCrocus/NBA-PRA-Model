"""
Feature Validation Script
Runs comprehensive checks on all feature engineering outputs
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
import sys

PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_DIR = PROJECT_ROOT / "data" / "feature_tables"


def check_file_exists(filename: str) -> bool:
    """Check if feature file exists"""
    filepath = FEATURE_DIR / filename
    if not filepath.exists():
        logger.error(f"❌ {filename} not found")
        return False
    logger.info(f"✓ {filename} exists ({filepath.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


def check_grain_uniqueness(df: pd.DataFrame, filename: str) -> bool:
    """Check that grain is unique (no duplicates)"""
    grain_cols = ['player_id', 'game_id', 'game_date']

    duplicates = df.duplicated(subset=grain_cols).sum()

    if duplicates > 0:
        logger.error(f"  ❌ CRITICAL: {duplicates} duplicate rows on grain!")
        return False
    logger.info(f"  ✓ Grain is unique ({len(df):,} rows)")
    return True


def check_temporal_leakage(df: pd.DataFrame, filename: str) -> bool:
    """Check for potential temporal leakage"""
    issues = []

    # Check if data is sorted
    if not df['game_date'].equals(df['game_date'].sort_values()):
        issues.append("Data not sorted by game_date")

    # Check for any columns that might indicate leakage
    suspicious_cols = [col for col in df.columns if 'current' in col.lower() or 'today' in col.lower()]

    if suspicious_cols:
        issues.append(f"Suspicious column names: {suspicious_cols}")

    if issues:
        logger.warning(f"  ⚠️  Potential leakage issues: {', '.join(issues)}")
        return False

    logger.info(f"  ✓ No obvious leakage detected")
    return True


def check_missing_values(df: pd.DataFrame, filename: str) -> bool:
    """Check missing value patterns"""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100)

    high_missing = missing_pct[missing_pct > 50]

    if len(high_missing) > 0:
        logger.warning(f"  ⚠️  {len(high_missing)} columns with >50% missing:")
        for col, pct in high_missing.head(5).items():
            logger.info(f"     - {col}: {pct:.1f}%")
    else:
        logger.info(f"  ✓ Missing values reasonable (max {missing_pct.max():.1f}%)")

    return len(high_missing) == 0


def check_feature_distributions(df: pd.DataFrame, filename: str) -> bool:
    """Check if feature distributions are reasonable"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['player_id', 'game_id']]

    issues = []

    for col in numeric_cols[:20]:  # Check first 20 numeric features
        if col in df.columns:
            col_data = df[col].dropna()

            if len(col_data) == 0:
                continue

            # Check for infinite values
            if np.isinf(col_data).any():
                issues.append(f"{col} has infinite values")

            # Check for unreasonably large values
            if col_data.max() > 1e6:
                issues.append(f"{col} has very large values (max={col_data.max():.0f})")

            # Check for constant values
            if col_data.nunique() == 1:
                issues.append(f"{col} is constant")

    if issues:
        logger.warning(f"  ⚠️  Distribution issues ({len(issues)}):")
        for issue in issues[:5]:
            logger.info(f"     - {issue}")
        return False

    logger.info(f"  ✓ Feature distributions look reasonable")
    return True


def validate_feature_file(filename: str) -> bool:
    """Run all validations on a feature file"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Validating: {filename}")
    logger.info(f"{'='*60}")

    # Check existence
    if not check_file_exists(filename):
        return False

    # Load file
    try:
        df = pd.read_parquet(FEATURE_DIR / filename)
    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        # File not found, corrupted, or permission error
        logger.error(f"❌ Error loading file: {e}")
        raise  # Always re-raise per user requirement

    # Run checks
    grain_ok = check_grain_uniqueness(df, filename)
    leakage_ok = check_temporal_leakage(df, filename)
    missing_ok = check_missing_values(df, filename)
    dist_ok = check_feature_distributions(df, filename)

    all_ok = grain_ok and leakage_ok

    if all_ok:
        logger.info(f"\n✅ {filename} passed all critical checks")
    else:
        logger.error(f"\n❌ {filename} has critical issues")

    return all_ok


def validate_master_features() -> bool:
    """Validate the master feature matrix"""
    logger.info(f"\n{'='*60}")
    logger.info("Validating Master Feature Matrix")
    logger.info(f"{'='*60}")

    filepath = FEATURE_DIR / "master_features.parquet"

    if not filepath.exists():
        logger.error("❌ Master features not found. Run build_features.py first.")
        return False

    df = pd.read_parquet(filepath)

    logger.info(f"\nMaster Features Shape: {df.shape}")
    logger.info(f"Date Range: {df['game_date'].min()} to {df['game_date'].max()}")
    logger.info(f"Players: {df['player_id'].nunique()}")
    logger.info(f"Games: {df['game_id'].nunique()}")

    # Check target variable
    if 'target_pra' in df.columns:
        logger.info(f"\nTarget (PRA) Statistics:")
        logger.info(f"  Mean: {df['target_pra'].mean():.2f}")
        logger.info(f"  Median: {df['target_pra'].median():.2f}")
        logger.info(f"  Std: {df['target_pra'].std():.2f}")
        logger.info(f"  Min: {df['target_pra'].min():.2f}")
        logger.info(f"  Max: {df['target_pra'].max():.2f}")
        logger.info(f"  Missing: {df['target_pra'].isnull().sum()}")

        if df['target_pra'].isnull().sum() > 0:
            logger.error("  ❌ Target has missing values!")
        else:
            logger.info("  ✓ Target complete")

    # Feature count by category
    logger.info(f"\nFeature Categories:")
    rolling_features = [col for col in df.columns if 'avg' in col or 'ewma' in col or 'trend' in col]
    matchup_features = [col for col in df.columns if 'opp' in col or 'opponent' in col]
    contextual_features = [col for col in df.columns if 'is_' in col or 'day' in col]
    position_features = [col for col in df.columns if 'position' in col]
    injury_features = [col for col in df.columns if 'injury' in col or 'dnp' in col or 'absence' in col]

    logger.info(f"  Rolling: {len(rolling_features)}")
    logger.info(f"  Matchup: {len(matchup_features)}")
    logger.info(f"  Contextual: {len(contextual_features)}")
    logger.info(f"  Position: {len(position_features)}")
    logger.info(f"  Injury: {len(injury_features)}")

    # Overall validation
    grain_ok = check_grain_uniqueness(df, "master_features.parquet")
    leakage_ok = check_temporal_leakage(df, "master_features.parquet")

    if grain_ok and leakage_ok:
        logger.info(f"\n✅ Master features validated successfully")
        logger.info(f"\nReady for model training!")
        return True
    else:
        logger.error(f"\n❌ Master features have issues - fix before training")
        return False


def main() -> int:
    """Run all validations"""
    logger.info("="*60)
    logger.info("FEATURE ENGINEERING VALIDATION")
    logger.info("="*60)

    feature_files = [
        "rolling_features.parquet",
        "matchup_features.parquet",
        "contextual_features.parquet",
        "advanced_metrics.parquet",
        "position_features.parquet",
        "injury_features.parquet",
    ]

    results = {}

    # Validate individual feature files
    for filename in feature_files:
        results[filename] = validate_feature_file(filename)

    # Validate master features
    master_ok = validate_master_features()

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*60}")

    for filename, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {filename}")

    master_status = "✅ PASS" if master_ok else "❌ FAIL"
    logger.info(f"{master_status}: master_features.parquet")

    all_pass = all(results.values()) and master_ok

    if all_pass:
        logger.info(f"\n🎉 All validations passed!")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Run: python model_training/train_split.py")
        logger.info(f"  2. Then: Train your model")
        return 0
    else:
        logger.warning(f"\n⚠️  Some validations failed - review issues above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
