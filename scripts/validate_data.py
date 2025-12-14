"""
Data validation script for CI/CD pipeline.
Validates data quality, schema, and basic statistics.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def validate_data_schema(
    df: pd.DataFrame, expected_columns: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that the dataset contains expected columns.

    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names

    Returns:
        Tuple of (is_valid, list of missing columns)
    """
    missing_columns = [col for col in expected_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0

    return is_valid, missing_columns


def validate_data_quality(
    df: pd.DataFrame, target_column: str = "genre"
) -> Tuple[bool, List[str]]:
    """
    Validate data quality metrics.

    Args:
        df: DataFrame to validate
        target_column: Name of the target column

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check for empty dataset
    if len(df) == 0:
        issues.append("Dataset is empty")
        return False, issues

    # Check for target column
    if target_column not in df.columns:
        issues.append(f"Target column '{target_column}' not found")
        return False, issues

    # Check for sufficient samples per class
    if target_column in df.columns:
        class_counts = df[target_column].value_counts()
        min_samples_per_class = 10

        for class_name, count in class_counts.items():
            if count < min_samples_per_class:
                issues.append(
                    f"Class '{class_name}' has only {count} samples "
                    f"(minimum required: {min_samples_per_class})"
                )

    # Check for excessive missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    high_missing_cols = missing_percentage[missing_percentage > 50]

    if len(high_missing_cols) > 0:
        for col, pct in high_missing_cols.items():
            issues.append(
                f"Column '{col}' has {pct:.1f}% missing values " f"(threshold: 50%)"
            )

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > len(df) * 0.1:  # More than 10% duplicates
        issues.append(
            f"Dataset has {duplicate_count} duplicate rows "
            f"({duplicate_count/len(df)*100:.1f}% of data)"
        )

    is_valid = len(issues) == 0
    return is_valid, issues


def validate_feature_distributions(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate feature distributions for anomalies.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check for constant features (zero variance)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() == 1:
            issues.append(f"Column '{col}' is constant (zero variance)")

    # Check for features with extreme outliers
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR > 0:
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            if outliers > len(df) * 0.05:  # More than 5% outliers
                issues.append(
                    f"Column '{col}' has {outliers} extreme outliers "
                    f"({outliers/len(df)*100:.1f}% of data)"
                )

    is_valid = len(issues) == 0
    return is_valid, issues


def main(data_path: str, target_column: str = "genre"):
    """
    Main validation function.

    Args:
        data_path: Path to the data file
        target_column: Name of the target column
    """
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    # Load data
    try:
        print(f"\nLoading data from: {data_path}")
        df = pd.read_csv(data_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Data file not found: {data_path}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load data: {e}")
        sys.exit(1)

    # Expected columns (adjust based on your dataset)
    expected_columns = [
        "artist_name",
        "track_name",
        "genre",
        "danceability",
        "energy",
        "acousticness",
        "instrumentalness",
        "loudness",
        "liveness",
        "valence",
        "tempo",
        "key",
        "year",
        "duration_ms",
    ]

    all_valid = True

    # 1. Schema validation
    print("\n" + "-" * 60)
    print("1. SCHEMA VALIDATION")
    print("-" * 60)
    schema_valid, missing_cols = validate_data_schema(df, expected_columns)

    if schema_valid:
        print("✅ All expected columns are present")
    else:
        print(f"❌ Missing columns: {missing_cols}")
        all_valid = False

    # 2. Data quality validation
    print("\n" + "-" * 60)
    print("2. DATA QUALITY VALIDATION")
    print("-" * 60)
    quality_valid, quality_issues = validate_data_quality(df, target_column)

    if quality_valid:
        print("✅ Data quality checks passed")
    else:
        print("❌ Data quality issues found:")
        for issue in quality_issues:
            print(f"   - {issue}")
        all_valid = False

    # 3. Feature distribution validation
    print("\n" + "-" * 60)
    print("3. FEATURE DISTRIBUTION VALIDATION")
    print("-" * 60)
    dist_valid, dist_issues = validate_feature_distributions(df)

    if dist_valid:
        print("✅ Feature distribution checks passed")
    else:
        print("⚠️  Feature distribution issues found:")
        for issue in dist_issues:
            print(f"   - {issue}")
        # Distribution issues are warnings, not failures

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if all_valid:
        print("✅ All critical validations passed!")
        sys.exit(0)
    else:
        print("❌ Validation failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate dataset for ML pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/spotify_data_reduced.csv",
        help="Path to the data file",
    )
    parser.add_argument(
        "--target-column", type=str, default="genre", help="Name of the target column"
    )

    args = parser.parse_args()
    main(args.data_path, args.target_column)
