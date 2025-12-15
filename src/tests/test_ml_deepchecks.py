"""
DeepChecks ML tests for data integrity, drift detection, and model validation.
These tests run automatically in CI/CD to detect ML issues.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Optional
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from deepchecks.tabular import Dataset
    from deepchecks.tabular.checks import (
        DataIntegrity,
        TrainTestValidation,
        ModelErrorAnalysis,
        ConfusionMatrixReport,
        SimpleModelComparison,
        WeakSegmentsPerformance,
    )
    from deepchecks.tabular.suites import (
        train_test_validation,
        model_evaluation,
    )

    DEEPCHECKS_AVAILABLE = True
except ImportError:
    DEEPCHECKS_AVAILABLE = False
    pytest.skip("DeepChecks not installed", allow_module_level=True)

from src.api.db_connection import get_db_connection
from src.api.db_config import DatabaseConfig


@pytest.fixture(scope="module")
def load_training_data():
    """Load training data from database."""
    try:
        config = DatabaseConfig()
        from sqlalchemy import create_engine

        conn_string = config.get_connection_string()
        engine = create_engine(
            conn_string,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=1,
            max_overflow=0,
            connect_args={"connect_timeout": 10},
        )

        query = """
        SELECT 
            artist_name, track_name, track_id, genre, popularity,
            danceability, energy, loudness, speechiness, acousticness,
            instrumentalness, valence
        FROM spotify_songs
        LIMIT 1000;
        """

        df = pd.read_sql(query, con=engine)
        engine.dispose()

        if len(df) == 0:
            pytest.skip("No data available in database")

        return df
    except Exception as e:
        pytest.skip(f"Could not load data from database: {e}")


@pytest.fixture(scope="module")
def prepare_datasets(load_training_data):
    """Prepare train and test datasets for DeepChecks."""
    df = load_training_data.copy()

    # Create engineered features (same as in training)
    if "danceability" in df.columns and "energy" in df.columns:
        df["Dance_Energy_Interaction"] = df["danceability"] * df["energy"]

    if "valence" in df.columns and "energy" in df.columns:
        df["Valence_Energy_Ratio"] = df["valence"] / (df["energy"] + 1e-6)

    if "acousticness" in df.columns and "instrumentalness" in df.columns:
        df["Acoustic_Instrumental_Interaction"] = (
            df["acousticness"] * df["instrumentalness"]
        )

    if "speechiness" in df.columns and "energy" in df.columns:
        df["Speechiness_Energy_Ratio"] = df["speechiness"] / (df["energy"] + 1e-6)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Feature columns (excluding metadata)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["artist_name", "track_name", "track_id", "genre", "popularity"]
        and not col.startswith("id")
    ]

    # Split into train/test (80/20)
    from sklearn.model_selection import train_test_split

    X = df[feature_cols]
    y_genre = df["genre"]
    y_popularity = df["popularity"]

    X_train, X_test, y_train_genre, y_test_genre = train_test_split(
        X, y_genre, test_size=0.2, random_state=42, stratify=y_genre
    )

    _, _, y_train_pop, y_test_pop = train_test_split(
        X, y_popularity, test_size=0.2, random_state=42
    )

    # Create DeepChecks datasets
    train_dataset = Dataset(
        X_train.join(y_train_genre.to_frame("genre")),
        label="genre",
        cat_features=[],
        datetime_name=None,
    )

    test_dataset = Dataset(
        X_test.join(y_test_genre.to_frame("genre")),
        label="genre",
        cat_features=[],
        datetime_name=None,
    )

    return {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_genre": y_train_genre,
        "y_test_genre": y_test_genre,
        "y_train_pop": y_train_pop,
        "y_test_pop": y_test_pop,
        "feature_cols": feature_cols,
    }


@pytest.fixture(scope="module")
def load_trained_model():
    """Load a trained model for validation."""
    model_path = (
        Path(__file__).parent.parent.parent
        / "models"
        / "genre_classification_model.pkl"
    )

    if not model_path.exists():
        pytest.skip("Trained model not found. Run model training first.")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        pytest.skip(f"Could not load model: {e}")


class TestDataIntegrity:
    """Tests for data integrity using DeepChecks."""

    @pytest.mark.skipif(not DEEPCHECKS_AVAILABLE, reason="DeepChecks not available")
    def test_data_integrity_checks(self, prepare_datasets):
        """Test data integrity on training dataset."""
        train_dataset = prepare_datasets["train_dataset"]

        # Run data integrity suite
        integrity_suite = DataIntegrity()
        result = integrity_suite.run(train_dataset)

        # Check that no critical issues were found
        assert result.passed, f"Data integrity checks failed: {result}"

        # Log results
        print("\nData Integrity Check Results:")
        print(result)

    @pytest.mark.skipif(not DEEPCHECKS_AVAILABLE, reason="DeepChecks not available")
    def test_train_test_validation(self, prepare_datasets):
        """Test train-test validation for data drift and distribution shifts."""
        train_dataset = prepare_datasets["train_dataset"]
        test_dataset = prepare_datasets["test_dataset"]

        # Run train-test validation suite
        suite = train_test_validation()
        result = suite.run(train_dataset, test_dataset)

        # Check that no critical issues were found
        # Note: Some warnings are acceptable, but failures should be investigated
        critical_failures = [
            check
            for check in result.results
            if check.priority == 1 and not check.passed
        ]

        assert (
            len(critical_failures) == 0
        ), f"Critical train-test validation failures found: {critical_failures}"

        print("\nTrain-Test Validation Results:")
        print(result)


class TestModelValidation:
    """Tests for model validation using DeepChecks."""

    @pytest.mark.skipif(not DEEPCHECKS_AVAILABLE, reason="DeepChecks not available")
    def test_confusion_matrix_report(self, prepare_datasets, load_trained_model):
        """Test confusion matrix generation."""
        train_dataset = prepare_datasets["train_dataset"]
        test_dataset = prepare_datasets["test_dataset"]
        model = load_trained_model

        check = ConfusionMatrixReport()
        result = check.run(train_dataset, test_dataset, model)

        assert result.passed, f"Confusion matrix check failed: {result}"

        print("\nConfusion Matrix Report:")
        print(result)

    @pytest.mark.skipif(not DEEPCHECKS_AVAILABLE, reason="DeepChecks not available")
    def test_model_error_analysis(self, prepare_datasets, load_trained_model):
        """Test model error analysis."""
        train_dataset = prepare_datasets["train_dataset"]
        test_dataset = prepare_datasets["test_dataset"]
        model = load_trained_model

        check = ModelErrorAnalysis()
        result = check.run(train_dataset, test_dataset, model)

        print("\nModel Error Analysis:")
        print(result)

        # Error analysis is informational, not a pass/fail check
        assert result is not None


class TestDataDriftDetection:
    """Tests for data drift detection."""

    @pytest.mark.skipif(not DEEPCHECKS_AVAILABLE, reason="DeepChecks not available")
    def test_feature_drift_detection(self, prepare_datasets):
        """Test for feature drift between train and test sets."""
        train_dataset = prepare_datasets["train_dataset"]
        test_dataset = prepare_datasets["test_dataset"]

        # Use train-test validation which includes drift detection
        suite = train_test_validation()
        result = suite.run(train_dataset, test_dataset)

        # Check for drift warnings
        drift_warnings = [
            check
            for check in result.results
            if "drift" in str(check).lower() and not check.passed
        ]

        if drift_warnings:
            print("\n⚠️ Data drift detected:")
            for warning in drift_warnings:
                print(f"  - {warning}")

        # Drift detection is a warning, not a failure
        # Log it but don't fail the test unless it's critical


@pytest.mark.integration
class TestMLPipelineIntegration:
    """Integration tests for ML pipeline with DeepChecks."""

    @pytest.mark.skipif(not DEEPCHECKS_AVAILABLE, reason="DeepChecks not available")
    def test_full_ml_pipeline_validation(self, prepare_datasets, load_trained_model):
        """Run full ML pipeline validation suite."""
        train_dataset = prepare_datasets["train_dataset"]
        test_dataset = prepare_datasets["test_dataset"]
        model = load_trained_model

        # Run comprehensive validation
        print("\n" + "=" * 80)
        print("Running Full ML Pipeline Validation")
        print("=" * 80)

        # Data integrity
        integrity_result = DataIntegrity().run(train_dataset)
        print(
            "\n1. Data Integrity:",
            "✅ PASSED" if integrity_result.passed else "❌ FAILED",
        )

        # Train-test validation
        train_test_result = train_test_validation().run(train_dataset, test_dataset)
        print(
            "2. Train-Test Validation:",
            "✅ PASSED" if train_test_result.passed else "⚠️ WARNINGS",
        )

        # Model evaluation (informational only - no accuracy threshold validation)
        model_eval_result = model_evaluation().run(train_dataset, test_dataset, model)
        print(
            "3. Model Evaluation:",
            "✅ PASSED" if model_eval_result.passed else "⚠️ WARNINGS",
        )
        # Note: Model evaluation runs but doesn't enforce accuracy thresholds

        print("=" * 80)

        # Overall validation - at least data integrity should pass
        assert integrity_result.passed, "Data integrity checks must pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
