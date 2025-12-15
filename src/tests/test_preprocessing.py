"""
Unit tests for preprocessing modules.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.feature_engineering import create_new_features
from src.preprocessing.handle_missing_data import handle_missing_values
from src.preprocessing.dimensionality_reduction import reduce_dimensionality


class TestFeatureEngineering:
    """Tests for feature engineering module."""

    def test_create_new_features_adds_expected_features(self):
        """Test that create_new_features adds the expected 4 new features."""
        # Create sample data
        df = pd.DataFrame(
            {
                "danceability": [0.5, 0.7, 0.9],
                "energy": [0.6, 0.8, 0.4],
                "loudness": [-10, -5, -15],
                "duration_ms": [200000, 180000, 220000],
                "tempo": [120, 130, 110],
                "valence": [0.5, 0.8, 0.3],
            }
        )

        result = create_new_features(df)

        # Check that new features were added
        assert "Dance_Energy_Interaction" in result.columns
        assert "Loudness_Duration_Ratio" in result.columns
        assert "Tempo_vs_120" in result.columns
        assert "Valence_Energy_Ratio" in result.columns

        # Check that original features are preserved
        assert "danceability" in result.columns
        assert "energy" in result.columns

    def test_create_new_features_handles_zero_energy(self):
        """Test that Valence_Energy_Ratio handles zero energy correctly."""
        df = pd.DataFrame(
            {
                "danceability": [0.5],
                "energy": [0.0],  # Zero energy
                "loudness": [-10],
                "duration_ms": [200000],
                "tempo": [120],
                "valence": [0.5],
            }
        )

        result = create_new_features(df)

        # Should not have NaN or inf values
        assert not result["Valence_Energy_Ratio"].isna().any()
        assert not np.isinf(result["Valence_Energy_Ratio"]).any()

    def test_create_new_features_handles_inf_values(self):
        """Test that infinite values are handled correctly."""
        df = pd.DataFrame(
            {
                "danceability": [0.5],
                "energy": [0.0],
                "loudness": [-10],
                "duration_ms": [0],  # This could cause division issues
                "tempo": [120],
                "valence": [0.5],
            }
        )

        result = create_new_features(df)

        # Should not have inf or NaN values
        for col in result.select_dtypes(include=[np.number]).columns:
            assert not np.isinf(result[col]).any(), f"Column {col} contains inf values"


class TestHandleMissingData:
    """Tests for missing data handling module."""

    def test_handle_missing_values_drops_rows_with_missing_text(self):
        """Test that rows with missing artist_name or track_name are dropped."""
        df = pd.DataFrame(
            {
                "artist_name": ["Artist1", None, "Artist3"],
                "track_name": ["Track1", "Track2", None],
                "danceability": [0.5, 0.6, 0.7],
                "energy": [0.6, 0.7, 0.8],
            }
        )

        result = handle_missing_values(df)

        # Should drop rows with missing text columns
        assert len(result) <= len(df)
        assert result["artist_name"].notna().all()
        assert result["track_name"].notna().all()

    def test_handle_missing_values_imputes_numerical_values(self):
        """Test that numerical missing values are imputed."""
        df = pd.DataFrame(
            {
                "artist_name": ["Artist1", "Artist2", "Artist3"],
                "track_name": ["Track1", "Track2", "Track3"],
                "Danceability": [0.5, None, 0.7],
                "Energy": [0.6, 0.7, None],
            }
        )

        result = handle_missing_values(df)

        # Should not have missing values in numerical columns
        assert result["Danceability"].notna().all()
        assert result["Energy"].notna().all()


class TestDimensionalityReduction:
    """Tests for dimensionality reduction module."""

    def test_reduce_dimensionality_keeps_important_features(self):
        """Test that important features are kept."""
        df = pd.DataFrame(
            {
                "year": [2020, 2021, 2022],
                "duration_ms": [200000, 180000, 220000],
                "acousticness": [0.5, 0.6, 0.4],
                "danceability": [0.7, 0.8, 0.6],
                "energy": [0.6, 0.7, 0.5],
                "artist_name": ["Artist1", "Artist2", "Artist3"],
                "track_name": ["Track1", "Track2", "Track3"],
                "genre": ["Pop", "Rock", "Jazz"],
                "unimportant_feature": [1, 2, 3],  # Should be dropped
            }
        )

        result = reduce_dimensionality(df)

        # Should keep important features
        assert "year" in result.columns
        assert "duration_ms" in result.columns
        assert "acousticness" in result.columns

        # Should keep metadata
        assert "artist_name" in result.columns
        assert "genre" in result.columns

        # Should drop unimportant features
        assert "unimportant_feature" not in result.columns
