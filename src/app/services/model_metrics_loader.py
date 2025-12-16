"""
Service to compute or load model evaluation metrics.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)


class ModelMetricsLoader:
    """Load or compute model evaluation metrics."""

    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.metrics_dir = self.models_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def load_metrics(self, model_name: str) -> Optional[Dict]:
        """Load saved metrics from JSON file."""
        metrics_file = self.metrics_dir / f"{model_name}_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    return json.load(f)
            except:
                pass
        return None

    def get_genre_model_metrics(self) -> Dict:
        """Get metrics for genre classification model."""
        metrics = self.load_metrics("genre_classification")
        if metrics:
            return metrics

        # Try to compute from model if possible
        model_file = self.models_dir / "genre_classification_model.pkl"
        if not model_file.exists():
            return {}

        try:
            with open(model_file, "rb") as f:
                model = pickle.load(f)

            # Can't compute accuracy without test data, but return model info
            return {
                "accuracy": None,
                "note": "Accuracy requires test data. Re-train model to save metrics.",
                "model_info": {
                    "n_classes": (
                        int(model.n_classes_) if hasattr(model, "n_classes_") else None
                    ),
                    "n_estimators": (
                        int(model.n_estimators)
                        if hasattr(model, "n_estimators")
                        else None
                    ),
                },
            }
        except:
            return {}

    def get_energy_regression_metrics(self) -> Dict:
        """Get metrics for energy regression model."""
        metrics = self.load_metrics("energy_regression")
        if metrics:
            return metrics

        model_file = self.models_dir / "energy_regression_model.pkl"
        if not model_file.exists():
            return {}

        return {
            "r2_score": None,
            "mse": None,
            "rmse": None,
            "mae": None,
            "note": "Metrics require test data. Re-train model to save metrics.",
        }

    def get_popularity_regression_metrics(self) -> Dict:
        """Get metrics for popularity regression model."""
        metrics = self.load_metrics("popularity_regression")
        if metrics:
            return metrics

        model_file = self.models_dir / "popularity_regression_model.pkl"
        if not model_file.exists():
            return {}

        return {
            "r2_score": None,
            "mse": None,
            "rmse": None,
            "mae": None,
            "note": "Metrics require test data. Re-train model to save metrics.",
        }

    def get_clustering_metrics(self) -> Dict:
        """Get metrics for clustering model."""
        metrics = self.load_metrics("clustering")
        if metrics:
            return metrics

        model_file = self.models_dir / "clustering_model.pkl"
        if not model_file.exists():
            return {}

        return {
            "silhouette_score": None,
            "note": "Silhouette score requires data. Re-train model to save metrics.",
        }

    def get_similar_songs_metrics(self) -> Dict:
        """Get metrics for similar songs model."""
        metrics = self.load_metrics("similar_songs")
        if metrics:
            return metrics

        # Similar songs model doesn't have traditional metrics
        return {"note": "KNN similarity model - metrics not applicable"}
