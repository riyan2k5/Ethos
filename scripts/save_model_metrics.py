"""
Script to save model metrics after training.
This should be called after training each model to save evaluation metrics.
"""
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.model_metrics_loader import ModelMetricsLoader


def save_genre_metrics(accuracy: float, precision: float = None, recall: float = None, f1: float = None):
    """Save genre classification metrics."""
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision) if precision else None,
        "recall": float(recall) if recall else None,
        "f1_score": float(f1) if f1 else None
    }
    
    loader = ModelMetricsLoader()
    metrics_file = loader.metrics_dir / "genre_classification_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved genre classification metrics to {metrics_file}")


def save_regression_metrics(model_name: str, r2: float, mse: float, rmse: float, mae: float):
    """Save regression model metrics."""
    metrics = {
        "r2_score": float(r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae)
    }
    
    loader = ModelMetricsLoader()
    metrics_file = loader.metrics_dir / f"{model_name}_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved {model_name} metrics to {metrics_file}")


def save_clustering_metrics(silhouette_score: float, cluster_distribution: dict = None):
    """Save clustering model metrics."""
    metrics = {
        "silhouette_score": float(silhouette_score),
        "cluster_distribution": cluster_distribution or {}
    }
    
    loader = ModelMetricsLoader()
    metrics_file = loader.metrics_dir / "clustering_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved clustering metrics to {metrics_file}")


if __name__ == "__main__":
    print("This script is meant to be imported and used in training scripts.")
    print("Example usage:")
    print("  from scripts.save_model_metrics import save_genre_metrics")
    print("  save_genre_metrics(accuracy=0.85)")

