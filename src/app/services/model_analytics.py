"""
Service for extracting model analytics and metadata.
"""
import pickle
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from src.app.services.model_metrics_loader import ModelMetricsLoader


class ModelAnalytics:
    """Extract analytics from trained models."""
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent.parent / "models"
        self.metrics_loader = ModelMetricsLoader()
    
    def get_file_size(self, file_path: Path) -> str:
        """Get human-readable file size."""
        if not file_path.exists():
            return "N/A"
        size = file_path.stat().st_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} TB"
    
    def get_file_date(self, file_path: Path) -> str:
        """Get file modification date."""
        if not file_path.exists():
            return "N/A"
        mtime = file_path.stat().st_mtime
        return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    
    def get_genre_model_info(self) -> Dict:
        """Get info about genre classification model."""
        model_file = self.models_dir / "genre_classification_model.pkl"
        features_file = self.models_dir / "genre_classification_features.pkl"
        
        info = {
            "name": "Genre Classification",
            "algorithm": "Random Forest Classifier",
            "type": "Classification",
            "model_file": self.get_file_size(model_file),
            "model_date": self.get_file_date(model_file),
            "exists": model_file.exists()
        }
        
        if features_file.exists():
            try:
                with open(features_file, 'rb') as f:
                    features = pickle.load(f)
                    info["num_features"] = len(features) if isinstance(features, list) else "N/A"
                    info["features"] = features[:10] if isinstance(features, list) else []
            except:
                info["num_features"] = "N/A"
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    if hasattr(model, 'n_estimators'):
                        info["n_estimators"] = model.n_estimators
                    if hasattr(model, 'n_classes_'):
                        info["num_classes"] = model.n_classes_
                    if hasattr(model, 'classes_'):
                        info["classes"] = list(model.classes_)[:10]  # First 10 genres
            except Exception as e:
                info["error"] = str(e)
        
        # Add metrics
        metrics = self.metrics_loader.get_genre_model_metrics()
        info["metrics"] = metrics
        
        return info
    
    def get_clustering_model_info(self) -> Dict:
        """Get info about clustering model."""
        model_file = self.models_dir / "clustering_model.pkl"
        features_file = self.models_dir / "clustering_features.pkl"
        scaler_file = self.models_dir / "clustering_scaler.pkl"
        
        info = {
            "name": "Song Clustering",
            "algorithm": "KMeans",
            "type": "Clustering",
            "model_file": self.get_file_size(model_file),
            "scaler_file": self.get_file_size(scaler_file) if scaler_file.exists() else "N/A",
            "model_date": self.get_file_date(model_file),
            "exists": model_file.exists()
        }
        
        if features_file.exists():
            try:
                with open(features_file, 'rb') as f:
                    features = pickle.load(f)
                    info["num_features"] = len(features) if isinstance(features, list) else "N/A"
            except:
                info["num_features"] = "N/A"
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    if hasattr(model, 'n_clusters'):
                        info["n_clusters"] = model.n_clusters
                    if hasattr(model, 'cluster_centers_'):
                        info["num_clusters"] = len(model.cluster_centers_)
            except Exception as e:
                info["error"] = str(e)
        
        # Add metrics
        metrics = self.metrics_loader.get_clustering_metrics()
        info["metrics"] = metrics
        
        return info
    
    def get_similar_songs_model_info(self) -> Dict:
        """Get info about similar songs model."""
        model_file = self.models_dir / "similar_songs_model.pkl"
        features_file = self.models_dir / "similar_songs_features.pkl"
        scaler_file = self.models_dir / "similar_songs_scaler.pkl"
        
        info = {
            "name": "Similar Songs",
            "algorithm": "K-Nearest Neighbors (KNN)",
            "type": "Similarity Search",
            "model_file": self.get_file_size(model_file),
            "scaler_file": self.get_file_size(scaler_file) if scaler_file.exists() else "N/A",
            "model_date": self.get_file_date(model_file),
            "exists": model_file.exists()
        }
        
        if features_file.exists():
            try:
                with open(features_file, 'rb') as f:
                    features = pickle.load(f)
                    info["num_features"] = len(features) if isinstance(features, list) else "N/A"
            except:
                info["num_features"] = "N/A"
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        knn_model = model_data.get('knn_model')
                        track_info = model_data.get('track_info')
                        if knn_model and hasattr(knn_model, 'n_neighbors'):
                            info["n_neighbors"] = knn_model.n_neighbors
                        if track_info is not None:
                            try:
                                info["num_tracks"] = len(track_info) if hasattr(track_info, '__len__') else "N/A"
                            except:
                                pass
            except Exception as e:
                info["error"] = str(e)
        
        # Add metrics
        metrics = self.metrics_loader.get_similar_songs_metrics()
        info["metrics"] = metrics
        
        return info
    
    def get_energy_regression_info(self) -> Dict:
        """Get info about energy regression model."""
        model_file = self.models_dir / "energy_regression_model.pkl"
        features_file = self.models_dir / "energy_regression_features.pkl"
        
        info = {
            "name": "Energy Regression",
            "algorithm": "Random Forest Regressor",
            "type": "Regression",
            "model_file": self.get_file_size(model_file),
            "model_date": self.get_file_date(model_file),
            "exists": model_file.exists()
        }
        
        if features_file.exists():
            try:
                with open(features_file, 'rb') as f:
                    features = pickle.load(f)
                    info["num_features"] = len(features) if isinstance(features, list) else "N/A"
            except:
                info["num_features"] = "N/A"
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    if hasattr(model, 'n_estimators'):
                        info["n_estimators"] = model.n_estimators
            except Exception as e:
                info["error"] = str(e)
        
        # Add metrics
        metrics = self.metrics_loader.get_energy_regression_metrics()
        info["metrics"] = metrics
        
        return info
    
    def get_popularity_regression_info(self) -> Dict:
        """Get info about popularity regression model."""
        model_file = self.models_dir / "popularity_regression_model.pkl"
        features_file = self.models_dir / "popularity_regression_features.pkl"
        
        info = {
            "name": "Popularity Regression",
            "algorithm": "Random Forest Regressor",
            "type": "Regression",
            "model_file": self.get_file_size(model_file),
            "model_date": self.get_file_date(model_file),
            "exists": model_file.exists()
        }
        
        if features_file.exists():
            try:
                with open(features_file, 'rb') as f:
                    features = pickle.load(f)
                    info["num_features"] = len(features) if isinstance(features, list) else "N/A"
            except:
                info["num_features"] = "N/A"
        
        if model_file.exists():
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                    if hasattr(model, 'n_estimators'):
                        info["n_estimators"] = model.n_estimators
            except Exception as e:
                info["error"] = str(e)
        
        # Add metrics
        metrics = self.metrics_loader.get_popularity_regression_metrics()
        info["metrics"] = metrics
        
        return info
    
    def get_all_analytics(self) -> Dict:
        """Get analytics for all models."""
        return {
            "models": [
                self.get_genre_model_info(),
                self.get_clustering_model_info(),
                self.get_similar_songs_model_info(),
                self.get_energy_regression_info(),
                self.get_popularity_regression_info()
            ],
            "dataset_info": {
                "current_dataset_size": self._get_current_dataset_size()
            }
        }
    
    def _get_current_dataset_size(self) -> int:
        """Get current dataset size."""
        try:
            from src.app.services.data_loader import DataLoader
            dl = DataLoader()
            dl.load_dataset()
            return len(dl.df) if dl.df is not None else 0
        except:
            return 0

