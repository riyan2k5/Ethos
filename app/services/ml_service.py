"""
ML service for making predictions using trained models.
"""
import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from typing import Dict, List, Optional

# Suppress sklearn feature name warnings (we handle this properly)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class MLService:
    """Service for ML model predictions."""
    
    def __init__(self):
        self.models_dir = Path(__file__).parent.parent.parent / "models"
        self.genre_model = None
        self.genre_features = None
        self.similar_songs_model = None
        self.similar_songs_scaler = None
        self.similar_songs_features = None
        self.similar_songs_track_info = None
    
    def load_models(self):
        """Load all ML models."""
        # Load genre classification model
        genre_model_file = self.models_dir / "genre_classification_model.pkl"
        genre_features_file = self.models_dir / "genre_classification_features.pkl"
        
        if genre_model_file.exists() and genre_features_file.exists():
            with open(genre_model_file, 'rb') as f:
                self.genre_model = pickle.load(f)
            with open(genre_features_file, 'rb') as f:
                self.genre_features = pickle.load(f)
            print("✅ Loaded genre classification model")
        
        # Load similar songs model
        similar_model_file = self.models_dir / "similar_songs_model.pkl"
        similar_scaler_file = self.models_dir / "similar_songs_scaler.pkl"
        similar_features_file = self.models_dir / "similar_songs_features.pkl"
        
        if all(f.exists() for f in [similar_model_file, similar_scaler_file, similar_features_file]):
            with open(similar_model_file, 'rb') as f:
                model_data = pickle.load(f)
                self.similar_songs_model = model_data.get('knn_model')
                self.similar_songs_track_info = model_data.get('track_info')
            
            with open(similar_scaler_file, 'rb') as f:
                self.similar_songs_scaler = pickle.load(f)
            
            with open(similar_features_file, 'rb') as f:
                self.similar_songs_features = pickle.load(f)
            
            print("✅ Loaded similar songs model")
    
    def _create_engineered_features(self, features: Dict) -> Dict:
        """Create engineered features from base features."""
        engineered = features.copy()
        
        if 'danceability' in features and 'energy' in features:
            engineered['Dance_Energy_Interaction'] = features['danceability'] * features['energy']
        
        if 'valence' in features and 'energy' in features:
            engineered['Valence_Energy_Ratio'] = features['valence'] / (features['energy'] + 1e-6)
        
        if 'acousticness' in features and 'instrumentalness' in features:
            engineered['Acoustic_Instrumental_Interaction'] = features['acousticness'] * features['instrumentalness']
        
        if 'speechiness' in features and 'energy' in features:
            engineered['Speechiness_Energy_Ratio'] = features['speechiness'] / (features['energy'] + 1e-6)
        
        # Handle infinities
        for key, value in engineered.items():
            if isinstance(value, (int, float)) and (np.isinf(value) or np.isnan(value)):
                engineered[key] = 0.0
        
        return engineered
    
    def predict_genre(self, song: Dict) -> Dict:
        """Predict genre for a song."""
        if self.genre_model is None or self.genre_features is None:
            return {"predicted": "Unknown", "probabilities": {}, "top_predictions": []}
        
        try:
            # Extract features
            features = {}
            engineered = self._create_engineered_features(song)
            
            for feat in self.genre_features:
                if feat in song:
                    try:
                        features[feat] = float(song[feat])
                    except (ValueError, TypeError):
                        # Try engineered features
                        if feat in engineered:
                            features[feat] = float(engineered[feat])
                        else:
                            # Use default value
                            features[feat] = 0.0
                elif feat in engineered:
                    features[feat] = float(engineered[feat])
                else:
                    # Missing feature - use default
                    features[feat] = 0.0
            
            # Create feature vector in correct order as a DataFrame with column names
            feature_vector = [features.get(feat, 0.0) for feat in self.genre_features]
            # Create DataFrame with proper column names to match training data
            feature_df = pd.DataFrame([feature_vector], columns=self.genre_features)
            
            # Predict (using DataFrame to match training format)
            predicted = self.genre_model.predict(feature_df)[0]
            probabilities = self.genre_model.predict_proba(feature_df)[0]
            classes = self.genre_model.classes_
            
            # Get top predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            top_predictions = [
                {"name": classes[idx], "score": float(probabilities[idx])}
                for idx in top_indices
            ]
            
            return {
                "predicted": str(predicted),
                "probabilities": {str(classes[i]): float(probabilities[i]) for i in range(len(classes))},
                "top_predictions": top_predictions
            }
        except Exception as e:
            print(f"Error in predict_genre: {e}")
            import traceback
            traceback.print_exc()
            return {"predicted": "Unknown", "probabilities": {}, "top_predictions": []}
    
    def find_similar_songs(self, song: Dict, limit: int = 10, available_track_ids: set = None) -> List[Dict]:
        """Find similar songs using KNN model based on audio features.
        
        Args:
            song: Song dictionary with features
            limit: Number of similar songs to return
            available_track_ids: Set of track IDs that exist in current dataset (optional)
        """
        if self.similar_songs_model is None or self.similar_songs_track_info is None:
            return []
        
        track_id = song.get('track_id')
        if not track_id:
            return []
        
        try:
            # Prepare features from the song (even if it's not in the model)
            features = {}
            engineered = self._create_engineered_features(song)
            
            for feat in self.similar_songs_features:
                if feat in song:
                    try:
                        features[feat] = float(song[feat])
                    except (ValueError, TypeError):
                        if feat in engineered:
                            features[feat] = float(engineered[feat])
                        else:
                            features[feat] = 0.0
                elif feat in engineered:
                    features[feat] = float(engineered[feat])
                else:
                    # Missing feature - use default
                    features[feat] = 0.0
            
            feature_vector = [features.get(feat, 0.0) for feat in self.similar_songs_features]
            # Create DataFrame with proper column names
            feature_df = pd.DataFrame([feature_vector], columns=self.similar_songs_features)
            
            # Scale and find neighbors using the song's features
            feature_array = feature_df.values
            feature_scaled = self.similar_songs_scaler.transform(feature_array)
            
            # Find many neighbors (we'll filter to available tracks)
            # Since only ~17% of dataset overlaps with model, search much more
            # Search for 50x more to ensure we get enough after filtering
            search_limit = min(limit * 50, len(self.similar_songs_track_info))
            if available_track_ids:
                # If filtering, search even more aggressively
                search_limit = min(limit * 100, len(self.similar_songs_track_info))
            
            distances, indices = self.similar_songs_model.kneighbors(
                feature_scaled, n_neighbors=search_limit
            )
            
            # Convert distances to similarities
            similar_indices = indices[0]
            similar_distances = distances[0]
            similarities = 1 - similar_distances  # Convert cosine distance to similarity
            
            # Build results, filtering by available track IDs if provided
            results = []
            seen_track_ids = {track_id}  # Don't include the song itself
            
            for idx, sim_score in zip(similar_indices, similarities):
                try:
                    similar_track = self.similar_songs_track_info.iloc[idx]
                    sim_track_id = similar_track.get('track_id', '')
                    
                    # Skip if we've already seen this track or it's the same track
                    if sim_track_id in seen_track_ids:
                        continue
                    
                    # If available_track_ids is provided, only include tracks that exist in current dataset
                    if available_track_ids is not None:
                        if sim_track_id not in available_track_ids:
                            continue  # Skip tracks not in current dataset
                    
                    seen_track_ids.add(sim_track_id)
                    results.append({
                        'track_id': sim_track_id,
                        'track_name': similar_track.get('track_name', 'Unknown'),
                        'artist_name': similar_track.get('artist_name', 'Unknown'),
                        'similarity': float(sim_score)
                    })
                    
                    # Stop when we have enough results
                    if len(results) >= limit:
                        break
                except Exception as e:
                    print(f"Error processing similar track at index {idx}: {e}")
                    continue
            
            return results
        except Exception as e:
            print(f"Error in find_similar_songs: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def predict_all(self, song: Dict) -> Dict:
        """Get all predictions for a song."""
        return {
            "genre": self.predict_genre(song),
            "similar_count": len(self.find_similar_songs(song, limit=1))
        }

