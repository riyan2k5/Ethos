"""
Utility script to predict cluster assignment for new songs using the saved clustering model.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.db_connection import get_db_connection, test_connection
from src.api.db_config import DatabaseConfig
from sqlalchemy import create_engine

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR.parent / "models"
MODEL_FILE = MODELS_DIR / "clustering_model.pkl"
SCALER_FILE = MODELS_DIR / "clustering_scaler.pkl"
FEATURES_FILE = MODELS_DIR / "clustering_features.pkl"
TABLE_NAME = "spotify_songs"


def predict_cluster(features: dict):
    """
    Predict cluster assignment for a song given its features.
    
    Args:
        features: Dictionary with feature values (danceability, energy, etc.)
    
    Returns:
        Cluster ID (0 or 1)
    """
    # Load model components
    print(f"Loading clustering model from '{MODEL_FILE}'...")
    with open(MODEL_FILE, 'rb') as f:
        kmeans = pickle.load(f)
    
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(FEATURES_FILE, 'rb') as f:
        feature_columns = pickle.load(f)
    
    # Prepare feature vector
    feature_vector = []
    for col in feature_columns:
        if col in features:
            feature_vector.append(features[col])
        else:
            raise ValueError(f"Missing required feature: {col}")
    
    feature_array = np.array(feature_vector).reshape(1, -1)
    
    # Scale and predict
    feature_scaled = scaler.transform(feature_array)
    cluster_id = kmeans.predict(feature_scaled)[0]
    
    print(f"\n📊 Cluster Prediction:")
    print(f"   Predicted Cluster: {cluster_id}")
    print(f"   Distance to Cluster Centers:")
    distances = kmeans.transform(feature_scaled)[0]
    for i, dist in enumerate(distances):
        print(f"     Cluster {i}: {dist:.3f}")
    
    return cluster_id


def predict_cluster_from_track_id(track_id: str):
    """
    Predict cluster for a track in the database.
    
    Args:
        track_id: Track ID in the database
    
    Returns:
        Cluster ID
    """
    # Load data from database
    config = DatabaseConfig()
    conn_string = config.get_connection_string()
    engine = create_engine(conn_string)
    
    query = f"""
    SELECT danceability, energy, loudness, speechiness, acousticness,
           instrumentalness, valence
    FROM {TABLE_NAME}
    WHERE track_id = %s;
    """
    
    df = pd.read_sql(query, con=engine, params=(track_id,))
    engine.dispose()
    
    if len(df) == 0:
        raise ValueError(f"Track ID '{track_id}' not found in database")
    
    features = df.iloc[0].to_dict()
    return predict_cluster(features)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict cluster for a song")
    parser.add_argument("--track-id", type=str, help="Track ID in database")
    parser.add_argument("--danceability", type=float, help="Danceability value")
    parser.add_argument("--energy", type=float, help="Energy value")
    parser.add_argument("--loudness", type=float, help="Loudness value")
    parser.add_argument("--speechiness", type=float, help="Speechiness value")
    parser.add_argument("--acousticness", type=float, help="Acousticness value")
    parser.add_argument("--instrumentalness", type=float, help="Instrumentalness value")
    parser.add_argument("--valence", type=float, help="Valence value")
    
    args = parser.parse_args()
    
    try:
        if args.track_id:
            predict_cluster_from_track_id(args.track_id)
        else:
            features = {
                'danceability': args.danceability,
                'energy': args.energy,
                'loudness': args.loudness,
                'speechiness': args.speechiness,
                'acousticness': args.acousticness,
                'instrumentalness': args.instrumentalness,
                'valence': args.valence,
            }
            predict_cluster(features)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

