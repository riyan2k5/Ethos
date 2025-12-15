"""
Utility script to predict energy level for new songs using the saved regression model.
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
MODEL_FILE = MODELS_DIR / "energy_regression_model.pkl"
FEATURES_FILE = MODELS_DIR / "energy_regression_features.pkl"
TABLE_NAME = "spotify_songs"


def predict_energy(features: dict):
    """
    Predict energy level for a song given its features.
    
    Args:
        features: Dictionary with feature values (danceability, loudness, etc.)
    
    Returns:
        Predicted energy value (0.0 to 1.0)
    """
    # Load model components
    print(f"Loading regression model from '{MODEL_FILE}'...")
    with open(MODEL_FILE, 'rb') as f:
        regressor = pickle.load(f)
    
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
    
    # Predict
    predicted_energy = regressor.predict(feature_array)[0]
    
    # Clamp to valid range
    predicted_energy = max(0.0, min(1.0, predicted_energy))
    
    print(f"\n📊 Energy Prediction:")
    print(f"   Predicted Energy: {predicted_energy:.3f} (Range: 0.0 to 1.0)")
    
    return predicted_energy


def predict_energy_from_track_id(track_id: str):
    """
    Predict energy for a track in the database.
    
    Args:
        track_id: Track ID in the database
    
    Returns:
        Predicted energy value
    """
    # Load data from database
    config = DatabaseConfig()
    conn_string = config.get_connection_string()
    engine = create_engine(conn_string)
    
    query = f"""
    SELECT danceability, loudness, speechiness, acousticness,
           instrumentalness, valence, energy
    FROM {TABLE_NAME}
    WHERE track_id = %s;
    """
    
    df = pd.read_sql(query, con=engine, params=(track_id,))
    engine.dispose()
    
    if len(df) == 0:
        raise ValueError(f"Track ID '{track_id}' not found in database")
    
    features = df.iloc[0].to_dict()
    actual_energy = features.pop('energy', None)
    
    predicted = predict_energy(features)
    
    if actual_energy is not None:
        print(f"   Actual Energy: {actual_energy:.3f}")
        print(f"   Prediction Error: {abs(actual_energy - predicted):.3f}")
    
    return predicted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict energy level for a song")
    parser.add_argument("--track-id", type=str, help="Track ID in database")
    parser.add_argument("--danceability", type=float, help="Danceability value")
    parser.add_argument("--loudness", type=float, help="Loudness value")
    parser.add_argument("--speechiness", type=float, help="Speechiness value")
    parser.add_argument("--acousticness", type=float, help="Acousticness value")
    parser.add_argument("--instrumentalness", type=float, help="Instrumentalness value")
    parser.add_argument("--valence", type=float, help="Valence value")
    
    args = parser.parse_args()
    
    try:
        if args.track_id:
            predict_energy_from_track_id(args.track_id)
        else:
            features = {
                'danceability': args.danceability,
                'loudness': args.loudness,
                'speechiness': args.speechiness,
                'acousticness': args.acousticness,
                'instrumentalness': args.instrumentalness,
                'valence': args.valence,
            }
            predict_energy(features)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

