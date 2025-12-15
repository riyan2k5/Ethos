"""
Utility script to predict genre for new songs using the saved classification model.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db_connection import get_db_connection, test_connection
from api.db_config import DatabaseConfig
from sqlalchemy import create_engine

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR.parent / "models"
MODEL_FILE = MODELS_DIR / "genre_classification_model.pkl"
FEATURES_FILE = MODELS_DIR / "genre_classification_features.pkl"
TABLE_NAME = "spotify_songs"

# Core audio features (from database)
CORE_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "valence",
]


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates engineered features from available audio features."""
    df = df.copy()

    if "danceability" in df.columns and "energy" in df.columns:
        df["Dance_Energy_Interaction"] = df["danceability"] * df["energy"]

    if "valence" in df.columns and "energy" in df.columns:
        df["Valence_Energy_Ratio"] = df["valence"] / (df["energy"] + 1e-6)

    if "acousticness" in df.columns and "instrumentalness" in df.columns:
        df["Acoustic_Instrumental_Interaction"] = df["acousticness"] * df["instrumentalness"]

    if "speechiness" in df.columns and "energy" in df.columns:
        df["Speechiness_Energy_Ratio"] = df["speechiness"] / (df["energy"] + 1e-6)

    df = df.replace([np.inf, -np.inf], np.nan)
    
    engineered_cols = [
        "Dance_Energy_Interaction",
        "Valence_Energy_Ratio",
        "Acoustic_Instrumental_Interaction",
        "Speechiness_Energy_Ratio",
    ]
    
    for col in engineered_cols:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


def predict_genre(features: dict):
    """
    Predict genre for a song given its features.
    
    Args:
        features: Dictionary with feature values
    
    Returns:
        Predicted genre and probability scores
    """
    # Load model components
    print(f"Loading classification model from '{MODEL_FILE}'...")
    with open(MODEL_FILE, 'rb') as f:
        classifier = pickle.load(f)
    
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
    predicted_genre = classifier.predict(feature_array)[0]
    probabilities = classifier.predict_proba(feature_array)[0]
    genre_classes = classifier.classes_
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    print(f"\n📊 Genre Prediction:")
    print(f"   Predicted Genre: {predicted_genre}")
    print(f"\n   Top 3 Genre Probabilities:")
    for idx in top_indices:
        genre = genre_classes[idx]
        prob = probabilities[idx]
        print(f"     {genre}: {prob:.2%}")
    
    return predicted_genre, dict(zip(genre_classes, probabilities))


def predict_genre_from_track_id(track_id: str):
    """
    Predict genre for a track in the database.
    
    Args:
        track_id: Track ID in the database
    
    Returns:
        Predicted genre
    """
    # Load data from database
    config = DatabaseConfig()
    conn_string = config.get_connection_string()
    engine = create_engine(conn_string)
    
    query = f"""
    SELECT danceability, energy, loudness, speechiness, acousticness,
           instrumentalness, valence, genre
    FROM {TABLE_NAME}
    WHERE track_id = %s;
    """
    
    df = pd.read_sql(query, con=engine, params=(track_id,))
    engine.dispose()
    
    if len(df) == 0:
        raise ValueError(f"Track ID '{track_id}' not found in database")
    
    # Create engineered features
    df = create_engineered_features(df)
    
    features = df.iloc[0].to_dict()
    actual_genre = features.pop('genre', None)
    
    predicted, probabilities = predict_genre(features)
    
    if actual_genre is not None:
        print(f"\n   Actual Genre: {actual_genre}")
        print(f"   Prediction: {'✅ Correct' if predicted == actual_genre else '❌ Incorrect'}")
    
    return predicted


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict genre for a song")
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
            predict_genre_from_track_id(args.track_id)
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
            predict_genre(features)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

