"""
Utility script to find similar songs using the saved similarity model.
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
MODEL_FILE = MODELS_DIR / "similar_songs_model.pkl"
SCALER_FILE = MODELS_DIR / "similar_songs_scaler.pkl"
FEATURES_FILE = MODELS_DIR / "similar_songs_features.pkl"
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


def find_similar_songs(track_id: str, n_similar: int = 10):
    """
    Find similar songs to a given track_id using KNN model.
    
    Args:
        track_id: The track_id to find similar songs for
        n_similar: Number of similar songs to return
    
    Returns:
        DataFrame with similar songs
    """
    # Load model components
    print(f"Loading similarity model from '{MODEL_FILE}'...")
    with open(MODEL_FILE, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    
    knn_model = model_data['knn_model']
    track_info = model_data['track_info']
    feature_columns = model_data['feature_columns']
    
    # Find track index
    track_mask = track_info['track_id'] == track_id
    if not track_mask.any():
        raise ValueError(f"Track ID '{track_id}' not found in the model data")
    
    track_idx = track_info[track_mask].index[0]
    track = track_info.iloc[track_idx]
    
    print(f"\n🎵 Finding songs similar to:")
    print(f"   '{track['track_name']}' by {track['artist_name']}")
    print(f"   Genre: {track['genre']}")
    
    # Load core features from database (engineered features are created locally)
    config = DatabaseConfig()
    conn_string = config.get_connection_string()
    engine = create_engine(conn_string)
    
    query = f"""
    SELECT {', '.join(CORE_FEATURES)}
    FROM {TABLE_NAME}
    WHERE track_id = %s;
    """
    
    track_features_df = pd.read_sql(query, con=engine, params=(track_id,))
    engine.dispose()
    
    if len(track_features_df) == 0:
        raise ValueError(f"Track ID '{track_id}' not found in database")
    
    # Create engineered features (same as training)
    track_features_df = create_engineered_features(track_features_df)
    
    # Prepare and scale features using the same feature columns as training
    # Ensure features are in the same order as training
    track_features = track_features_df[feature_columns].fillna(track_features_df[feature_columns].median())
    track_features_scaled = scaler.transform(track_features)
    
    # Find similar songs using KNN
    distances, indices = knn_model.kneighbors(
        track_features_scaled, 
        n_neighbors=n_similar + 1  # +1 to exclude itself
    )
    
    # Exclude the first result (itself) and convert distances to similarities
    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]
    similarities = 1 - similar_distances  # Convert cosine distance to cosine similarity
    
    # Create results DataFrame
    results = []
    for idx, sim_score in zip(similar_indices, similarities):
        similar_track = track_info.iloc[idx]
        results.append({
            'track_id': similar_track['track_id'],
            'track_name': similar_track['track_name'],
            'artist_name': similar_track['artist_name'],
            'genre': similar_track['genre'],
            'popularity': similar_track['popularity'],
            'similarity_score': sim_score
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n📊 Top {n_similar} Similar Songs:")
    print("=" * 80)
    for i, row in results_df.iterrows():
        print(f"{i+1}. '{row['track_name']}' by {row['artist_name']}")
        print(f"   Genre: {row['genre']}, Similarity: {row['similarity_score']:.3f}, "
              f"Popularity: {row['popularity']}")
    
    return results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Find similar songs")
    parser.add_argument("--track-id", type=str, required=True, help="Track ID to find similar songs for")
    parser.add_argument("--n-similar", type=int, default=10, help="Number of similar songs to return")
    
    args = parser.parse_args()
    
    try:
        find_similar_songs(args.track_id, args.n_similar)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

