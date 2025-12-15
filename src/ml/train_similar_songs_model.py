"""
Model for finding similar songs based on audio features.
Uses KNN with cosine similarity to find songs similar to a given track.
Memory-efficient approach that doesn't store full similarity matrix.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sys
import time
import os
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.db_connection import get_db_connection, test_connection
from src.api.db_config import DatabaseConfig
from sqlalchemy import create_engine

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
TABLE_NAME = "spotify_songs"
MODEL_FILE = MODELS_DIR / "similar_songs_model.pkl"
SCALER_FILE = MODELS_DIR / "similar_songs_scaler.pkl"
FEATURES_FILE = MODELS_DIR / "similar_songs_features.pkl"

# Check if DATABASE_URL is set
if not os.getenv("DATABASE_URL") and not os.getenv("DB_CONNECTION_STRING"):
    print("=" * 60)
    print("⚠️  WARNING: DATABASE_URL not set!")
    print("=" * 60)
    print("Please set your database connection string:")
    print(
        "  export DATABASE_URL='postgresql://user:password@host:port/database?sslmode=require'"
    )
    print("=" * 60)
    print()

# Core audio features used for similarity
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
        df["Acoustic_Instrumental_Interaction"] = (
            df["acousticness"] * df["instrumentalness"]
        )

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


def get_feature_columns(df: pd.DataFrame) -> list:
    """Returns the list of feature columns to use."""
    feature_cols = []

    for col in CORE_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    engineered_cols = [
        "Dance_Energy_Interaction",
        "Valence_Energy_Ratio",
        "Acoustic_Instrumental_Interaction",
        "Speechiness_Energy_Ratio",
    ]
    for col in engineered_cols:
        if col in df.columns:
            feature_cols.append(col)

    return feature_cols


def load_data_from_database(table_name: str) -> pd.DataFrame:
    """Load data from PostgreSQL database."""
    print(f"Loading data from database table '{table_name}'...")

    if not test_connection():
        raise ConnectionError(
            "Database connection failed. Please check your configuration."
        )

    try:
        config = DatabaseConfig()
        conn_string = config.get_connection_string()

        engine = create_engine(
            conn_string,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=1,
            max_overflow=0,
            connect_args={"connect_timeout": 10},
        )

        query = f"""
        SELECT 
            id, artist_name, track_name, track_id, genre, popularity,
            danceability, energy, loudness, speechiness, acousticness,
            instrumentalness, valence
        FROM {table_name};
        """

        print("   Fetching data...")
        df = pd.read_sql(query, con=engine)
        engine.dispose()

        print(f"   ✅ Data loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        print(f"   ⚠️  SQLAlchemy method failed: {e}")
        raise ConnectionError(f"Failed to load data from database: {e}")


def train_similar_songs_model(table_name: str):
    """
    Trains a similarity model for finding similar songs.
    Saves the model, scaler, and feature matrix for future use.
    """
    print(f"--- Starting Similar Songs Model Training Pipeline ---")

    config = DatabaseConfig()
    print(f"Database Configuration:")
    print(f"  Host: {config.host}")
    print(f"  Database: {config.database}")
    print(f"  User: {config.user}")
    if "neon.tech" in config.host:
        print(f"  ✅ Using Neon Database")
    else:
        print(f"  ⚠️  Using local database (set DATABASE_URL for Neon)")
    print(f"Data Source: PostgreSQL Database (table: {table_name})")
    print()

    try:
        # 1. Load the Data from Database
        df = load_data_from_database(table_name)

        print(f"\nData Summary:")
        print(f"  Total rows: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")

        # 2. Create Engineered Features
        print("\nCreating engineered features...")
        df = create_engineered_features(df)

        # 3. Feature Selection
        feature_cols = get_feature_columns(df)
        X = df[feature_cols].copy()

        # Store track info for later retrieval
        track_info = df[
            ["id", "track_id", "artist_name", "track_name", "genre", "popularity"]
        ].copy()

        print(f"\nFeatures used for similarity ({len(feature_cols)}):")
        print(feature_cols)

        # 4. Handle missing values
        print("\nPreprocessing data...")
        X = X.fillna(X.median())

        # 5. Scale features for cosine similarity
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 6. Train KNN model (memory-efficient, computes similarities on-demand)
        print("Training KNN model for similarity search...")
        start_time = time.time()
        # Use cosine metric for similarity, n_neighbors=50 to get good coverage
        knn_model = NearestNeighbors(
            n_neighbors=min(50, len(X_scaled)),
            metric="cosine",
            algorithm="brute",  # Brute force is fine for cosine similarity
            n_jobs=-1,
        )
        knn_model.fit(X_scaled)
        end_time = time.time()
        print(f"   ✅ KNN model trained in {end_time - start_time:.2f} seconds")
        print(f"   Data shape: {X_scaled.shape}")

        # 7. Save model components
        print(f"\nSaving model to '{MODEL_FILE}'...")
        # Store track_info with reset index for easier lookup
        track_info_reset = track_info.reset_index(drop=True)

        model_data = {
            "knn_model": knn_model,
            "track_info": track_info_reset,
            "feature_columns": feature_cols,
        }

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model_data, f)

        with open(SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)

        with open(FEATURES_FILE, "wb") as f:
            pickle.dump(feature_cols, f)

        print(f"   ✅ Model saved successfully")
        print(f"   ✅ Scaler saved to '{SCALER_FILE}'")
        print(f"   ✅ Features saved to '{FEATURES_FILE}'")
        print(
            f"   Note: Using KNN model (memory-efficient, no full similarity matrix stored)"
        )

        # 8. Evaluation - Test similarity on a few examples
        print("\n" + "=" * 60)
        print("MODEL EVALUATION - Sample Similarity Results")
        print("=" * 60)

        # Test with first 5 songs
        test_indices = min(5, len(df))
        for i in range(test_indices):
            track = track_info.iloc[i]
            query_point = X_scaled[i : i + 1]  # Reshape for KNN

            # Find similar songs
            distances, indices = knn_model.kneighbors(
                query_point, n_neighbors=6
            )  # 6 to exclude itself
            # Exclude the first result (itself)
            similar_indices = indices[0][1:]
            similar_distances = distances[0][1:]
            # Convert distance to similarity (cosine distance -> cosine similarity)
            similarities = 1 - similar_distances

            print(f"\n🎵 Song: '{track['track_name']}' by {track['artist_name']}")
            print(f"   Genre: {track['genre']}")
            print(f"   Top 5 Similar Songs:")
            for idx, sim_score in zip(similar_indices[:5], similarities[:5]):
                similar_track = track_info.iloc[idx]
                print(
                    f"     - '{similar_track['track_name']}' by {similar_track['artist_name']} "
                    f"(similarity: {sim_score:.3f}, genre: {similar_track['genre']})"
                )

        # Clear X_scaled from memory after evaluation
        del X_scaled

        print("\n" + "=" * 60)
        print("✅ Similar Songs Model Training Completed Successfully!")
        print("=" * 60)
        print(f"\nModel saved to: {MODEL_FILE}")
        print(
            f"To use this model, load it and call find_similar_songs(track_id, n_similar=10)"
        )

    except ConnectionError as e:
        print(f"\n❌ Database Connection Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    train_similar_songs_model(TABLE_NAME)
