"""
Model for clustering songs into two genre groups using KMeans clustering.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from pathlib import Path
import sys
import time
import os
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db_connection import get_db_connection, test_connection
from api.db_config import DatabaseConfig
from sqlalchemy import create_engine

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
TABLE_NAME = "spotify_songs"
MODEL_FILE = MODELS_DIR / "clustering_model.pkl"
SCALER_FILE = MODELS_DIR / "clustering_scaler.pkl"
FEATURES_FILE = MODELS_DIR / "clustering_features.pkl"
N_CLUSTERS = 2  # Two genre clusters

# Check if DATABASE_URL is set
if not os.getenv("DATABASE_URL") and not os.getenv("DB_CONNECTION_STRING"):
    print("=" * 60)
    print("⚠️  WARNING: DATABASE_URL not set!")
    print("=" * 60)
    print("Please set your database connection string:")
    print("  export DATABASE_URL='postgresql://user:password@host:port/database?sslmode=require'")
    print("=" * 60)
    print()

# Core audio features used for clustering
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
        raise ConnectionError("Database connection failed. Please check your configuration.")
    
    try:
        config = DatabaseConfig()
        conn_string = config.get_connection_string()
        
        engine = create_engine(
            conn_string,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=1,
            max_overflow=0,
            connect_args={"connect_timeout": 10}
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


def train_clustering_model(table_name: str, n_clusters: int = 2):
    """
    Trains a KMeans clustering model to group songs into two clusters.
    Saves the model, scaler, and feature columns for future use.
    """
    print(f"--- Starting Clustering Model Training Pipeline ---")
    print(f"Target: {n_clusters} clusters (two genre groups)")
    
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
        print(f"  Genres: {df['genre'].nunique()}")
        print(f"  Genre distribution:")
        genre_counts = df['genre'].value_counts()
        for genre, count in genre_counts.items():
            print(f"    {genre}: {count}")

        # 2. Create Engineered Features
        print("\nCreating engineered features...")
        df = create_engineered_features(df)

        # 3. Feature Selection
        feature_cols = get_feature_columns(df)
        X = df[feature_cols].copy()
        
        # Store track info
        track_info = df[['id', 'track_id', 'artist_name', 'track_name', 'genre', 'popularity']].copy()
        
        print(f"\nFeatures used for clustering ({len(feature_cols)}):")
        print(feature_cols)

        # 4. Handle missing values
        print("\nPreprocessing data...")
        X = X.fillna(X.median())
        
        # 5. Scale features
        print("Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 6. Train KMeans model
        print(f"\nTraining KMeans model with {n_clusters} clusters...")
        start_time = time.time()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        end_time = time.time()
        print(f"   ✅ Model trained in {end_time - start_time:.2f} seconds")

        # 7. Save model components
        print(f"\nSaving model to '{MODEL_FILE}'...")
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(kmeans, f)
        
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(FEATURES_FILE, 'wb') as f:
            pickle.dump(feature_cols, f)
        
        print(f"   ✅ Model saved successfully")
        print(f"   ✅ Scaler saved to '{SCALER_FILE}'")
        print(f"   ✅ Features saved to '{FEATURES_FILE}'")

        # 8. Evaluation
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        sys.stdout.flush()  # Ensure output is flushed
        
        try:
            # Add cluster labels to track info (reset index to ensure alignment)
            print("\nPreparing evaluation data...", flush=True)
            sys.stdout.flush()
            track_info_eval = track_info.reset_index(drop=True).copy()
            track_info_eval['cluster'] = cluster_labels
            print(f"   Track info shape: {track_info_eval.shape}")
            print(f"   Cluster labels shape: {cluster_labels.shape}")
            
            # Silhouette score (use sample for large datasets to avoid long computation)
            print("\nComputing silhouette score...", flush=True)
            sys.stdout.flush()
            if len(X_scaled) > 5000:
                print("   (Large dataset detected, using sample for faster computation...)")
                sample_size = min(5000, len(X_scaled))
                sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                X_sample = X_scaled[sample_indices]
                labels_sample = cluster_labels[sample_indices]
                silhouette_avg = silhouette_score(X_sample, labels_sample)
                print(f"   ✅ Silhouette Score (on {sample_size} sample): {silhouette_avg:.3f}")
            else:
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                print(f"   ✅ Silhouette Score: {silhouette_avg:.3f}")
            print("   (Range: -1 to 1, higher is better)")
            
            # Cluster distribution
            print(f"\nCluster Distribution:")
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            print(f"   Total clusters found: {len(cluster_counts)}")
            for cluster_id, count in cluster_counts.items():
                print(f"  Cluster {cluster_id}: {count} songs ({count/len(df)*100:.1f}%)")
            
            # Save metrics
            try:
                import json
                metrics_dir = Path(__file__).parent.parent / "models" / "metrics"
                metrics_dir.mkdir(exist_ok=True)
                metrics = {
                    "silhouette_score": float(silhouette_avg),
                    "cluster_distribution": {str(k): int(v) for k, v in cluster_counts.items()}
                }
                with open(metrics_dir / "clustering_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
                print(f"\n✅ Metrics saved to {metrics_dir / 'clustering_metrics.json'}")
            except Exception as e:
                print(f"\n⚠️  Could not save metrics: {e}")
            
            # Genre distribution per cluster
            print(f"\nGenre Distribution by Cluster:")
            for cluster_id in range(n_clusters):
                cluster_data = track_info_eval[track_info_eval['cluster'] == cluster_id]
                if len(cluster_data) > 0:
                    print(f"\n  Cluster {cluster_id}:")
                    genre_dist = cluster_data['genre'].value_counts()
                    for genre, count in genre_dist.items():
                        print(f"    {genre}: {count} ({count/len(cluster_data)*100:.1f}%)")
                else:
                    print(f"\n  Cluster {cluster_id}: No songs assigned")
            
            # Cluster centroids (average feature values)
            print(f"\nCluster Centroids (Average Feature Values):")
            X_eval = X.reset_index(drop=True) if hasattr(X, 'index') else X
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_features = X_eval[cluster_mask]
                if len(cluster_features) > 0:
                    print(f"\n  Cluster {cluster_id}:")
                    for feature in feature_cols:
                        if feature in cluster_features.columns:
                            avg_value = cluster_features[feature].mean()
                            print(f"    {feature}: {avg_value:.3f}")
                else:
                    print(f"\n  Cluster {cluster_id}: No songs assigned")
                    
        except Exception as eval_error:
            print(f"\n⚠️  Error during evaluation: {eval_error}")
            import traceback
            traceback.print_exc()
            print("\nModel was saved successfully, but evaluation failed.")

        print("\n" + "=" * 60)
        print("✅ Clustering Model Training Completed Successfully!")
        print("=" * 60)
        print(f"\nModel saved to: {MODEL_FILE}")
        print(f"To use this model, load it and call predict_cluster(features)")

    except ConnectionError as e:
        print(f"\n❌ Database Connection Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_clustering_model(TABLE_NAME, N_CLUSTERS)

