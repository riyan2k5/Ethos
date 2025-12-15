import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import sys
import time
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.db_connection import get_db_connection, test_connection
from api.db_config import DatabaseConfig
from sqlalchemy import create_engine

# --- Configuration ---
SCRIPT_DIR = Path(__file__).parent.resolve()
TABLE_NAME = "spotify_songs"  # Database table name
TARGET_COLUMN = "genre"

# Check if DATABASE_URL is set (required for Neon)
if not os.getenv("DATABASE_URL") and not os.getenv("DB_CONNECTION_STRING"):
    print("=" * 60)
    print("⚠️  WARNING: DATABASE_URL not set!")
    print("=" * 60)
    print("Please set your database connection string:")
    print("  export DATABASE_URL='postgresql://user:password@host:port/database?sslmode=require'")
    print()
    print("Example format:")
    print("  export DATABASE_URL='postgresql://username:password@hostname:5432/database_name?sslmode=require'")
    print("=" * 60)
    print()

# Core audio features used for classification
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
    """
    Creates engineered features from available audio features.
    Works with reduced data that only contains core features.
    """
    df = df.copy()

    # 1. Danceability-Energy Interaction Term
    # Represents tracks that are both high-energy AND suitable for dancing.
    if "danceability" in df.columns and "energy" in df.columns:
        df["Dance_Energy_Interaction"] = df["danceability"] * df["energy"]

    # 2. Positivity-Energy Ratio
    # Tracks that are high-valence (happy) but low-energy (chill) vs high-energy (exciting).
    if "valence" in df.columns and "energy" in df.columns:
        df["Valence_Energy_Ratio"] = df["valence"] / (
            df["energy"] + 1e-6
        )  # Add small epsilon to prevent division by zero

    # 3. Acousticness-Instrumentalness Interaction
    # May help distinguish acoustic instrumental tracks from electronic tracks.
    if "acousticness" in df.columns and "instrumentalness" in df.columns:
        df["Acoustic_Instrumental_Interaction"] = df["acousticness"] * df["instrumentalness"]

    # 4. Speechiness-Energy Ratio
    # Tracks with high speechiness (spoken word) relative to energy.
    if "speechiness" in df.columns and "energy" in df.columns:
        df["Speechiness_Energy_Ratio"] = df["speechiness"] / (
            df["energy"] + 1e-6
        )  # Add small epsilon to prevent division by zero

    # Handle any infinite or NaN values in engineered features
    engineered_cols = [
        "Dance_Energy_Interaction",
        "Valence_Energy_Ratio",
        "Acoustic_Instrumental_Interaction",
        "Speechiness_Energy_Ratio",
    ]

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in engineered_cols:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Returns the list of feature columns to use for training.
    Includes core features and any engineered features that exist.
    """
    feature_cols = []

    # Add core features
    for col in CORE_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    # Add engineered features if they exist
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
    """
    Load data from PostgreSQL database using SQLAlchemy.
    All credentials are read from environment variables - NEVER hardcoded.
    
    Args:
        table_name: Name of the table to load from
    
    Returns:
        pd.DataFrame: DataFrame containing the data
    """
    print(f"Loading data from database table '{table_name}'...")
    
    # Test connection first
    if not test_connection():
        raise ConnectionError("Database connection failed. Please check your configuration.")
    
    # Use SQLAlchemy with proper configuration for Neon
    print("   Executing query...")
    try:
        config = DatabaseConfig()
        conn_string = config.get_connection_string()
        
        # Create engine with pool_pre_ping and pool_recycle for Neon compatibility
        # Disable connection pooling to avoid hanging issues
        engine = create_engine(
            conn_string,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=300,    # Recycle connections after 5 minutes
            pool_size=1,         # Minimal pool size
            max_overflow=0,      # No overflow connections
            connect_args={"connect_timeout": 10}  # 10 second timeout
        )
        
        # Query to get all data (excluding id and created_at columns)
        query = f"""
        SELECT 
            artist_name, track_name, track_id, genre, popularity,
            danceability, energy, loudness, speechiness, acousticness,
            instrumentalness, valence
        FROM {table_name};
        """
        
        print("   Fetching data...")
        # Use pandas read_sql with SQLAlchemy engine (no warning)
        df = pd.read_sql(query, con=engine)
        engine.dispose()  # Close all connections
        
        print(f"   ✅ Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"   ⚠️  SQLAlchemy method failed: {e}")
        print("   Trying fallback method...")
        # Fallback: use direct psycopg2 connection
        try:
            import psycopg2
            config = DatabaseConfig()
            conn_string = config.get_connection_string()
            
            conn = psycopg2.connect(conn_string)
            
            query = f"""
            SELECT 
                artist_name, track_name, track_id, genre, popularity,
                danceability, energy, loudness, speechiness, acousticness,
                instrumentalness, valence
            FROM {table_name};
            """
            
            print("   Fetching data...")
            # Use pandas read_sql_query to avoid warning
            df = pd.read_sql_query(query, con=conn)
            conn.close()
            
            print(f"   ✅ Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e2:
            raise ConnectionError(f"Failed to load data from database: {e2}")


def train_and_evaluate_model(table_name: str, target_col: str):
    """
    Loads the dataset from database, creates engineered features, trains a Random Forest 
    Classifier, and prints detailed performance metrics by genre.
    All credentials are read from environment variables - NEVER hardcoded.
    
    Args:
        table_name: Name of the database table to load data from
        target_col: Name of the target column (genre)
    """
    print(f"--- Starting Model Training Pipeline ---")
    
    # Show database configuration
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
        
        # Show data summary
        print(f"\nData Summary:")
        print(f"  Total rows: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Genres: {df[target_col].nunique()}")
        print(f"  Genre distribution:")
        genre_counts = df[target_col].value_counts()
        for genre, count in genre_counts.items():
            print(f"    {genre}: {count}")

        # 2. Create Engineered Features
        print("\nCreating engineered features...")
        df = create_engineered_features(df)

        # 3. Feature Selection
        feature_cols = get_feature_columns(df)

        X = df[feature_cols]
        y = df[target_col]

        print(f"\nFeatures used for training ({len(feature_cols)}):")
        print(feature_cols)

        # 4. Train/Test Split (80/20)
        print("\nSplitting data (80% Train, 20% Test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        # Note: stratify=y ensures each genre is represented proportionally in train and test sets

        # 5. Initialize and Train the Model
        print("Training Random Forest Classifier... (This may take a moment)")
        start_time = time.time()

        # Using 100 trees (n_estimators=100) is a standard starting point
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")

        # 6. Predictions
        print("Generating predictions on test set...")
        y_pred = clf.predict(X_test)

        # 7. Evaluation
        overall_acc = accuracy_score(y_test, y_pred)

        print("\n" + "=" * 60)
        print(f"✅ MODEL RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy: {overall_acc:.2%}")
        print("-" * 60)
        print("\nDetailed Report by Genre (Precision, Recall, F1-Score):\n")

        # This generates the table you asked for
        print(classification_report(y_test, y_pred))

        print("=" * 60)

    except ConnectionError as e:
        print(f"\n❌ Database Connection Error: {e}")
        print("\nPlease ensure:")
        print("  1. DATABASE_URL environment variable is set")
        print("  2. Database is accessible")
        print("  3. Table exists in the database")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    train_and_evaluate_model(TABLE_NAME, TARGET_COLUMN)
