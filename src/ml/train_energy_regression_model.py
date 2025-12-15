"""
Model for predicting energy level of songs using regression.
Uses Random Forest Regressor to predict energy based on audio features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
TARGET_COLUMN = "energy"
MODEL_FILE = MODELS_DIR / "energy_regression_model.pkl"
FEATURES_FILE = MODELS_DIR / "energy_regression_features.pkl"

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

# Core audio features used for regression (excluding energy as it's the target)
CORE_FEATURES = [
    "danceability",
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
    """Returns the list of feature columns to use (excluding target)."""
    feature_cols = []

    for col in CORE_FEATURES:
        if col in df.columns:
            feature_cols.append(col)

    # Add engineered features (but remove ones that depend on energy)
    engineered_cols = [
        "Acoustic_Instrumental_Interaction",
        # Note: Excluding Dance_Energy_Interaction, Valence_Energy_Ratio,
        # and Speechiness_Energy_Ratio as they depend on energy (target variable)
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


def train_energy_regression_model(table_name: str, target_col: str):
    """
    Trains a Random Forest Regressor to predict energy levels.
    Saves the model and feature columns for future use.
    """
    print(f"--- Starting Energy Regression Model Training Pipeline ---")

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
    print(f"Target Variable: {target_col}")
    print()

    try:
        # 1. Load the Data from Database
        df = load_data_from_database(table_name)

        print(f"\nData Summary:")
        print(f"  Total rows: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Target ({target_col}) statistics:")
        print(f"    Mean: {df[target_col].mean():.3f}")
        print(f"    Std: {df[target_col].std():.3f}")
        print(f"    Min: {df[target_col].min():.3f}")
        print(f"    Max: {df[target_col].max():.3f}")

        # 2. Create Engineered Features
        print("\nCreating engineered features...")
        df = create_engineered_features(df)

        # 3. Feature Selection (excluding target)
        feature_cols = get_feature_columns(df)
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        print(f"\nFeatures used for regression ({len(feature_cols)}):")
        print(feature_cols)

        # 4. Handle missing values
        print("\nPreprocessing data...")
        X = X.fillna(X.median())
        y = y.fillna(y.median())

        # 5. Train/Test Split (80/20)
        print("\nSplitting data (80% Train, 20% Test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 6. Train Random Forest Regressor
        print("Training Random Forest Regressor...")
        start_time = time.time()

        regressor = RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, max_depth=10
        )
        regressor.fit(X_train, y_train)

        end_time = time.time()
        print(f"   ✅ Model trained in {end_time - start_time:.2f} seconds")

        # 7. Predictions
        print("Generating predictions on test set...")
        y_pred = regressor.predict(X_test)

        # 8. Evaluation
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # 9. Save model components
        print(f"\nSaving model to '{MODEL_FILE}'...")

        with open(MODEL_FILE, "wb") as f:
            pickle.dump(regressor, f)

        with open(FEATURES_FILE, "wb") as f:
            pickle.dump(feature_cols, f)

        print(f"   ✅ Model saved successfully")
        print(f"   ✅ Features saved to '{FEATURES_FILE}'")

        # 10. Display Results
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        print(f"\nRegression Metrics:")
        print(f"  R² Score: {r2:.4f} (Range: -∞ to 1, higher is better)")
        print(f"  Mean Squared Error (MSE): {mse:.6f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")

        # Save metrics
        try:
            import json
            from pathlib import Path

            metrics_dir = Path(__file__).parent.parent.parent / "models" / "metrics"
            metrics_dir.mkdir(exist_ok=True)
            metrics = {
                "r2_score": float(r2),
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
            }
            with open(metrics_dir / "energy_regression_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            print(
                f"\n✅ Metrics saved to {metrics_dir / 'energy_regression_metrics.json'}"
            )
        except Exception as e:
            print(f"\n⚠️  Could not save metrics: {e}")

        # Feature importance
        print(f"\nTop 10 Most Important Features:")
        feature_importance = pd.DataFrame(
            {"feature": feature_cols, "importance": regressor.feature_importances_}
        ).sort_values("importance", ascending=False)

        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        # Sample predictions
        print(f"\nSample Predictions (first 10 test samples):")
        sample_indices = min(10, len(y_test))
        for i in range(sample_indices):
            print(
                f"  Actual: {y_test.iloc[i]:.3f}, Predicted: {y_pred[i]:.3f}, "
                f"Error: {abs(y_test.iloc[i] - y_pred[i]):.3f}"
            )

        print("\n" + "=" * 60)
        print("✅ Energy Regression Model Training Completed Successfully!")
        print("=" * 60)
        print(f"\nModel saved to: {MODEL_FILE}")
        print(f"To use this model, load it and call predict_energy(features)")

    except ConnectionError as e:
        print(f"\n❌ Database Connection Error: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    train_energy_regression_model(TABLE_NAME, TARGET_COLUMN)
