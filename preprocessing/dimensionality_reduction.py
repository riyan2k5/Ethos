import pandas as pd
import numpy as np

# --- Configuration ---
INPUT_FILE = "../data/spotify_data_engineered_features.csv"
OUTPUT_FILE = "../data/spotify_data_reduced.csv"

# Features to keep (based on importance scores)
FEATURES_TO_KEEP = [
    "year",
    "duration_ms",
    "acousticness",
    "instrumentalness",
    "loudness",
    "danceability",
    "energy",
    "valence",
    "liveness",
    "Valence_Energy_Ratio",
    "Dance_Energy_Interaction",
    "tempo",
    "Tempo_vs_120",
    "key",
]

# Non-numeric columns to preserve (metadata)
METADATA_COLUMNS = ["artist_name", "track_name", "track_id", "genre", "popularity"]


def reduce_dimensionality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the most important features based on feature importance scores.
    """
    print("=" * 60)
    print("--- Starting Dimensionality Reduction ---")
    print("=" * 60)
    initial_shape = df.shape
    print(f"Initial data shape: {initial_shape}")

    # Determine which columns to keep
    columns_to_keep = []

    # Add metadata columns if they exist
    for col in METADATA_COLUMNS:
        if col in df.columns:
            columns_to_keep.append(col)

    # Add important features if they exist
    for col in FEATURES_TO_KEEP:
        if col in df.columns:
            columns_to_keep.append(col)
        else:
            print(f"Warning: Feature '{col}' not found in dataset")

    # Select only the columns we want to keep
    df_reduced = df[columns_to_keep].copy()

    final_shape = df_reduced.shape
    print("\n" + "=" * 60)
    print("--- Dimensionality Reduction Complete ---")
    print("=" * 60)
    print(f"Final data shape: {final_shape}")
    print(
        f"Features reduced: {initial_shape[1]} -> {final_shape[1]} "
        f"({(1 - final_shape[1]/initial_shape[1])*100:.1f}% reduction)"
    )
    print(f"Rows: {initial_shape[0]} -> {final_shape[0]}")
    print(
        f"\nKept {len(FEATURES_TO_KEEP)} important features + {len(METADATA_COLUMNS)} metadata columns"
    )

    return df_reduced


def main():
    """
    Main function to execute the dimensionality reduction process.
    """
    try:
        # 1. Load the dataset
        print(f"Loading data from '{INPUT_FILE}'...")
        df = pd.read_csv(INPUT_FILE)
        print(f"Data loaded successfully. Shape: {df.shape}")

        # Drop the "Unnamed: 0" column if it exists (it's just a row index)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
            print("Dropped 'Unnamed: 0' column (row index)")

        # 2. Reduce dimensionality
        df_reduced = reduce_dimensionality(df)

        # 3. Save the reduced data
        df_reduced.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Successfully saved reduced data to '{OUTPUT_FILE}'")
        print(f"\nFinal columns: {list(df_reduced.columns)}")

    except FileNotFoundError:
        print(f"\nERROR: The file '{INPUT_FILE}' was not found.")
        print("Please run feature_engineering.py first to create the input file.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
