import pandas as pd
import numpy as np

# --- Configuration ---
INPUT_FILE = "../data/spotify_data_mapped_genres.csv"
OUTPUT_FILE = "../data/spotify_data_engineered_features.csv"


def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates 4 new meaningful features from existing audio features.
    """
    print("--- Starting Feature Engineering ---")
    initial_features = len(df.columns)
    print(f"Initial number of features: {initial_features}")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # 1. Danceability-Energy Interaction Term (New Feature 1)
    # Represents tracks that are both high-energy AND suitable for dancing.
    df["Dance_Energy_Interaction"] = df["danceability"] * df["energy"]

    # 2. Loudness-Duration Ratio (New Feature 2)
    # May indicate compression/mastering style related to track length.
    df["Loudness_Duration_Ratio"] = df["loudness"] / (df["duration_ms"] / 1000)

    # 3. Tempo Difference from 120 BPM (New Feature 3)
    # 120 BPM is a common electronic/dance tempo. Closeness to it may be important.
    df["Tempo_vs_120"] = abs(df["tempo"] - 120)

    # 4. Positivity-Energy Ratio (New Feature 4)
    # Tracks that are high-valence (happy) but low-energy (chill) vs high-energy (exciting).
    df["Valence_Energy_Ratio"] = df["valence"] / (
        df["energy"] + 1e-6
    )  # Add small epsilon to prevent division by zero

    # Handle any infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values in new numerical columns with median
    new_cols = [
        "Dance_Energy_Interaction",
        "Loudness_Duration_Ratio",
        "Tempo_vs_120",
        "Valence_Energy_Ratio",
    ]
    for col in new_cols:
        if col in df.columns and df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    final_features = len(df.columns)
    new_features = final_features - initial_features
    print(f"Final number of features: {final_features}")
    print(f"New features created: {new_features}")
    print("--- Feature Engineering Complete ---\n")

    return df


def main():
    """
    Main function to execute the feature engineering process.
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

        # 2. Create new features
        df_engineered = create_new_features(df)

        # 3. Save the engineered data
        df_engineered.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Successfully saved engineered data to '{OUTPUT_FILE}'")
        print(f"Final Data Shape: {df_engineered.shape}")

        # 4. Print summary of new features
        print("\n--- Summary of New Features ---")
        original_cols = set(df.columns)
        new_cols = [col for col in df_engineered.columns if col not in original_cols]
        print(f"New features created: {len(new_cols)}")
        for i, col in enumerate(new_cols, 1):
            print(f"  {i}. {col}")

    except FileNotFoundError:
        print(f"\nERROR: The file '{INPUT_FILE}' was not found.")
        print("Please ensure the file is in the same directory as this script.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
