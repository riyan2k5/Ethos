import pandas as pd
import numpy as np

# Define the input and output file names
INPUT_FILE = "../data/spotify_data.csv"
OUTPUT_FILE = "../data/spotify_data_cleaned_dropped.csv"  # Changed output name to reflect the drop operation


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads a Spotify dataset, handles missing numerical values via imputation,
    and drops rows with missing 'artist_name' or 'track_name'.
    """
    initial_rows = len(df)
    print(f"--- Initial Data Shape: {df.shape} ---")

    # Identify the number of missing values per column
    missing_before = df.isnull().sum()
    print("\n--- Missing Values Before Handling ---")
    print(missing_before[missing_before > 0])

    # =======================================================
    # 1. Handling Text/Object Columns (Dropping Rows)
    # =======================================================

    text_cols_to_drop = ["artist_name", "track_name"]

    # Drop rows where any of the specified text columns are null
    df.dropna(subset=text_cols_to_drop, inplace=True)

    rows_dropped = initial_rows - len(df)
    print(
        f"\nDropped {rows_dropped} rows due to missing values in 'artist_name' or 'track_name'."
    )
    print(f"Current Data Shape after dropping: {df.shape}")

    # =======================================================
    # 2. Handling Numerical Audio Features (Imputation)
    # =======================================================

    # Features appropriate for Mean Imputation
    mean_impute_cols = [
        "Popularity",
        "Year",
        "Danceability",
        "Energy",
        "Loudness",
        "Speechiness",
        "Acousticness",
        "Instrumentalness",
        "Liveness",
        "Valence",
        "Tempo",
        "Duration_ms",
    ]

    # Features appropriate for Mode Imputation
    mode_impute_cols = ["Key", "Mode", "Time_signature"]

    # Apply Imputation

    # Note: We check df[col].isnull().any() to ensure we only process columns that actually have NaNs

    # Apply Mean Imputation
    for col in mean_impute_cols:
        if col in df.columns and df[col].isnull().any():
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)
            print(
                f"Filled missing numerical values in '{col}' with Mean: {mean_value:.2f}"
            )

    # Apply Mode Imputation
    for col in mode_impute_cols:
        if col in df.columns and df[col].isnull().any():
            mode_value = df[col].mode().iloc[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Filled missing numerical values in '{col}' with Mode: {mode_value}")

    # =======================================================
    # 3. Verification
    # =======================================================

    missing_after = df.isnull().sum()
    remaining_missing = missing_after[missing_after > 0]

    if remaining_missing.empty:
        print("\n✅ Successfully handled ALL missing values.")
    else:
        print("\n⚠️ Note: The following columns still contain missing values:")
        print(remaining_missing)

    print(f"\n--- Final Data Shape: {df.shape} ---")

    return df


def main():
    """
    Main function to execute the missing value handling process.
    """
    try:
        # 1. Load the dataset
        print(f"Attempting to load data from '{INPUT_FILE}'...")
        df = pd.read_csv(INPUT_FILE)
        print("Data loaded successfully.")

        # 2. Handle missing values
        df_cleaned = handle_missing_values(df.copy())  # Use a copy for safe operations

        # 3. Save the cleaned data to a new CSV
        df_cleaned.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved cleaned data to '{OUTPUT_FILE}'")

    except FileNotFoundError:
        print(f"\nERROR: The file '{INPUT_FILE}' was not found.")
        print("Please ensure 'spotify_data.csv' is in the 'data/' folder.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
