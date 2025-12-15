import pandas as pd
import numpy as np

# --- Configuration ---
# Set this to the name of your data file
INPUT_FILE = "../data/spotify_data_cleaned_dropped.csv"

# IMPORTANT: Change 'genre' to the actual name of your genre column if different
GENRE_COLUMN = "genre"


def print_genre_counts(input_path: str, genre_col: str):
    """
    Loads the dataset and prints all unique genres along with the count of tracks
    belonging to each genre, sorted by count (most frequent first).
    """
    print(f"--- Attempting to load data from '{input_path}' ---")

    try:
        # 1. Load the dataset
        df = pd.read_csv(input_path)
        print(f"Data loaded successfully. Total tracks: {len(df)}")

        # 2. Check if the specified genre column exists
        if genre_col not in df.columns:
            print(f"\nERROR: Column '{genre_col}' not found in the dataset.")
            print("Please check the capitalization or column name in your CSV file.")
            print(f"Available columns are: {list(df.columns)}")
            return

        # 3. Calculate value counts and sort them
        # Fill NaN genres with a placeholder before counting
        df[genre_col].fillna("UNKNOWN_GENRE", inplace=True)

        # Get the count of each unique genre, sorted descending by count
        genre_counts = df[genre_col].value_counts()

        # 4. Print the results
        print(f"\n--- Genre Counts (Total Unique Genres: {len(genre_counts)}) ---")
        print("Genre \t\t\t | Count")
        print("---------------------------------")

        # Print the counts
        for genre, count in genre_counts.items():
            if genre == "UNKNOWN_GENRE":
                print(f"{genre} \t\t\t | {count} (from missing values)")
            else:
                # Use string formatting to align the output
                print(f"{genre:<20} | {count}")

        print("\nSuccessfully listed all unique genres and their counts.")

    except FileNotFoundError:
        print(f"\nERROR: The file '{input_path}' was not found.")
        print("Please ensure your data file is in the 'data/' folder.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    print_genre_counts(INPUT_FILE, GENRE_COLUMN)
