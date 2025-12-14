import pandas as pd
import numpy as np
import io 

# --- Configuration ---
INPUT_FILE = '../data/spotify_data_cleaned_dropped.csv' 
OUTPUT_FILE = '../data/spotify_data_mapped_genres.csv' # Changed output name to reflect finality
GENRE_COLUMN = 'genre' 

# Define the comprehensive mapping from specific sub-genres to the 9 target categories.
GENRE_MAP = {
    # POP
    'pop': 'POP', 'power-pop': 'POP', 'pop-film': 'POP', 'cantopop': 'POP', 
    'k-pop': 'POP', 'indie-pop': 'POP', 
    
    # ROCK (Including Metal/Punk subgenres)
    'alt-rock': 'ROCK', 'rock-n-roll': 'ROCK', 'hard-rock': 'ROCK', 'psych-rock': 'ROCK', 
    'rock': 'ROCK', 'punk-rock': 'ROCK', 'punk': 'ROCK', 'metal': 'ROCK', 
    'heavy-metal': 'ROCK', 'death-metal': 'ROCK', 'black-metal': 'ROCK', 
    'metalcore': 'ROCK', 'goth': 'ROCK', 'emo': 'ROCK', 'hardcore': 'ROCK', 
    'industrial': 'ROCK', 'garage': 'ROCK', 
    
    # HIP-HOP
    'hip-hop': 'HIP-HOP', 
    
    # DANCE (Including Samba, Salsa, party, etc.)
    'dance': 'DANCE', 'samba': 'DANCE', 'salsa': 'DANCE', 'forro': 'DANCE', 
    'tango': 'DANCE', 'disco': 'DANCE', 'dancehall': 'DANCE', 'ska': 'DANCE', 
    'groove': 'DANCE', 'party': 'DANCE', 
    
    # CLASSICAL
    'classical': 'CLASSICAL', 'opera': 'CLASSICAL', 'piano': 'CLASSICAL', 
    
    # JAZZ (Including blues)
    'jazz': 'JAZZ', 'blues': 'JAZZ', 
    
    # COUNTRY
    'country': 'COUNTRY', 'sertanejo': 'COUNTRY', 'folk': 'COUNTRY', 
    'singer-songwriter': 'COUNTRY', 'songwriter': 'COUNTRY', 
    
    # DESI
    'indian': 'DESI', 
    
    # ELECTRONIC (Including various EDM/House/Techno subgenres and ambient/mood genres)
    'ambient': 'ELECTRONIC', 'deep-house': 'ELECTRONIC', 'trance': 'ELECTRONIC', 
    'electronic': 'ELECTRONIC', 'techno': 'ELECTRONIC', 'house': 'ELECTRONIC', 
    'chicago-house': 'ELECTRONIC', 'detroit-techno': 'ELECTRONIC', 
    'minimal-techno': 'ELECTRONIC', 'club': 'ELECTRONIC', 'drum-and-bass': 'ELECTRONIC', 
    'breakbeat': 'ELECTRONIC', 'hardstyle': 'ELECTRONIC', 'electro': 'ELECTRONIC', 
    'trip-hop': 'ELECTRONIC', 'dubstep': 'ELECTRONIC', 'dub': 'ELECTRONIC', 
    'edm': 'ELECTRONIC', 'progressive-house': 'ELECTRONIC', 
    'new-age': 'ELECTRONIC', 'chill': 'ELECTRONIC', 'sleep': 'ELECTRONIC'
}

# Genres to be dropped entirely (Non-core music, mood, language, or user-excluded)
DROPPED_GENRES = [
    'gospel', 'acoustic', 'comedy', 'spanish', 'french', 'german', 
    'swedish', 'show-tunes', 'soul', 'romance', 'sad', 'guitar',
    
    # New exclusions requested by user:
    'funk', 'grindcore', 'afrobeat' 
]


def map_and_count_genres(input_path: str, output_path: str, genre_col: str):
    """
    Loads data, maps sub-genres to main categories, drops unused genres,
    and prints the final count table.
    """
    print(f"--- Starting Final Genre Mapping Process ---")

    try:
        df = pd.read_csv(input_path)
        
        if genre_col not in df.columns:
            print(f"\nERROR: Column '{genre_col}' not found. Check column name.")
            return

        # 1. Drop rows corresponding to the excluded genres
        initial_rows = len(df)
        df = df[~df[genre_col].isin(DROPPED_GENRES)].copy()
        rows_dropped = initial_rows - len(df)
        print(f"\nDropped {rows_dropped} rows belonging to all excluded genres.")

        # 2. Apply the genre mapping
        df[genre_col] = df[genre_col].replace(GENRE_MAP, regex=False)
        print("Genre mapping applied successfully.")

        # 3. Handle any remaining UNMAPPED or NaN values
        df[genre_col].fillna('UNKNOWN_GENRE', inplace=True)
        
        # 4. Calculate the final value counts
        final_counts = df[genre_col].value_counts().reset_index()
        final_counts.columns = ['Category', 'Count']
        
        # 5. Print the final results table
        print("\n--- Final Aggregated Genre Counts ---")
        print(final_counts.to_markdown(index=False))

        # 6. Save the new DataFrame
        df.to_csv(output_path, index=False)
        print(f"\n✅ Successfully saved FINAL data with mapped genres to '{output_path}'")
        print(f"Final Data Shape: {df.shape}")


    except FileNotFoundError:
        print(f"\nERROR: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    map_and_count_genres(INPUT_FILE, OUTPUT_FILE, GENRE_COLUMN)