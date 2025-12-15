"""
Utility script to list tracks from the database.
Helps you find track_ids to use with prediction scripts.
"""
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.db_connection import test_connection
from src.api.db_config import DatabaseConfig
from sqlalchemy import create_engine

# --- Configuration ---
TABLE_NAME = "spotify_songs"


def list_tracks(limit: int = 20, genre: str = None):
    """
    List tracks from the database.
    
    Args:
        limit: Number of tracks to show
        genre: Optional genre filter
    """
    if not test_connection():
        print("❌ Database connection failed. Please check your configuration.")
        return
    
    try:
        config = DatabaseConfig()
        conn_string = config.get_connection_string()
        engine = create_engine(conn_string)
        
        query = f"""
        SELECT track_id, track_name, artist_name, genre, popularity
        FROM {TABLE_NAME}
        """
        
        if genre:
            query += f" WHERE genre = '{genre}'"
        
        query += f" ORDER BY popularity DESC LIMIT {limit};"
        
        df = pd.read_sql(query, con=engine)
        engine.dispose()
        
        if len(df) == 0:
            print("No tracks found.")
            return
        
        print(f"\n📊 Found {len(df)} tracks:")
        print("=" * 100)
        for idx, row in df.iterrows():
            print(f"{idx+1}. Track ID: {row['track_id']}")
            print(f"   '{row['track_name']}' by {row['artist_name']}")
            print(f"   Genre: {row['genre']}, Popularity: {row['popularity']}")
            print()
        
        print("=" * 100)
        print("\nTo find similar songs, use:")
        print(f"  python ml/predict_similar_songs.py --track-id \"{df.iloc[0]['track_id']}\" --n-similar 10")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="List tracks from database")
    parser.add_argument("--limit", type=int, default=20, help="Number of tracks to show")
    parser.add_argument("--genre", type=str, help="Filter by genre")
    
    args = parser.parse_args()
    
    list_tracks(args.limit, args.genre)

