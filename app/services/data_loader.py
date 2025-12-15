"""
Data loader for Spotify dataset.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pickle


class DataLoader:
    """Loads and manages the Spotify dataset."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.dataset_path = Path(__file__).parent.parent.parent / "data" / "fortest" / "Spotify_Dataset_V3.csv"
    
    def load_dataset(self):
        """Load the Spotify dataset from CSV."""
        print(f"Loading dataset from {self.dataset_path}...")
        
        # Read CSV with semicolon separator
        self.df = pd.read_csv(self.dataset_path, sep=';', encoding='utf-8', low_memory=False)
        
        # Clean column names (remove spaces, lowercase)
        self.df.columns = self.df.columns.str.strip().str.lower()
        
        # Normalize column names - handle various formats
        column_mapping = {
            'title': 'track_name',
            'artists': 'artist_name',
            'id': 'track_id',
            'song url': 'spotify_url',
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.rename(columns={old_name: new_name}, inplace=True)
        
        # Extract track_id from URL if needed
        if 'track_id' not in self.df.columns and 'spotify_url' in self.df.columns:
            self.df['track_id'] = self.df['spotify_url'].apply(
                lambda x: x.split('/')[-1] if pd.notna(x) and isinstance(x, str) else None
            )
        
        # Ensure track_id exists
        if 'track_id' not in self.df.columns and 'id' in self.df.columns:
            self.df['track_id'] = self.df['id']
        
        # Handle loudness (normalize to proper range -60 to 0)
        if 'loudness' in self.df.columns:
            # Convert to proper range (-60 to 0)
            def normalize_loudness(x):
                try:
                    val = float(x)
                    if abs(val) > 60:
                        return max(-60, min(0, val / 100))
                    return max(-60, min(0, val))
                except (ValueError, TypeError):
                    return -12.0  # Default value
            self.df['loudness'] = self.df['loudness'].apply(normalize_loudness)
        
        # Remove duplicates based on track_id (do this early for performance)
        self.df = self.df.drop_duplicates(subset=['track_id'], keep='first')
        
        # Fill missing values (only for columns we actually use)
        numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness', 
                       'acousticness', 'instrumentalness', 'valence']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                # Use fillna with method='ffill' then 'bfill' for faster operation
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
        
        # Optimize data types for memory and speed
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('float32')
        
        print(f"✅ Loaded {len(self.df)} songs")
        print(f"Columns: {list(self.df.columns)}")
    
    def get_song_by_id(self, track_id: str) -> Optional[Dict]:
        """Get a song by track_id."""
        if self.df is None:
            return None
        
        if 'track_id' not in self.df.columns:
            return None
        
        song = self.df[self.df['track_id'] == track_id]
        if len(song) == 0:
            return None
        
        return song.iloc[0].to_dict()
    
    def search_songs(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for songs by title or artist."""
        if self.df is None:
            return []
        
        query_lower = query.lower()
        
        # Handle case where columns might not exist
        track_col = 'track_name' if 'track_name' in self.df.columns else 'title'
        artist_col = 'artist_name' if 'artist_name' in self.df.columns else 'artists'
        
        mask = (
            self.df[track_col].astype(str).str.lower().str.contains(query_lower, na=False) |
            self.df[artist_col].astype(str).str.lower().str.contains(query_lower, na=False)
        )
        
        results = self.df[mask].head(limit)
        return results.to_dict('records')
    
    def get_all_songs(self) -> pd.DataFrame:
        """Get all songs."""
        return self.df.copy() if self.df is not None else pd.DataFrame()
    
    def get_songs_by_ids(self, track_ids: List[str]) -> List[Dict]:
        """Get multiple songs by track_ids."""
        if self.df is None:
            return []
        
        results = self.df[self.df['track_id'].isin(track_ids)]
        return results.to_dict('records')

