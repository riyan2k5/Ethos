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
        self.dataset_path = (
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "fortest"
            / "Spotify_Dataset_V3.csv"
        )
        self.custom_dataset: Optional[pd.DataFrame] = None  # For user-uploaded datasets
        self.use_custom: bool = False  # Flag to use custom dataset

    def load_dataset(self):
        """Load the Spotify dataset from CSV."""
        # If using custom dataset, don't reload default
        if self.use_custom and self.custom_dataset is not None:
            self.df = self.custom_dataset
            return

        if self.df is not None:
            return

        print(f"Loading dataset from {self.dataset_path}...")

        # Read CSV with semicolon separator
        self.df = pd.read_csv(
            self.dataset_path, sep=";", encoding="utf-8", low_memory=False
        )

        # Clean column names (remove spaces, lowercase)
        self.df.columns = self.df.columns.str.strip().str.lower()

        # Normalize column names - handle various formats
        column_mapping = {
            "title": "track_name",
            "artists": "artist_name",
            "id": "track_id",
            "song url": "spotify_url",
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.rename(columns={old_name: new_name}, inplace=True)

        # Extract track_id from URL if needed
        if "track_id" not in self.df.columns and "spotify_url" in self.df.columns:
            self.df["track_id"] = self.df["spotify_url"].apply(
                lambda x: (
                    x.split("/")[-1] if pd.notna(x) and isinstance(x, str) else None
                )
            )

        # Ensure track_id exists
        if "track_id" not in self.df.columns and "id" in self.df.columns:
            self.df["track_id"] = self.df["id"]

        # Handle loudness (normalize to proper range -60 to 0)
        if "loudness" in self.df.columns:
            # Convert to proper range (-60 to 0)
            def normalize_loudness(x):
                try:
                    val = float(x)
                    if abs(val) > 60:
                        return max(-60, min(0, val / 100))
                    return max(-60, min(0, val))
                except (ValueError, TypeError):
                    return -12.0  # Default value

            self.df["loudness"] = self.df["loudness"].apply(normalize_loudness)

        # Remove duplicates based on track_id (do this early for performance)
        self.df = self.df.drop_duplicates(subset=["track_id"], keep="first")

        # Fill missing values (only for columns we actually use)
        numeric_cols = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "valence",
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                # Use fillna with method='ffill' then 'bfill' for faster operation
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)

        # Optimize data types for memory and speed
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype("float32")

        print(f"✅ Loaded {len(self.df)} songs")
        print(f"Columns: {list(self.df.columns)}")

    def get_song_by_id(self, track_id: str) -> Optional[Dict]:
        """Get a song by track_id."""
        if self.df is None:
            return None

        if "track_id" not in self.df.columns:
            return None

        song = self.df[self.df["track_id"] == track_id]
        if len(song) == 0:
            return None

        return song.iloc[0].to_dict()

    def search_songs(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for songs by title or artist."""
        if self.df is None:
            return []

        query_lower = query.lower()

        # Handle case where columns might not exist
        track_col = "track_name" if "track_name" in self.df.columns else "title"
        artist_col = "artist_name" if "artist_name" in self.df.columns else "artists"

        mask = self.df[track_col].astype(str).str.lower().str.contains(
            query_lower, na=False
        ) | self.df[artist_col].astype(str).str.lower().str.contains(
            query_lower, na=False
        )

        results = self.df[mask].head(limit)
        return results.to_dict("records")

    def get_all_songs(self) -> pd.DataFrame:
        """Get all songs."""
        return self.df.copy() if self.df is not None else pd.DataFrame()

    def get_songs_by_ids(self, track_ids: List[str]) -> List[Dict]:
        """Get multiple songs by track_ids."""
        if self.df is None:
            return []

        results = self.df[self.df["track_id"].isin(track_ids)]
        return results.to_dict("records")

    def load_custom_dataset(self, file_content: bytes, filename: str) -> Dict[str, any]:
        """Load a custom dataset from uploaded file."""
        try:
            import io

            # Try to detect encoding, fallback to utf-8
            encoding = "utf-8"
            try:
                import chardet

                detected = chardet.detect(file_content)
                if detected and detected.get("encoding"):
                    encoding = detected.get("encoding", "utf-8")
            except ImportError:
                # chardet not available, try common encodings
                pass

            # Try to read as CSV with different delimiters
            df = None
            for delimiter in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(
                        io.BytesIO(file_content),
                        delimiter=delimiter,
                        encoding=encoding,
                        low_memory=False,
                    )
                    print(f"Successfully read CSV with '{delimiter}' delimiter")
                    break
                except:
                    continue

            if df is None:
                raise ValueError(
                    "Could not parse CSV file. Please ensure it's a valid CSV."
                )

            # Normalize column names (same as default dataset)
            df.columns = [
                col.lower()
                .replace(" ", "_")
                .replace(".", "")
                .replace("#_of", "num_of")
                .replace("(ind)", "ind")
                for col in df.columns
            ]

            # Map common column names
            column_mapping = {
                "title": "track_name",
                "artists": "artist_name",
                "id": "track_id",
                "song url": "spotify_url",
            }
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)

            # Ensure track_id exists
            if "track_id" not in df.columns:
                # Try to find similar column
                possible_ids = [
                    col
                    for col in df.columns
                    if "id" in col.lower() or "track" in col.lower()
                ]
                if possible_ids:
                    df["track_id"] = df[possible_ids[0]].astype(str)
                else:
                    # Generate IDs if none exist
                    df["track_id"] = [f"custom_{i}" for i in range(len(df))]

            df["track_id"] = df["track_id"].astype(str)
            df.drop_duplicates(subset=["track_id"], inplace=True)

            # Fill missing values
            numerical_cols = [
                "danceability",
                "energy",
                "loudness",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "valence",
            ]
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col].fillna(median_val, inplace=True)
                    else:
                        df[col].fillna(0.0, inplace=True)

            # Store custom dataset
            self.custom_dataset = df
            self.use_custom = True
            self.df = df

            return {
                "success": True,
                "message": f"Successfully loaded {len(df)} songs from {filename}",
                "count": len(df),
                "columns": list(df.columns),
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"success": False, "message": f"Error loading dataset: {str(e)}"}

    def reset_to_default(self):
        """Reset to default dataset."""
        self.use_custom = False
        self.custom_dataset = None
        self.df = None
        self.load_dataset()
