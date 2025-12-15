"""
Recommendation service for generating personalized recommendations.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import random
from src.app.services.data_loader import DataLoader
from src.app.services.ml_service import MLService
from src.app.services.spotify_service import SpotifyService


class RecommendationService:
    """Service for generating recommendations."""

    def __init__(
        self,
        data_loader: DataLoader,
        ml_service: MLService,
        spotify_service: SpotifyService,
    ):
        self.data_loader = data_loader
        self.ml_service = ml_service
        self.spotify_service = spotify_service
        self.user_interactions: Dict[str, List[str]] = defaultdict(list)

    def track_interaction(self, user_id: str, track_id: str, action: str = "click"):
        """Track user interaction for personalization."""
        if action == "click" or action == "like":
            self.user_interactions[user_id].append(track_id)
            # Keep only last 50 interactions
            self.user_interactions[user_id] = self.user_interactions[user_id][-50:]

    def get_recommendations(self, user_id: str, limit: int = 20) -> List[Dict]:
        """Get personalized recommendations."""
        df = self.data_loader.get_all_songs()
        if df.empty:
            return []

        # If user has interactions, use collaborative filtering
        if (
            user_id in self.user_interactions
            and len(self.user_interactions[user_id]) > 0
        ):
            # Get songs similar to user's liked songs
            liked_tracks = self.user_interactions[user_id][-10:]  # Last 10 interactions
            recommendations = []
            seen_track_ids = set(liked_tracks)

            for track_id in liked_tracks:
                song = self.data_loader.get_song_by_id(track_id)
                if song:
                    similar = self.ml_service.find_similar_songs(song, limit=5)
                    for sim_song in similar:
                        sim_track_id = sim_song.get("track_id")
                        if sim_track_id and sim_track_id not in seen_track_ids:
                            # Get full song data
                            full_song = self.data_loader.get_song_by_id(sim_track_id)
                            if full_song:
                                full_song["similarity"] = sim_song.get("similarity", 0)
                                recommendations.append(full_song)
                                seen_track_ids.add(sim_track_id)

            # Sort by similarity
            recommendations.sort(key=lambda x: x.get("similarity", 0), reverse=True)

            if len(recommendations) >= limit:
                return recommendations[:limit]

        # Fallback to popularity-based recommendations
        # Use rank or points if available, otherwise random
        if "rank" in df.columns:
            df_sorted = df.sort_values("rank", ascending=True)
        elif "points (total)" in df.columns:
            df_sorted = df.sort_values("points (total)", ascending=False)
        else:
            # Random sample with some variety
            df_sorted = df.sample(frac=1, random_state=42).reset_index(drop=True)

        recommendations = df_sorted.head(limit).to_dict("records")
        return recommendations

    def get_clustered_songs(self, limit: int = 15) -> Dict[str, List[Dict]]:
        """Get songs grouped by clusters."""
        df = self.data_loader.get_all_songs()
        if df.empty:
            return {}

        # Try to use clustering model if available
        clusters: Dict[int, List[Dict]] = defaultdict(list)

        # Sample for performance (reduced sample size)
        sample_size = min(500, len(df))
        sample_df = df.sample(n=sample_size, random_state=42)

        for _, row in sample_df.iterrows():
            song = row.to_dict()

            # Try to predict cluster using ML service
            try:
                # Use energy and danceability as simple clustering
                energy = float(song.get("energy", 0.5))
                danceability = float(song.get("danceability", 0.5))

                # Simple clustering: high energy + high danceability = cluster 0, else cluster 1
                if energy > 0.6 and danceability > 0.6:
                    cluster_id = 0
                else:
                    cluster_id = 1

                clusters[cluster_id].append(song)

                # Stop if we have enough songs in both clusters
                if all(len(clusters[i]) >= limit for i in [0, 1]):
                    break
            except Exception as e:
                # Fallback to hash-based clustering
                cluster_id = hash(str(song.get("track_id", ""))) % 2
                clusters[cluster_id].append(song)

        # Convert to named clusters
        result = {}
        cluster_names = ["High Energy", "Chill Vibes"]
        for i in [0, 1]:
            if i in clusters and len(clusters[i]) > 0:
                cluster_name = (
                    cluster_names[i] if i < len(cluster_names) else f"Cluster {i}"
                )
                result[cluster_name] = clusters[i][:limit]

        return result

    def get_similar_songs(self, track_id: str, limit: int = 10) -> List[Dict]:
        """Get similar songs for a given track."""
        try:
            song = self.data_loader.get_song_by_id(track_id)
            if not song:
                return []

            # Get all songs from current dataset
            df = self.data_loader.get_all_songs()
            if df.empty or "track_id" not in df.columns:
                return []

            # Try using ML model first (if track exists in model)
            available_track_ids = set(df["track_id"].tolist())
            similar = self.ml_service.find_similar_songs(
                song, limit=limit, available_track_ids=available_track_ids
            )

            # If ML model didn't find enough, compute similarity directly from dataset
            if len(similar) < limit:
                similar_from_dataset = self._find_similar_in_dataset(
                    song, df, limit, track_id
                )
                # Combine results, avoiding duplicates
                seen_ids = {s.get("track_id") for s in similar}
                for sim in similar_from_dataset:
                    if sim.get("track_id") not in seen_ids:
                        similar.append(sim)
                        seen_ids.add(sim.get("track_id"))
                    if len(similar) >= limit:
                        break

            # Ensure we have full song info from our dataset and exclude the original track
            results = []
            seen_track_ids = {
                track_id
            }  # Track IDs we've already added (including the current song)

            for sim_song in similar:
                sim_track_id = sim_song.get("track_id")
                if not sim_track_id or sim_track_id == track_id:
                    continue  # Skip if no track_id or it's the same song

                # Skip duplicates
                if sim_track_id in seen_track_ids:
                    continue

                # Get full song data from our dataset
                full_song = self.data_loader.get_song_by_id(sim_track_id)
                if full_song:
                    seen_track_ids.add(sim_track_id)
                    results.append(
                        {
                            "track_id": sim_track_id,
                            "track_name": full_song.get(
                                "track_name",
                                full_song.get(
                                    "title", sim_song.get("track_name", "Unknown")
                                ),
                            ),
                            "artist_name": full_song.get(
                                "artist_name",
                                full_song.get(
                                    "artists", sim_song.get("artist_name", "Unknown")
                                ),
                            ),
                            "similarity": sim_song.get("similarity", 0.0),
                        }
                    )

                    # Stop when we have enough unique results (but don't require exactly 'limit')
                    if len(results) >= limit:
                        break

            # Return whatever we have (even if less than limit) - no duplicates, no padding
            return results
        except Exception as e:
            print(f"Error in get_similar_songs: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _find_similar_in_dataset(
        self, song: Dict, df, limit: int, exclude_track_id: str
    ) -> List[Dict]:
        """Find similar songs by computing cosine similarity directly in the dataset."""
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        try:
            # Sample dataset for faster computation (use 2000 songs max)
            if len(df) > 2000:
                df_sample = df.sample(n=2000, random_state=42)
            else:
                df_sample = df.copy()

            # Get feature columns
            feature_cols = [
                "danceability",
                "energy",
                "loudness",
                "speechiness",
                "acousticness",
                "instrumentalness",
                "valence",
            ]

            # Prepare query song features
            query_features = []
            for col in feature_cols:
                if col in song:
                    try:
                        val = float(song[col])
                        # Normalize loudness
                        if col == "loudness":
                            val = (val + 60) / 60  # Normalize to 0-1
                        query_features.append(val)
                    except (ValueError, TypeError):
                        query_features.append(0.0)
                else:
                    query_features.append(0.0)

            # Get features for sampled songs
            df_features = df_sample[feature_cols].fillna(0).values

            # Normalize loudness for all songs
            loudness_idx = feature_cols.index("loudness")
            df_features[:, loudness_idx] = (df_features[:, loudness_idx] + 60) / 60

            # Compute cosine similarity
            query_array = np.array(query_features).reshape(1, -1)
            similarities = cosine_similarity(query_array, df_features)[0]

            # Get top similar songs (excluding the song itself)
            df_with_sim = df_sample.copy()
            df_with_sim["similarity"] = similarities

            # Remove duplicates by track_id (keep first occurrence)
            df_with_sim = df_with_sim.drop_duplicates(subset=["track_id"], keep="first")

            # Exclude the song itself
            df_with_sim = df_with_sim[df_with_sim["track_id"] != exclude_track_id]

            # Sort by similarity (descending)
            df_with_sim = df_with_sim.sort_values("similarity", ascending=False)

            # Return top results
            results = []
            for _, row in df_with_sim.head(limit).iterrows():
                results.append(
                    {
                        "track_id": row["track_id"],
                        "track_name": row.get("track_name", "Unknown"),
                        "artist_name": row.get("artist_name", "Unknown"),
                        "similarity": float(row["similarity"]),
                    }
                )

            return results
        except Exception as e:
            print(f"Error in _find_similar_in_dataset: {e}")
            import traceback

            traceback.print_exc()
            return []
