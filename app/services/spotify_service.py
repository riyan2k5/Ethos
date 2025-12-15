"""
Spotify API service for fetching cover art and metadata.
"""
import os
import base64
import ssl
import aiohttp
from typing import Optional, Dict
from dotenv import load_dotenv
from app.services.cache import cover_art_cache

load_dotenv()

# Create SSL context that doesn't verify certificates (for development)
# In production, you should use proper certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


class SpotifyService:
    """Service for interacting with Spotify API."""
    
    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[float] = None
    
    async def initialize(self):
        """Initialize Spotify service and get access token."""
        if not self.client_id or not self.client_secret:
            print("⚠️  Spotify credentials not found in .env file")
            print("   Cover art will not be available, but the app will still work.")
            return
        
        try:
            await self._refresh_token()
        except Exception as e:
            print(f"⚠️  Failed to initialize Spotify service: {e}")
            print("   Cover art will not be available, but the app will still work.")
            # Don't raise - let the app continue without Spotify
    
    async def _refresh_token(self):
        """Refresh Spotify access token."""
        if not self.client_id or not self.client_secret:
            return
        
        auth_string = f"{self.client_id}:{self.client_secret}"
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = base64.b64encode(auth_bytes).decode("utf-8")
        
        # Use connector with SSL context that doesn't verify (for development)
        # Note: In production, you should use proper SSL verification
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            try:
                async with session.post(
                    "https://accounts.spotify.com/api/token",
                    headers={
                        "Authorization": f"Basic {auth_base64}",
                        "Content-Type": "application/x-www-form-urlencoded"
                    },
                    data={"grant_type": "client_credentials"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.access_token = data.get("access_token")
                        expires_in = data.get("expires_in", 3600)
                        import time
                        self.token_expires_at = time.time() + expires_in - 60  # Refresh 1 min early
                        print("✅ Spotify access token obtained")
                    else:
                        error_text = await response.text()
                        print(f"⚠️  Failed to get Spotify token: {response.status}")
                        print(f"   Response: {error_text[:200]}")
            except Exception as e:
                print(f"⚠️  Error connecting to Spotify: {e}")
                raise
    
    async def _ensure_token(self):
        """Ensure we have a valid access token."""
        import time
        if not self.access_token or (self.token_expires_at and time.time() >= self.token_expires_at):
            await self._refresh_token()
    
    async def get_cover_art(self, track_id: str) -> Optional[str]:
        """Get cover art URL for a track (with caching)."""
        # Check cache first
        cached = cover_art_cache.get(f"cover_{track_id}")
        if cached is not None:
            return cached
        
        await self._ensure_token()
        
        if not self.access_token:
            return None
        
        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            # Use shorter timeout for faster failure
            timeout = aiohttp.ClientTimeout(total=1.5)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                async with session.get(
                    f"https://api.spotify.com/v1/tracks/{track_id}",
                    headers={"Authorization": f"Bearer {self.access_token}"}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        images = data.get("album", {}).get("images", [])
                        if images:
                            # Return the medium-sized image (640x640)
                            for img in images:
                                if img.get("height", 0) >= 300:
                                    cover_url = img.get("url")
                                    # Cache the result
                                    if cover_url:
                                        cover_art_cache.set(f"cover_{track_id}", cover_url)
                                    return cover_url
                            # Fallback to first image
                            cover_url = images[0].get("url")
                            if cover_url:
                                cover_art_cache.set(f"cover_{track_id}", cover_url)
                            return cover_url
                    elif response.status == 401:
                        # Token expired, refresh and retry once (with timeout)
                        await self._refresh_token()
                        if self.access_token:
                            retry_connector = aiohttp.TCPConnector(ssl=ssl_context)
                            retry_timeout = aiohttp.ClientTimeout(total=1.5)
                            async with aiohttp.ClientSession(connector=retry_connector, timeout=retry_timeout) as retry_session:
                                async with retry_session.get(
                                    f"https://api.spotify.com/v1/tracks/{track_id}",
                                    headers={"Authorization": f"Bearer {self.access_token}"}
                                ) as retry_response:
                                    if retry_response.status == 200:
                                        retry_data = await retry_response.json()
                                        retry_images = retry_data.get("album", {}).get("images", [])
                                        if retry_images:
                                            cover_url = retry_images[0].get("url")
                                            if cover_url:
                                                cover_art_cache.set(f"cover_{track_id}", cover_url)
                                            return cover_url
        except Exception as e:
            print(f"Error fetching cover art: {e}")
        
        return None
    
    async def search_track(self, query: str) -> Optional[Dict]:
        """Search for a track on Spotify."""
        await self._ensure_token()
        
        if not self.access_token:
            return None
        
        try:
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(
                    f"https://api.spotify.com/v1/search",
                    headers={"Authorization": f"Bearer {self.access_token}"},
                    params={"q": query, "type": "track", "limit": 1}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tracks = data.get("tracks", {}).get("items", [])
                        if tracks:
                            return tracks[0]
        except Exception as e:
            print(f"Error searching track: {e}")
        
        return None

