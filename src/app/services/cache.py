"""
Simple in-memory cache for cover arts and API responses.
"""
from typing import Optional, Dict
import time


class SimpleCache:
    """Simple in-memory cache with TTL."""
    
    def __init__(self, ttl: int = 3600):  # 1 hour default
        self.cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if time.time() < expiry:
                return value
            else:
                # Expired, remove it
                del self.cache[key]
        return None
    
    def set(self, key: str, value: any, ttl: Optional[int] = None):
        """Set value in cache with TTL."""
        expiry = time.time() + (ttl or self.ttl)
        self.cache[key] = (value, expiry)
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()


# Global cache instances
cover_art_cache = SimpleCache(ttl=86400)  # 24 hours for cover arts

