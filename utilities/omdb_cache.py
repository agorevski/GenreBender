"""
OMDB Cache System
=================

Caching system for OMDB API responses to avoid repeated API calls.
"""

import json
import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from utilities.omdb_models import OMDBMovie

logger = logging.getLogger(__name__)

class OMDBCache:
    """
    Cache system for OMDB API responses.
    
    Caches movie data to avoid repeated API calls. Cache files are stored
    in the output directory structure: outputs/<filename>/cache/omdb/
    
    Usage:
        cache = OMDBCache(output_dir="outputs/movie_name")
        movie = cache.get_or_fetch("Dumb and Dumber", client.get_movie_by_title)
    """
    
    def __init__(
        self,
        output_dir: str = "outputs",
        cache_subdir: str = "cache/omdb",
        ttl_seconds: int = 2592000  # 30 days
    ):
        """
        Initialize cache.
        
        Args:
            output_dir: Base output directory (e.g., "outputs/movie_name")
            cache_subdir: Subdirectory for OMDB cache (relative to output_dir)
            ttl_seconds: Time-to-live for cache entries in seconds (default: 30 days)
        """
        self.cache_dir = Path(output_dir) / cache_subdir
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized OMDB cache at: {self.cache_dir}")
    
    def _get_cache_key(self, identifier: str) -> str:
        """
        Generate cache key from movie identifier.
        
        Args:
            identifier: Movie title or IMDb ID
            
        Returns:
            Cache key (hash of identifier)
        """
        # Normalize identifier (lowercase, strip whitespace)
        normalized = identifier.lower().strip()
        # Generate hash for filename safety
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _get_cache_path(self, identifier: str) -> Path:
        """
        Get cache file path for identifier.
        
        Args:
            identifier: Movie title or IMDb ID
            
        Returns:
            Path to cache file
        """
        cache_key = self._get_cache_key(identifier)
        return self.cache_dir / f"{cache_key}.json"
    
    def _is_cache_valid(self, cache_data: Dict[str, Any]) -> bool:
        """
        Check if cache entry is still valid (not expired).
        
        Args:
            cache_data: Cache data dictionary
            
        Returns:
            True if cache is valid, False if expired
        """
        if 'cached_at' not in cache_data:
            return False
        
        try:
            cached_at = datetime.fromisoformat(cache_data['cached_at'])
            expiry_time = cached_at + timedelta(seconds=self.ttl_seconds)
            is_valid = datetime.now() < expiry_time
            
            if not is_valid:
                logger.debug(f"Cache expired (cached at {cached_at})")
            
            return is_valid
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid cache timestamp: {e}")
            return False
    
    def get(self, identifier: str) -> Optional[OMDBMovie]:
        """
        Get movie from cache.
        
        Args:
            identifier: Movie title or IMDb ID
            
        Returns:
            OMDBMovie object if cached and valid, None otherwise
        """
        cache_path = self._get_cache_path(identifier)
        
        if not cache_path.exists():
            logger.debug(f"Cache miss for: {identifier}")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            if not self._is_cache_valid(cache_data):
                logger.info(f"Cache expired for: {identifier}")
                # Delete expired cache file
                cache_path.unlink()
                return None
            
            movie_data = cache_data.get('movie_data')
            if not movie_data:
                return None
            
            logger.info(f"Cache hit for: {identifier}")
            return OMDBMovie.from_dict(movie_data)
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid cache file for {identifier}: {e}")
            # Delete corrupted cache file
            cache_path.unlink()
            return None
    
    def set(self, identifier: str, movie: OMDBMovie) -> None:
        """
        Store movie in cache.
        
        Args:
            identifier: Movie title or IMDb ID
            movie: OMDBMovie object to cache
        """
        cache_path = self._get_cache_path(identifier)
        
        cache_data = {
            'identifier': identifier,
            'cached_at': datetime.now().isoformat(),
            'movie_data': movie.to_dict()
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.info(f"Cached movie data for: {identifier}")
        except Exception as e:
            logger.error(f"Failed to cache movie data: {e}")
    
    def get_or_fetch(
        self,
        identifier: str,
        fetch_func: Callable[[str], OMDBMovie],
        **fetch_kwargs
    ) -> OMDBMovie:
        """
        Get movie from cache or fetch from API if not cached.
        
        Args:
            identifier: Movie title or IMDb ID
            fetch_func: Function to fetch movie data (e.g., client.get_movie_by_title)
            **fetch_kwargs: Additional keyword arguments for fetch function
            
        Returns:
            OMDBMovie object
            
        Example:
            cache = OMDBCache(output_dir="outputs/movie_name")
            client = OMDBClient()
            movie = cache.get_or_fetch("Dumb and Dumber", client.get_movie_by_title)
        """
        # Try cache first
        cached_movie = self.get(identifier)
        if cached_movie:
            return cached_movie
        
        # Fetch from API
        logger.info(f"Fetching from OMDB API: {identifier}")
        movie = fetch_func(identifier, **fetch_kwargs)
        
        # Cache the result
        self.set(identifier, movie)
        
        return movie
    
    def invalidate(self, identifier: str) -> bool:
        """
        Invalidate (delete) cache entry for identifier.
        
        Args:
            identifier: Movie title or IMDb ID
            
        Returns:
            True if cache entry was deleted, False if not found
        """
        cache_path = self._get_cache_path(identifier)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Invalidated cache for: {identifier}")
            return True
        
        return False
    
    def clear_all(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of cache entries deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Cleared {count} cache entries")
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.json"))
        total_entries = len(cache_files)
        valid_entries = 0
        expired_entries = 0
        corrupted_entries = 0
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                if self._is_cache_valid(cache_data):
                    valid_entries += 1
                else:
                    expired_entries += 1
            except (json.JSONDecodeError, KeyError):
                corrupted_entries += 1
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'corrupted_entries': corrupted_entries,
            'ttl_seconds': self.ttl_seconds
        }
