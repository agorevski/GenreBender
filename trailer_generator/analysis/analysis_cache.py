"""
Analysis cache management for storing and retrieving shot analysis results.
Prevents redundant API calls during development and re-runs.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AnalysisCache:
    """
    Manages caching of shot analysis results to avoid redundant API calls.
    Uses SHA-256 hash of keyframe as cache key.
    """
    
    def __init__(self, cache_dir: str = "cache", enabled: bool = True):
        """
        Initialize analysis cache.
        
        Args:
            cache_dir: Directory to store cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.cache_file = self.cache_dir / "analysis_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from disk.
        
        Returns:
            Dict: Cache dictionary containing previously stored analysis results.
        """
        if not self.enabled:
            return {}
        
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                return {}
        
        return {}
    
    def _save_cache(self):
        """Save the current cache state to disk.
        
        Writes the cache dictionary to a JSON file. Does nothing if caching
        is disabled. Logs an error if the save operation fails.
        """
        if not self.enabled:
            return
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.debug(f"Saved cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _compute_hash(self, keyframe_path: str) -> str:
        """Compute SHA-256 hash of keyframe image.
        
        Args:
            keyframe_path: Path to the keyframe image file.
            
        Returns:
            str: Hexadecimal string representation of the SHA-256 hash,
                or empty string if hashing fails.
        """
        try:
            with open(keyframe_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Failed to hash {keyframe_path}: {e}")
            return ""
    
    def get(self, shot: Dict) -> Optional[Dict]:
        """Get cached analysis for a shot.
        
        Args:
            shot: Shot dictionary containing at least a 'keyframe' key
                with the path to the keyframe image.
            
        Returns:
            Optional[Dict]: The cached analysis dictionary if found,
                or None if not in cache or caching is disabled.
        """
        if not self.enabled:
            return None
        
        keyframe_path = shot.get('keyframe')
        if not keyframe_path:
            return None
        
        cache_key = self._compute_hash(keyframe_path)
        if not cache_key:
            return None
        
        if cache_key in self.cache:
            logger.debug(f"Cache hit for shot {shot.get('id')}")
            return self.cache[cache_key]['analysis']
        
        logger.debug(f"Cache miss for shot {shot.get('id')}")
        return None
    
    def put(self, shot: Dict, analysis: Dict):
        """Store analysis result in cache.
        
        Args:
            shot: Shot dictionary containing at least a 'keyframe' key
                with the path to the keyframe image, and optionally an 'id'.
            analysis: Analysis result dictionary to cache.
        """
        if not self.enabled:
            return
        
        keyframe_path = shot.get('keyframe')
        if not keyframe_path:
            return
        
        cache_key = self._compute_hash(keyframe_path)
        if not cache_key:
            return
        
        self.cache[cache_key] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'shot_id': shot.get('id'),
            'keyframe': keyframe_path
        }
        
        logger.debug(f"Cached analysis for shot {shot.get('id')}")
        
        # Save every 10 entries to avoid excessive I/O
        if len(self.cache) % 10 == 0:
            self._save_cache()
    
    def get_batch(self, shots: list) -> tuple[list, list]:
        """Check cache for multiple shots at once.
        
        Args:
            shots: List of shot dictionaries, each containing at least
                a 'keyframe' key with the path to the keyframe image.
            
        Returns:
            tuple[list, list]: A tuple containing:
                - cached_shots: List of shots with their cached analysis added.
                - uncached_shots: List of shots not found in cache.
        """
        cached = []
        uncached = []
        
        for shot in shots:
            analysis = self.get(shot)
            if analysis:
                shot_with_analysis = shot.copy()
                shot_with_analysis['analysis'] = analysis
                cached.append(shot_with_analysis)
            else:
                uncached.append(shot)
        
        logger.info(f"Batch cache check: {len(cached)} cached, {len(uncached)} uncached")
        
        return cached, uncached
    
    def put_batch(self, shots: list):
        """Store multiple analysis results in cache.
        
        Args:
            shots: List of shot dictionaries, each containing an 'analysis'
                field with the analysis result to cache.
        """
        for shot in shots:
            if 'analysis' in shot:
                self.put(shot, shot['analysis'])
        
        # Save after batch
        self._save_cache()
    
    def clear(self):
        """Clear all cache entries.
        
        Removes all entries from the in-memory cache and deletes the
        cache file from disk if it exists.
        """
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics.
        
        Returns:
            Dict: Dictionary containing cache statistics with keys:
                - total_entries: Number of cached entries.
                - enabled: Whether caching is enabled.
                - cache_file: Path to the cache file.
                - size_bytes: Size of the cache file in bytes.
        """
        return {
            'total_entries': len(self.cache),
            'enabled': self.enabled,
            'cache_file': str(self.cache_file),
            'size_bytes': self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }
    
    def prune_old_entries(self, days: int = 30):
        """Remove cache entries older than specified days.
        
        Args:
            days: Age threshold in days. Entries older than this will
                be removed. Defaults to 30.
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(days=days)
        
        old_count = len(self.cache)
        
        self.cache = {
            k: v for k, v in self.cache.items()
            if datetime.fromisoformat(v['timestamp']) > cutoff
        }
        
        pruned = old_count - len(self.cache)
        
        if pruned > 0:
            logger.info(f"Pruned {pruned} old cache entries")
            self._save_cache()
