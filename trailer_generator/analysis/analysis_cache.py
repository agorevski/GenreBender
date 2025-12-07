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
        """
        Load cache from disk.
        
        Returns:
            Cache dictionary
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
        """Save cache to disk."""
        if not self.enabled:
            return
        
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            logger.debug(f"Saved cache with {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _compute_hash(self, keyframe_path: str) -> str:
        """
        Compute SHA-256 hash of keyframe image.
        
        Args:
            keyframe_path: Path to keyframe image
            
        Returns:
            Hex string of hash
        """
        try:
            with open(keyframe_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash
        except Exception as e:
            logger.error(f"Failed to hash {keyframe_path}: {e}")
            return ""
    
    def get(self, shot: Dict) -> Optional[Dict]:
        """
        Get cached analysis for a shot.
        
        Args:
            shot: Shot dictionary with keyframe path
            
        Returns:
            Cached analysis or None if not found
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
        """
        Store analysis result in cache.
        
        Args:
            shot: Shot dictionary with keyframe path
            analysis: Analysis result to cache
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
        """
        Check cache for multiple shots at once.
        
        Args:
            shots: List of shot dictionaries
            
        Returns:
            Tuple of (cached_shots, uncached_shots)
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
        """
        Store multiple analysis results in cache.
        
        Args:
            shots: List of shot dictionaries with 'analysis' field
        """
        for shot in shots:
            if 'analysis' in shot:
                self.put(shot, shot['analysis'])
        
        # Save after batch
        self._save_cache()
    
    def clear(self):
        """Clear all cache entries."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'total_entries': len(self.cache),
            'enabled': self.enabled,
            'cache_file': str(self.cache_file),
            'size_bytes': self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }
    
    def prune_old_entries(self, days: int = 30):
        """
        Remove cache entries older than specified days.
        
        Args:
            days: Age threshold in days
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
