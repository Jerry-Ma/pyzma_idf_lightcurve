"""Server-side cache for LightcurveStorage objects.

This module provides a singleton cache to store loaded LightcurveStorage objects
on the server side, avoiding the need to reload large datasets in every callback.
"""

import logging
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class StorageCache:
    """Singleton cache for LightcurveStorage objects.
    
    This cache stores loaded storage objects on the server side, indexed by
    a unique key combining storage path and mode. This avoids reloading
    large datasets in every callback.
    
    Usage:
        cache = StorageCache.get_instance()
        
        # Store loaded storage
        cache.set(storage_path, mode, storage_object)
        
        # Retrieve cached storage
        storage = cache.get(storage_path, mode)
        
        # Clear cache when needed
        cache.clear()
    """
    
    _instance: Optional['StorageCache'] = None
    
    def __init__(self):
        """Initialize the cache. Use get_instance() instead of direct instantiation."""
        self._cache: dict[str, Any] = {}  # Values are LightcurveStorage objects
        self._access_count: dict[str, int] = {}
        logger.info("StorageCache initialized")
    
    @classmethod
    def get_instance(cls) -> 'StorageCache':
        """Get the singleton instance of StorageCache."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @staticmethod
    def _make_key(storage_path: str | Path, mode: str) -> str:
        """Create a unique cache key from storage path and mode.
        
        Args:
            storage_path: Path to the storage directory
            mode: Storage mode ('read' or 'write')
            
        Returns:
            Unique cache key string
        """
        return f"{Path(storage_path).resolve()}:{mode}"
    
    def get(self, storage_path: str | Path, mode: str) -> Optional[Any]:
        """Retrieve a cached storage object.
        
        Args:
            storage_path: Path to the storage directory
            mode: Storage mode ('read' or 'write')
            
        Returns:
            Cached LightcurveStorage object if found, None otherwise
        """
        key = self._make_key(storage_path, mode)
        storage = self._cache.get(key)
        
        if storage is not None:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            logger.debug(f"Cache HIT: {key} (accessed {self._access_count[key]} times)")
        else:
            logger.debug(f"Cache MISS: {key}")
        
        return storage
    
    def set(self, storage_path: str | Path, mode: str, storage: Any) -> None:
        """Store a storage object in the cache.
        
        Args:
            storage_path: Path to the storage directory
            mode: Storage mode ('read' or 'write')
            storage: LightcurveStorage object to cache
        """
        key = self._make_key(storage_path, mode)
        self._cache[key] = storage
        self._access_count[key] = 0
        
        # Log cache status
        data_shape = storage.get_storage_info().get('data_shape', {})
        logger.info(f"Cache SET: {key} (shape={data_shape}, total cached={len(self._cache)})")
    
    def invalidate(self, storage_path: str | Path, mode: Optional[str] = None) -> None:
        """Invalidate cache entries for a specific storage path.
        
        Args:
            storage_path: Path to the storage directory
            mode: Optional mode to invalidate. If None, invalidates all modes.
        """
        if mode is None:
            # Invalidate all modes for this path
            keys_to_remove = [
                k for k in self._cache.keys()
                if k.startswith(f"{Path(storage_path).resolve()}:")
            ]
        else:
            keys_to_remove = [self._make_key(storage_path, mode)]
        
        for key in keys_to_remove:
            if key in self._cache:
                del self._cache[key]
                del self._access_count[key]
                logger.info(f"Cache INVALIDATE: {key}")
    
    def clear(self) -> None:
        """Clear all cached storage objects."""
        count = len(self._cache)
        self._cache.clear()
        self._access_count.clear()
        logger.info(f"Cache CLEARED: removed {count} entries")
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'size': len(self._cache),
            'keys': list(self._cache.keys()),
            'access_counts': dict(self._access_count),
            'total_accesses': sum(self._access_count.values()),
        }
