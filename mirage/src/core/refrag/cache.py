"""
Compression Cache
Stores compressed fragments to avoid recompression
Uses Redis for distributed caching
"""

import json
import hashlib
from typing import Optional, Dict, Any
from loguru import logger


class CompressionCache:
    """Cache for storing compressed text fragments"""

    def __init__(self, redis_client=None, ttl: int = 86400):
        """
        Initialize compression cache

        Args:
            redis_client: Redis client (optional, uses in-memory if None)
            ttl: Time-to-live in seconds (default: 24 hours)
        """
        self.redis_client = redis_client
        self.ttl = ttl
        
        # In-memory fallback cache
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            f"Initialized CompressionCache "
            f"(redis={'yes' if redis_client else 'memory-only'}, ttl={ttl}s)"
        )

    def get(
        self,
        text: str,
        compression_ratio: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Get compressed version from cache

        Args:
            text: Original text
            compression_ratio: Target compression ratio

        Returns:
            Cached compression result or None
        """
        cache_key = self._generate_key(text, compression_ratio)

        # Try Redis first
        if self.redis_client:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    logger.debug(f"Cache hit (Redis): {cache_key[:16]}...")
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        # Fallback to memory cache
        if cache_key in self._memory_cache:
            logger.debug(f"Cache hit (memory): {cache_key[:16]}...")
            return self._memory_cache[cache_key]

        logger.debug(f"Cache miss: {cache_key[:16]}...")
        return None

    def set(
        self,
        text: str,
        compression_ratio: float,
        compressed_data: Dict[str, Any],
    ):
        """
        Store compressed version in cache

        Args:
            text: Original text
            compression_ratio: Compression ratio used
            compressed_data: Compression result to cache
        """
        cache_key = self._generate_key(text, compression_ratio)

        # Store in Redis if available
        if self.redis_client:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.ttl,
                    json.dumps(compressed_data),
                )
                logger.debug(f"Cached to Redis: {cache_key[:16]}...")
            except Exception as e:
                logger.warning(f"Redis cache error: {e}")

        # Always store in memory cache
        self._memory_cache[cache_key] = compressed_data
        logger.debug(f"Cached to memory: {cache_key[:16]}...")

    def _generate_key(self, text: str, compression_ratio: float) -> str:
        """
        Generate cache key from text and parameters

        Args:
            text: Original text
            compression_ratio: Compression ratio

        Returns:
            Cache key
        """
        # Hash the text for consistent key
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        
        # Include compression ratio in key
        key = f"refrag:v1:{text_hash}:{compression_ratio:.2f}"
        
        return key

    def clear(self):
        """Clear all cached compressions"""
        self._memory_cache.clear()
        
        if self.redis_client:
            try:
                # Delete all refrag keys
                for key in self.redis_client.scan_iter("refrag:v1:*"):
                    self.redis_client.delete(key)
                logger.info("Cleared Redis cache")
            except Exception as e:
                logger.warning(f"Error clearing Redis cache: {e}")
        
        logger.info("Cleared memory cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "memory_cache_size": len(self._memory_cache),
            "redis_enabled": self.redis_client is not None,
        }

        if self.redis_client:
            try:
                # Count refrag keys in Redis
                redis_count = sum(
                    1 for _ in self.redis_client.scan_iter("refrag:v1:*")
                )
                stats["redis_cache_size"] = redis_count
            except Exception as e:
                logger.warning(f"Error getting Redis stats: {e}")
                stats["redis_cache_size"] = 0

        return stats
