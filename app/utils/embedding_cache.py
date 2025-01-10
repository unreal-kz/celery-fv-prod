import logging
import hashlib
import pickle
import torch
import numpy as np
from typing import Optional, Tuple
from .redis_manager import RedisManager
from ..config import settings

logger = logging.getLogger(__name__)

class EmbeddingCache:
    def __init__(self):
        """Initialize the embedding cache."""
        self.redis = RedisManager.get_client()
        self.cache_ttl = settings.EMBEDDING_CACHE_TTL
        self.prefix = settings.EMBEDDING_CACHE_PREFIX
        self.max_cache_size = settings.MAX_CACHE_SIZE
    
    def _generate_key(self, url: str) -> str:
        """Generate a cache key for a given URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        return f"{self.prefix}{url_hash}"
    
    def get_embedding(self, url: str) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
        """
        Get embedding from cache.
        
        Args:
            url: Image URL
            
        Returns:
            Optional[Tuple[torch.Tensor, np.ndarray]]: Cached embedding and aligned face if found
        """
        try:
            key = self._generate_key(url)
            cached_data = self.redis.get(key)
            
            if cached_data:
                logger.info(f"Cache hit for URL: {url}")
                embedding_dict = pickle.loads(cached_data)
                embedding = torch.tensor(embedding_dict['embedding'])
                aligned_face = embedding_dict['aligned_face']
                
                # Refresh TTL on access
                self.redis.expire(key, self.cache_ttl)
                
                return embedding, aligned_face
            
            logger.info(f"Cache miss for URL: {url}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving embedding from cache: {str(e)}")
            return None
    
    def set_embedding(self, url: str, embedding: torch.Tensor, aligned_face: np.ndarray) -> bool:
        """
        Store embedding in cache.
        
        Args:
            url: Image URL
            embedding: Face embedding tensor
            aligned_face: Aligned face image array
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check cache size
            if self.redis.zcard(f"{self.prefix}urls") >= self.max_cache_size:
                # Remove oldest entry
                oldest_url = self.redis.zpopmin(f"{self.prefix}urls")[0][0]
                self.redis.delete(self._generate_key(oldest_url))
                logger.info(f"Removed oldest cache entry for URL: {oldest_url}")
            
            key = self._generate_key(url)
            
            # Store embedding and aligned face
            embedding_dict = {
                'embedding': embedding.cpu().numpy(),
                'aligned_face': aligned_face
            }
            self.redis.setex(
                key,
                self.cache_ttl,
                pickle.dumps(embedding_dict)
            )
            
            # Add URL to sorted set with current timestamp
            self.redis.zadd(f"{self.prefix}urls", {url: float(torch.cuda.current_stream().cuda_stream)})
            
            logger.info(f"Cached embedding for URL: {url}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching embedding: {str(e)}")
            return False
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        try:
            total_entries = self.redis.zcard(f"{self.prefix}urls")
            memory_usage = sum(
                len(self.redis.get(self._generate_key(url)) or b'')
                for url in self.redis.zrange(f"{self.prefix}urls", 0, -1)
            )
            
            return {
                "total_entries": total_entries,
                "memory_usage_bytes": memory_usage,
                "max_entries": self.max_cache_size,
                "ttl_seconds": self.cache_ttl
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {
                "error": str(e)
            }
    
    def clear_cache(self) -> bool:
        """Clear all cached embeddings."""
        try:
            # Get all URLs
            urls = self.redis.zrange(f"{self.prefix}urls", 0, -1)
            
            # Delete all embedding entries
            for url in urls:
                self.redis.delete(self._generate_key(url))
            
            # Delete URL sorted set
            self.redis.delete(f"{self.prefix}urls")
            
            logger.info("Cleared embedding cache")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False

# Create singleton instance
embedding_cache = EmbeddingCache()
