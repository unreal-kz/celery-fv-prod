import redis
from typing import Optional
import logging
from contextlib import contextmanager
from urllib.parse import urlparse
from ..config import settings

logger = logging.getLogger(__name__)

class RedisManager:
    _pool: Optional[redis.ConnectionPool] = None
    _client: Optional[redis.Redis] = None
    
    @classmethod
    def get_pool(cls) -> redis.ConnectionPool:
        """Get or create Redis connection pool."""
        if cls._pool is None:
            logger.info("Initializing Redis connection pool")
            
            # Parse Redis URL
            parsed_url = urlparse(settings.REDIS_URL)
            
            # Create pool with parsed connection details
            cls._pool = redis.ConnectionPool(
                host=parsed_url.hostname or 'localhost',
                port=parsed_url.port or 6379,
                db=int(parsed_url.path.lstrip('/') or 0),
                password=parsed_url.password,
                username=parsed_url.username,
                max_connections=settings.REDIS_POOL_MAX_CONNECTIONS,
                socket_timeout=settings.REDIS_POOL_TIMEOUT,
                socket_connect_timeout=settings.REDIS_POOL_CONNECT_TIMEOUT,
                health_check_interval=settings.REDIS_POOL_HEALTH_CHECK_INTERVAL,
                decode_responses=True
            )
            logger.info("Redis connection pool initialized successfully")
        return cls._pool
    
    @classmethod
    def get_client(cls) -> redis.Redis:
        """Get Redis client from the connection pool."""
        if cls._client is None:
            cls._client = redis.Redis(connection_pool=cls.get_pool())
        return cls._client
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        """Get a Redis connection from the pool with context management."""
        client = None
        try:
            client = cls.get_client()
            yield client
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {str(e)}")
            if cls._pool:
                cls.close_pool()  # Reset pool on connection error
            raise
        except Exception as e:
            logger.error(f"Error in Redis operation: {str(e)}")
            raise
        finally:
            if client:
                try:
                    client.connection_pool.reset()  # Reset connection instead of disconnecting
                except Exception as e:
                    logger.error(f"Error resetting Redis connection: {str(e)}")
    
    @classmethod
    def get_pool_stats(cls) -> dict:
        """Get current pool statistics."""
        if cls._pool is None:
            return {"status": "not_initialized"}
            
        try:
            return {
                "max_connections": cls._pool.max_connections,
                "current_connections": len(cls._pool._connections),
                "in_use_connections": len(cls._pool._in_use_connections),
                "status": "active"
            }
        except Exception as e:
            logger.error(f"Error getting pool stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    @classmethod
    def close_pool(cls):
        """Close all connections in the pool."""
        if cls._pool is not None:
            logger.info("Closing Redis connection pool")
            try:
                if cls._client:
                    cls._client.close()
                    cls._client = None
                cls._pool.disconnect()
                cls._pool = None
            except Exception as e:
                logger.error(f"Error closing Redis pool: {str(e)}")
                raise
