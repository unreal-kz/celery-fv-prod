from pydantic_settings import BaseSettings
import os
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """Application settings."""
    
    # Redis configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_TASKS_KEY: str = "face_verification_tasks"  # Key for the task queue
    REDIS_RESULTS_KEY: str = "face_verification_results:"  # Prefix for results
    REDIS_POOL_MAX_CONNECTIONS: int = int(os.getenv("REDIS_POOL_MAX_CONNECTIONS", "50"))
    REDIS_POOL_TIMEOUT: float = float(os.getenv("REDIS_POOL_TIMEOUT", "5.0"))
    REDIS_POOL_CONNECT_TIMEOUT: float = float(os.getenv("REDIS_POOL_CONNECT_TIMEOUT", "1.0"))
    REDIS_POOL_HEALTH_CHECK_INTERVAL: int = int(os.getenv("REDIS_POOL_HEALTH_CHECK_INTERVAL", "30"))
    
    # Cache configuration
    EMBEDDING_CACHE_TTL: int = 3600  # 1 hour in seconds
    EMBEDDING_CACHE_PREFIX: str = "embedding:"
    MAX_CACHE_SIZE: int = 10000  # Maximum number of embeddings to cache
    
    # Resource limits
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_IMAGE_DIMENSION: int = 1920
    MAX_CONCURRENT_TASKS: int = 10
    WORKER_MEMORY_LIMIT_MB: int = 1024
    
    # Performance settings
    BATCH_SIZE: int = 10
    DOWNLOAD_TIMEOUT: int = 30
    VERIFICATION_THRESHOLD: float = 0.7
    
    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 60  # seconds
    
    # API configuration
    API_TITLE: str = "Face Verification API"
    API_DESCRIPTION: str = "API for verifying face similarity using deep learning"
    API_VERSION: str = "1.0.0"
    
    # Celery configuration
    CELERY_TASK_TIMEOUT: int = 30  # seconds
    
    # File storage configuration
    UPLOAD_DIR: str = str(Path(__file__).parent.parent / "uploads")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # AWS S3 Configuration
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    
    # Verification thresholds and confidence calculation
    COSINE_SIMILARITY_THRESHOLD: float = float(os.getenv("COSINE_SIMILARITY_THRESHOLD", "0.7"))
    EUCLIDEAN_DISTANCE_THRESHOLD: float = float(os.getenv("EUCLIDEAN_DISTANCE_THRESHOLD", "0.8"))
    MAX_EUCLIDEAN_DISTANCE: float = float(os.getenv("MAX_EUCLIDEAN_DISTANCE", "2.0"))  # Maximum expected L2 distance for normalized vectors
    MIN_CONFIDENCE_THRESHOLD: float = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.5"))  # Minimum confidence to consider a match
    
    class Config:
        env_file = ".env"
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create upload directory if it doesn't exist
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

settings = Settings()
