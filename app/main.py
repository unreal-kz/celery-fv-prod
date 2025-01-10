"""
Face Verification API

This module implements a FastAPI-based REST API for face verification services.
It provides endpoints for single and batch face verification, task status checking,
and system monitoring.

Key Features:
    - Face Verification:
        * Single pair verification
        * Batch verification support
        * Async task processing
        * Result caching
    
    - Monitoring:
        * Health check endpoint
        * System metrics
        * Redis connection stats
        * Cache performance metrics
    
    - Error Handling:
        * Rate limiting
        * Input validation
        * Detailed error messages
        * Request tracking
    
Technical Details:
    - API Version: 1.0.0
    - Authentication: None (internal service)
    - Rate Limiting: Configurable per endpoint
    - Response Format: JSON
    - HTTP Methods: GET, POST
    
Security:
    - CORS enabled with configurable origins
    - File size limits
    - Request validation
    - Safe error messages
    
Dependencies:
    - FastAPI: Web framework
    - Celery: Task queue
    - Redis: Cache and task storage
    - Pydantic: Data validation
"""

from fastapi import FastAPI, HTTPException, status, Request, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, List, Tuple, Any, Optional
import time
from pydantic import BaseModel

from .config import settings
from .worker import verify_faces_task, verify_faces_batch_task
from .models.schemas import (
    VerificationRequest,
    VerificationResponse,
    TaskStatus,
    VerificationResult
)
from .utils.task_manager import task_manager
from .utils.metrics import metrics
from .utils.rate_limiter import rate_limiter
from .utils.redis_manager import RedisManager
from .utils.embedding_cache import embedding_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_metrics(request: Request, call_next):
    """Middleware to collect request metrics."""
    start_time = time.time()
    try:
        response = await call_next(request)
        metrics.record_success(request.url.path)
        return response
    except Exception as e:
        metrics.record_error(request.url.path)
        raise
    finally:
        duration = time.time() - start_time
        metrics.record_request_time(request.url.path, duration)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    try:
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    return metrics.get_metrics()

@app.get("/metrics/redis", tags=["monitoring"])
async def get_redis_metrics():
    """Get Redis connection pool metrics."""
    return RedisManager.get_pool_stats()

@app.get("/metrics/cache", tags=["monitoring"])
async def get_cache_metrics():
    """Get embedding cache metrics."""
    return embedding_cache.get_cache_stats()

@app.post("/verify", response_model=VerificationResponse)
async def verify_faces(
    request: Request,
    image1_url: str = Form(...),
    image2_url: str = Form(...)
) -> VerificationResponse:
    """
    Endpoint to verify if two faces match.
    
    This endpoint initiates an asynchronous face verification task and returns
    a task ID that can be used to check the result status.
    
    Args:
        request: FastAPI request object
        image1_url: URL of first image
        image2_url: URL of second image
        
    Returns:
        VerificationResponse: Response containing:
            - task_id: Unique identifier for the verification task
            - status: Current task status
            - message: Optional status message
    
    Technical Details:
        - Image Requirements:
            * Supported formats: JPG, PNG
            * Maximum size: 10MB
            * Maximum dimension: 1920px
            * Must contain exactly one face
        
        - Processing:
            * Async task creation
            * Redis-based task queue
            * Configurable timeout
        
        - Rate Limiting:
            * Per-IP limits
            * Configurable window
            * Burst allowance
    
    Error Responses:
        - 400 Bad Request:
            * Invalid image URLs
            * Unsupported image format
            * Image too large
        - 429 Too Many Requests:
            * Rate limit exceeded
        - 500 Internal Server Error:
            * Processing pipeline failure
            * Redis connectivity issues
    
    Example:
        ```bash
        curl -X POST "http://api/verify" \\
             -F "image1_url=http://example.com/face1.jpg" \\
             -F "image2_url=http://example.com/face2.jpg"
        ```
        
        Response:
        ```json
        {
            "task_id": "verify_12345",
            "status": "PENDING",
            "message": "Task created successfully"
        }
        ```
    """
    try:
        logger.info(f"Received verification request for images: {image1_url}, {image2_url}")
        
        # Check rate limit
        try:
            rate_limiter.check_rate_limit(request.client.host)
        except HTTPException as he:
            # Add rate limit headers before raising
            response = JSONResponse(
                status_code=he.status_code,
                content={"detail": he.detail}
            )
            response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, rate_limiter.requests_per_minute - 
                    len(rate_limiter.requests.get(request.client.host, [])))
            )
            raise HTTPException(
                status_code=he.status_code,
                detail=he.detail,
                headers=response.headers
            )
            
        # Validate URLs
        if not image1_url or not image2_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both image URLs are required"
            )
            
        # Submit task with image URLs
        try:
            task = verify_faces_task.delay(image1_url, image2_url)
            logger.info(f"Created task with ID: {task.id}")
        except Exception as e:
            logger.error(f"Failed to create Celery task: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create task: {str(e)}"
            )
        
        # Register task for monitoring
        try:
            task_manager.register_task(task.id)
            logger.info(f"Registered task {task.id} with task manager")
        except Exception as e:
            logger.error(f"Failed to register task: {str(e)}", exc_info=True)
            # Continue even if registration fails
            
        response = VerificationResponse(
            task_id=task.id,
            status="pending",
            message="Face verification task submitted successfully"
        )
        
        # Add rate limit headers to successful response
        headers = {
            "X-RateLimit-Limit": str(rate_limiter.requests_per_minute),
            "X-RateLimit-Remaining": str(
                max(0, rate_limiter.requests_per_minute - 
                    len(rate_limiter.requests.get(request.client.host, [])))
            )
        }
        
        return JSONResponse(
            content=response.dict(),
            headers=headers
        )
        
    except HTTPException as he:
        logger.error(f"HTTP error in verification request: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Error processing verification request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing request: {str(e)}"
        )

class BatchVerificationRequest(BaseModel):
    """Request model for batch face verification."""
    image_pairs: List[Tuple[str, str]]

class BatchVerificationResponse(BaseModel):
    """Response model for batch face verification."""
    task_id: str
    status: str
    message: Optional[str] = None

@app.post("/verify/batch", response_model=BatchVerificationResponse)
async def verify_faces_batch(
    request: Request,
    batch_request: BatchVerificationRequest
):
    """
    Endpoint to verify multiple pairs of faces in batch.
    
    Args:
        request: FastAPI request object
        batch_request: Batch verification request containing image pairs
        
    Returns:
        BatchVerificationResponse: Response containing task ID and status
    """
    try:
        # Apply rate limiting
        await rate_limiter.check_rate_limit(request)
        
        # Validate batch size
        if len(batch_request.image_pairs) > settings.BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size exceeds maximum limit of {settings.BATCH_SIZE}"
            )
        
        # Start batch verification task
        task = verify_faces_batch_task.delay(batch_request.image_pairs)
        
        # Register task
        task_manager.register_task(task.id)
        
        # Record metrics
        metrics.record_request_time("/verify/batch", time.time())
        
        return BatchVerificationResponse(
            task_id=task.id,
            status="pending",
            message=f"Processing {len(batch_request.image_pairs)} image pairs"
        )
        
    except Exception as e:
        logger.error(f"Error starting batch verification task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting batch verification: {str(e)}"
        )

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_task_status(task_id: str) -> TaskStatus:
    """
    Get the status of a verification task.
    
    This endpoint checks the current status of a face verification task
    and returns the result if the task is complete.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        TaskStatus: Current status containing:
            - task_id: Task identifier
            - status: Task status (PENDING, SUCCESS, FAILURE)
            - result: Verification result if complete
            - error: Error message if failed
    
    Technical Details:
        - Status Types:
            * PENDING: Task is queued or processing
            * SUCCESS: Task completed successfully
            * FAILURE: Task failed with error
        
        - Result Format (on success):
            ```json
            {
                "is_match": bool,
                "confidence": float,
                "metrics": {
                    "cosine_similarity": float,
                    "euclidean_distance": float
                },
                "processing_time": float
            }
            ```
        
        - Caching:
            * Results cached for 1 hour
            * Cache size limit: 10000 items
            * LRU eviction policy
    
    Error Responses:
        - 404 Not Found:
            * Invalid task ID
            * Expired task result
        - 500 Internal Server Error:
            * Redis connectivity issues
    
    Example:
        ```bash
        curl "http://api/status/verify_12345"
        ```
        
        Response:
        ```json
        {
            "task_id": "verify_12345",
            "status": "SUCCESS",
            "result": {
                "is_match": true,
                "confidence": 0.85,
                "metrics": {
                    "cosine_similarity": 0.9,
                    "euclidean_distance": 0.3
                },
                "processing_time": 1.23
            }
        }
        ```
    """
    try:
        result = task_manager.get_task_result(task_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found"
            )
            
        return TaskStatus(
            task_id=task_id,
            **result
        )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error checking task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking task status: {str(e)}"
        )

@app.post("/cache/clear", tags=["maintenance"])
async def clear_cache():
    """Clear the embedding cache."""
    success = embedding_cache.clear_cache()
    if success:
        return {"status": "success", "message": "Cache cleared"}
    return {"status": "error", "message": "Failed to clear cache"}

@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    # Clean up old tasks
    task_manager.cleanup_old_tasks()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application")
    RedisManager.close_pool()
