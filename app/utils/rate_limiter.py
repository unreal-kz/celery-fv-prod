import time
from typing import Dict, Tuple
import logging
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, requests_per_minute: int = 120, burst_limit: int = 30):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Number of requests allowed per minute per IP
            burst_limit: Maximum burst of requests allowed
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.requests: Dict[str, list] = {}
        
    def _cleanup_old_requests(self, ip: str):
        """Remove requests older than 1 minute."""
        current_time = time.time()
        self.requests[ip] = [t for t in self.requests[ip] 
                           if current_time - t < 60]
    
    def check_rate_limit(self, ip: str):
        """
        Check if request is within rate limits.
        
        Args:
            ip: IP address of the requester
            
        Raises:
            HTTPException: If rate limit is exceeded
        """
        current_time = time.time()
        
        # Initialize if IP not seen before
        if ip not in self.requests:
            self.requests[ip] = []
            
        # Cleanup old requests
        self._cleanup_old_requests(ip)
        
        # Add current request
        self.requests[ip].append(current_time)
        
        # Check if exceeding requests per minute
        if len(self.requests[ip]) > self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP: {ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )
            
        # Check burst limit (requests in last second)
        recent_requests = len([t for t in self.requests[ip] 
                             if current_time - t < 1])
        if recent_requests > self.burst_limit:
            logger.warning(f"Burst limit exceeded for IP: {ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )

# Create singleton instance with higher limits since we're using Celery
rate_limiter = RateLimiter(requests_per_minute=120, burst_limit=30)
