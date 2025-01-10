import time
from typing import Dict, List
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        """Initialize metrics collector."""
        self.request_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
    def record_request_time(self, endpoint: str, duration: float):
        """Record request processing time."""
        self.request_times[endpoint].append(duration)
        
        # Keep only last 1000 requests
        if len(self.request_times[endpoint]) > 1000:
            self.request_times[endpoint] = self.request_times[endpoint][-1000:]
            
    def record_error(self, endpoint: str):
        """Record an error."""
        self.error_counts[endpoint] += 1
        
    def record_success(self, endpoint: str):
        """Record a success."""
        self.success_counts[endpoint] += 1
        
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current metrics."""
        metrics = {}
        
        for endpoint in self.request_times:
            times = self.request_times[endpoint]
            if not times:
                continue
                
            total_requests = self.success_counts[endpoint] + self.error_counts[endpoint]
            error_rate = (self.error_counts[endpoint] / total_requests) if total_requests > 0 else 0
            
            metrics[endpoint] = {
                "avg_response_time": sum(times) / len(times),
                "min_response_time": min(times),
                "max_response_time": max(times),
                "error_rate": error_rate,
                "success_rate": 1 - error_rate,
                "total_requests": total_requests
            }
            
        return metrics

# Create singleton instance
metrics = MetricsCollector()
