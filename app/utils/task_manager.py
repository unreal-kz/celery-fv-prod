import logging
from typing import Optional, Dict, Any
from celery.result import AsyncResult
from datetime import datetime, timedelta
from .redis_manager import RedisManager

logger = logging.getLogger(__name__)

class TaskManager:
    def __init__(self, max_task_age_hours: int = 24):
        """
        Initialize task manager.
        
        Args:
            max_task_age_hours: Maximum age of tasks to keep in result backend
        """
        self.max_task_age = timedelta(hours=max_task_age_hours)
        self.tasks: Dict[str, datetime] = {}
        
    def register_task(self, task_id: str) -> None:
        """Register a new task."""
        try:
            self.tasks[task_id] = datetime.now()
            with RedisManager.get_connection() as redis_client:
                # Store task registration time in Redis
                redis_client.hset(
                    "task_registry",
                    task_id,
                    datetime.now().isoformat()
                )
            logger.info(f"Task {task_id} registered successfully")
        except Exception as e:
            logger.error(f"Error registering task {task_id}: {str(e)}")
            raise
        
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task result with proper error handling.
        
        Args:
            task_id: ID of the task
            
        Returns:
            Optional[Dict[str, Any]]: Task result or None if not found
        """
        try:
            # Check if task exists in registry
            with RedisManager.get_connection() as redis_client:
                if not redis_client.hexists("task_registry", task_id):
                    logger.warning(f"Task {task_id} not found in registry")
                    return {
                        "status": "error",
                        "error": "Task not found"
                    }
            
            result = AsyncResult(task_id)
            
            # Check task state
            if result.ready():
                if result.successful():
                    return {
                        "status": "completed",
                        "result": result.get()
                    }
                else:
                    # Task failed
                    error = str(result.result) if result.result else "Unknown error"
                    logger.error(f"Task {task_id} failed: {error}")
                    return {
                        "status": "error",
                        "error": error
                    }
            else:
                # Task still running
                return {
                    "status": "pending",
                    "message": "Task is still processing"
                }
                
        except Exception as e:
            logger.error(f"Error getting task result for {task_id}: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": f"Error retrieving task result: {str(e)}"
            }
            
    def cleanup_old_tasks(self) -> None:
        """Clean up old tasks from registry."""
        try:
            current_time = datetime.now()
            with RedisManager.get_connection() as redis_client:
                # Get all tasks from registry
                tasks = redis_client.hgetall("task_registry")
                
                for task_id, registered_time in tasks.items():
                    try:
                        task_time = datetime.fromisoformat(registered_time)
                        if current_time - task_time > self.max_task_age:
                            # Remove old task
                            redis_client.hdel("task_registry", task_id)
                            if task_id in self.tasks:
                                del self.tasks[task_id]
                            logger.info(f"Cleaned up old task: {task_id}")
                    except Exception as e:
                        logger.error(f"Error cleaning up task {task_id}: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error during task cleanup: {str(e)}")

# Create singleton instance
task_manager = TaskManager()
