from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any

class VerificationRequest(BaseModel):
    """Request model for face verification."""
    image1_url: HttpUrl
    image2_url: HttpUrl

class VerificationResponse(BaseModel):
    """Response model for face verification."""
    task_id: str
    status: str
    message: Optional[str] = None

class TaskStatus(BaseModel):
    """Model for task status responses."""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class VerificationResult(BaseModel):
    """Model for verification results."""
    status: str
    is_match: Optional[bool] = None
    confidence: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    message: Optional[str] = None
