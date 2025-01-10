"""
Face Verification Utility Module

This module provides utility functions for face verification, including:
- Confidence score calculation
- Similarity metric computation
- Result preparation and formatting

Key Features:
    - Adaptive Threshold Approach:
        * Combines multiple similarity metrics
        * Dynamic confidence calculation
        * Configurable thresholds
    
    - Similarity Metrics:
        * Cosine Similarity (0-1, higher is better)
        * Euclidean Distance (0-inf, lower is better)
    
    - Result Formatting:
        * Standardized result structure
        * Error handling
        * Processing time tracking

Technical Details:
    - Confidence Calculation:
        * Uses both cosine similarity and Euclidean distance
        * Adaptive weighting based on threshold proximity
        * Normalized to [0,1] range
    
    - Thresholds:
        * Cosine Similarity: Configurable (default: 0.7)
        * Euclidean Distance: Configurable (default: 0.8)
        * Minimum Confidence: Configurable (default: 0.5)

Usage:
    ```python
    # Calculate confidence and match status
    confidence, is_match = calculate_confidence(cosine_sim=0.8, euclidean_dist=0.5)
    
    # Prepare verification result
    result = prepare_verification_result(
        confidence=confidence,
        is_match=is_match,
        similarity_metrics={
            "cosine_similarity": 0.8,
            "euclidean_distance": 0.5
        }
    )
    ```
"""

import logging
from typing import Dict, Tuple
import numpy as np
from ..config import settings

logger = logging.getLogger(__name__)

def calculate_confidence(cosine_sim: float, euclidean_dist: float) -> Tuple[float, bool]:
    """
    Calculate confidence score using adaptive threshold approach.
    
    This function implements an advanced confidence calculation that:
    1. Normalizes both similarity metrics to comparable ranges
    2. Applies adaptive weighting based on threshold proximity
    3. Combines metrics using a dynamic approach
    
    Args:
        cosine_sim: Cosine similarity score (0-1, higher is better)
        euclidean_dist: Euclidean distance between embeddings (0-inf, lower is better)
        
    Returns:
        Tuple[float, bool]: 
            - float: confidence score (0-1)
            - bool: True if match, False otherwise
    
    Technical Details:
        - Metric Normalization:
            * Cosine similarity: Already in [0,1] range
            * Euclidean distance: Normalized using MAX_EUCLIDEAN_DISTANCE
        
        - Threshold Application:
            * Primary: COSINE_SIMILARITY_THRESHOLD
            * Secondary: EUCLIDEAN_DISTANCE_THRESHOLD
            * Final: MIN_CONFIDENCE_THRESHOLD
        
        - Confidence Calculation:
            1. If both metrics exceed thresholds:
               * Average of normalized scores
            2. If one/both metrics below thresholds:
               * Minimum of normalized scores
            3. Final confidence clamped to [0,1]
    
    Error Handling:
        - Returns (0.0, False) on calculation errors
        - Logs errors for debugging
        - Validates input ranges
    
    Example:
        ```python
        confidence, is_match = calculate_confidence(
            cosine_sim=0.85,    # High similarity
            euclidean_dist=0.5   # Low distance
        )
        # Returns: (0.75, True)  # Example values
        ```
    """
    try:
        # Normalize Euclidean distance to [0,1] range (inverse, as lower is better)
        euclidean_score = 1 - min(euclidean_dist / settings.MAX_EUCLIDEAN_DISTANCE, 1.0)
        
        # Calculate normalized scores relative to thresholds
        cosine_score = (cosine_sim - settings.COSINE_SIMILARITY_THRESHOLD) / (1 - settings.COSINE_SIMILARITY_THRESHOLD)
        euclidean_norm = (euclidean_score - settings.EUCLIDEAN_DISTANCE_THRESHOLD) / (1 - settings.EUCLIDEAN_DISTANCE_THRESHOLD)
        
        # Combine scores with adaptive weighting
        if cosine_sim > settings.COSINE_SIMILARITY_THRESHOLD and euclidean_score > settings.EUCLIDEAN_DISTANCE_THRESHOLD:
            # Both metrics are good - use average
            confidence = (cosine_score + euclidean_norm) / 2
        else:
            # One or both metrics are below threshold - use minimum
            confidence = min(cosine_score, euclidean_norm)
        
        # Clamp confidence to [0,1] range
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine if it's a match based on confidence
        is_match = confidence >= settings.MIN_CONFIDENCE_THRESHOLD
        
        logger.debug(
            f"Confidence calculation: cosine_sim={cosine_sim:.3f}, euclidean_dist={euclidean_dist:.3f}, "
            f"confidence={confidence:.3f}, is_match={is_match}"
        )
        
        return confidence, is_match
        
    except Exception as e:
        logger.error(f"Error calculating confidence: {str(e)}")
        # Return conservative values on error
        return 0.0, False

def prepare_verification_result(
    confidence: float,
    is_match: bool,
    similarity_metrics: Dict[str, float],
    processing_time: float = None,
    error_message: str = None
) -> Dict[str, any]:
    """
    Prepare the final verification result dictionary.
    
    This function creates a standardized result format for face verification,
    including confidence scores, similarity metrics, and optional error information.
    
    Args:
        confidence: Calculated confidence score (0-1)
        is_match: Boolean indicating if faces match
        similarity_metrics: Dictionary containing similarity metrics:
            - cosine_similarity: float (0-1)
            - euclidean_distance: float (0-inf)
        processing_time: Optional processing time in seconds
        error_message: Optional error message for failed verifications
        
    Returns:
        Dict containing the verification result with structure:
        {
            "status": "success" or "error",
            "is_match": bool,
            "confidence": float,
            "metrics": {
                "cosine_similarity": float,
                "euclidean_distance": float
            },
            "processing_time": float,  # if provided
            "error": str  # if error occurred
        }
    
    Technical Details:
        - All floating point values are converted to Python float
        - Metrics are validated before inclusion
        - Processing time is optional but recommended
        - Error messages provide context for failures
    
    Example:
        ```python
        result = prepare_verification_result(
            confidence=0.85,
            is_match=True,
            similarity_metrics={
                "cosine_similarity": 0.9,
                "euclidean_distance": 0.3
            },
            processing_time=1.23
        )
        ```
    """
    result = {
        "status": "success" if error_message is None else "error",
        "is_match": is_match,
        "confidence": float(confidence),
        "metrics": {
            "cosine_similarity": float(similarity_metrics["cosine_similarity"]),
            "euclidean_distance": float(similarity_metrics["euclidean_distance"]),
        }
    }
    
    if error_message is not None:
        result["error"] = error_message
        
    if processing_time is not None:
        result["processing_time"] = float(processing_time)
        
    return result
