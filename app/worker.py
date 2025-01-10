"""
Face Verification Worker Module

This module handles the core face verification functionality, including:
- Face detection and alignment using MTCNN
- Face embedding generation using FaceNet (InceptionResnetV1)
- Asynchronous task processing with Celery
- Caching of face embeddings
- Error handling and retry logic

Key Components:
    - MTCNN: Multi-task Cascaded Convolutional Networks for face detection and alignment
    - FaceNet: Deep learning model for generating face embeddings
    - Celery: Distributed task queue for asynchronous processing
    - Redis: Cache storage for embeddings and task results

Technical Details:
    - Face Detection:
        * Uses MTCNN with thresholds [0.6, 0.75, 0.9]
        * Minimum face size: 60 pixels
        * Rejects images with 0 or multiple faces
    
    - Face Alignment:
        * Output size: 160x160 pixels
        * Margin: 5 pixels
        * Uses 5-point facial landmarks
    
    - Embedding Generation:
        * Model: InceptionResnetV1 (pretrained on VGGFace2)
        * Embedding size: 512 dimensions
        * Normalized output

Error Handling:
    - Retries failed tasks with exponential backoff
    - Proper cleanup of temporary files
    - Detailed error logging
    - Graceful degradation on resource exhaustion

Usage:
    The module is designed to be used with Celery workers:
    ```bash
    celery -A app.worker:celery_app worker --loglevel=info
    ```
"""

import io
import torch
import numpy as np
from PIL import Image
from celery import Celery, signals
from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
from typing import Dict, Any, Tuple, Optional, List
import tempfile
import os
import requests
import asyncio
import aiohttp
from pathlib import Path
import psutil
from .config import settings
from .utils.embedding_cache import embedding_cache
from .utils.verification import calculate_confidence, prepare_verification_result
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "face_verification",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    task_track_started=True,
    task_time_limit=settings.CELERY_TASK_TIMEOUT,
    worker_max_tasks_per_child=settings.MAX_CONCURRENT_TASKS,
    worker_prefetch_multiplier=1  # Prevent worker from prefetching too many tasks
)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Global variables for models
mtcnn = None
resnet = None

def init_models():
    """
    Initialize the face detection and recognition models.
    
    This function lazily initializes MTCNN and FaceNet models when first needed.
    Models are initialized only once per worker process and kept in memory
    for subsequent requests.
    
    Technical Details:
        - MTCNN Configuration:
            * image_size: 160 (output face image size)
            * margin: 5 (pixels to add around detected face)
            * min_face_size: 60 (minimum face size to detect)
            * thresholds: [0.6, 0.75, 0.9] (detection confidence thresholds)
            * factor: 0.709 (pyramid scale factor)
        
        - FaceNet Configuration:
            * Architecture: InceptionResnetV1
            * Weights: Pretrained on VGGFace2
            * Mode: eval (inference mode)
    
    Global Variables Modified:
        mtcnn: MTCNN model instance
        resnet: FaceNet model instance
    
    Raises:
        RuntimeError: If model initialization fails
        torch.cuda.OutOfMemoryError: If GPU memory is insufficient
    """
    global mtcnn, resnet
    
    if mtcnn is None:
        logger.info("Initializing MTCNN model...")
        mtcnn = MTCNN(
            image_size=160,
            margin=5,
            min_face_size=60,
            thresholds=[0.6, 0.75, 0.9],
            factor=0.709,
            device=device
        )
        
    if resnet is None:
        logger.info("Initializing FaceNet model...")
        resnet = InceptionResnetV1(
            pretrained='vggface2',
            device=device
        ).eval()
    
    logger.info("Models initialized successfully")

@signals.worker_init.connect
def init_worker(**kwargs):
    """Initialize worker process."""
    try:
        init_models()
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise

async def download_image_async(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Download image asynchronously."""
    try:
        async with session.get(url, timeout=settings.DOWNLOAD_TIMEOUT) as response:
            if response.status != 200:
                logger.error(f"Failed to download image. Status: {response.status}")
                return None
                
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
                logger.warning(f"Image too large: {url}")
                return None
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(await response.read())
                return tmp_file.name
                
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        return None

def process_image(image_path: str) -> Optional[Tuple[torch.Tensor, np.ndarray]]:
    """
    Process image and extract face embedding.
    Returns None if no face or multiple faces are detected.
    
    This function handles the complete pipeline of face processing:
    1. Image loading and preprocessing
    2. Face detection and validation
    3. Face alignment
    4. Embedding generation
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Optional[Tuple[torch.Tensor, np.ndarray]]: If successful, returns:
            - torch.Tensor: 512-dimensional face embedding
            - np.ndarray: Aligned face image (160x160x3)
            Returns None if:
            - No face is detected
            - Multiple faces are detected
            - Face alignment fails
            
    Technical Details:
        - Image Preprocessing:
            * Converts RGBA to RGB
            * Resizes large images maintaining aspect ratio
            * Maximum dimension: settings.MAX_IMAGE_DIMENSION
        
        - Face Detection:
            * Uses MTCNN to detect faces
            * Validates exactly one face is present
            * Minimum face size: 60 pixels
        
        - Face Alignment:
            * Uses 5-point facial landmarks
            * Output size: 160x160 pixels
            * Adds 5-pixel margin
        
        - Embedding Generation:
            * Uses FaceNet (InceptionResnetV1)
            * Output: 512-dimensional normalized vector
    
    Error Handling:
        - Returns None for invalid images
        - Logs warnings for detection/alignment failures
        - Proper cleanup of loaded images
    """
    try:
        # Ensure models are initialized
        init_models()
        
        # Load and preprocess image
        image = Image.open(image_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Resize if needed
        if max(image.size) > settings.MAX_IMAGE_DIMENSION:
            ratio = settings.MAX_IMAGE_DIMENSION / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Detect faces
        boxes, probs = mtcnn.detect(image)
        
        if boxes is None or len(boxes) == 0:
            logger.warning(f"No faces detected in {image_path}")
            return None
        elif len(boxes) > 1:
            logger.warning(f"Multiple faces ({len(boxes)}) detected in {image_path}")
            return None
            
        # Get aligned face
        aligned_face = mtcnn(image)
        if aligned_face is None:
            logger.warning(f"Failed to align face in {image_path}")
            return None
        
        # Get embedding
        with torch.no_grad():
            embedding = resnet(aligned_face.unsqueeze(0))
            
        return embedding[0], aligned_face.cpu().numpy()
        
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        return None

def calculate_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> Dict[str, float]:
    """Calculate similarity metrics between embeddings."""
    try:
        # Convert to numpy for calculations
        emb1_np = emb1.cpu().numpy()
        emb2_np = emb2.cpu().numpy()
        
        # Calculate cosine similarity
        cos_sim = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
        
        # Calculate Euclidean distance
        l2_distance = np.linalg.norm(emb1_np - emb2_np)
        
        return {
            "cosine_similarity": float(cos_sim),
            "euclidean_distance": float(l2_distance)
        }
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        raise

@celery_app.task(bind=True, name="verify_faces", max_retries=settings.MAX_RETRIES)
def verify_faces_task(self, image1_url: str, image2_url: str) -> Dict[str, Any]:
    """Process a single face verification task."""
    start_time = time.time()
    temp_files = []
    
    try:
        # Validate URLs
        if not image1_url or not image2_url:
            raise ValueError("Both image URLs are required")
            
        logger.info(f"Starting verification task for images: {image1_url}, {image2_url}")
        
        embeddings = []
        aligned_faces = []
        
        # Process each image
        for url in [image1_url, image2_url]:
            try:
                # Check cache first
                cached_result = embedding_cache.get_embedding(url)
                
                if cached_result is not None:
                    embedding, aligned_face = cached_result
                    embeddings.append(embedding)
                    aligned_faces.append(aligned_face)
                    logger.info(f"Using cached embedding for {url}")
                    continue
                
                # Download and process image
                temp_file = None
                try:
                    async def download():
                        async with aiohttp.ClientSession() as session:
                            return await download_image_async(session, url)
                    temp_file = asyncio.run(download())
                except Exception as e:
                    logger.error(f"Failed to download image from {url}: {str(e)}")
                    raise RuntimeError(f"Failed to download image from {url}")
                
                if temp_file is None:
                    raise RuntimeError(f"Failed to download image from {url}")
                    
                temp_files.append(temp_file)
                
                # Process image
                result = process_image(temp_file)
                if result is None:
                    # Return low confidence result for invalid face detection
                    processing_time = time.time() - start_time
                    return prepare_verification_result(
                        confidence=0.0,
                        is_match=False,
                        similarity_metrics={
                            "cosine_similarity": 0.0,
                            "euclidean_distance": float('inf')
                        },
                        processing_time=processing_time,
                        error_message="Invalid number of faces detected"
                    )
                    
                embedding, aligned_face = result
                embeddings.append(embedding)
                aligned_faces.append(aligned_face)
                
                # Cache the result
                embedding_cache.set_embedding(url, embedding, aligned_face)
                logger.info(f"Successfully processed and cached embedding for {url}")
                
            except Exception as e:
                logger.error(f"Error processing image {url}: {str(e)}")
                raise
        
        # Calculate similarity metrics
        similarity_metrics = calculate_similarity(embeddings[0], embeddings[1])
        
        # Calculate confidence and match status
        confidence, is_match = calculate_confidence(
            similarity_metrics["cosine_similarity"],
            similarity_metrics["euclidean_distance"]
        )
        
        # Prepare final result
        processing_time = time.time() - start_time
        result = prepare_verification_result(
            confidence=confidence,
            is_match=is_match,
            similarity_metrics=similarity_metrics,
            processing_time=processing_time
        )
        
        logger.info(
            f"Task completed successfully in {processing_time:.2f} seconds. "
            f"Match: {is_match}, Confidence: {confidence:.3f}"
        )
        return result
        
    except Exception as e:
        error_msg = f"Error in verification task: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Retry the task if appropriate
        if self.request.retries < settings.MAX_RETRIES:
            logger.info(f"Retrying task. Attempt {self.request.retries + 1} of {settings.MAX_RETRIES}")
            raise self.retry(
                exc=e,
                countdown=settings.RETRY_DELAY * (2 ** self.request.retries)  # Exponential backoff
            )
            
        return prepare_verification_result(
            confidence=0.0,
            is_match=False,
            similarity_metrics={
                "cosine_similarity": 0.0,
                "euclidean_distance": float('inf')
            },
            error_message=error_msg
        )
        
    finally:
        # Cleanup temp files
        for temp_file in temp_files:
            if temp_file:
                try:
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
                except Exception as e:
                    logger.error(f"Error cleaning up file {temp_file}: {str(e)}")

async def download_images_parallel(urls: List[str]) -> List[Optional[str]]:
    """Download multiple images in parallel."""
    async with aiohttp.ClientSession() as session:
        tasks = [download_image_async(session, url) for url in urls]
        return await asyncio.gather(*tasks)

def preprocess_image(image_path: str) -> Optional[Image.Image]:
    """Preprocess image for face detection."""
    try:
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Resize if too large
        if image.size[0] > settings.MAX_IMAGE_DIMENSION or image.size[1] > settings.MAX_IMAGE_DIMENSION:
            image.thumbnail((settings.MAX_IMAGE_DIMENSION, settings.MAX_IMAGE_DIMENSION))
        
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image {image_path}: {str(e)}")
        return None

@celery_app.task(bind=True, name="verify_faces_batch")
def verify_faces_batch_task(self, image_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """Process multiple face verification pairs in batch."""
    results = []
    embeddings_cache = {}  # Local memory cache during batch processing
    
    # Get unique URLs
    unique_urls = list(set([url for pair in image_pairs for url in pair]))
    
    # Download all images in parallel
    temp_files = asyncio.run(download_images_parallel(unique_urls))
    url_to_file = dict(zip(unique_urls, temp_files))
    
    try:
        for image1_url, image2_url in image_pairs:
            try:
                # Get files
                file1 = url_to_file[image1_url]
                file2 = url_to_file[image2_url]
                
                if not file1 or not file2:
                    results.append({
                        "status": "error",
                        "message": "Failed to download one or both images"
                    })
                    continue
                
                # Process images
                result1 = process_image(file1)
                result2 = process_image(file2)
                
                if result1 is None or result2 is None:
                    results.append({
                        "status": "error",
                        "message": "Failed to detect faces in one or both images"
                    })
                    continue
                
                # Calculate similarity
                embedding1, aligned1 = result1
                embedding2, aligned2 = result2
                
                similarity_metrics = calculate_similarity(embedding1, embedding2)
                is_match = similarity_metrics["cosine_similarity"] > settings.VERIFICATION_THRESHOLD
                
                results.append({
                    "status": "success",
                    "is_match": is_match,
                    "confidence": similarity_metrics["cosine_similarity"],
                    "metrics": similarity_metrics
                })
                
            except Exception as e:
                results.append({
                    "status": "error",
                    "message": str(e)
                })
                
    finally:
        # Cleanup temp files
        for file_path in temp_files:
            if file_path:
                cleanup_file(file_path)
    
    return results

def cleanup_file(filepath: str):
    """
    Safely clean up a temporary file.
    
    Args:
        filepath: Path to file to clean up
    """
    try:
        if os.path.exists(filepath):
            # Add small delay to ensure file is not in use
            time.sleep(0.1)
            os.unlink(filepath)
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {str(e)}")
        # If file is still in use, schedule it for deletion on next startup
        try:
            with open(os.path.join(settings.UPLOAD_DIR, 'cleanup.txt'), 'a') as f:
                f.write(f"{filepath}\n")
        except Exception:
            pass