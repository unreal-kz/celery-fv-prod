# Face Verification API with Celery

A high-performance face verification service using FastAPI and Celery for asynchronous processing. The system uses MTCNN for face detection/alignment and FaceNet (InceptionResnetV1) for generating face embeddings.

## Key Features

### Face Processing
- Advanced face detection using MTCNN with optimized thresholds
- Precise face alignment using 5-point landmarks
- High-quality face embeddings using FaceNet
- Strict single-face validation (rejects images with 0 or multiple faces)
- Support for various image formats and sizes

### Verification
- Dual-metric similarity calculation:
  * Cosine similarity for angular distance
  * Euclidean distance for absolute difference
- Adaptive confidence scoring
- Configurable verification thresholds
- Detailed similarity metrics in results

### Performance
- Asynchronous task processing with Celery
- Redis-based result caching
- Embedding cache with TTL
- Batch processing support
- GPU acceleration (when available)

### Monitoring & Reliability
- Comprehensive health check endpoints
- Detailed performance metrics
- Redis connection monitoring
- Cache statistics
- Request tracking

### Security
- Input validation
- Rate limiting
- File size restrictions
- Safe error handling
- CORS support

## Technical Specifications

### Face Detection (MTCNN)
- Input image size: Up to 1920px (auto-scaled)
- Minimum face size: 60px
- Detection thresholds: [0.6, 0.75, 0.9]
- Margin: 5px
- Output size: 160x160px

### Face Recognition (FaceNet)
- Model: InceptionResnetV1
- Pretrained on: VGGFace2
- Embedding size: 512 dimensions
- Normalized output vectors

### Verification Thresholds
- Cosine similarity threshold: 0.7
- Euclidean distance threshold: 0.8
- Minimum confidence threshold: 0.5
- Maximum Euclidean distance: 2.0

## Requirements

- Python 3.8+
- Redis server
- CUDA-capable GPU (optional but recommended)
- 2GB+ RAM
- 10GB+ disk space

## Installation

1. Install Redis server:
   ```bash
   # Windows (using WSL or Docker)
   docker run --name redis -d -p 6379:6379 redis

   # Linux
   sudo apt-get install redis-server
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory:
```env
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_MAX_CONNECTIONS=50
REDIS_POOL_TIMEOUT=5.0

# Performance Settings
MAX_IMAGE_SIZE_MB=10
MAX_IMAGE_DIMENSION=1920
MAX_CONCURRENT_TASKS=10
WORKER_MEMORY_LIMIT_MB=2048

# Verification Thresholds
COSINE_SIMILARITY_THRESHOLD=0.7
EUCLIDEAN_DISTANCE_THRESHOLD=0.8
MAX_EUCLIDEAN_DISTANCE=2.0
MIN_CONFIDENCE_THRESHOLD=0.5

# Task Settings
CELERY_TASK_TIMEOUT=30
MAX_RETRIES=3
RETRY_DELAY=60
```

## Running the Application

1. Start Redis server (if not running):
   ```bash
   redis-server
   ```

2. Start Celery worker:
   ```bash
   celery -A app.worker:celery_app worker --loglevel=info
   ```

3. Start FastAPI server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

### POST /verify
Verify two faces from URLs.

Request:
```bash
curl -X POST "http://localhost:8000/verify" \
     -F "image1_url=http://example.com/face1.jpg" \
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

### GET /status/{task_id}
Check verification result.

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

### POST /verify/batch
Batch verify multiple face pairs.

Request:
```json
{
    "image_pairs": [
        ["http://example.com/face1.jpg", "http://example.com/face2.jpg"],
        ["http://example.com/face3.jpg", "http://example.com/face4.jpg"]
    ]
}
```

### GET /health
Check service health.

### GET /metrics
Get system metrics.

## Error Handling

The API returns appropriate HTTP status codes:

- 400: Bad Request (invalid input)
- 404: Not Found (invalid task ID)
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error

Error Response:
```json
{
    "status": "error",
    "error": "Detailed error message",
    "code": "ERROR_CODE"
}
```

## Monitoring

Access metrics at:
- `/metrics` - System metrics
- `/metrics/redis` - Redis stats
- `/metrics/cache` - Cache performance

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
