# Face Verification API with Celery

This project implements a face verification API using FastAPI and Celery for asynchronous processing. It uses MTCNN for face detection and FaceNet for generating face embeddings.

## Features

- Face detection using MTCNN
- Face embedding generation using FaceNet (InceptionResnetV1)
- Asynchronous processing with Celery and Redis
- GPU support when available
- Frontal face checking
- Similarity scoring using both Euclidean distance and Cosine similarity

## Requirements

- Python 3.8+
- Redis server
- CUDA-capable GPU (optional but recommended)

## Installation

1. Install Redis server
2. Create a virtual environment:
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

Create a `.env` file in the root directory with the following settings:
```env
REDIS_URL=redis://localhost:6379/0
CELERY_TASK_TIMEOUT=30
```

## Running the Application

1. Start Redis server:
   ```bash
   redis-server
   ```

2. Start Celery worker:
   ```bash
   celery -A app.worker:celery_app worker --loglevel=info --pool=solo
   ```

3. Start FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Endpoints

### POST /verify
Upload two face images for verification.

Request:
- `image1`: First image file
- `image2`: Second image file

Response:
```json
{
    "task_id": "face_verification_abc123"
}
```

### GET /status/{task_id}
Check the status of a verification task.

Response:
```json
{
    "task_id": "face_verification_abc123",
    "status": "SUCCESS",
    "result": {
        "verified": true,
        "verification_score": 0.85,
        "euclidean_distance": 0.5,
        "cosine_similarity": 0.95,
        "face1_frontal": true,
        "face2_frontal": true,
        "confidence1": 0.99,
        "confidence2": 0.98
    }
}
```

## Error Handling

The API includes comprehensive error handling for:
- Invalid image formats
- Missing or multiple faces
- Non-frontal faces
- Processing failures

## Performance Considerations

- Uses thread-local storage for models
- GPU memory optimization
- Worker task limits to prevent memory leaks
- Automatic retries for failed tasks
