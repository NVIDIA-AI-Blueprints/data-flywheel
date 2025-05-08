# API Service

A FastAPI-based service that provides a simple job queue system using Celery for asynchronous task processing.

## Features

- RESTful API endpoints for job management
- Asynchronous task processing using Celery
- Job status tracking and result retrieval
- Redis-based message broker for Celery

## Prerequisites

- Python 3.10 or higher
- Redis server (for Celery message broker)
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. Clone the repository
2. Navigate to the API service directory:
   ```bash
   cd services/api
   ```
3. Install dependencies:
   ```bash
   uv sync
   ```

## Development Setup

For development, install additional dependencies:
```bash
uv sync --dev
```

## Running the Service

1. Start the Redis server (if not already running)
2. Start the Celery worker:
   ```bash
   celery -A src.tasks worker --loglevel=info
   ```
3. Start the FastAPI server:
   ```bash
   uvicorn src.api:app --reload
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### Create a Job
- **POST** `/jobs`
- Request body:
  ```json
  {
    "message": "Your message here"
  }
  ```
- Response:
  ```json
  {
    "job_id": "task-id",
    "status": "queued",
    "message": "Your message here"
  }
  ```

### List All Jobs
- **GET** `/jobs`
- Response:
  ```json
  {
    "jobs": {
      "task-id": {
        "id": "task-id",
        "name": "message"
      }
    }
  }
  ```

### Get Job Status
- **GET** `/jobs/{job_id}`
- Response (pending):
  ```json
  {
    "job_id": "task-id",
    "status": "pending"
  }
  ```
- Response (completed):
  ```json
  {
    "job_id": "task-id",
    "status": "completed",
    "result": "echoed message"
  }
  ```

## Docker Support

The service includes a Dockerfile for containerized deployment. To build and run:

```bash
docker build -f Dockerfile.api -t api-service .
docker run -p 8000:8000 api-service
```

## Development Tools

- **Flower**: Celery monitoring tool (available in dev dependencies)
- **Watchdog**: File system event monitoring for development

## License

[Add your license information here]
