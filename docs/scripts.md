# Scripts Directory

Efficiently manage and maintain the Data Flywheel Blueprint application using the following scripts. These utilities help automate cleanup, manage resources, and streamline your workflow.

## Prerequisites

- Docker and Docker Compose must be installed.
- The `uv` package manager must be installed.
- Always stop all Docker Compose services before running any cleanup or resource-modifying scripts.
- Scripts assume you are running in the project root directory unless otherwise specified.

---

## Scripts

### Cleanup Running Resources Script

The `src/scripts/cleanup_running_resources.py` is a comprehensive cleanup script that removes all running resources from the Data Flywheel Blueprint application.

**What it Does:**

1. **Finds all running flywheel runs** from MongoDB that are in `PENDING` or `RUNNING` status.
2. **Identifies running NVIDIA Inference Microservices (NIMs)**—NIMs—with `RUNNING` deployment status.
3. **Cleans up associated resources** for each NIM:
   - Deletes all evaluation jobs
   - Cancels running customization jobs
   - Shuts down NIM deployments
4. **Shuts down LLM judge** deployments (local only)
5. **Marks all resources as cancelled** in the database

#### Limitations & Warnings

> **Warning:** This script will permanently delete evaluation jobs and shut down deployments. Cancelled customization jobs may leave customized models in storage; these are not deleted by this script.
>
> The script only affects resources marked as "running" in the database. Remote LLM judges are not shut down, as they are external services.

#### Usage

##### Option 1: Use the Wrapper Script (Recommended)

```bash
# Stop Docker Compose first (if running)
cd deploy
docker compose down

# Run the cleanup script using the wrapper
cd ..
./scripts/cleanup_resources.sh
```

##### Option 2: Run Directly with uv

```bash
# Stop Docker Compose first (if running)
cd deploy
docker compose down

# Run the cleanup script directly
cd ..
uv run python src/scripts/cleanup_running_resources.py
```

#### Safety Features

- **Checks Docker Compose status**: Refuses to run if services are still running
- **Dependency management**: Automatically installs required packages with `uv`
- **Temporary service startup**: Only starts MongoDB temporarily to get information
- **Confirmation prompt**: Asks for user confirmation before proceeding
- **Error handling**: Collects and reports all errors encountered during cleanup
- **Graceful shutdown**: Always stops temporary services even if errors occur

#### What Happens During Execution

1. Checks that Docker Compose services are down
2. Verifies `uv` is installed and installs dependencies if needed
3. Prompts for user confirmation
4. Starts MongoDB service temporarily
5. Connects to MongoDB and queries for running resources
6. For each running flywheel run:
   - Finds associated NIMs with RUNNING status
   - Cancels customization jobs
   - Deletes evaluation jobs
   - Shuts down NIM deployments
   - Shuts down LLM judge (if local)
   - Marks all resources as cancelled in the database
7. Stops MongoDB service
8. Reports cleanup results

#### Error Handling

The script is robust and continues cleanup even if individual operations fail:

- Collects and reports errors at the end
- Failing to clean up one resource doesn't stop the cleanup of others
- Always stops the MongoDB service, even if cleanup fails
- Logs all errors with detailed information

#### Example Output

```sh
============================================================
Data Flywheel Blueprint - Cleanup Running Resources
============================================================

Checking Docker Compose status...
Docker Compose services are down. Proceeding with cleanup...

Checking Python dependencies...
Starting cleanup process...

This will clean up all running resources. Continue? (y/N): y
2025-01-27 10:30:00 - cleanup_script - INFO - Starting cleanup of running resources...
2025-01-27 10:30:00 - cleanup_script - INFO - Starting MongoDB service temporarily...
2025-01-27 10:30:10 - cleanup_script - INFO - Connected to MongoDB
2025-01-27 10:30:10 - cleanup_script - INFO - Finding running flywheel runs...
2025-01-27 10:30:10 - cleanup_script - INFO - Found 2 running flywheel runs
2025-01-27 10:30:10 - cleanup_script - INFO - Cleaning up flywheel run 507f1f77bcf86cd799439011
2025-01-27 10:30:10 - cleanup_script - INFO - Found 1 running NIMs for flywheel run 507f1f77bcf86cd799439011
2025-01-27 10:30:10 - cleanup_script - INFO - Processing NIM llama-3.1-8b-instruct (ID: 507f1f77bcf86cd799439012)
2025-01-27 10:30:10 - cleanup_script - INFO - Found 0 customizations for NIM 507f1f77bcf86cd799439012
2025-01-27 10:30:10 - cleanup_script - INFO - Found 2 evaluations for NIM 507f1f77bcf86cd799439012
2025-01-27 10:30:11 - cleanup_script - INFO - Deleted evaluation job eval_job_123
2025-01-27 10:30:11 - cleanup_script - INFO - Deleted evaluation job eval_job_124
2025-01-27 10:30:12 - cleanup_script - INFO - Shutdown NIM deployment for llama-3.1-8b-instruct
2025-01-27 10:30:12 - cleanup_script - INFO - LLM judge is remote, no shutdown needed
2025-01-27 10:30:12 - cleanup_script - INFO - Marked all resources as cancelled for flywheel run 507f1f77bcf86cd799439011
2025-01-27 10:30:15 - cleanup_script - INFO - Stopping MongoDB service...
2025-01-27 10:30:20 - cleanup_script - INFO - MongoDB service stopped
2025-01-27 10:30:20 - cleanup_script - INFO - Cleanup completed successfully with no errors!

Cleanup process completed!
```

### `cleanup_resources.sh`

A robust shell wrapper that ensures a safe environment before running the Python cleanup script. It:

- Checks that all Docker Compose services are stopped before proceeding.
- Verifies that the `uv` Python package manager is installed, and provides installation instructions if not.
- Checks for required Python dependencies (`pymongo`, `bson`), and installs them automatically if missing.
- Runs the main cleanup script (`src/scripts/cleanup_running_resources.py`) using `uv` for dependency isolation.
- Exits with clear error messages if any prerequisite is not met.

### `generate_openapi.py`

Python script to generate the OpenAPI specification for the API.

- Imports the FastAPI app and writes the OpenAPI schema to `openapi.json` (or a user-specified path).
- Validates the output path for safety.
- Can be run as `python scripts/generate_openapi.py [output_path.json]`.

### `run.sh`

- Stops any running containers, then starts the main application stack using Docker Compose.
- Builds images as needed.
- Runs MongoDB in detached mode without attaching logs, to reduce log noise.

### `run-dev.sh`

- Stops any running containers, then starts the application stack with both the main and development Docker Compose files.
- Builds images as needed.
- Runs MongoDB, Elasticsearch, and Kibana in detached mode (no logs attached).
- Ensures development UIs are available.

### `stop.sh`

- Stops all running containers for both the main and development Docker Compose files.

### Volume Cleanup Scripts

- `clear_es_volume.sh`, `clear_redis_volume.sh`, `clear_mongodb_volume.sh`—Each script:
  - Stops the relevant service container (Elasticsearch, Redis, or MongoDB).
  - Removes the associated Docker volume to clear all stored data.
  - Restarts the service container to ensure the service is running with a fresh, empty volume.
  - Prints status messages for each step.
- `clear_all_volumes.sh`—A convenience script to clear all persistent data volumes used by the application. It sequentially calls the above scripts and restarts all services.

### `check_requirements.sh`

A script to ensure your `requirements.txt` is in sync with your `pyproject.toml`:

- Uses `uv` to generate a temporary list of installed packages.
- Compares this list to `requirements.txt`.
- If out of sync, prints a diff and instructions to update.
- Exits with an error if not up to date, otherwise confirms success.

### `quick-test.sh`

A minimal script to quickly verify that the API is running and responsive:

- Sends a POST request to `http://localhost:8001/jobs` with a test payload.
- Useful for smoke-testing the local API after startup.
