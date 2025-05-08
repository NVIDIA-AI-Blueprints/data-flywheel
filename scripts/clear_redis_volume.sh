#!/bin/bash

# Stop any running Redis containers
echo "Stopping Redis containers..."
docker compose -f ./deploy/docker-compose.yaml stop redis

# Remove the Redis volume
echo "Removing Redis volume..."
docker compose -f ./deploy/docker-compose.yaml down -v redis

# Start Redis again
echo "Starting Redis..."
docker compose -f ./deploy/docker-compose.yaml up -d redis

echo "Redis volume cleared and container restarted."
