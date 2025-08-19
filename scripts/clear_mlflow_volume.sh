#!/bin/bash

# Stop any running MLflow containers
echo "Stopping MLflow containers..."
docker compose -f ./deploy/docker-compose.yaml --profile mlflow stop mlflow

# Remove the MLflow volume
echo "Removing MLflow volume..."
docker compose -f ./deploy/docker-compose.yaml --profile mlflow down -v mlflow

# Start MLflow again
echo "Starting MLflow..."
docker compose -f ./deploy/docker-compose.yaml --profile mlflow up -d mlflow

echo "MLflow volume cleared and container restarted."
