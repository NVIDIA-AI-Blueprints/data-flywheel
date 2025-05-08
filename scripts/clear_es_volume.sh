#!/bin/bash

# Stop any running Elasticsearch containers
echo "Stopping Elasticsearch containers..."
docker compose -f ./deploy/docker-compose.yaml stop elasticsearch

# Remove the Elasticsearch volume
echo "Removing Elasticsearch volume..."
docker compose -f ./deploy/docker-compose.yaml down -v elasticsearch

# Start Elasticsearch again
echo "Starting Elasticsearch..."
docker compose -f ./deploy/docker-compose.yaml up -d elasticsearch

echo "Elasticsearch volume cleared and container restarted."
