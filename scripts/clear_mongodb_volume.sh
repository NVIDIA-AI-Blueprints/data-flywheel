#!/bin/bash

# Stop any running MongoDB containers
echo "Stopping MongoDB containers..."
docker compose -f ./deploy/docker-compose.yaml stop mongodb

# Remove the MongoDB volume
echo "Removing MongoDB volume..."
docker compose -f ./deploy/docker-compose.yaml down -v mongodb

# Start MongoDB again
echo "Starting MongoDB..."
docker compose -f ./deploy/docker-compose.yaml up -d mongodb

echo "MongoDB volume cleared and container restarted."
