#!/bin/bash

# Run w/ dev docker compose so we get the UIs
# MongoDB logs too noisy
docker compose -f ./deploy/docker-compose.yaml -f ./deploy/docker-compose.dev.yaml down && \
  docker compose -f ./deploy/docker-compose.yaml -f ./deploy/docker-compose.dev.yaml up --build \
  --no-attach mongodb --no-attach elasticsearch --no-attach kibana
