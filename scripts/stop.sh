#!/bin/bash

docker compose -f ./deploy/docker-compose.yaml -f ./deploy/docker-compose.dev.yaml down
