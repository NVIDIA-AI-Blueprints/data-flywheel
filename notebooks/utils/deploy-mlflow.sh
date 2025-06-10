#!/bin/bash

case "$1" in
  deploy)
    docker compose -f mlflow-compose.yml up -d
    ;;
  stop)
    docker compose -f mlflow-compose.yml down
    ;;
  *)
    echo "Usage: $0 {deploy|stop}"
    exit 1
esac