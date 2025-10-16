#!/bin/bash

# Clear Redis data in Kubernetes

set -e

# Configuration
NAMESPACE="${NAMESPACE:-nv-nvidia-blueprint-data-flywheel}"

echo "🔄 Clearing Redis data in namespace: $NAMESPACE"

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "❌ Namespace '$NAMESPACE' does not exist"
    exit 1
fi

# Retrieve the Redis pod name
REDIS_POD=$(kubectl get pods -l app=df-redis-deployment -o jsonpath='{.items[0].metadata.name}')

# Flush all data from Redis
echo "🔄 Flushing all data from Redis..."
response=$(kubectl exec $REDIS_POD -n $NAMESPACE -- redis-cli FLUSHALL)
echo "🔄 Redis response: $response"

# Check if the response is "OK" (trim whitespace and control characters)
if [[ "$(echo "$response" | tr -d '\r\n ')" != "OK" ]]; then
    echo "❌ Redis data clearing failed"
    exit 1
else
    echo "✅ Redis data cleared successfully!"
fi
