#!/bin/bash

set -e

# Configuration
NAMESPACE="${NAMESPACE:-nv-nvidia-blueprint-data-flywheel}"

echo "🔄 Clearing Elasticsearch data in namespace: $NAMESPACE"

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "❌ Namespace '$NAMESPACE' does not exist"
    exit 1
fi

# Retrieve the Elasticsearch pod name
ES_POD=$(kubectl get pods -l app=df-elasticsearch-deployment -o jsonpath='{.items[0].metadata.name}')

# Execute the http request to the elasticsearch pod
response=$(kubectl exec -it "$ES_POD" -n "$NAMESPACE" -- curl -X DELETE "http://localhost:9200/_all")
echo "🔄 Response: $response"

# Check if the request was successful based on the response (should be {"acknowledged: true"})
if [[ "$response" == *"{\"acknowledged\":true}"* ]]; then
    echo "✅ Elasticsearch data cleared successfully!"
else
    echo "❌ Failed to clear Elasticsearch data"
    exit 1
fi
