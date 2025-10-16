#!/bin/bash

# Clear MongoDB data in Kubernetes

set -e

# Configuration
NAMESPACE="${NAMESPACE:-nv-nvidia-blueprint-data-flywheel}"
MONGODB_DB="flywheel"
MONGODB_CLI="mongosh"

echo "🔄 Clearing MongoDB data in namespace: $NAMESPACE"

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "❌ Namespace '$NAMESPACE' does not exist"
    exit 1
fi

# Retrieve the MongoDB pod name
MONGODB_POD=$(kubectl get pods -l app=df-mongodb-deployment -o jsonpath='{.items[0].metadata.name}')

# Get all databases
current_dbs=$(kubectl exec -it $MONGODB_POD -n $NAMESPACE -- $MONGODB_CLI --eval "show dbs")

# Check if the database exists
if ! echo "$current_dbs" | grep -q "$MONGODB_DB"; then
    echo "❌ Database '$MONGODB_DB' does not exist"
    exit 1
fi

# Use the database
kubectl exec -it $MONGODB_POD -n $NAMESPACE -- $MONGODB_CLI --eval "use $MONGODB_DB"

# List all collections in the database
collections=$(kubectl exec -it $MONGODB_POD -n $NAMESPACE -- $MONGODB_CLI $MONGODB_DB --eval "show collections")
echo "🔄 Collections in the database:"
echo "$collections"

# Drop all collections in the database
echo "🔄 Clearing all collections in the database..."
kubectl exec -it $MONGODB_POD -n $NAMESPACE -- $MONGODB_CLI $MONGODB_DB --eval "db.getCollectionNames().forEach(c => {
    db[c].deleteMany({});
    print('🔄 ' + c + ' cleared');
})"

# Check if the request was successful based on the response (should be {"ok": 1})
echo "✅ MongoDB data cleared successfully!" 