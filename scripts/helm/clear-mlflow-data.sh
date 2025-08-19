#!/bin/bash

# Clear MLflow data in Kubernetes

set -e

# Configuration
NAMESPACE="${NAMESPACE:-nv-nvidia-blueprint-data-flywheel}"
MLFLOW_DEPLOYMENT="df-mlflow-deployment"

echo "🔄 Clearing MLflow data in namespace: $NAMESPACE"

# Check if namespace exists
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
    echo "❌ Namespace '$NAMESPACE' does not exist"
    exit 1
fi

# Check if MLflow is enabled/deployed
if ! kubectl get deployment "$MLFLOW_DEPLOYMENT" -n "$NAMESPACE" &>/dev/null; then
    echo "❌  MLflow deployment not found. It might be disabled or not deployed yet. Exiting..."
    exit 0
fi

# Retrieve the MLflow pod name
MLFLOW_POD=$(kubectl get pods -n $NAMESPACE -l app=df-mlflow-deployment -o jsonpath='{.items[0].metadata.name}')

# Delete all files and folder in path /mlruns of the MLflow pod
echo "🔄  Removing all contents from /mlruns directory..."
kubectl exec $MLFLOW_POD -n $NAMESPACE -- sh -c 'find /mlruns -mindepth 1 -delete'

echo "✅ /mlruns successfully cleared"
