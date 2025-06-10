#!/bin/bash

# Data Flywheel Blueprint - Cleanup Script Wrapper
# This script provides an easy way to run the cleanup process

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================================"
echo "Data Flywheel Blueprint - Cleanup Running Resources"
echo "============================================================"
echo

# Change to project root
cd "$PROJECT_ROOT"

# Check if docker compose is running
echo "Checking docker compose status..."
if docker compose -f deploy/docker-compose.yaml ps --format json | grep -q .; then
    echo "ERROR: Docker compose services are still running!"
    echo "Please stop them first with:"
    echo "  cd deploy && docker compose down"
    echo
    exit 1
fi

echo "Docker compose services are down. Proceeding with cleanup..."
echo

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv is not available. Please ensure uv is installed."
    echo "You can install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  or follow instructions at: https://docs.astral.sh/uv/getting-started/installation/"
    echo
    exit 1
fi

# Check if required Python packages are available
echo "Checking Python dependencies..."
if ! uv run python -c "import pymongo, bson" 2>/dev/null; then
    echo "ERROR: Required Python packages (pymongo) are not installed."
    echo "Installing required dependencies..."
    uv add pymongo || {
        echo "Failed to install pymongo. You can try manually with:"
        echo "  uv add pymongo"
        echo
        exit 1
    }
fi

# Run the cleanup script
echo "Starting cleanup process..."
echo
exec uv run python src/scripts/cleanup_running_resources.py
