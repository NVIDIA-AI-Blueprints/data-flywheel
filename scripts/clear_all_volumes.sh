#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Clear Elasticsearch volume
echo "Clearing Elasticsearch volume..."
"$SCRIPT_DIR/clear_es_volume.sh"

# Clear Redis volume
echo "Clearing Redis volume..."
"$SCRIPT_DIR/clear_redis_volume.sh"

# Clear MongoDB volume
echo "Clearing MongoDB volume..."
"$SCRIPT_DIR/clear_mongodb_volume.sh"

echo "All volumes have been cleared and services restarted."
