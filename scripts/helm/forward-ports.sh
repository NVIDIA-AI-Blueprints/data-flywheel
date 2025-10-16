#!/bin/bash

# Data Flywheel Port Forwarding Script
# Usage: ./forward-ports.sh [<service_name> <desired_port> ...] [-h|--help]

set -e

# Color codes
RED='\033[0;31m'
ORANGE='\033[0;33m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Associative array of allowed services with their default ports
declare -A ALLOWED_SERVICES=(
    ["df-api-service"]="8000"
    ["df-elasticsearch-service"]="9200"
    ["df-mlflow-service"]="5000"
    ["df-mongodb-service"]="27017"
    ["df-redis-service"]="6379"
    ["df-kibana-service"]="5601"
    ["df-flower-service"]="5555"
)

# Function to display help
show_help() {
    cat << EOF
Data Flywheel Port Forwarding Script

USAGE:
    ./forward-ports.sh [<service_name> <desired_port> ...]
    ./forward-ports.sh [-h|--help]

DESCRIPTION:
    This script forwards ports for specified Kubernetes services of Data Flywheel using kubectl port-forward.
    You can specify multiple service/port pairs as consecutive arguments.

ARGUMENTS:
    <service_name>    Name of the Kubernetes service to forward
    <desired_port>    Local port number to forward to

OPTIONS:
    -h, --help        Show this help message and exit

ALLOWED SERVICES:
    - df-api-service             (original: 8000)
    - df-elasticsearch-service   (original: 9200)
    - df-mlflow-service          (original: 5000)
    - df-redis-service           (original: 6379)
    - df-mongodb-service         (original: 27017)
    - df-redis-service           (original: 6379)
    - df-kibana-service          (original: 5601)

EXAMPLES:
    ./forward-ports.sh --help
    ./forward-ports.sh df-api-service 8000
    ./forward-ports.sh df-api-service 8000 df-mlflow-service 5000
    ./forward-ports.sh df-api-service 8001 df-mlflow-service 5001 (change port numbers as you wish)

NOTES:
    - Services must be from the allowed list above
    - Each port forwarding runs in the background
    - Arguments must be in pairs: service_name followed by port_number

EOF
}

# Function to check if service is allowed
is_service_allowed() {
    local service="$1"
    [[ -n "${ALLOWED_SERVICES[$service]}" ]]
}

# Function to validate port number
is_valid_port() {
    local port="$1"
    if [[ "$port" =~ ^[0-9]+$ ]] && [ "$port" -ge 1 ] && [ "$port" -le 65535 ]; then
        return 0
    fi
    return 1
}

is_port_free() {
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -q ":$1$" && return 1 || return 0;
}

# Function to forward port for a service
forward_port() {
    local service="$1"
    local port="$2"

    # Get the original port for the service
    local original_port="${ALLOWED_SERVICES[$service]}"
    
    # Start port forwarding with output capture using stdbuf to disable buffering
    {
        stdbuf -oL -eL kubectl port-forward service/"$service" "$port:$original_port" 2>&1 | while IFS= read -r line; do
            if [[ "$line" =~ ^Forwarding\ from ]]; then
                echo -e "${GREEN}[success] $line${NC}"
            elif [[ "$line" =~ ^Error\ from\ server ]]; then
                echo -e "${RED}[error] $line${NC}"
            elif [[ "$line" =~ ^error|^Error ]]; then
                echo -e "${RED}[error] $line${NC}"
            else
                echo -e "${YELLOW}$line${NC}"
            fi
        done
    } &
    local pid=$!
    
    echo "Port forwarding started for $service on port $port"
    
    # Store PID for cleanup
    PIDS+=($pid)
}

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "Stopping all port forwarding processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "Stopped process $pid"
        fi
    done
    exit 0
}

# Array to store background process PIDs
PIDS=()

# Set up trap for cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Main script logic
main() {
    # Check for help flag
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_help
        exit 0
    fi
    
    # Check if no arguments provided
    if [[ $# -eq 0 ]]; then
        echo -e "${RED}[error] No arguments provided.${NC}"
        echo -e "${ORANGE}[ Use -h or --help for usage information. ]${NC}"
        exit 1
    fi
    
    # Check if number of arguments is even (pairs of service and port)
    if [[ $(($# % 2)) -ne 0 ]]; then
        echo -e "${RED}[error] Invalid number of arguments.${NC}"
        echo -e "${ORANGE}[ Arguments must be in pairs: <service_name> <desired_port> ]${NC}"
        echo -e "${ORANGE}[ Use -h or --help for usage information. ]${NC}"
        exit 1
    fi
    
    # Process arguments in pairs
    while [[ $# -gt 0 ]]; do
        local service="$1"
        local port="$2"
        shift 2
        
        # Validate service name
        if ! is_service_allowed "$service"; then
            echo -e "${RED}[error] Service '$service' is not in the allowed list.${NC}"
            echo -e "${ORANGE}Allowed services with original ports:${NC}"
            for svc in "${!ALLOWED_SERVICES[@]}"; do
                echo -e "${ORANGE}  - $svc (original: ${ALLOWED_SERVICES[$svc]})${NC}"
            done
            exit 1
        fi
        
        # Validate port number
        if ! is_valid_port "$port"; then
            echo -e "${RED}[error] Invalid port number '$port'.${NC}"
            echo -e "${ORANGE}[ Port must be a number between 1 and 65535. ]${NC}"
            exit 1
        fi
        
        # Check if port is free
        if ! is_port_free "$port"; then
            echo -e "${RED}[error] Port '$port' is already in use.${NC}"
            exit 1
        fi

        # Forward the port
        forward_port "$service" "$port"
    done
    
    # Check if any services were processed
    if [[ ${#PIDS[@]} -eq 0 ]]; then
        echo -e "${RED}[error] No valid service/port pairs found.${NC}"
        echo -e "${ORANGE}[ Use -h or --help for usage information. ]${NC}"
        exit 1
    fi
    
    echo ""
    echo -e "${ORANGE}👉 Press Ctrl+C to stop all port forwarding${NC}"
    
    # Wait for all background processes
    wait
}

# Run main function with all arguments
main "$@"
