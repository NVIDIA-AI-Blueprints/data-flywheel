#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'

# === Debug Settings ===
# Show exit codes and command context for all failures
trap 'echo "Error on line $LINENO: Command failed with exit code $?" >&2' ERR
# Print each command before execution
PS4='+(${BASH_SOURCE}:${LINENO}): ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
#set -x

# === Config ===
NAMESPACE="default"
REQUIRED_DISK_GB=200
REQUIRED_GPUS=2
NGC_API_KEY="${NGC_API_KEY:-}"
HELM_CHART_URL="https://helm.ngc.nvidia.com/nvidia/nemo-microservices/charts/nemo-microservices-helm-chart-25.4.0.tgz"
ADDITIONAL_VALUES_FILES=(demo-values.yaml)

# === Utility ===
log() { echo -e "\033[1;32m[INFO]\033[0m $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $*"; }
err() { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }
die() { err "$*"; exit 1; }

# Check if running as root
is_root() {
  [[ $EUID -eq 0 ]] || [[ $(id -u) -eq 0 ]]
}

# Run a command with sudo if not root
maybe_sudo() {
  if is_root; then
    "$@"
  else
    sudo "$@"
  fi
}

# === Dependency Installation ===
# Install a dependency if it's missing - optimized for Linux
install_dependency() {
  local tool=$1

  log "Installing $tool using verified methods..."

  case "$tool" in
    minikube)
      log "Installing minikube via direct download..."
      curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
      maybe_sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64
      ;;

    kubectl)
      log "Installing kubectl via direct download..."
      curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
      maybe_sudo mv kubectl /usr/local/bin/
      maybe_sudo chmod +x /usr/local/bin/kubectl
      ;;

    helm)
      log "Installing Helm via direct download..."
      curl -fsSL -o helm.tar.gz https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz
      tar -zxvf helm.tar.gz
      maybe_sudo mv linux-amd64/helm /usr/local/bin/helm
      rm -rf helm.tar.gz linux-amd64
      ;;

    huggingface-cli)
      log "Installing huggingface_hub via pip..."
      if command -v pip3 >/dev/null 2>&1; then
        pip3 install --user --upgrade huggingface_hub
      elif command -v pip >/dev/null 2>&1; then
        pip install --user --upgrade huggingface_hub
      else
        # Try to install pip first
        log "pip not found, attempting to install python3-pip..."
        maybe_sudo apt-get update && maybe_sudo apt-get install -y python3-pip
        pip3 install --user --upgrade huggingface_hub
      fi

      # Add ~/.local/bin to PATH immediately after installation
      export PATH="$HOME/.local/bin:$PATH"
      log "Added $HOME/.local/bin to PATH"
      ;;

    jq)
      log "Installing jq..."
      maybe_sudo apt-get update && maybe_sudo apt-get install -y jq || {
        # If apt fails, try direct download
        log "apt installation failed, trying direct download..."
        local tmp_dir=$(mktemp -d)
        curl -L https://github.com/jqlang/jq/releases/download/jq-1.7/jq-linux64 -o "$tmp_dir/jq"
        chmod +x "$tmp_dir/jq"
        maybe_sudo mv "$tmp_dir/jq" /usr/local/bin/jq
        rm -rf "$tmp_dir"
      }
      ;;

    yq)
      log "Installing yq via direct download..."
      curl -L https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -o yq
      chmod +x yq
      maybe_sudo mv yq /usr/local/bin/
      ;;

    *)
      err "No installation method available for $tool."
      return 1
      ;;
  esac

  # Verify installation
  if command -v "$tool" >/dev/null 2>&1; then
    log "$tool installed successfully."
    return 0
  else
    err "Failed to install $tool. Please try installing it manually."
    return 1
  fi
}

show_help() {
  cat << EOF
Usage: $0 [OPTIONS]

Setup and deploy NeMo microservices on Minikube.

Options:
  --helm-chart-url URL    Override the default helm chart URL
                          Default: $HELM_CHART_URL
  --values-file FILE      Path to a values file (can be specified multiple times). Not required.
  --help                  Show this help message

Environment Variables:
  NGC_API_KEY            NVIDIA NGC API key for authentication
                         Can be set in environment or will be prompted if not set

Requirements:
  - NVIDIA Container Toolkit v1.16.2 or higher
  - NVIDIA GPU Driver 560.35.03 or higher
  - At least $REQUIRED_GPUS A100 80GB, H100 80GB, RTX 6000, or RTX 5880 GPUs
  - At least $REQUIRED_DISK_GB GB free disk space
  - minikube v1.33.0 or higher
  - Docker v27.0.0 or higher
  - kubectl
  - helm
  - huggingface-cli
  - jq
  - yq

Example:
  $0 --values-file /path/to/values1.yaml --values-file /path/to/values2.yaml
EOF
}

# === Argument Parsing ===
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      --helm-chart-url)
        HELM_CHART_URL="$2"
        shift 2
        ;;
      --values-file)
        ADDITIONAL_VALUES_FILES+=("$2")
        shift 2
        ;;
      --help)
        show_help
        exit 0
        ;;
      *)
        err "Unknown argument: $1"
        show_help
        exit 1
        ;;
    esac
  done
}

# === Diagnostic Functions ===
collect_pod_diagnostics() {
  local pod=$1
  local namespace=$2
  local err_dir=$3
  local pod_dir="$err_dir/$pod"

  mkdir -p "$pod_dir"

  # Collect pod logs
  log "Collecting logs for pod $pod..."
  kubectl logs --all-containers "$pod" -n "$namespace" > "$pod_dir/logs.txt" 2>&1 || true
  kubectl logs --all-containers "$pod" -n "$namespace" --previous > "$pod_dir/logs.previous.txt" 2>&1 || true

  # Collect pod description
  log "Collecting pod description for $pod..."
  kubectl describe pod "$pod" -n "$namespace" > "$pod_dir/describe.txt" 2>&1 || true

  # Collect pod events
  log "Collecting events for pod $pod..."
  kubectl get events --field-selector involvedObject.name="$pod" -n "$namespace" > "$pod_dir/events.txt" 2>&1 || true

  # Check for image pull issues
  if kubectl describe pod "$pod" -n "$namespace" | grep -q "ImagePullBackOff\|ErrImagePull"; then
    log "Detected image pull issues for pod $pod"
    kubectl describe pod "$pod" -n "$namespace" | grep -A 10 "ImagePullBackOff\|ErrImagePull" > "$pod_dir/image_pull_issues.txt" 2>&1 || true
  fi

  # Collect container status
  log "Collecting container status for pod $pod..."
  kubectl get pod "$pod" -n "$namespace" -o json | jq '.status.containerStatuses' > "$pod_dir/container_status.json" 2>&1 || true
}

check_image_pull_secrets() {
  local namespace=$1
  log "Verifying image pull secrets..."

  # Check if the secret exists
  if ! kubectl get secret nvcrimagepullsecret -n "$namespace" &>/dev/null; then
    err "Image pull secret 'nvcrimagepullsecret' not found in namespace $namespace"
    return 1
  fi

  # Check if the secret is properly configured
  if ! kubectl get secret nvcrimagepullsecret -n "$namespace" -o json | jq -e '.data[".dockerconfigjson"]' &>/dev/null; then
    err "Image pull secret 'nvcrimagepullsecret' is not properly configured"
    return 1
  fi

  log "Image pull secrets verified successfully"
  return 0
}

# === Phase 0: Preflight Checks ===
check_prereqs() {
  log "Checking system requirements..."

  # Check jq
  if ! command -v jq >/dev/null; then
    warn "jq is required but not found. Attempting to install..."
    if ! install_dependency "jq"; then
      die "Please install jq manually and try again"
    fi
  fi

  # Check yq
  if ! command -v yq >/dev/null; then
    warn "yq is required but not found. Attempting to install..."
    if ! install_dependency "yq"; then
      die "Please install yq manually and try again"
    fi
  fi

  # Check NVIDIA Container Toolkit version
  if command -v nvidia-ctk >/dev/null 2>&1; then
    nvidia_ctk_version=$(nvidia-ctk --version 2>/dev/null | head -n1 | awk '{print $6}')
    if [[ "$nvidia_ctk_version" == "0.0.0" ]]; then
      die "nvidia-ctk is installed but version check failed. Please ensure it's properly installed."
    fi
    if [[ "$(printf '%s\n' "1.16.2" "$nvidia_ctk_version" | sort -V | head -n1)" != "1.16.2" ]]; then
      warn "NVIDIA Container Toolkit v1.16.2 or higher is recommended. Found: $nvidia_ctk_version"
    fi
    log "NVIDIA Container Toolkit version: $nvidia_ctk_version"
  else
    die "nvidia-ctk is not installed. Please install it first."
  fi

  # Check GPU models
  valid_gpus=0
  while IFS= read -r gpu; do
    log "Checking GPU: $gpu"
    if [[ "$gpu" == *"A100"*"80GB"* ]] || [[ "$gpu" == *"H100"*"80GB"* ]] || [[ "$gpu" == *"6000"* ]] || [[ "$gpu" == *"5880"* ]]; then
      log "Found valid GPU: $gpu"
      valid_gpus=$((valid_gpus + 1))
    fi
  done < <(nvidia-smi --query-gpu=name --format=csv,noheader)
  log "Total valid GPUs found: $valid_gpus"
  (( valid_gpus >= REQUIRED_GPUS )) || warn "At least $REQUIRED_GPUS A100 80GB, H100 80GB, RTX 6000, or RTX 5880 GPUs are required. We could not confirm that you have the correct set of GPUs. This could be a script error. Please check that you have the right set of GPUs. Found: $valid_gpus"

  # Check filesystem type
  filesystem_type=$(df -T / | awk 'NR==2 {print $2}')
  if [[ "$filesystem_type" != "ext4" ]]; then
    warn "Warning: Filesystem type is $filesystem_type. EXT4 is recommended for proper file locking support."
  fi

  # Check free disk space
  free_space_gb=$(df / | awk 'NR==2 {print int($4 / 1024 / 1024)}')
  (( free_space_gb >= REQUIRED_DISK_GB )) || warn "Warning: Your root filesystem does not have enough free disk space.\n\
  This may not be a problem if you have other filesystems mounted,\n\
  but you should check the output of df (below) to ensure that you\n\
  have enough space for images and PVCs. Required: ${REQUIRED_DISK_GB} GB\n`df -kP`"

  # Check minikube version
  if ! command -v minikube >/dev/null; then
    warn "minikube is required but not found. Attempting to install..."
    if ! install_dependency "minikube"; then
      die "Please install minikube v1.33.0 or higher manually and try again"
    fi
  fi
  minikube_version=$(minikube version --short 2>/dev/null | cut -d'v' -f2)
  if [[ "$(printf '%s\n' "1.33.0" "$minikube_version" | sort -V | head -n1)" != "1.33.0" ]]; then
    err "minikube v1.33.0 or higher is required. Found: $minikube_version"
    warn "Attempting to upgrade minikube..."
    if ! install_dependency "minikube"; then
      die "Please upgrade minikube manually and try again"
    fi
    # Recheck version after installation
    minikube_version=$(minikube version --short 2>/dev/null | cut -d'v' -f2)
    if [[ "$(printf '%s\n' "1.33.0" "$minikube_version" | sort -V | head -n1)" != "1.33.0" ]]; then
      die "Failed to upgrade minikube to required version. Please upgrade manually."
    fi
  fi

  # Check Docker version
  docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null)
  if [[ "$(printf '%s\n' "27.0.0" "$docker_version" | sort -V | head -n1)" != "27.0.0" ]]; then
    die "Docker v27.0.0 or higher is required. Found: $docker_version"
  fi

  # Check kubectl version
  if ! command -v kubectl >/dev/null; then
    warn "kubectl is required but not found. Attempting to install..."
    if ! install_dependency "kubectl"; then
      die "Please install kubectl manually and try again"
    fi
  fi
  kubectl_version=$(kubectl version --client -o json | jq -r '.clientVersion.gitVersion' | sed 's/^v//')
  if [[ -z "$kubectl_version" ]]; then
    die "Could not determine kubectl version"
  fi

  # Check helm version
  if ! command -v helm >/dev/null; then
    warn "helm is required but not found. Attempting to install..."
    if ! install_dependency "helm"; then
      die "Please install helm manually and try again"
    fi
  fi
  helm_version=$(helm version --template='{{.Version}}' | sed 's/^v//')
  if [[ -z "$helm_version" ]]; then
    die "Could not determine helm version"
  fi

  # Check huggingface-cli
  if ! command -v huggingface-cli >/dev/null; then
    # Check in ~/.local/bin explicitly as a fallback
    if [[ -x "$HOME/.local/bin/huggingface-cli" ]]; then
      log "huggingface-cli found in $HOME/.local/bin, adding to PATH"
      export PATH="$HOME/.local/bin:$PATH"
    else
      warn "huggingface-cli is required but not found. Attempting to install..."
      if ! install_dependency "huggingface-cli"; then
        die "Please install huggingface-cli manually and try again"
      fi
    fi
  fi

  log "All prerequisites are met."
}

# === Phase 1: Minikube Setup ===
start_minikube() {
  log "Checking Minikube status..."
  if minikube status &>/dev/null; then
    die "Minikube is already running. For this script we need a clean installation of minikube. Please delete your current minikube cluster by running 'minikube delete'"
  fi

  log "Starting Minikube with GPU support..."

  # Add --force flag if running as root
  local extra_args=""
  if is_root; then
    extra_args="--force"
    log "Running as root, adding --force flag to minikube command"
  fi

  minikube start \
    --driver=docker \
    --container-runtime=docker \
    --cpus=no-limit \
    --memory=no-limit \
    --gpus=all \
    $extra_args

  log "Enabling ingress addon..."
  minikube addons enable ingress
}

# === Phase 2: NGC Auth and Helm Setup ===
setup_ngc_and_helm() {
  [[ -n "$NGC_API_KEY" ]] || read -rsp "Enter your NGC API Key: " NGC_API_KEY && echo

  export NGC_API_KEY

  log "Creating Kubernetes secrets for NGC access..."
  kubectl create secret docker-registry nvcrimagepullsecret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password="$NGC_API_KEY"

  kubectl create secret generic ngc-api \
    --from-literal=NGC_API_KEY="$NGC_API_KEY"
}

# === Phase 3: Deploy Helm Chart ===
download_helm_chart() {
  log "Downloading NeMo microservices Helm chart..."


  # Check if demo-values.yaml exists, create if missing
  if [[ ! -f "demo-values.yaml" ]]; then
    log "Creating demo-values.yaml with default content..."
    cat > demo-values.yaml << 'EOF'
##############################################################
# example values for use with tutorial workflow and minikube #
##############################################################

existingSecret: ngc-api
existingImagePullSecret: nvcrimagepullsecret

entity-store:
  enabled: true
  appConfig:
    BASE_URL_NIM: http://nemo-nim-proxy:8000
    BASE_URL_DATASTORE: http://nemo-data-store:3000/v1/hf

data-store:
  enabled: true
  external:
    rootUrl: http://data-store.test
    domain: data-store.test
  persistence:
    size: 2Gi

customizer:
  enabled: true
  modelsStorage:
    storageClassName: standard
  customizerConfig:
    models:
      meta/llama-3.2-1b-instruct:
        enabled: true
      meta/llama-3.1-8b-instruct:
        enabled: true
        model_path: llama-3_1-8b-instruct
        training_options:
        - finetuning_type: lora
          num_gpus: 1
          training_type: sft
    training:
      pvc:
        storageClass: "standard"
        volumeAccessMode: "ReadWriteOnce"
    wandb:
      project: data-flywheel-baked
    env:
      WANDB_API_KEY: $WANDB_API_KEY

evaluator:
  enabled: true
  milvus:
    enabled: false
  argoServiceAccount:
    create: true
    name: workflow-executor

guardrails:
  enabled: true
  env:
    DEMO: "True"
    DEFAULT_CONFIG_ID: self-check
    NIM_ENDPOINT_URL: http://nemo-nim-proxy:8000/v1

ingress:
  enabled: true
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: 100m

deployment-management:
  deployments:
    defaultStorageClass: standard
EOF
  fi


  local helm_args=()
  for values_file in "${ADDITIONAL_VALUES_FILES[@]}"; do
    if [[ ! -f "$values_file" ]]; then
      die "Values file not found: $values_file"
    fi
    helm_args+=("-f" "$values_file")
  done

  # Need to fetch and untar for the volcano installation
  helm fetch --untar "$HELM_CHART_URL" \
      --username='$oauthtoken' \
      --password=$NGC_API_KEY
}

install_nemo_microservices () {
  log "Installing NeMo microservices Helm chart..."
  volcano_version=$(yq '.dependencies[] | select(.name=="volcano") | .version' < nemo-microservices-helm-chart/Chart.yaml)
  kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v${volcano_version}/installer/volcano-development.yaml

  sleep 15

  helm install nemo "$HELM_CHART_URL" -f demo-values.yaml --namespace "$NAMESPACE" \
    --username='$oauthtoken' \
    --password=$NGC_API_KEY

  sleep 20
}

wait_for_pods() {
  log "Waiting for pods to initialize (up to 30 minutes)... You may seem some CrashLoops, but that's okay."
  log "They will go away eventually. The errors you want to look out for are ImagePullBackOff and ErrImagePull."
  log "The script will automatically fail if that happens."

  local old_err_trap=$(trap -p ERR)
  trap 'echo "Interrupted by user. Exiting."; exit 1;' SIGINT

  local start_time=$(date +%s)
  local end_time=$((start_time + 1800))

  while true; do
    # Get current pod statuses
    local pod_statuses
    if ! pod_statuses=$(kubectl get pods -n "$NAMESPACE" --no-headers 2>/dev/null); then
        warn "Failed to get pod statuses from kubectl. Retrying..."
        sleep 5
        continue
    fi

    # --- Premature exit for ImagePull errors ---
    local image_pull_errors
    image_pull_errors=$(echo "$pod_statuses" | grep -E "ImagePullBackOff|ErrImagePull" || true)
    if [[ -n "$image_pull_errors" ]]; then
      err "Detected ImagePull errors!"
      echo "$image_pull_errors" >&2 # Show the specific pods with errors
      warn "Gathering diagnostics for pods with ImagePull errors..."
      # Extract pod names with errors and collect diagnostics
      local error_pods=($(echo "$image_pull_errors" | awk '{print $1}'))
      local err_dir="nemo-errors-$(date +%s)"
      mkdir -p "$err_dir" || warn "Could not create error directory: $err_dir"
      for pod in "${error_pods[@]}"; do
        collect_pod_diagnostics "$pod" "$NAMESPACE" "$err_dir"
      done
      # Restore trap before dying
      eval "$old_err_trap"
      trap - SIGINT
      die "Exiting due to ImagePull errors. Diagnostics collected to $err_dir (if possible)."
    fi
    # --- End ImagePull check ---

    # Check if any non-Completed pods are in other problematic states
    if ! echo "$pod_statuses" | grep -v "Completed" | grep -qE "0/|Pending|CrashLoop|Error"; then
      log "All necessary pods are ready or succeeded."
      break
    fi

    # Check for timeout
    local current_time=$(date +%s)
    if (( current_time >= end_time )); then
      warn "Timeout waiting for pods to stabilize. Gathering diagnostics..."
      check_pod_health # Attempt to collect info before exiting
      # Restore trap before dying
      eval "$old_err_trap"
      trap - SIGINT
      die "Timeout waiting for pods to stabilize after 30 minutes. Diagnostics collected (if possible)."
    fi

    sleep 10
  done

  # Restore the original ERR trap and remove the SIGINT trap
  eval "$old_err_trap"
  trap - SIGINT

  log "Pods have stabilized."
}

# === Phase 4: Pod Health Verification ===
check_pod_health() {
  log "Checking pod health and collecting errors if needed..."
  local err_dir="nemo-errors-$(date +%s)"
  # Try to create the directory, but continue even if it fails (e.g., permissions)
  mkdir -p "$err_dir" || warn "Could not create error directory: $err_dir"

  # Check for image pull issues first
  # Use a temporary variable to store the return code
  local secrets_ok=0
  check_image_pull_secrets "$NAMESPACE" || secrets_ok=$?
  if [[ $secrets_ok -ne 0 ]]; then
    # If secrets check fails, we likely can't proceed usefully, but we already logged.
    # Let's still try to get pod status before potentially dying.
    warn "Image pull secret issues detected. Pods might fail to start."
  fi

  # Get all pods in the namespace
  # Use process substitution and check kubectl's exit code
  local all_pods=()
  if ! mapfile -t all_pods < <(kubectl get pods -n "$NAMESPACE" --no-headers -o custom-columns=NAME:.metadata.name); then
      warn "Failed to get pod list from kubectl."
      # Optionally, decide if this is fatal or if we can continue
      # For now, we'll just warn and might have an empty list
  fi

  # Track unhealthy pods
  local unhealthy_pods=()
  local pending_pods=()

  for pod in "${all_pods[@]}"; do
    local pod_status=""
    # Get status, handle potential kubectl errors
    if ! pod_status=$(kubectl get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null); then
        warn "Failed to get status for pod: $pod"
        unhealthy_pods+=("$pod") # Treat error getting status as unhealthy
        continue
    fi

    if [[ "$pod_status" != "Running" && "$pod_status" != "Succeeded" ]]; then
      if [[ "$pod_status" == "Pending" ]]; then
        pending_pods+=("$pod")
      # Add Failed status as unhealthy explicitly
      elif [[ "$pod_status" == "Failed" ]]; then
        warn "Pod $pod is in Failed state."
        unhealthy_pods+=("$pod")
      else
        # Catch other non-Running/Succeeded/Pending states (like Unknown)
        warn "Pod $pod is in unexpected state: $pod_status"
        unhealthy_pods+=("$pod")
      fi
    fi
  done

  # Handle pending pods first
  if (( ${#pending_pods[@]} > 0 )); then
    warn "Detected ${#pending_pods[@]} pending pods. Checking if they eventually run..."
    local still_pending=()
    for pod in "${pending_pods[@]}"; do
      # Give pending pods a short time to resolve (e.g., 60 seconds)
      timeout 60 bash -c "while kubectl get pod $pod -n $NAMESPACE -o jsonpath='{.status.phase}' 2>/dev/null | grep -q Pending; do sleep 5; done" || {
        warn "Pod $pod remained in Pending state."
        unhealthy_pods+=("$pod") # Add to unhealthy if it stays pending
      }
    done
  fi

  # Handle unhealthy pods
  if (( ${#unhealthy_pods[@]} > 0 )); then
    warn "Detected ${#unhealthy_pods[@]} unhealthy pods. Gathering diagnostics..."
    # De-duplicate unhealthy list before collecting diagnostics
    local unique_unhealthy=($(printf "%s\n" "${unhealthy_pods[@]}" | sort -u))

    for pod in "${unique_unhealthy[@]}"; do
      collect_pod_diagnostics "$pod" "$NAMESPACE" "$err_dir"
    done

    # Collect cluster-wide events
    log "Collecting cluster-wide events..."
    kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' > "$err_dir/cluster_events.txt" 2>/dev/null || warn "Failed to get cluster events."

    warn "Diagnostics written to $err_dir (if possible)"
    # This function is now just for checking, not dying. The caller decides.
    return 1 # Indicate unhealthy state
  else
    log "All pods are healthy (Running or Succeeded)."
    return 0 # Indicate healthy state
  fi
}

# === Phase 5: DNS Configuration ===
configure_dns() {
  log "Configuring DNS for ingress..."
  minikube_ip=$(minikube ip)

  # Backup /etc/hosts
  log "Creating backup of /etc/hosts..."
  maybe_sudo cp /etc/hosts "/etc/hosts.backup.$(date +%Y%m%d%H%M%S)"

  # Check if entries already exist
  if grep -q "nemo.test" /etc/hosts; then
    warn "Existing nemo.test entry found in /etc/hosts"
    if ! grep -q "$minikube_ip.*nemo.test" /etc/hosts; then
      warn "IP address mismatch for nemo.test. Updating..."
      maybe_sudo sed -i.bak "/nemo.test/d" /etc/hosts
    fi
  fi

  # Add new entries
  {
    echo "# Added by NeMo setup script"
    echo "$minikube_ip nemo.test"
    echo "$minikube_ip nim.test"
    echo "$minikube_ip data-store.test"
  } | maybe_sudo tee -a /etc/hosts > /dev/null

  log "Hosts file updated successfully."
}

# === Main Entrypoint ===
main() {
  parse_args "$@"
  check_prereqs
  download_helm_chart
  start_minikube
  # Ingress needs a few more seconds after it reports ready before the containers can get installed
  sleep 10
  setup_ngc_and_helm
  install_nemo_microservices
  wait_for_pods
  check_pod_health || die "Base cluster is not healthy after waiting. Investigate and re-run."
  configure_dns
  log "Nemo Microservices Platform successfully deployed."
}

main "$@"
