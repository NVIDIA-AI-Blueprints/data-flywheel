# Helm Installation Guide

Learn how to deploy the Data Flywheel Blueprint on Kubernetes using Helm charts for scalable, production-ready environments.

## Overview

The Data Flywheel Blueprint provides a comprehensive Helm chart for Kubernetes deployment, enabling scalable data processing workflows with integrated NeMo microservices, experiment tracking, and monitoring capabilities.

### What This Guide Covers

- Complete Kubernetes deployment using Helm 3
- Configuration management for production environments
- Integration with NeMo microservices
- Post-installation setup and verification
- Operations, scaling, and maintenance procedures

## Prerequisites

> **📖 For complete prerequisites:** See [System Requirements](03-configuration.md#system-requirements)

### Tools and Access

| Tool/Access | Version | Purpose | Note |
|-------------|---------|---------|---------|
| **Kubernetes cluster** | 1.25+ | Target deployment environment | Required |
| **Helm** | 3.8+ | Chart installation and management | Required |
| **kubectl** | Latest | Cluster interaction and verification | Required |
| **NGC API key** | Current | Access to NVIDIA container registry and models | Required |
| **NVIDIA API key** | Current | Remote LLM judge access via NVIDIA API catalog | Required |
| **Hugging Face token** | Current | Model and dataset access | Required |
| **LLM Judge API key** | Current | API key for remote LLM judge services | Optional |
| **EMB API Key** | Current | Model API key for remote embedding services | Optional |

> **📖 For API key setup instructions:** See [Required API Keys and Access](03-configuration.md#required-api-keys-and-access)

#### NVIDIA API Key Source

To get your **NVIDIA API key**, visit [build.nvidia.com](https://build.nvidia.com) and generate an API key for accessing NVIDIA's API catalog and remote LLM services.

### Environment Verification

Verify your environment meets the requirements:

```bash
# Check Kubernetes version
kubectl version --client
```

```bash
# Check Helm version
helm version --short
```

```bash
# Verify cluster access and node resources
kubectl get nodes -o wide
```

```bash
# Check available storage classes
kubectl get storageclass
```

```bash
# Verify GPU nodes (in a standard Kubernetes cluster)
kubectl get nodes -l nvidia.com/gpu.present=true

# or if in minikube cluster:
export NODE=$(kubectl get node -o jsonpath='{.items[0].metadata.name}')
kubectl describe node $NODE | grep nvidia.com/gpu
```

## Configuration Overview

The Data Flywheel Blueprint Helm chart uses a `values.yaml` file that defines all configuration options. This file must be updated with your specific settings before deployment:

- **Model Deployments**: All candidate models (NIMs) are deployed locally in your cluster for evaluation and customization
- **LLM Judge**: Can be configured as either:
  - **Local deployment**: Runs as a NIM in your cluster (requires additional GPU resources)
  - **Remote deployment**: Uses external API endpoints (recommended for resource efficiency)
- **Evaluation Settings**: Controls how models are evaluated, including data splitting and in-context learning
- **Training Parameters**: Defines fine-tuning settings for model customization
- **Resource Allocation**: Kubernetes resource requests, limits, and scaling configurations
- **Service Integration**: NeMo microservices, MLflow, and monitoring stack configuration

> **📖 For complete configuration details:** See the [Configuration Guide](03-configuration.md)

## Chart Download and Setup

### Method 1: From GitHub Repository

```bash
# Clone the repository
git clone https://github.com/NVIDIA-AI-Blueprints/data-flywheel.git
cd data-flywheel

# Install Git LFS if not already installed
git lfs install

# Pull large files including datasets
git lfs pull

# Navigate to helm chart directory
cd deploy/helm/data-flywheel
```

### Method 2: From NGC Private Registry

```bash
# Download chart from NGC registry
helm fetch https://helm.ngc.nvidia.com/nvidia/blueprint/charts/nvidia-blueprint-data-flywheel-0.3.1.tgz \
    --username='$oauthtoken' \
    --password=$NGC_API_KEY

# Extract chart
tar -xvf nvidia-blueprint-data-flywheel-0.3.1.tgz
cd nvidia-blueprint-data-flywheel
```

> **Note**: The NGC private registry contains pre-packaged charts with all dependencies included.

### Chart Dependencies Setup

> **Note**: Skip this step if you downloaded the chart from NGC registry, as all sub-charts are included.

Add required Helm repositories:

```bash
# Add Bitnami repository for infrastructure services
helm repo add bitnami https://charts.bitnami.com/bitnami

# Add NeMo microservices repository
helm repo add nemo-microservices https://helm.ngc.nvidia.com/nvidia/nemo-microservices \
    --username '$oauthtoken' \
    --password $NGC_API_KEY

# Update repository indexes
helm repo update

# Navigate to chart directory (if cloning from repository)
cd deploy/helm/data-flywheel

# Update chart dependencies
helm dependency update
```

Verify dependencies are downloaded:

```bash
# Check Chart.lock file was created
ls -la Chart.lock

# Verify dependency charts in charts/ directory
ls -la charts/
```

## Configuration Management

### Understanding the Chart Structure

**Source**: `deploy/helm/data-flywheel/`

```text
deploy/helm/data-flywheel/
├── Chart.yaml                    # Chart metadata and dependencies
├── values.yaml                   # Default configuration values
├── templates/
│   ├── _helpers.tpl               # Helm template helpers
│   ├── api-deployment.yaml       # FastAPI server deployment
│   ├── api-service.yaml          # API service exposure
│   ├── celeryParentWorker-deployment.yaml  # Main orchestration worker
│   ├── celeryWorker-deployment.yaml        # Task processing workers
│   ├── config-configmap.yaml     # Configuration management
│   ├── secrets.yaml              # Sensitive data management
│   ├── flower-deployment.yaml    # Task monitoring UI
│   ├── flower-service.yaml       # Flower service
│   ├── kibana-deployment.yaml    # Log visualization
│   ├── kibana-service.yaml       # Kibana service
│   ├── mlflow-deployment.yaml    # Experiment tracking
│   ├── mlflow-service.yaml       # MLflow service
│   └── volcano-install.yaml      # Job scheduling for GPU workloads
├── endpoints.md                   # Service endpoint documentation
└── README.md                     # Installation instructions
```

### Core Configuration Sections

#### API Keys and Secrets Configuration

The chart expects three main secrets to be provided during installation:

```yaml
# Secrets values - set during deployment
secrets:
  ngcApiKey: ""  # Set this to your NGC API key
  nvidiaApiKey: "" # Set this to your NVIDIA API key
  hfToken: "" # Set this to your HF Token
  llmJudgeApiKey: "" # Set this to your LLM Judge API key
  embApiKey: "" # Set this to your Embedding API key
```

#### Resource and Profile Configuration

```yaml
# Deployment profile settings
profile:
  production:
    enabled: true    # Set to false for development (enables Kibana/Flower)
  mlflow:
    COMPOSE_PROFILES: "mlflow"    # Set to "" (blank) if not using the mlflow

# Namespace configuration
namespace: "nv-nvidia-blueprint-data-flywheel"
```

#### Infrastructure Services Configuration

```yaml
# Elasticsearch configuration
elasticsearch:
  enabled: true
  resources:
    ...
  env:
    # Environment variables to configure your Elasticsearch, read more: https://www.elastic.co/docs/reference/elasticsearch/configuration-reference/.
    ...

# MongoDB configuration  
mongodb:
  enabled: true
  aresources:
    ...
  env:
    ...

# Redis configuration
redis:
  enabled: true
  resources:
    ...
  env:
    ...
```

#### Data Flywheel Server Configuration

```yaml
# Main application configuration
foundationalFlywheelServer:
  image:
    repository: nvcr.io/nvidia/blueprint/foundational-flywheel-server
    tag: "0.3.1"
  
  deployments:
    api:
      enabled: true              # Main API server
    celeryWorker:
      enabled: true              # Task processing workers
    celeryParentWorker:
      enabled: true              # Orchestration worker
    mlflow:
      # mlflow server, controlled by `profile.mlflow.COMPOSE_PROFILES`
      ...
    flower:
      # flower server, controlled by `profile.production.enabled`

  config:
    # Core configuration sections
    nmp_config:
      # NeMo platform configuration
    logging_config:
      # Logging settings
    llm_judge_config:
      # LLM judge configuration
    mlflow_config:
      # mlflow configuration
```

### Custom Configuration Examples

#### Development Environment Configuration

Create a custom values file for development:

```yaml
# values-dev.yaml
profile:
  production:
    enabled: false               # Enables Kibana and Flower for debugging

# Reduced resource requirements for development
elasticsearch:
  resources:
    requests:
      memory: 1Gi
      cpu: 1
      ephemeral-storage: "2Gi"
    limits:
      memory: 2Gi
      cpu: 2
      ephemeral-storage: "5Gi"

# Enable development tools (Flower is automatically enabled when profile.production.enabled is false)
profile:
  production:
    enabled: false             # Enables development tools including Flower task monitoring UI
```

#### Production Environment Configuration

Create a custom values file for production:

```yaml
# values-prod.yaml
profile:
  production:
    enabled: true                # Disables development tools

# Enhanced resource allocation
elasticsearch:
  resources:
    requests:
      memory: 2Gi
      cpu: 1
      ephemeral-storage: "10Gi"
    limits:
      memory: 4Gi
      cpu: 2
      ephemeral-storage: "20Gi"

# Production storage configuration (for NeMo Microservices components)
nemo-microservices-helm-chart:
  data-store:
    persistence:
      enabled: true
      storageClass: "fast-ssd"
      size: 100Gi
```

> **Note**: The Data Flywheel server itself currently uses ephemeral storage (emptyDir volumes). Only NeMo microservices components (data-store, customizer training) support persistent volumes. The Data Flywheel server relies on external databases (MongoDB, Elasticsearch, Redis) for data persistence.

#### GPU and NIM Configuration

```yaml
# Custom NIM deployment configuration
foundationalFlywheelServer:
  config:
    nims:
      - model_name: "meta/llama-3.1-8b-instruct"
        context_length: 32768
        gpus: 2                  # Increased GPU allocation
        pvc_size: 50Gi          # Larger storage for model
        tag: "1.8.3"
        customization_enabled: true
```

## Installation Process

### Step 1: Environment Preparation

Set your API keys as environment variables:

> **📖 For environment variable setup:** See [Environment Variables](03-configuration.md#environment-variables)

```bash
# Set required API keys
export NGC_API_KEY="your_ngc_api_key_here"
export NVIDIA_API_KEY="your_nvidia_api_key_here" 
export HF_TOKEN="your_hugging_face_token_here"

# Optional: Set custom API keys for specific services (if different from NVIDIA_API_KEY)
export LLM_JUDGE_API_KEY="your_llm_judge_api_key_here"
export EMB_API_KEY="your_emb_api_key_here"

# Verify keys are set
echo "NGC_API_KEY: ${NGC_API_KEY:0:10}..."
echo "NVIDIA_API_KEY: ${NVIDIA_API_KEY:0:10}..."
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."
echo "LLM_JUDGE_API_KEY: ${LLM_JUDGE_API_KEY:0:10}..."
echo "EMB_API_KEY: ${EMB_API_KEY:0:10}..."
```

> **API Key Explanations:**
> - **`LLM_JUDGE_API_KEY`**: API key used specifically for remote LLM judge services when `llm_judge_config.deployment_type` is set to "remote". If not provided, defaults to `NVIDIA_API_KEY`. Can be from any provider (OpenAI, Anthropic, etc.) depending on your remote LLM judge endpoint configuration.
> - **`EMB_API_KEY`**: API key used specifically for remote embedding services when using semantic similarity for ICL example selection. If not provided, defaults to `NVIDIA_API_KEY`. Can be from any provider depending on your embedding service endpoint configuration.

Configure Docker registry access:

```bash
# Log in to NVIDIA Container Registry
docker login nvcr.io
# Username: $oauthtoken
# Password: <your_ngc_api_key>
```

### Step 2: Pre-Installation Requirements

For minikube deployments, ensure proper GPU runtime configuration:

```bash
# Configure nvidia-ctk runtime for Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Start minikube with GPU support
minikube start \
   --driver docker \
   --container-runtime docker \
   --cpus no-limit \
   --memory no-limit \
   --gpus all

# Enable ingress addon
minikube addons enable ingress
```

### Step 3: Kubernetes Namespace Setup

Create and configure the deployment namespace:

```bash
# Create dedicated namespace
kubectl create namespace nv-nvidia-blueprint-data-flywheel

# Switch to the new namespace
kubectl config set-context --current --namespace=nv-nvidia-blueprint-data-flywheel

# Verify current namespace
kubectl config view --minify --output 'jsonpath={..namespace}'
```

### Step 4: DNS Resolution Config

Retrieve IP address of minikube:
```bash
export NEMO_HOST=$(minikube ip)
```

Add host name entries in the /etc/hosts file for the *.test ingress hosts to use the accessible IP address.
Make a backup of the /etc/hosts file before you make the changes:
```bash
sudo cp /etc/hosts /etc/hosts.bak
echo -e "$NEMO_HOST nemo.test\n$NEMO_HOST nim.test\n$NEMO_HOST data-store.test\n" | sudo tee -a /etc/hosts
```

> **Note:** Retrieve IP address and Add host name entries sections are necessary for the Data Flywheel server to access NeMo microservices, see more: [NMP Configure DNS Resoluttion](https://docs.nvidia.com/nemo/microservices/latest/get-started/setup/minikube-manual.html#configure-dns-resolution).

### Step 5: Volcano Installation
Install Volcano scheduler before installing the chart:
```bash
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.9.0/installer/volcano-development.yaml
```

After installing the Volcano scheduler, verify that the `volcano-monitoring` and `volcano-system` namespaces exist:
```bash
$ kubectl get namespaces
NAME                                STATUS   AGE
default                             Active   17m
ingress-nginx                       Active   17m
kube-node-lease                     Active   17m
kube-public                         Active   17m
kube-system                         Active   17m
nv-nvidia-blueprint-data-flywheel   Active   17m
volcano-monitoring                  Active   16m  👈
volcano-system                      Active   16m  👈
```

### Step 6: Chart Installation

#### Basic Installation

```bash
# Navigate to chart directory
cd deploy/helm/data-flywheel

# Install with default configuration
helm upgrade --install data-flywheel . \
  --set secrets.ngcApiKey=$NGC_API_KEY \
  --set secrets.nvidiaApiKey=$NVIDIA_API_KEY \
  --set secrets.hfToken=$HF_TOKEN \
  --set secrets.llmJudgeApiKey=$LLM_JUDGE_API_KEY \
  --set secrets.embApiKey=$EMB_API_KEY \
  --namespace nv-nvidia-blueprint-data-flywheel \
  --timeout 20m
```

#### Production Installation with Custom Values

```bash
# Install with production configuration
helm upgrade --install data-flywheel . \
  --values values-prod.yaml \
  --set secrets.ngcApiKey=$NGC_API_KEY \
  --set secrets.nvidiaApiKey=$NVIDIA_API_KEY \
  --set secrets.hfToken=$HF_TOKEN \
  --set secrets.llmJudgeApiKey=$LLM_JUDGE_API_KEY \
  --set secrets.embApiKey=$EMB_API_KEY \
  --set profile.production.enabled=true \
  --namespace nv-nvidia-blueprint-data-flywheel \
  --timeout 30m
```

#### Development Installation

```bash
# Install with development configuration
helm upgrade --install data-flywheel . \
  --values values-dev.yaml \
  --set secrets.ngcApiKey=$NGC_API_KEY \
  --set secrets.nvidiaApiKey=$NVIDIA_API_KEY \
  --set secrets.hfToken=$HF_TOKEN \
  --set secrets.llmJudgeApiKey=$LLM_JUDGE_API_KEY \
  --set secrets.embApiKey=$EMB_API_KEY \
  --set profile.production.enabled=false \
  --namespace nv-nvidia-blueprint-data-flywheel \
  --timeout 20m
```

### Step 7: Installation Verification

Monitor the deployment progress:

```bash
# Watch pod creation and status
kubectl get pods -w

# Check all resources
kubectl get all

# Check secrets creation
kubectl get secrets

# Verify persistent volumes (for NeMo Microservices components only)
kubectl get pv,pvc
```

Wait for all pods to reach `Running` status:

> **⏱️ Expected Startup Times:**
> - **Total deployment time**: 10-15 minutes for all pods to be running and ready
> - **Infrastructure services** (Elasticsearch, MongoDB, Redis): 2-3 minutes
> - **NeMo microservices** (customizer, data-store, etc.): 5-8 minutes
> - **Data Flywheel services** (API, workers): 1-2 minutes
> - **LLM Judge model deployment**: Might take 10-15 minutes (if using local deployment), but jobs can start immediately as the system will wait for deployment completion

```bash
# Check pod status with details
kubectl get pods -o wide

# View pod logs if there are issues
kubectl logs -l app=df-api-deployment --tail=50

# Describe problematic pods for troubleshooting
kubectl describe pod <pod-name>
```

Expected pod status after successful installation:
▶️ Profile: production
```text
NAME                                                              READY   STATUS    RESTARTS        AGE
data-flywheel-argo-workflows-server-xxx                           1/1     Running   0               7m1s
data-flywheel-argo-workflows-workflow-controller-xxx              1/1     Running   0               7m1s
data-flywheel-customizer-xxx                                      1/1     Running   2 (6m56s ago)   7m2s
data-flywheel-customizerdb-0                                      1/1     Running   0               6m59s
data-flywheel-data-store-xxx                                      1/1     Running   0               7m2s
data-flywheel-deployment-management-xxx                           1/1     Running   0               7m1s
data-flywheel-entity-storexxx                                     1/1     Running   0               7m1s
data-flywheel-entity-storedb-0                                    1/1     Running   0               6m59s
data-flywheel-evaluator-xxx                                       1/1     Running   0               7m1s
data-flywheel-evaluatordb-0                                       1/1     Running   0               6m59s
data-flywheel-guardrails-xxx                                      1/1     Running   1 (6m57s ago)   7m1s
data-flywheel-nemo-operator-controller-manager-xxx                2/2     Running   0               7m
data-flywheel-nim-operator-xxx                                    1/1     Running   0               7m
data-flywheel-nim-proxy-xxx                                       1/1     Running   0               7m
data-flywheel-postgresql-0                                        1/1     Running   0               6m59s
df-api-deployment-xxx                                             1/1     Running   0               7m
df-celery-parent-worker-deployment-xxx                            1/1     Running   2 (6m33s ago)   7m
df-celery-worker-deployment-xxx                                   1/1     Running   2 (6m24s ago)   6m59s
df-elasticsearch-deployment-xxx                                   1/1     Running   0               7m
df-mlflow-deployment-xxx                                          1/1     Running   0               6m59s
df-mongodb-xxx                                                    1/1     Running   0               7m2s
df-redis-deployment-xxx                                           1/1     Running   0               6m59s
modeldeployment-dfwbp-meta-llama-3-3-70b-instruct-xxx             0/1     Running   1 (34s ago)     6m50s
```

## Post-Installation Setup

### Data Loading and Initialization

#### Step 1: Copy Dataset to API Pod

Get the API pod name and copy your dataset:

```bash
# Get API pod name
export POD_NAME=$(kubectl get pods -l app=df-api-deployment -o jsonpath='{.items[0].metadata.name}')
echo "API Pod: $POD_NAME"

# Set path to your JSONL dataset
export LOCAL_DATA_PATH="/path/to/your/dataset.jsonl"

# For example, using provided AIVA dataset:
export LOCAL_DATA_PATH="$(pwd)/../../../data/aiva_primary_assistant_dataset.jsonl"

# If you cloned from GitHub, the path would be:
# export LOCAL_DATA_PATH="$(pwd)/../../../data/aiva_primary_assistant_dataset.jsonl"

# Copy dataset to the pod
kubectl cp $LOCAL_DATA_PATH $POD_NAME:/legal/source/data/datasets.jsonl
```

#### Step 2: Load Data into the System

```bash
# Load the dataset into the flywheel
kubectl exec -it deployment/df-api-deployment -- \
  uv run python /legal/source/src/scripts/load_test_data.py \
  --file /legal/source/data/datasets.jsonl

# Verify data was loaded successfully
kubectl exec -it $POD_NAME -- ls -la /legal/source/data/
```

### Service Access Configuration

#### Service Access via Port Forwarding Script
To automate Kubernetes service port forwarding, use the `forward-ports.sh` script located in `/scripts/helm/`

Synxtax: `forward-ports.sh [<service_name> <desired_port> ...] [-h|--help]`

Usage:
- Forward one or more services to specified local ports:\
`./scripts/helm/forward-ports.sh <service_name> <local_port> [...]`
- Display help:\
`./scripts/helm/forward-ports.sh -h`

Example:
```bash
./scripts/helm/forward-ports.sh --help
```

```bash
# Change port numbers as you wish
./scripts/helm/forward-ports.sh \
  df-api-service 8001 \
  df-mlflow-service 5001
```

```bash
# Forward all services for non-production
./scripts/helm/forward-ports.sh \
  df-api-service 8000 \
  df-elasticsearch-service 9200 \
  df-mlflow-service 5000 \
  df-kibana-service 5601 \
  df-flower-service 5555
```

```bash
# Forward essential services for production
./scripts/helm/forward-ports.sh \
  df-api-service 8000 \
  df-elasticsearch-service 9200 \
  df-mlflow-service 5000
```

#### Custom Access

##### API Access

```bash
# Port forward API service
kubectl port-forward service/df-api-service 8000:8000
```

```bash
# Test API connectivity (in a new terminal)
curl http://localhost:8000/api/jobs -s | jq .
```

```bash
# Test API documentation access (if you're in Cursor or other IDEs...)
open http://localhost:8000/docs
```

##### MLflow Access (if using mlflow profile)

```bash
# Port forward MLflow service
kubectl port-forward service/df-mlflow-service 5000:5000
```

```bash
# Access MLflow UI (if you're in Cursor or other IDEs...)
open http://localhost:5000
```

> **💡 Tips:** If the deployment is running on a VM, ensure that forwarded ports are exposed and accessible from your local machine:
> ```bash
> # Example SSH tunneling commands
> ssh -L 8000:localhost:8000 user@vm-ip-address  # For API server
> ssh -L 5000:localhost:5000 user@vm-ip-address  # For MLflow 
> ```

##### Flower Access (development mode)

```bash
# Port forward Flower service (if enabled)
kubectl port-forward service/df-flower-service 5555:5555
```

```bash
# Access Flower UI (if you're in Cursor or other IDEs...)
open http://localhost:5555
```

##### Kibana Access (development mode)

```bash
# Port forward Kibana service (if enabled)
kubectl port-forward service/df-kibana-service 5601:5601
```

```bash
# Access Kibana UI (if you're in Cursor or other IDEs...)
open http://localhost:5601
```

### Initial Testing and Verification

#### Test API Functionality

```bash
# List available jobs
curl http://localhost:8000/api/jobs -s | jq .

# Create a test job
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"workload_id": "primary_assistant", "client_id": "aiva-1"}' \
  -s | jq .

# Monitor job progress
JOB_ID=$(curl -s http://localhost:8000/api/jobs | jq -r '.jobs[-1].id')
curl http://localhost:8000/api/jobs/$JOB_ID -s | jq .
```

#### Verify NeMo Platform Integration
```bash
# Launch a temporary BusyBox pod for HTTP requests
# (Press Ctrl+D to exit — the pod will be automatically deleted)
kubectl run tmp-busybox --rm -it --image=busybox -- sh

# Check NeMo service connectivity
# (inside temporary BusyBox shell)
wget -qO- http://nemo.test/v1/models

# Verify NIM proxy access
# (inside temporary BusyBox shell)
wget -qO- http://nim.test/v1/models
```

#### Monitor System Health

```bash
# Check all service health
kubectl get pods --all-namespaces | grep -E "(data-flywheel|nemo)"

# View recent logs
kubectl logs deployment/df-api-deployment --tail=20

# Check resource usage
kubectl top pods
```

## Configuration Examples

### MLflow Integration Configuration

To enable MLflow profile:
```bash
profile:
  ...
  mlflow:
    COMPOSE_PROFILES: "mlflow"  # ⬅️ leave blank if not using mlflow
```

Customize MLflow experiment tracking:

```yaml
# In values.yaml or custom values file
foundationalFlywheelServer:
  config:
    mlflow_config:
      tracking_uri: "http://df-mlflow-service:5000"
      experiment_name_prefix: "data-flywheel"
      artifact_location: "./mlruns"
```

Access MLflow experiments:

```bash
# Port forward MLflow service
kubectl port-forward service/df-mlflow-service 5000:5000

# MLflow experiments will be accessible at http://localhost:5000
# Job completion messages will include direct links like:
# "mlflow_uri": "http://df-mlflow-service:5000/#/experiments/602724461284860314"
# Replace df-mlflow-service with localhost and 5000 with your port-forwarded (current case is 5000 as above):
# http://localhost:5000/#/experiments/602724461284860314
```

### LLM Judge Configuration

#### Remote LLM Judge (Recommended)

```yaml
foundationalFlywheelServer:
  config:
    llm_judge_config:
      deployment_type: "remote"
      url: "https://integrate.api.nvidia.com/v1/chat/completions"
      model_name: "meta/llama-3.3-70b-instruct"
```

#### Local NIM Deployment for LLM Judge

```yaml
foundationalFlywheelServer:
  config:
    llm_judge_config:
      deployment_type: "local"
      model_name: "meta/llama-3.3-70b-instruct"
      context_length: 32768
      gpus: 4
      pvc_size: 25Gi
      tag: "1.8.5"
```

### NIM Deployment Configuration

Configure local NIMs for model customization:

```yaml
foundationalFlywheelServer:
  config:
    nims:
      - model_name: "meta/llama-3.2-1b-instruct"
        context_length: 8192
        gpus: 1
        pvc_size: 25Gi
        tag: "1.8.3"
        customization_enabled: true
        customizer_configs:
          target: "meta/llama-3.2-1b-instruct@2.0"
          gpus: 1
          max_seq_length: 8192
      
      - model_name: "meta/llama-3.1-8b-instruct"
        context_length: 32768
        gpus: 2                               # More GPUs for larger model
        pvc_size: 50Gi                        # More storage for larger model
        tag: "1.8.3"
        customizer_configs:
          target: "meta/llama-3.1-8b-instruct@2.0"
          gpus: 2
          max_seq_length: 8192
```

### Data Processing Configuration

Customize data split and ICL (In-Context Learning) parameters:

```yaml
foundationalFlywheelServer:
  config:
    data_split_config:
      eval_size: 200                          # Evaluation dataset size
      val_ratio: 0.15                         # Validation split ratio
      min_total_records: 100                  # Minimum required records
      limit: 2000                             # Maximum records to process
      parse_function_arguments: true          # Parse function calling syntax

    icl_config:
      max_context_length: 32768               # Maximum context window
      reserved_tokens: 4096                   # Tokens reserved for response
      max_examples: 5                         # Maximum ICL examples
      min_examples: 2                         # Minimum ICL examples
      example_selection: "semantic_similarity" # Selection strategy
```

### Training and Customization Configuration

Configure LoRA fine-tuning parameters:

```yaml
foundationalFlywheelServer:
  config:
    training_config:
      training_type: "sft"                    # Supervised fine-tuning
      finetuning_type: "lora"                 # LoRA adaptation
      epochs: 3                               # Training epochs
      batch_size: 32                          # Batch size
      learning_rate: 0.0002                   # Learning rate

    lora_config:
      adapter_dim: 64                         # LoRA adapter dimension
      adapter_dropout: 0.1                    # Dropout rate
```

## Troubleshooting

### Installation Issues

#### Chart Dependency Failures

**Problem**: `helm dependency update` fails or charts don't download

```bash
# Clear Helm cache and retry
helm repo remove bitnami nemo-microservices
rm -rf charts/ Chart.lock

# Re-add repositories with authentication
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add nemo-microservices https://helm.ngc.nvidia.com/nvidia/nemo-microservices \
    --username '$oauthtoken' \
    --password $NGC_API_KEY

helm repo update
helm dependency update
```

#### Registry Access Problems

**Problem**: Image pull failures from nvcr.io

```bash
# Verify Docker login
docker login nvcr.io
# Username: $oauthtoken  
# Password: <NGC_API_KEY>

# Test image pull manually
docker pull nvcr.io/nvidia/blueprint/foundational-flywheel-server:0.3.1

# Check Kubernetes secret creation
kubectl get secret nvcrimagepullsecret -o yaml
```

### Runtime Issues

#### Pod Startup Failures

**Problem**: Pods failing to start or crashing

```bash
# Check pod events and logs
kubectl describe pod <pod-name>
kubectl logs <pod-name> --previous

# Common issues and solutions:

# 1. Configuration errors
kubectl get configmap df-config -o yaml

# 2. Secret access issues  
kubectl get secrets
kubectl describe secret ngc-api nvidia-api hf-secret

# 3. Storage issues (NeMo Microservices components)
kubectl get pv,pvc
kubectl describe pvc <pvc-name>
```

#### Service Connectivity Problems

**Problem**: Services can't communicate with each other

```bash
# Test internal DNS resolution
kubectl run busybox --rm -it --restart=Never -n nv-nvidia-blueprint-data-flywheel \
  --image=debian:stable-slim -- \
  getent hosts df-elasticsearch-service

# Check service endpoints
kubectl get endpoints

# Test service connectivity
kubectl run curlbox --rm -it --restart=Never -n nv-nvidia-blueprint-data-flywheel \
  --image=curlimages/curl -- \
  curl -s http://df-elasticsearch-service:9200/_cluster/health

# Verify network policies (if any)
kubectl get networkpolicies
```

#### GPU Allocation Issues

**Problem**: NIMs not getting GPU resources

```bash
# Check GPU node availability
kubectl get nodes -l nvidia.com/gpu.present=true

# Verify GPU scheduling
kubectl describe node <gpu-node-name> | grep -A 10 "Allocated resources"

# Check GPU plugin installation
kubectl get daemonset -n kube-system | grep nvidia

# Test GPU access in pod
kubectl exec -it <nim-pod-name> -- nvidia-smi
```

### Monitoring and Debugging

#### Enable Debug Logging

```yaml
# In values file or upgrade command
foundationalFlywheelServer:
  config:
    logging_config:
      level: "DEBUG"                          # Increase logging verbosity
```

#### Health Check Commands

```bash
# Comprehensive system health check
kubectl get pods,svc,pvc,secrets -o wide

# Check resource usage
kubectl top pods
kubectl top nodes

# View recent events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check ingress (if configured)
kubectl get ingress
```

## Operations and Maintenance

### Monitoring and Health Checks

#### Log Monitoring

```bash
# Monitor API logs
kubectl logs -f deployment/df-api-deployment

# Monitor worker logs
kubectl logs -f deployment/df-celery-worker-deployment

# Monitor infrastructure logs
kubectl logs -f deployment/df-elasticsearch-deployment
```

### Updates and Rollbacks

#### Configuration Updates
If Data Flywheel changes are happen after a helm deployment (in `foundationalFlywheelServer.config`):
```bash
helm upgrade --install data-flywheel . \
  --set secrets.ngcApiKey=$NGC_API_KEY \
  --set secrets.nvidiaApiKey=$NVIDIA_API_KEY \
  --set secrets.hfToken=$HF_TOKEN \
  -n nv-nvidia-blueprint-data-flywheel
```

```bash
# Restart specific deployments after config changes
kubectl rollout restart deployment/df-api-deployment
kubectl rollout restart deployment/df-celery-parent-worker-deployment
kubectl rollout restart deployment/df-celery-worker-deployment
```

#### Check upgrade status
```bash
helm status data-flywheel
helm history data-flywheel
```

#### Rollback Procedures

```bash
# View deployment history
helm history data-flywheel

# Rollback to previous version
helm rollback data-flywheel

# Rollback to specific revision
helm rollback data-flywheel 2

# Verify rollback success
helm status data-flywheel
kubectl get pods
```

### Scaling Operations

#### Horizontal Scaling

```bash
# Scale Celery workers
kubectl scale deployment df-celery-worker-deployment --replicas=5

# Scale Elasticsearch (if clustered)
helm upgrade data-flywheel . \
  --reuse-values \
  --set elasticsearch.master.replicaCount=3

# Scale API servers (with load balancer)
kubectl scale deployment df-api-deployment --replicas=3
```

#### Vertical Scaling

```bash
# Increase resource limits for API server
kubectl patch deployment df-api-deployment -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "df-api",
          "resources": {
            "limits": {
              "memory": "8Gi",
              "cpu": "4"
            },
            "requests": {
              "memory": "4Gi", 
              "cpu": "2"
            }
          }
        }]
      }
    }
  }
}'
```

#### Auto-scaling Configuration

```bash
# Create Horizontal Pod Autoscaler for API
kubectl autoscale deployment df-api-deployment \
  --cpu-percent=70 \
  --min=2 \
  --max=10

# Create HPA for Celery workers
kubectl autoscale deployment df-celery-worker-deployment \
  --cpu-percent=80 \
  --min=2 \
  --max=20

# Check autoscaler status
kubectl get hpa
```

### Backup

#### Data Backup

```bash
# Backup Elasticsearch indices
export ES_POD=$(kubectl get pods -l app=df-elasticsearch-deployment -o jsonpath='{.items[0].metadata.name}')

kubectl exec $ES_POD -- \
  curl -X PUT "localhost:9200/_snapshot/backup" \
  -H 'Content-Type: application/json' \
  -d '{"type": "fs", "settings": {"location": "/backup"}}'

# Backup MongoDB data
```bash
kubectl exec deployment/df-mongodb-deployment -- \
  mongodump --out /tmp/mongodb-backup/mongodb-$(date +%Y%m%d)
```

# Backup MLflow artifacts
```bash
kubectl exec deployment/df-mlflow-deployment -- mkdir -p /tmp/mlflow-backup
```

```bash
kubectl exec deployment/df-mlflow-deployment -- \
  tar -czf /tmp/mlflow-backup/mlflow-$(date +%Y%m%d).tar.gz /mlruns
```

#### Configuration Backup

```bash
# Export all configurations
kubectl get configmaps,secrets -o yaml > dataflywheel-config-backup-$(date +%Y%m%d).yaml

# Export Helm values
helm get values data-flywheel > values-backup-$(date +%Y%m%d).yaml
```

### Helm Chart Cleanup

#### Data clearing

File structure of `scripts/helm/`
```
scripts/helm/
├── clear-all-data.sh            # Clears data from all services
├── clear-es-data.sh             # Clears Elasticsearch indices
├── clear-mlflow-data.sh         # Clears MLflow experiments and artifacts
├── clear-mongodb-data.sh        # Clears MongoDB collections
└── clear-redis-data.sh          # Clears Redis cache data
```

##### Usage

All scripts accept an optional `NAMESPACE` environment variable. If not provided, they default to `nv-nvidia-blueprint-data-flywheel`.

Run the specific script to clear the data in specific service base on your need.

For axample:
> For all-in-one approach:
```bash
./scripts/helm/clear-all-data.sh
```

> For specific resource:
```bash
./scripts/helm/clear-mongodb-data.sh
```

#### Helm graceful cleanup
```bash
export NAMESPACE_DELETED=nv-nvidia-blueprint-data-flywheel
```

```bash
helm uninstall data-flywheel -n $NAMESPACE_DELETED

kubectl get nimservices -n $NAMESPACE_DELETED -o name | xargs -I {} kubectl patch {} -p '{"metadata":{"finalizers":[]}}' --type=merge

kubectl delete nimservices --all -n $NAMESPACE_DELETED --force --grace-period=0

kubectl delete all --all -n $NAMESPACE_DELETED --force --grace-period=0

kubectl delete pvc --all -n $NAMESPACE_DELETED --force --grace-period=0

kubectl patch namespace "$NAMESPACE_DELETED" -p '{"spec":{"finalizers":null}}' --type=merge

kubectl delete ns $NAMESPACE_DELETED
```

#### Helm complete cleanup
To completely reset your environment:
```bash
minikube delete
```

> **Note:** This is also a clean starting point if you want to perform a fresh installation.

## Related Documentation

- **[Getting Started](02-quickstart.md)**: Local development setup using Docker Compose
- **[Configuration Guide](03-configuration.md)**: Detailed configuration options and examples
- **[Production Deployment](10-production-deployment.md)**: Production architecture and advanced deployment strategies
- **[NeMo Platform Integration](09-nemo-platform-integration.md)**: Integration with NeMo microservices
- **[API Reference](07-api-reference.md)**: Complete API documentation and examples

## Additional Resources

- **[Endpoint Configuration](../deploy/helm/data-flywheel/endpoints.md)**: Service endpoint reference
- **[NeMo microservices Documentation](https://docs.nvidia.com/nemo/microservices/)**: Platform integration guide
- **[Kubernetes Documentation](https://kubernetes.io/docs/)**: Kubernetes concepts and operations
- **[Helm Documentation](https://helm.sh/docs/)**: Helm chart management and best practices 