# NVIDIA Data Flywheel Blueprint on OpenShift

This guide provides instructions for deploying the NVIDIA Data Flywheel Blueprint on Red Hat OpenShift Container Platform (OCP), including integration with NVIDIA NeMo Microservices.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Step 1: Deploy NeMo Microservices](#step-1-deploy-nemo-microservices)
  - [Step 2: Deploy Data Flywheel](#step-2-deploy-data-flywheel)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## Overview

The Data Flywheel Blueprint on OpenShift provides:
- **OpenShift-native deployment** with Security Context Constraints (SCC) compliance
- **Route-based external access** using OpenShift Routes with TLS
- **Shared secret management** with NeMo Microservices
- **Storage class flexibility** for OpenShift-specific storage backends
- **Restricted security contexts** compliant with OpenShift's security policies

## Prerequisites

### Cluster Requirements
- Red Hat OpenShift Container Platform 4.x
- GPU-enabled worker nodes (for model training and inference)
- Cluster administrator access for creating namespaces and RBAC resources
- Available storage class for persistent volumes (e.g., `gp3-csi`)

### Required Tools
- `oc` CLI configured and authenticated to your cluster
- `helm` 3.x installed locally
- `git` for cloning repositories

### API Keys
You will need the following API keys:

| Key | Purpose | How to Obtain |
|-----|---------|---------------|
| **NGC_API_KEY** | Pull NVIDIA container images and download models | [Generate at NGC](https://ngc.nvidia.com/setup/api-key) |
| **NVIDIA_API_KEY** | Access NVIDIA API Catalog for remote inference | [Generate at NVIDIA API Catalog](https://build.nvidia.com/explore/discover) |
| **HF_TOKEN** | Access Hugging Face models and datasets | [Generate at Hugging Face](https://huggingface.co/settings/tokens) |

### Storage Requirements

The deployment requires persistent storage across multiple components:

| Component | Purpose | Default Size | Adjustable |
|-----------|---------|--------------|------------|
| NeMo Customizer (model cache) | Base model storage | 100 Gi | Yes |
| NeMo Customizer (workspace) | Training workspace | 50 Gi | Yes |
| Vector DB (Milvus) | Embedding vectors | 100 Gi | Yes |
| NIMCache | Model inference cache | 100 Gi | Yes |
| PostgreSQL databases | Multiple DB instances | ~10 Gi total | Yes |
| Elasticsearch | Logging and data | Ephemeral | Via resources |
| MongoDB | Metadata storage | Ephemeral | Via resources |
| Redis | Task queueing | Ephemeral | Via resources |

**Total estimated storage:** ~360 Gi for a minimal deployment with persistent storage.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OpenShift Routes                     │
│  (TLS Edge Termination - External Access)               │
│  • df-api-route      → Data Flywheel API                │
│  • df-mlflow-route   → MLflow UI                        │
│  • df-kibana-route   → Kibana UI                        │
│  • df-flower-route   → Celery Flower UI                 │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│              Data Flywheel Services                     │
│  • df-api            → FastAPI REST API                 │
│  • df-celery-worker  → Task execution workers           │
│  • df-celery-parent  → Task orchestration               │
│  • df-mlflow         → Experiment tracking              │
│  • df-flower         → Task monitoring                  │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│           Infrastructure Services                       │
│  • df-elasticsearch  → Logging and data storage         │
│  • df-mongodb        → Metadata storage                 │
│  • df-redis          → Task queue                       │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│         NeMo Microservices Platform                     │
│  • nemo-gateway      → API Gateway                      │
│  • nemodatastore     → Data management                  │
│  • nemocustomizer    → Model fine-tuning                │
│  • nemoevaluator     → Model evaluation                 │
│  • nemoguardrails    → Safety guardrails                │
│  • NIM services      → Model inference                  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                Shared Kubernetes Secrets                │
│  • nvcrimagepullsecret → NGC container registry auth    │
│  • ngc-api             → NGC API key                    │
│  • nvidia-api          → NVIDIA API key                 │
│  • hf-secret           → Hugging Face token             │
└─────────────────────────────────────────────────────────┘
```

## Installation

### Step 1: Deploy NeMo Microservices

The Data Flywheel Blueprint depends on NeMo Microservices for model training, evaluation, and inference capabilities.

**📖 Follow the complete installation guide:**
**[NeMo Microservices on OpenShift Installation Guide](https://github.com/RHEcosystemAppEng/NeMo-Microservices)**

This guide includes OpenShift-specific configuration, Security Context Constraints, GPU operator setup, secret creation, and verification steps.

**Important:** The secrets created during NeMo installation (`nvcrimagepullsecret`, `ngc-api`, `nvidia-api`, `hf-secret`) are shared with Data Flywheel. Do not delete or modify these secrets.

### Step 2: Deploy Data Flywheel

Once NeMo Microservices is running, deploy the Data Flywheel Blueprint.

#### 2.1 Clone the Repository

```bash
git clone https://github.com/NVIDIA-AI-Blueprints/nvidia-data-flywheel.git
cd nvidia-data-flywheel
```

#### 2.2 Install Data Flywheel using Helm

Since NeMo Microservices has already created the required secrets, you can deploy Data Flywheel without re-creating them:

```bash
# Set your namespace (same as NeMo installation)
export NAMESPACE="data-flywheel"

# Install Data Flywheel
helm install data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  --set openshift.enabled=true \
  --set namespace=$NAMESPACE \
  --set foundationalFlywheelServer.config.nmp_config.nemo_base_url="http://nemo-gateway" \
  --set foundationalFlywheelServer.config.nmp_config.nmp_namespace="$NAMESPACE" \
  --set nemo-microservices-helm-chart.enabled=false \
  --set mongodb.image.repository="docker.io/mongo" \
  --skip-crds
```

**Command Breakdown:**
- `--set openshift.enabled=true`: Enables OpenShift-specific features (Routes, SCCs, security contexts)
- `--set namespace=$NAMESPACE`: Sets the deployment namespace
- `--set foundationalFlywheelServer.config.nmp_config.nemo_base_url="http://nemo-gateway"`: Points to NeMo gateway service
- `--set foundationalFlywheelServer.config.nmp_config.nmp_namespace="$NAMESPACE"`: Tells Data Flywheel where NeMo resources are located
- `--set nemo-microservices-helm-chart.enabled=false`: Disables NeMo subchart (already installed separately)
- `--set mongodb.image.repository="docker.io/mongo"`: Uses Docker Hub MongoDB image (OpenShift compatible)
- `--skip-crds`: Skips CRD installation (volcano CRDs already installed by NeMo)

#### 2.3 Alternative: Using a Values File

For production deployments, you can create a custom values file:

```bash
# Create a custom values file
cat > values-openshift.yaml <<EOF
# OpenShift configuration
openshift:
  enabled: true
  storageClass: "gp3-csi"  # Adjust to your cluster's storage class

# Namespace configuration
namespace: "$NAMESPACE"

# Shared secrets (already created by NeMo installation)
# Leave empty to use existing secrets
secrets:
  ngcApiKey: ""
  nvidiaApiKey: ""
  hfToken: ""
  llmJudgeApiKey: ""
  embApiKey: ""

# NeMo Microservices integration
foundationalFlywheelServer:
  config:
    nmp_config:
      nemo_base_url: "http://nemo-gateway"
      nim_base_url: "http://nemo-gateway"
      datastore_base_url: "http://nemodatastore-sample:8000"
      nmp_namespace: "$NAMESPACE"

# Disable NeMo subchart (already installed)
nemo-microservices-helm-chart:
  enabled: false

# Use Docker Hub for MongoDB (OpenShift compatible)
mongodb:
  image:
    repository: "docker.io/mongo"
EOF

# Install using the values file
helm install data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  -f values-openshift.yaml \
  --skip-crds
```

## Configuration

### OpenShift-Specific Settings

The Helm chart includes several OpenShift-specific configurations enabled via `openshift.enabled=true`:

#### Security Context Constraints (SCC)

All deployments use the `restricted-v2` SCC with:
- **Pod-level security:**
  - `runAsNonRoot: true`
  - `seccompProfile.type: RuntimeDefault`
  - UID/GID automatically assigned by OpenShift from namespace range

- **Container-level security:**
  - `allowPrivilegeEscalation: false`
  - `runAsNonRoot: true`
  - `capabilities.drop: [ALL]`

#### OpenShift Routes

External access is provided via OpenShift Routes with TLS edge termination:

| Route Name | Service | Default Port | Purpose |
|------------|---------|--------------|---------|
| `df-api-route` | df-api-service | 8000 | Data Flywheel REST API |
| `df-mlflow-route` | df-mlflow-service | 5000 | MLflow experiment tracking UI |
| `df-kibana-route` | df-kibana-service | 5601 | Kibana logging visualization |
| `df-flower-route` | df-flower-service | 5555 | Celery task monitoring |

Access routes with:
```bash
# Get route URLs
oc get routes -n $NAMESPACE

# Example: Access API
API_URL=$(oc get route df-api-route -n $NAMESPACE -o jsonpath='{.spec.host}')
curl https://$API_URL/health
```

#### Storage Classes

By default, the chart uses `gp3-csi` storage class for OpenShift. Adjust this in `values.yaml` or via command-line:

```bash
--set openshift.storageClass="your-storage-class"
```

### Secret Management

The Data Flywheel chart creates secrets conditionally. When installing on top of existing NeMo Microservices:

**Shared Secrets (created by NeMo):**
- `nvcrimagepullsecret`: NGC container registry authentication
- `ngc-api`: NGC API key for model downloads
- `nvidia-api`: NVIDIA API key for remote inference
- `hf-secret`: Hugging Face token for datasets

**Data Flywheel-Only Secrets (optional):**
- `llm-judge-api`: LLM Judge API key (if using separate service)
- `emb-api`: Embedding API key (if using separate service)

To create Data Flywheel-specific secrets:
```bash
# Only if you need separate API keys for LLM Judge or Embeddings
oc create secret generic llm-judge-api \
  --from-literal=LLM_JUDGE_API_KEY="<your-key>" \
  -n $NAMESPACE

oc create secret generic emb-api \
  --from-literal=EMB_API_KEY="<your-key>" \
  -n $NAMESPACE
```

### Resource Configuration

Adjust resources based on your cluster capacity. Example for resource-constrained environments:

```yaml
# In values-openshift.yaml
elasticsearch:
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1"

mongodb:
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"

foundationalFlywheelServer:
  deployments:
    celeryWorker:
      resources:
        requests:
          memory: "4Gi"
          cpu: "2"
        limits:
          memory: "8Gi"
          cpu: "4"
```

## Verification

### Check Pod Status

```bash
# View all Data Flywheel pods
oc get pods -n $NAMESPACE | grep df-

# Expected output (all Running):
# df-api-<hash>                  1/1     Running
# df-celery-worker-<hash>        1/1     Running
# df-celery-parent-worker-<hash> 1/1     Running
# df-elasticsearch-<hash>        1/1     Running
# df-mongodb-<hash>              1/1     Running
# df-redis-<hash>                1/1     Running
# df-mlflow-<hash>               1/1     Running (if mlflow enabled)
# df-flower-<hash>               1/1     Running (if not production)
# df-kibana-<hash>               1/1     Running (if not production)
```

### Check Services

```bash
# View services
oc get svc -n $NAMESPACE | grep df-

# Expected services:
# df-api-service
# df-elasticsearch-service
# df-mongodb-service
# df-redis-service
# df-mlflow-service
# df-flower-service
# df-kibana-service
```

### Check Routes

```bash
# View routes
oc get routes -n $NAMESPACE

# Test API health
API_URL=$(oc get route df-api-route -n $NAMESPACE -o jsonpath='{.spec.host}')
curl -k https://$API_URL/health

# Expected response:
# {"status": "healthy"}
```

### Check Secrets

```bash
# Verify all required secrets exist
oc get secrets -n $NAMESPACE | grep -E "(nvcrimagepullsecret|ngc-api|nvidia-api|hf-secret)"

# Expected output:
# nvcrimagepullsecret   kubernetes.io/dockerconfigjson   1      Xm
# ngc-api               Opaque                           1      Xm
# nvidia-api            Opaque                           1      Xm
# hf-secret             Opaque                           1      Xm
```

### Verify NeMo Integration

```bash
# Check if Data Flywheel can reach NeMo gateway
oc exec -n $NAMESPACE deployment/df-api -- curl -s http://nemo-gateway/health

# Check NeMo datastore
oc exec -n $NAMESPACE deployment/df-api -- curl -s http://nemodatastore-sample:8000/health
```

### Access Web Interfaces

```bash
# Get all route URLs
echo "Data Flywheel API:    https://$(oc get route df-api-route -n $NAMESPACE -o jsonpath='{.spec.host}')"
echo "MLflow UI:            https://$(oc get route df-mlflow-route -n $NAMESPACE -o jsonpath='{.spec.host}')"
echo "Kibana UI:            https://$(oc get route df-kibana-route -n $NAMESPACE -o jsonpath='{.spec.host}')"
echo "Flower UI:            https://$(oc get route df-flower-route -n $NAMESPACE -o jsonpath='{.spec.host}')"
```

## Troubleshooting

### Pods in ImagePullBackOff

**Symptom:** Pods fail to pull NVIDIA images from `nvcr.io`

**Solution:**
```bash
# Verify NGC secret exists and is correct
oc get secret nvcrimagepullsecret -n $NAMESPACE -o yaml

# Recreate if needed
oc delete secret nvcrimagepullsecret -n $NAMESPACE
oc create secret docker-registry nvcrimagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  -n $NAMESPACE
```

### Pods in CreateContainerConfigError

**Symptom:** Pods cannot find required secrets

**Solution:**
```bash
# Check which secret is missing
oc describe pod <pod-name> -n $NAMESPACE | grep -A 5 "Secret"

# Verify all required secrets exist
oc get secrets -n $NAMESPACE | grep -E "(ngc-api|nvidia-api|hf-secret)"

# Create missing secrets (these should have been created by NeMo installation)
# Refer to the NeMo Microservices installation guide
```

### SCC Permission Errors

**Symptom:** Pods fail with security context constraint violations

**Solution:**
```bash
# Verify openshift.enabled=true was set during installation
helm get values data-flywheel -n $NAMESPACE | grep "enabled: true"

# If not, upgrade with correct settings
helm upgrade data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  --set openshift.enabled=true \
  --reuse-values
```

### Cannot Connect to NeMo Services

**Symptom:** Data Flywheel cannot reach `nemo-gateway` or other NeMo services

**Solution:**
```bash
# Verify NeMo gateway service exists
oc get svc nemo-gateway -n $NAMESPACE

# If not found, check if NeMo is in a different namespace
oc get svc --all-namespaces | grep nemo-gateway

# Update the nemo_base_url if in different namespace
helm upgrade data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  --set foundationalFlywheelServer.config.nmp_config.nemo_base_url="http://nemo-gateway.<nemo-namespace>.svc.cluster.local" \
  --set foundationalFlywheelServer.config.nmp_config.nmp_namespace="<nemo-namespace>" \
  --reuse-values
```

### Elasticsearch Out of Memory

**Symptom:** Elasticsearch pod crashes or is OOMKilled

**Solution:**
```bash
# Increase memory limits
helm upgrade data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  --set elasticsearch.resources.requests.memory="4Gi" \
  --set elasticsearch.resources.limits.memory="8Gi" \
  --set elasticsearch.env.ES_JAVA_OPTS="-Xms4g -Xmx4g" \
  --reuse-values
```

### Celery Workers Not Processing Tasks

**Symptom:** Tasks stuck in pending state, visible in Flower UI

**Solution:**
```bash
# Check worker logs
oc logs -n $NAMESPACE deployment/df-celery-worker --tail=100

# Check Redis connectivity
oc exec -n $NAMESPACE deployment/df-celery-worker -- redis-cli -h df-redis-service ping

# Restart workers
oc rollout restart deployment/df-celery-worker -n $NAMESPACE
oc rollout restart deployment/df-celery-parent-worker -n $NAMESPACE
```

### Routes Not Accessible

**Symptom:** Cannot access services via route URLs

**Solution:**
```bash
# Verify routes exist
oc get routes -n $NAMESPACE

# Check route TLS settings
oc get route df-api-route -n $NAMESPACE -o yaml

# Test internal service first
oc exec -n $NAMESPACE deployment/df-api -- curl -s http://df-api-service:8000/health

# Check router logs
oc logs -n openshift-ingress deployment/router-default
```

### Helm Adoption Errors

**Symptom:** `Secret 'xyz' exists and cannot be imported into the current release`

**Solution:**
This occurs when secrets exist without Helm ownership labels. The chart has been updated to create secrets conditionally. Ensure you're not passing secret values during installation if they already exist from NeMo:

```bash
# Don't pass --set secrets.* flags if secrets already exist
helm install data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  --set openshift.enabled=true \
  --set namespace=$NAMESPACE \
  # ... other flags but NO --set secrets.ngcApiKey, etc.
```

## Advanced Configuration

### Using Custom Storage Classes

```bash
helm install data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  --set openshift.enabled=true \
  --set openshift.storageClass="fast-ssd" \
  # ... other flags
```

### Disabling Non-Production Services

For production deployments, disable Kibana and Flower:

```bash
helm install data-flywheel ./deploy/helm/data-flywheel \
  -n $NAMESPACE \
  --set profile.production.enabled=true \
  # ... other flags
```

This automatically disables:
- Kibana UI and route
- Flower UI and route

### Using Local NIMs Instead of Remote API

By default, Data Flywheel uses remote NVIDIA API for LLM judge and embeddings. To use local NIMs:

```yaml
# In values-openshift.yaml
foundationalFlywheelServer:
  config:
    llm_judge_config:
      deployment_type: "local"
      model_name: "meta/llama-3.3-70b-instruct"
      context_length: 32768
      gpus: 4
      pvc_size: 25Gi
      tag: "1.8.5"

    icl_config:
      similarity_config:
        embedding_nim_config:
          deployment_type: "local"
          model_name: "nvidia/llama-3.2-nv-embedqa-1b-v2"
          context_length: 32768
          gpus: 1
          pvc_size: "25Gi"
          tag: "1.9.0"
```

### Scaling Workers

Adjust Celery worker concurrency based on your workload:

```yaml
# In values-openshift.yaml
foundationalFlywheelServer:
  deployments:
    celeryWorker:
      resources:
        requests:
          memory: "10Gi"
          cpu: "4"
        limits:
          memory: "20Gi"
          cpu: "8"
      command:
        - "uv"
        - "run"
        - "celery"
        - "-A"
        - "src.tasks.cli:celery_app"
        - "worker"
        - "--loglevel=info"
        - "--concurrency=100"  # Increase from default 50
        - "--queues=celery"
        - "-n"
        - "main_worker@%h"
        - "--purge"
```

## References

- [NeMo Microservices on OpenShift](https://github.com/RHEcosystemAppEng/NeMo-Microservices) - Complete OpenShift installation guide for NeMo
- [Data Flywheel Blueprint Documentation](https://github.com/NVIDIA-AI-Blueprints/nvidia-data-flywheel/tree/main/docs)
- [NVIDIA NeMo Microservices Documentation](https://docs.nvidia.com/nemo/microservices/latest/index.html)
- [OpenShift Security Context Constraints](https://docs.openshift.com/container-platform/latest/authentication/managing-security-context-constraints.html)
- [OpenShift Routes](https://docs.openshift.com/container-platform/latest/networking/routes/route-configuration.html)

## Support

For issues specific to:
- **OpenShift deployment**: Create an issue in this repository
- **NeMo Microservices on OpenShift**: Refer to [NeMo Microservices on OpenShift repo](https://github.com/RHEcosystemAppEng/NeMo-Microservices)
- **Data Flywheel functionality**: Create an issue in the [main Data Flywheel repository](https://github.com/NVIDIA-AI-Blueprints/data-flywheel)
