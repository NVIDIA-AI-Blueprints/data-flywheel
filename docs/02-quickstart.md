# Getting Started With Data Flywheel Blueprint

Learn how to set up and deploy the Data Flywheel Blueprint using the steps in this guide.

This quickstart provides an initial [AIVA dataset](../data/aiva_primary_assistant_dataset.jsonl) to help you get started working with the services.

## Prerequisites

### Review Minimum System Requirements

| Requirement Type | Details |
|-------------------------|---------|
| Minimum GPU | **Self-hosted LLM Judge**: 6× (NVIDIA H100 or A100 GPUs)<br>**Remote LLM Judge**: 2× (NVIDIA H100 or A100 GPUs) |
| Cluster | Single-node NVIDIA GPU cluster on Linux with cluster-admin permissions |
| Disk Space | At least 200 GB free |
| Software | Python 3.11<br>Docker Engine<br>Docker Compose v2 |

> **📖 For complete system requirements:** See [System Requirements](03-configuration.md#system-requirements)

### Obtain an NGC API Key and Log In

You must [generate a personal API key](https://org.ngc.nvidia.com/setup/api-keys) with the `NGC catalog` and `Public API Endpoints` services selected.

> **📖 For all required API keys:** See [Required API Keys and Access](03-configuration.md#required-api-keys-and-access)

### Install and Configure Git LFS

You must have Git Large File Storage (LFS) installed and configured to download the dataset files.

1. Download and install Git LFS by following the [installation instructions](https://git-lfs.com/).
2. Initialize Git LFS in your environment.

   ```bash
   git lfs install
   ```

3. Pull the dataset into the current repository.

   ```bash
   git lfs pull
   ```

---

## Set Up the Data Flywheel Blueprint

### 1. Log In to NGC

Authenticate with NGC using `NGC login`. For detailed instructions, see the [NGC Private Registry User Guide on Accessing NGC Registry](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#accessing-ngc-registry).

### 2. Deploy NMP

To deploy NMP, follow the [NeMo Microservices Platform Prerequisites](https://docs.nvidia.com/nemo/microservices/25.8.0/get-started/setup/index.html) beginner tutorial. These instructions launch NMP using a local Minikube cluster.

> **Note**
> Data Flywheel Blueprint has been tested and is compatible with NeMo Microservices Platform (NMP) version 25.8.0.

**Use Manual Installation Only**

For the Data Flywheel Blueprint, use the [Install Manually](https://docs.nvidia.com/nemo/microservices/25.8.0/get-started/setup/minikube-manual.html) option. The deployment scripts option should be avoided as it deploys models outside the namespace of the Data Flywheel and can cause conflict.

**Use Override Values File (Bitnami Workaround)**

Due to Bitnami making PostgreSQL images private, you need to use an [override values file](../deploy/override-values.yaml) as a workaround when deploying NMP 25.8.0.

> **Note**
> This is a temporary workaround to resolve the Bitnami PostgreSQL image issue until a new NMP release is tested with the Data Flywheel Blueprint.

**Enable Customization for Models**

Before installing NMP, modify the `demo-values.yaml` file in your NMP deployment directory to enable customization for specific models:

> **Note**
> To enable customization for specific models, add the following customizer configuration to your `demo-values.yaml` file:
> 
> ```yaml
> customizer:
>   enabled: true
>   modelsStorage:
>     storageClassName: standard
>   customizationTargets:
>     overrideExistingTargets: true
>     targets:
>       meta/llama-3.2-1b-instruct@2.0:
>         enabled: true
>       meta/llama-3.2-3b-instruct@2.0:
>         enabled: true
>       meta/llama-3.1-8b-instruct@2.0:
>         enabled: true
>   customizerConfig:
>     training:
>       pvc:
>         storageClass: "standard"
>         volumeAccessMode: "ReadWriteOnce"
> ```

**Install NMP with Override Values**

When installing the NMP Helm chart, include both the `demo-values.yaml` and `override-values.yaml` files:

```bash
helm --namespace default install \
  nemo nmp/nemo-microservices-helm-chart \
  -f demo-values.yaml \
  -f override-values.yaml \
  --set guardrails.guardrails.nvcfAPIKeySecretName="nvidia-api"
```

> **Important**
> The Data Flywheel Blueprint automatically manages model deployment—spinning up or down models in the configured namespace. You don't need to intervene manually. The blueprint manages all aspects of the model lifecycle within the configured namespace.

### 3. Configure Data Flywheel

Before setting up environment variables, it's important to understand the key configuration concepts:

#### Configuration Overview

The Data Flywheel Blueprint uses a configuration file (`config/config.yaml`) that defines:

- **Model Deployments**: All candidate models (NIMs) are deployed locally in your cluster for evaluation and customization
- **LLM Judge**: Can be configured as either:
  - **Local deployment**: Runs as a NIM in your cluster (requires additional GPU resources)
  - **Remote deployment**: Uses external API endpoints (recommended for resource efficiency)
- **Evaluation Settings**: Controls how models are evaluated, including data splitting and in-context learning
- **Training Parameters**: Defines fine-tuning settings for model customization

> **Important:** After starting the Data Flywheel services, wait 4-5 minutes for all deployments to be ready before starting your first job. This delay is normal during the initialization phase as the system sets up model deployments and services.

> **📖 For complete configuration details:** See the [Configuration Guide](03-configuration.md)

#### Environment Setup

1. Set up the required environment variables:

   Create an NGC API key by following the instructions at [Generating NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-api-key).

   ```bash
   export NGC_API_KEY="<your-ngc-api-key>"
   ```

2. Set up the NVIDIA API key for remote configurations:

   Go to [build.nvidia.com](https://build.nvidia.com) and generate an NVIDIA API key, then export it:

   ```bash
   export NVIDIA_API_KEY="<your-nvidia-api-key>"
   ```

   > **Note:** The `NGC_API_KEY` is only needed for NGC login and container downloads. The `NVIDIA_API_KEY` is used for remote API access to NVIDIA services.

3. **Optional:** For remote configurations, if you want to use different API keys for LLM judge and embedding services, you can set custom environment variables:

   ```bash
   # For remote LLM judge services
   export LLM_JUDGE_API_KEY=<your-llm-judge-api-key>

   # For remote embedding services  
   export EMB_API_KEY=<your-embedding-api-key>
   ```

   > **Note:** If you don't set these custom variables, the system will use `NVIDIA_API_KEY` as the default for both remote services.
   >
   > **Tip:** You can use API keys from any provider (OpenAI, Anthropic, etc.) by setting them to `LLM_JUDGE_API_KEY` or `EMB_API_KEY` for remote configurations.

   > **📖 For complete environment setup:** See [Environment Variables](03-configuration.md#environment-variables)

2. Clone the repository:

   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/data-flywheel.git
   cd data-flywheel
   git checkout main
   ```

3. Review and modify the [configuration file](../config/config.yaml) according to your requirements.

   > **📖 For all configuration options:** See [Model Integration & Settings](03-configuration.md#model-integration)

### 4. Start Services

You have several options to start the services:

1. **Recommended:** Use the [launch script](../scripts/run.sh):

   ```bash
   ./scripts/run.sh
   ```

2. Use the [development script](../scripts/run-dev.sh):

   This script runs additional services for observability:

   - `flower`: A web UI for monitoring Celery tasks and workers
   - `kibana`: A visualization dashboard for exploring data stored in Elasticsearch

   ```bash
   ./scripts/run-dev.sh
   ```

3. Use Docker Compose directly:

   ```bash
   docker compose -f ./deploy/docker-compose.yaml up --build
   ```

   **To start with MLflow enabled:**

   ```bash
   export COMPOSE_PROFILES=mlflow && docker compose -f deploy/docker-compose.yaml up -d --build
   ```

   **Using Environment Files:**

   If you want to use an `.env` file to store all environment variables, you have two options:

   - **Option 1:** Store the `.env` file in the `deploy/` folder (Docker Compose default location):
     ```bash
     # Place your .env file at deploy/.env
     docker compose -f ./deploy/docker-compose.yaml up --build
     ```

   - **Option 2:** Use a custom `.env` file location with the `--env-file` argument:
     ```bash
     # Example: Using .env file at the root of the repo
     docker compose -f ./deploy/docker-compose.yaml --env-file .env up --build
     ```

   **Example `.env` file contents:**

   ```bash
   # API Keys for NVIDIA services
   NVIDIA_API_KEY=nvapi-your-nvidia-api-key-here
   NGC_API_KEY=nvapi-your-ngc-api-key-here
   
   # Hugging Face token for data uploading
   HF_TOKEN=hf_your-huggingface-token-here
   
   # Optional: Override API keys for specific services
   LLM_JUDGE_API_KEY=your-custom-llm-judge-api-key
   EMB_API_KEY=your-custom-embedding-api-key
   
   # Docker Compose profiles (enable MLflow)
   COMPOSE_PROFILES=mlflow
   
   # Optional: Database connection overrides
   MONGODB_URL=mongodb://localhost:27017
   REDIS_URL=redis://localhost:6379/0
   ELASTICSEARCH_URL=http://localhost:9200
   ```

   > **Note:** The `--env-file` argument allows you to specify any `.env` file location in your repository.

   > **MLflow Integration**
   >
   > MLflow is controlled by a single environment variable:
   > - Set `COMPOSE_PROFILES=mlflow` to enable both the MLflow Docker service and configuration
   > - The MLflow service will be available at `http://localhost:5000`
   > - MLflow configuration is automatically enabled when the profile is active

4. **Production Deployment:** Use Helm for Kubernetes deployment:

   For production environments or Kubernetes-based deployments, you can use Helm to deploy the Data Flywheel Blueprint.

   > **📖 For complete Helm deployment instructions:** See [Helm Installation Guide](11-helm-installation.md)

### 5. Load Data

You can feed data to the Flywheel in two ways:

1. **Manually:** For demo or short-lived environments, use the provided `load_test_data.py` script.
2. **Automatically:** For production environments where you deploy the blueprint to run continuously, use a [continuous log exportation flow](./01-architecture.md#how-production-logs-flow-into-the-flywheel). For production deployment setup, see the [Helm Installation Guide](11-helm-installation.md).

> **📖 For complete implementation guide:** See [Data Logging for AI Apps](data-logging.md)

Use the provided script and demo datasets to quickly experience the value of the Flywheel service.

#### Demo Dataset

Load test data using the provided scripts:

##### AIVA Dataset

```bash
uv run python src/scripts/load_test_data.py \
  --file aiva_primary_assistant_dataset.jsonl
```

#### Custom Data

To submit your own custom dataset, provide the loader with a file in [JSON Lines (JSONL)](https://jsonlines.org/) format. The JSONL file should contain one JSON object per line with the following structure:

#### Example Entry

```json
{"messages": [
  {"role": "user", "content": "Describe your issue here."},
  {"role": "assistant", "content": "Assistant's response goes here."}
]}
```

#### Example with Tool Calling

```json
{
  "request": {
    "model": "meta/llama-3.1-70b-instruct", 
    "messages": [
      {"role": "system", "content": "You are a chatbot that helps with purchase history."},
      {"role": "user", "content": "Has my bill been processed?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "structured_rag",
          "parameters": {
            "properties": {
              "query": {"type": "string"},
              "user_id": {"type": "string"}
            },
            "required": ["query", "user_id"]
          }
        }
      }
    ]
  },
  "response": {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
          "type": "function",
          "function": {
            "name": "structured_rag",
            "arguments": {"query": "account bill processed", "user_id": "4165"}
          }
        }]
      }
    }]
  },
  "workload_id": "primary_assistant",
  "client_id": "aiva-1",
  "timestamp": 1746138417
}
```

Each line in your dataset file should follow this structure, which is compatible with the OpenAI API request and response format.

> **Note**
> If `workload_id` and `client_id` aren't provided in the dataset entries, you can specify them when running a job.

---

## Job Operations

Now that you've got the Data Flywheel running and loaded with data, you can start running jobs.

### Quick Start Example

> **Important:** After starting the Data Flywheel services, wait 4-5 minutes for all deployments to be ready before starting your first job. This delay is normal during the initialization phase as the system sets up model deployments and services.

#### Start Job

```bash
curl -X POST http://localhost:8000/api/jobs \
-H "Content-Type: application/json" \
-d '{"workload_id": "primary_assistant", "client_id": "aiva-1"}'
```

#### Check Job Status

```bash
curl -X GET http://localhost:8000/api/jobs/:job-id -H "Content-Type: application/json"
```

> **📖 For complete API documentation:** See [API Reference](07-api-reference.md)
> 
> The API Reference includes detailed information on:
> - All available endpoints and parameters
> - Request/response schemas
> - Error handling
> - Job cancellation and deletion
> - Custom data split configuration
> - Python integration examples

### Using Notebooks

1. Launch Jupyter Lab using uv:

   ```bash
   uv run jupyter lab \
     --allow-root \
     --ip=0.0.0.0 \
     --NotebookApp.token='' \
     --port=8889 \
     --no-browser
   ```

2. Access Jupyter Lab in your browser at `http://<your-host-ip>:8889`.
3. Navigate to the `notebooks` directory.
4. Open the example notebook for running and monitoring jobs: [data-flywheel-bp-tutorial.ipynb](../notebooks/data-flywheel-bp-tutorial.ipynb)

Follow the instructions in the Jupyter Lab notebook to interact with the Data Flywheel services.

## Evaluate Results

> **📖 For detailed evaluation information:** See [Evaluation Types and Metrics](06-evaluation-types-and-metrics.md)

## Cleanup

### 1. Data Flywheel Services

When you're done using the services, you can stop them using the stop script:

```bash
./scripts/stop.sh
```

### 2. Resource Cleanup

#### Automatic Cleanup (During System Shutdown)

The system automatically cleans up running resources through the `CleanupManager` when Celery workers are shut down gracefully. This happens automatically when:
- Docker containers are stopped (`docker compose down`)
- Celery workers receive shutdown signals (SIGTERM, SIGINT)
- The system is restarted

The `CleanupManager` automatically:
1. **Finds all running flywheel runs** from the database with `PENDING` or `RUNNING` status
2. **Identifies running NIMs** with `RUNNING` deployment status for each flywheel run
3. **Cancels active customization jobs** for each running NIM
4. **Shuts down NIM deployments** using the DMS client
5. **Shuts down LLM judge deployments** (if running locally)
6. **Marks all resources as cancelled** in the database with appropriate error messages

The cleanup is triggered by the `worker_shutting_down` signal in the Celery worker, ensuring that all resources are properly cleaned up even during unexpected shutdowns.

For technical details about the automatic cleanup process, see the [Architecture Overview](01-architecture.md#automatic-resource-cleanup).

### 3. Clear Volumes

Then, you can clean up using the [clear volumes script](../scripts/clear_all_volumes.sh):

```bash
./scripts/clear_all_volumes.sh
```

This script clears all service volumes (Elasticsearch, Redis, and MongoDB).

### 4. NMP Cleanup

You can remove NMP when you're done using the platform by following the official [Uninstall NeMo Microservices Helm Chart](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-platform/uninstall-platform-helm-chart.html) guide.

## Troubleshooting

If you encounter any issues:

1. Check that all environment variables are properly set.
   - See the [Environment Variables section](03-configuration.md#environment-variables) for the complete list of required and optional variables.
2. Make sure all prerequisites are installed and configured correctly.
3. Verify that you have the necessary permissions and access to all required resources.

## Additional Resources

- [Data Flywheel Blueprint Repository](https://github.com/NVIDIA-AI-Blueprints/data-flywheel)

## Next Steps

Learn how to deploy the Data Flywheel Blueprint on Kubernetes using Helm charts for scalable, production-ready environments by following the [Helm Install Guide](11-helm-installation.md).
