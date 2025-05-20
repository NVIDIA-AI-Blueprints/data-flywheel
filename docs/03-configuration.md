# Configuration Guide

Learn how to configure the Data Flywheel Foundational Blueprint using this guide. It covers all available configuration options, their impacts, and recommended settings.

- [Configuration Guide](#configuration-guide)
  - [Before You Start](#before-you-start)
  - [Configuration File Location](#configuration-file-location)
  - [NMP Configuration](#nmp-configuration)
  - [Logging Configuration](#logging-configuration)
  - [Environment Variables](#environment-variables)
  - [Model Integration](#model-integration)
    - [Supported Models](#supported-models)
  - [Evaluation Settings](#evaluation-settings)
    - [LLM Judge Configuration](#llm-judge-configuration)
    - [Data Split Configuration](#data-split-configuration)
    - [ICL (In-Context Learning) Configuration](#icl-in-context-learning-configuration)
  - [Fine-tuning Options](#fine-tuning-options)
    - [Training Configuration](#training-configuration)
    - [LoRA Configuration](#lora-configuration)
  - [Data Infrastructure](#data-infrastructure)
    - [Storage Services](#storage-services)
    - [Processing Configuration](#processing-configuration)
  - [Deployment Options](#deployment-options)
    - [Development Environment](#development-environment)
    - [Production Environment](#production-environment)
    - [Resource Configuration](#resource-configuration)

## Before You Start

- Run the [Quickstart](./02-quickstart.md) to tour the functionality and default settings of the Flywheel service.
- Run the [Notebooks](../notebooks/README.md) to understand how the Flywheel supports different use cases.

> **Important**
>
> Advanced configuration such as Large-scale hyper-parameter sweeps, architecture search, or custom evaluation metrics must run directly in **NeMo Microservices Platform (NMP)**. The configurations in this guide are only for the blueprint itself.

> **Note**
> 
> For detailed NMP API documentation, refer to the [official documentation](https://docs.nvidia.com/nemo/microservices/latest/api/index.html).

## Configuration File Location

The Data Flywheel Foundational Blueprint uses a YAML-based configuration system. The primary configuration file is located at:

```bash
config/config.yaml
```

## NMP Configuration

The `nmp_config` section controls the NeMo Microservices Platform (NMP) integration:

```yaml
nmp_config:
  nemo_base_url: "http://nemo.test"
  nim_base_url: "http://nim.test"
  datastore_base_url: "http://data-store.test"
  nmp_namespace: "dfwbp"
```

| Option | Description | Default |
|--------|-------------|---------|
| `nemo_base_url` | Base URL for NeMo services | `http://nemo.test` |
| `nim_base_url` | Base URL for NIM services | `http://nim.test` |
| `datastore_base_url` | Base URL for datastore services | `http://data-store.test` |
| `nmp_namespace` | Namespace for NMP resources | "dfwbp" |

## Logging Configuration

The `logging_config` section controls the verbosity of log output across all services:

```yaml
logging_config:
  level: "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `level` | Log verbosity level | "INFO" | Controls detail level of application logs |

The `logging_config` section configures logging level. Available options:

- `CRITICAL`: Only critical errors
- `ERROR`: Error events 
- `WARNING`: Warning messages
- `INFO`: Informational messages (default)
- `DEBUG`: Detailed diagnostic information

## Environment Variables

The Data Flywheel Foundational Blueprint relies on several environment variables for configuration. These can be set in a `.env` file in the project root or as system environment variables.

| Variable | Description | Required | Default | Notes |
|----------|-------------|----------|---------|-------|
| `NGC_API_KEY` | API key for NVIDIA Cloud Foundation | Yes | None | Used for LLM judge when configured as remote |
| `HF_TOKEN` | Hugging Face authentication token | Yes | None | Required for data uploading functionality |
| `ES_COLLECTION_NAME` | Name of the Elasticsearch collection | No | "flywheel" | Used by Elasticsearch client |
| `ELASTICSEARCH_URL` | URL for Elasticsearch connection | No | "http://localhost:9200" | Used by Elasticsearch client |
| `MONGODB_URL` | URL for MongoDB connection | No | "mongodb://localhost:27017" | Used by MongoDB client |
| `MONGODB_DB` | Name of the MongoDB database | No | "flywheel" | Used by MongoDB client |
| `REDIS_URL` | URL for Redis connection | No | "redis://localhost:6379/0" | Used for task broker and results backend |

For full functionality, at minimum you should configure:

```bash
export NGC_API_KEY="your-key-here"
export HF_TOKEN="your-huggingface-token"
```

## Model Integration

The `nims` section configures which models to deploy and their settings:

```yaml
nims:
  - model_name: "meta/llama-3.2-1b-instruct"
    context_length: 32768
    gpus: 1
    pvc_size: 25Gi
    tag: "1.8.3"
    customization_enabled: true
```

| Option | Description | Required | Example |
|--------|-------------|----------|---------|
| `model_name` | Name of the model to deploy | Yes | "meta/llama-3.2-1b-instruct" |
| `context_length` | Maximum context length in tokens | Yes | 32768 |
| `gpus` | Number of GPUs to allocate | Yes | 1 |
| `pvc_size` | Persistent volume claim size | No | "25Gi" |
| `tag` | Model version tag | Yes | "1.8.3" |
| `customization_enabled` | Whether model can be fine-tuned | No | true |

### Supported Models

Currently supported models include:
- Meta Llama 3.1 8B Instruct
- Meta Llama 3.2 1B Instruct
- Meta Llama 3.2 3B Instruct
- Meta Llama 3.3 70B Instruct

Note: Not all models may be enabled by default in the configuration. Enable them by uncommenting and configuring the appropriate sections in `config/config.yaml`.

## Evaluation Settings

The `llm_judge_config`, `data_split_config`, and `icl_config` sections control evaluation processes:

### LLM Judge Configuration

The `llm_judge_config` section configures the LLM used for evaluating model outputs:

**NOTE: By default llm_judge_config is set to remote configuration.**

```yaml
llm_judge_config:
  type: "remote"
  url: "https://integrate.api.nvidia.com/v1/chat/completions"
  model_id: "meta/llama-3.3-70b-instruct"
  api_key_env: "NGC_API_KEY"
```

| Option | Description | Required | Example |
|--------|-------------|----------|---------|
| `type` | Deployment type (remote or local) | Yes | "remote" |
| `url` | API endpoint for remote LLM | Yes (if remote) | "https://integrate.api.nvidia.com/v1/chat/completions" |
| `model_id` | Model identifier | Yes | "meta/llama-3.3-70b-instruct" |
| `api_key_env` | Environment variable name containing API key | Yes (if remote) | "NGC_API_KEY" |

For local deployment, use the following configuration instead:

```yaml
llm_judge_config:
  type: "local"
  model_name: "meta/llama-3.3-70b-instruct"
  context_length: 32768
  gpus: 1
  pvc_size: 25Gi
  tag: "1.8.3"
```

| Option | Description | Required | Example |
|--------|-------------|----------|---------|
| `type` | Deployment type (remote or local) | Yes | "local" |
| `model_name` | Name of the model to deploy | Yes | "meta/llama-3.3-70b-instruct" |
| `context_length` | Maximum context length in tokens | Yes | 32768 |
| `gpus` | Number of GPUs to allocate | Yes | 1 |
| `pvc_size` | Persistent volume claim size | Yes | "25Gi" |
| `tag` | Model version tag | Yes | "1.8.3" |

### Data Split Configuration

```yaml
data_split_config:
  eval_size: 20
  val_ratio: 0.1
  min_total_records: 50
  random_seed: null
  limit: null
```

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `eval_size` | Number of examples for evaluation | 20 | Minimum size of evaluation set |
| `val_ratio` | Ratio of data used for validation | 0.1 | 10% of remaining data after eval |
| `min_total_records` | Minimum required records | 50 | Total dataset size requirement |
| `random_seed` | Seed for reproducible splits | null | Set for reproducible results |
| `limit` | Limit for evaluator | null | Set for evaluator config limit |

### ICL (In-Context Learning) Configuration

```yaml
icl_config:
  max_context_length: 32768
  reserved_tokens: 4096
  max_examples: 3
  min_examples: 1
```

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `max_context_length` | Maximum tokens in context | 32768 | Model dependent |
| `reserved_tokens` | Tokens reserved for system | 4096 | For prompts and metadata |
| `max_examples` | Maximum ICL examples | 3 | Upper limit per context |
| `min_examples` | Minimum ICL examples | 1 | Lower limit per context |

## Fine-tuning Options

The `training_config` and `lora_config` sections control model fine-tuning:

```yaml
training_config:
  training_type: "sft"
  finetuning_type: "lora"
  epochs: 2
  batch_size: 16
  learning_rate: 0.0001

lora_config:
  adapter_dim: 32
  adapter_dropout: 0.1
```

### Training Configuration

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `training_type` | Type of training | "sft" | Supervised Fine-Tuning |
| `finetuning_type` | Fine-tuning method | "lora" | Low-Rank Adaptation |
| `epochs` | Training epochs | 2 | Full passes through data |
| `batch_size` | Batch size | 16 | Samples per training step |
| `learning_rate` | Learning rate | 0.0001 | Training step size |

### LoRA Configuration

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `adapter_dim` | LoRA adapter dimension | 32 | Rank of adaptation |
| `adapter_dropout` | Dropout rate | 0.1 | Regularization parameter |

## Data Infrastructure

The Data Flywheel uses several services for data storage and processing:

### Storage Services

| Service | Purpose | Configuration Location |
|---------|---------|----------------------|
| Elasticsearch | Log storage | `deploy/docker-compose.yaml` |
| MongoDB | API data persistence | `deploy/docker-compose.yaml` |
| Redis | Task queue | `deploy/docker-compose.yaml` |

### Processing Configuration

| Component | Purpose | Configuration |
|-----------|---------|---------------|
| Celery Workers | Background processing | Configurable concurrency |
| API Server | REST endpoints | FastAPI configuration |

## Deployment Options

The deployment configuration is primarily managed through Docker Compose:

### Development Environment

```bash
./scripts/run-dev.sh
```

Includes additional services:
- Flower (Celery monitoring)
- Kibana (Elasticsearch visualization)

### Production Environment

```bash
./scripts/run.sh
```

Standard deployment with core services:
- API Server
- Celery Workers
- Redis
- MongoDB
- Elasticsearch

### Resource Configuration

| Resource | Configuration | Notes |
|----------|--------------|-------|
| Network Mode | `deploy/docker-compose.yaml` | Service networking |
| Volume Mounts | `deploy/docker-compose.yaml` | Persistent storage |
| Health Checks | `deploy/docker-compose.yaml` | Service monitoring |
| Environment | `.env` file or environment variables | API keys and URLs | 