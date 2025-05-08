# Configuration Guide

This guide provides detailed information about configuring the Data Flywheel Foundational Blueprint. It covers all available configuration options, their impacts, and recommended settings.

- [Configuration Guide](#configuration-guide)
  - [Overview](#overview)
  - [Configuration File Location](#configuration-file-location)
  - [NMP Configuration](#nmp-configuration)
  - [Logging Configuration](#logging-configuration)
  - [Model Integration](#model-integration)
    - [Supported Models](#supported-models)
  - [Evaluation Settings](#evaluation-settings)
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

## Overview

The Data Flywheel Foundational Blueprint uses a YAML-based configuration system. The main configuration file is located at `config/config.yaml`. This guide explains each configuration section and its options in detail.

## Configuration File Location

The primary configuration file is located at:
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
  nim_parallelism: 1
  nmp_namespace: "dfwbp"
```

| Option | Description | Default |
|--------|-------------|---------|
| `nemo_base_url` | Base URL for NeMo services | `http://nemo.test` |
| `nim_base_url` | Base URL for NIM services | `http://nim.test` |
| `datastore_base_url` | Base URL for datastore services | `http://data-store.test` |
| `nim_parallelism` | Maximum number of NIMs that can run in parallel | 1 |
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
| `gpus` | Number of GPUs to allocate | No | 1 |
| `pvc_size` | Persistent volume claim size | No | "25Gi" |
| `tag` | Model version tag | No | "1.8.3" |
| `customization_enabled` | Whether model can be fine-tuned | No | true |

### Supported Models

Currently supported models include:
- Meta Llama 3.1 8B Instruct
- Meta Llama 3.2 1B Instruct
- Meta Llama 3.2 3B Instruct
- Meta Llama 3.3 70B Instruct

Note: Not all models may be enabled by default in the configuration. Enable them by uncommenting and configuring the appropriate sections in `config/config.yaml`.

## Evaluation Settings

The `data_split_config` and `icl_config` sections control how data is split and used for evaluation:

```yaml
data_split_config:
  eval_size: 20
  val_ratio: 0.1
  min_total_records: 50
  random_seed: null

icl_config:
  max_context_length: 32768
  reserved_tokens: 4096
  max_examples: 3
  min_examples: 1
```

### Data Split Configuration

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `eval_size` | Number of examples for evaluation | 20 | Minimum size of evaluation set |
| `val_ratio` | Ratio of data used for validation | 0.1 | 10% of remaining data after eval |
| `min_total_records` | Minimum required records | 50 | Total dataset size requirement |
| `random_seed` | Seed for reproducible splits | null | Set for reproducible results |

### ICL (In-Context Learning) Configuration

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