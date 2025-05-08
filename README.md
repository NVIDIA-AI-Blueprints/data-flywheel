# Data Flywheel Foundational Blueprint

This is the NVIDIA Flywheel Foundational Blueprint. In real-world tests, NVIDIA has found that using a Flywheel can reduce inference costs by up to 98.6% by replacing `llama-3.1-70b-instruct` with a fine-tuned version of `llama-3.2-1b-instruct`. This Blueprint contains a reference implementation of a Flywheel service that you can deploy in your own infrastructure to achieve similar results. It leverages the NeMo Microservice Platform for managing datasets, running evaluations, and fine-tuning models.

- [Data Flywheel Foundational Blueprint](#data-flywheel-foundational-blueprint)
  - [Overview](#overview)
  - [Key Features](#key-features)
  - [What is a Flywheel?](#what-is-a-flywheel)
  - [Target Audience](#target-audience)
  - [Software Components](#software-components)
  - [Technical Diagrams](#technical-diagrams)
  - [Minimum System Requirements](#minimum-system-requirements)
    - [Software Requirements](#software-requirements)
    - [Service Requirements](#service-requirements)
    - [Resource Requirements](#resource-requirements)
    - [Development Environment](#development-environment)
    - [Production Environment](#production-environment)
  - [Next Steps](#next-steps)
  - [Available Customizations](#available-customizations)
  - [Contributing](#contributing)
  - [License](#license)

## Overview

The Data Flywheel Foundational Blueprint is a continuous improvement system for AI models. It provides a complete framework for collecting real-world usage data, generating evaluation and training data, and optimizing model performance through various strategies.

## Key Features

- Data Collection and Storage:
  - Elasticsearch for logging prompt/completion data
  - MongoDB for API and metadata storage
  - Redis for task queue management
- Model Integration:
  - Support for Meta Llama 3.2 1B Instruct model
  - Configurable context length up to 32768 tokens
- Training and Evaluation:
  - In-context learning (ICL) with configurable parameters
  - LoRA-based fine-tuning support
  - Automated data splitting for evaluation
- Deployment Infrastructure:
  - Docker Compose setup for development
  - Celery workers for background processing
  - Health monitoring for core services

## What is a Flywheel?

A Flywheel is a system you deploy alongside your Generative AI applications. It creates eval and fine tuning datasets from production prompt/completion logs, and automates the process of finding smaller NIMs that are as accurate as the reference NIMs used in production. It requires no data curation, no hand-labeled data, and no human intervention at all other than instrumenting your application to log data and then reading the Flywheel results to see if there is a model you want to promote to production to replace the NIM you are using in production.

```mermaid
flowchart LR

app["Your application"] --Prompt/completion logs--> log_store["Log Store"]
log_store --Datasets--> datasets["NeMo Datastore"]
datasets --"Fine-tuning datasets"--> customizer["NeMo Customizer"]
customizer --"Customized model"--> NIM
datasets --"Eval datasets"--> evaluator["NeMo Evaluator"]
evaluator --> NIM
NIM --> evaluator
evaluator --> results["Flywheel Results"]
```

You can also learn more about Flywheels here:

* [Enhance Your AI Agent with Data Flywheels Using NVIDIA NeMo Microservices](https://developer.nvidia.com/blog/enhance-your-ai-agent-with-data-flywheels-using-nvidia-nemo-microservices/)
* [Nvidia Releases NeMo Microservices To Streamline AI Agent Development](https://www.forbes.com/sites/janakirammsv/2025/04/25/nvidia-releases-nemo-microservices-to-streamline-ai-agent-development/)
* [Overview of NeMo Microservices](https://docs.nvidia.com/nemo/microservices/latest/about/index.html)
* [Enterprises Onboard AI Teammates Faster With NVIDIA NeMo Tools to Scale Employee Productivity](https://blogs.nvidia.com/blog/nemo-enterprises-ai-teammates-employee-productivity/)

## Target Audience

This blueprint is designed for:

- Generalists
- AI Application Engineers
- ML Engineers
- Data Scientists
- DevOps Engineers working with AI systems

## Software Components

The blueprint consists of the following implemented components:

- **API Layer**:
  - FastAPI-based REST endpoints (`src/api/endpoints.py`)
  - Data models and schemas (`src/api/models.py`, `src/api/schemas.py`)
  - Job service for task management (`src/api/job_service.py`)
- **Data Storage**:
  - Elasticsearch for log storage
  - MongoDB for API data persistence (`src/api/db.py`)
  - Redis for task queue
- **Task Processing**:
  - Celery workers for background jobs (`src/tasks/tasks.py`)
  - Configurable concurrency and monitoring
- **NeMo Microservices Integration**:
  - Datastore client for dataset management
  - Model evaluation and customization interfaces
  - Configurable NMP endpoints

## Technical Diagrams

For details on the architecture of a Flywheel and the components of this Blueprint, view the [Architecture Overview](./docs/01-architecture.md).

## Minimum System Requirements

**Note**: The following does not include requirements for NMP (NeMo Microservices Platform)

### Software Requirements

- Python 3.11
- Docker Engine
- Docker Compose v2

### Service Requirements

- Elasticsearch 8.12.2
- MongoDB 7.0
- Redis 7.2
- FastAPI (API server)
- Celery (task processing)

### Resource Requirements

- Minimum Memory: 1GB (512MB reserved for Elasticsearch)
- Storage: Varies based on log volume and model size
- Network: Ports 8000 (API), 9200 (Elasticsearch), 27017 (MongoDB), 6379 (Redis)

### Development Environment

- Docker Compose for local development with hot reloading
- Support for macOS (Darwin) and Linux
- Optional: GPU support for model inference

### Production Environment

- Kubernetes cluster (recommended for production)
- Resource requirements scale with workload
- Persistent volume support for data storage

## Next Steps

- Review the [Architecture Overview](./docs/01-architecture.md)
- Follow the [Quickstart Guide](./docs/02-quickstart.md) to deploy this blueprint
- Explore the full [API Specification](./openapi.json) to understand all available endpoints

## Available Customizations

The following are some of the customizations that you can make after you complete the steps in the [Quickstart Guide](./docs/02-quickstart.md).

| Category | Description | Available Options |
|----------|-------------|------------------|
| [Model Integration](docs/03-configuration.md#model-integration) | Configure and deploy LLM models | • **Currently Supported**: Meta Llama 3.2 1B Instruct<br>• **Context Length**: Up to 32768 tokens<br>• **Hardware Config**: GPU support (configurable), PVC size (configurable)<br>• **Version Control**: Model tags supported |
| [Evaluation Settings](docs/03-configuration.md#evaluation-settings) | Configure data splitting and evaluation parameters | • **Data Split**: Eval size (default: 20), validation ratio (0.1)<br>• **Minimum Records**: 50 records required<br>• **Reproducibility**: Optional random seed<br>• **ICL Settings**: Context length (max 32768), reserved tokens (4096), examples (min 1, max 3) |
| [Fine-tuning Options](docs/03-configuration.md#fine-tuning-options) | Customize model training | • **Training Type**: SFT (Supervised Fine-Tuning)<br>• **Method**: LoRA with configurable parameters<br>• **Parameters**: epochs (2), batch size (16), learning rate (0.0001)<br>• **LoRA Config**: adapter dimension (32), dropout (0.1) |
| [Data Infrastructure](docs/03-configuration.md#data-infrastructure) | Configure data storage and processing | • **Storage**: Elasticsearch for logs<br>• **Queue**: Redis for task processing<br>• **Database**: MongoDB for API data<br>• **Processing**: Celery workers with configurable concurrency |
| [Deployment Options](docs/03-configuration.md#deployment-options) | Infrastructure configuration | • **Development**: Docker Compose with hot reloading<br>• **Services**: API, Celery Worker, Redis, MongoDB, Elasticsearch<br>• **Resource Config**: Network mode, volume mounts, health checks<br>• **Environment**: Configurable URLs and API keys |

Refer to the [Configuration Guide](./docs/03-configuration.md) for more information.

## Contributing

1. Install development dependencies:

   ```sh
   uv sync --dev
   ```

   This command installs all dependencies needed to build the container and run the tests.

2. Start required services:

   ```sh
   ./scripts/run.sh
   ```

   This starts the necessary services via docker compose that are required for testing.

3. Run the tests:

   - For unit tests (requires MongoDB from docker compose):

     ```sh
     uv run pytest
     ```

   - For integration tests (with mocked NMP components):

     ```sh
     uv run pytest -m integration
     ```

4. Clean up after development:

   - Stop all services:

     ```sh
     ./scripts/stop.sh
     ```

   - (Optional) Clear all database volumes:

     ```sh
     ./scripts/clear_all_volumes.sh
     ```

If you modify the API, regenerate the openapi.json with the following command:

```sh
uv run python scripts/generate_openapi.py
```

## License

This NVIDIA AI BLUEPRINT is licensed under the [Apache License, Version 2.0.](./LICENSE) This project will download and install additional third-party open source software projects and containers. Review [the license terms of these open source projects](./LICENSE-3rd-party.txt) before use.

The software and materials are governed by the NVIDIA Software License Agreement (found at https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and the Product-Specific Terms for NVIDIA AI Products (found at https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/), except that models are governed by the AI Foundation Models Community License Agreement (found at NVIDIA Agreements | Enterprise Software | NVIDIA Community Model License) and NVIDIA dataset is governed by the NVIDIA Asset License Agreement found [here](./data/LICENSE.DATA).

For Meta/llama-3.1-70b-instruct model the Llama 3.1 Community License Agreement, for nvidia/llama-3.2-nv-embedqa-1b-v2model the Llama 3.2 Community License Agreement, and for the nvidia/llama-3.2-nv-rerankqa-1b-v2 model the Llama 3.2 Community License Agreement. Built with Llama.
