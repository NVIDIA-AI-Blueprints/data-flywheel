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
| Services | Elasticsearch 8.12.2<br>MongoDB 7.0<br>Redis 7.2<br>FastAPI (API server)<br>Celery (task processing) |
| Resource | **Minimum Memory**: 1 GB (512 MB reserved for Elasticsearch)<br>**Storage**: Varies by log volume or model size<br>**Network**: Ports 8000 (API), 9200 (Elasticsearch), 27017 (MongoDB), 6379 (Redis) |
| Development | Docker Compose for local development with hot reloading<br>Supports macOS (Darwin) and Linux<br>Optional: GPU support for model inference |
| Production | Kubernetes cluster (recommended)<br>Resources scale with workload<br>Persistent volume support for data storage |

### Obtain an NGC API Key and Log In

You must [generate a personal API key](https://org.ngc.nvidia.com/setup/api-keys) with the `NGC catalog` and `Public API Endpoints` services selected. This lets you:

- Complete deployment of NMP (NeMo Microservices Platform)
- Access NIM services
- Access models hosted in the NVIDIA API Catalog
- Download models on-premises

For detailed steps, see the official [NGC Private Registry User Guide](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-personal-api-key).

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

To deploy NMP, follow the [NeMo Microservices Platform Prerequisites](https://docs.nvidia.com/nemo/microservices/latest/get-started/setup/index.html) beginner tutorial. These instructions launch NMP using a local Minikube cluster.

**Use Manual Installation Only**

For the Data Flywheel Blueprint, use the [Install Manually](https://docs.nvidia.com/nemo/microservices/latest/get-started/setup/minikube-manual.html) option. The deployment scripts option should be avoided as it deploys models outside the namespace of the Data Flywheel and can cause conflict.

Enable customization for the models

> **Note**
> To enable customization for specific models, modify the `demo-values.yaml` file in your NMP deployment. Modify the customizer configuration with the models you want to enable for fine-tuning:
> 
> ```yaml
> customizer:
>   enabled: true
>   modelsStorage:
>     storageClassName: standard
>   customizerConfig:
>     models:
>       meta/llama-3.2-1b-instruct:
>         enabled: true
>       meta/llama-3.2-3b-instruct:
>         enabled: true
>         model_path: llama-3_2-3b-instruct
>         training_options:
>         - finetuning_type: lora
>           num_gpus: 1
>           training_type: sft
>       meta/llama-3.1-8b-instruct:
>         enabled: true
>         model_path: llama-3_1-8b-instruct
>         training_options:
>         - finetuning_type: lora
>           num_gpus: 1
>           training_type: sft
>     training:
>       pvc:
>         storageClass: "standard"
>         volumeAccessMode: "ReadWriteOnce"
> ```

> **Important**
> The Data Flywheel Blueprint automatically manages model deployment—spinning up or down models in the configured namespace. You don't need to intervene manually. The blueprint manages all aspects of the model lifecycle within the configured namespace.

### 3. Configure Data Flywheel

1. Set up the required environment variables:

   Create an NGC API key by following the instructions at [Generating NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-api-key).

   ```bash
   export NGC_API_KEY="<your-ngc-api-key>"
   ```

   For a complete list of environment variables and their descriptions, see the [Environment Variables section](03-configuration.md#environment-variables) in the Configuration Guide.

2. Clone the repository:

   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/data-flywheel.git
   cd data-flywheel
   git checkout main
   ```

3. Review and modify the [configuration file](../config/config.yaml) according to your requirements.

   **About the Configuration File**

   The `config.yaml` file controls which models (NIMs) are deployed and how the system runs. The main sections are:

   - `nmp_config`: URLs and namespace for your NMP deployment.
   - `nims`: List of models to deploy. Each entry lets you set the model name, context length, GPU count, and other options. Uncomment or add entries to test different models.
   - `data_split_config`: How your data is split for training, validation, and evaluation.
   - `icl_config`: Settings for in-context learning (ICL) examples.
   - `training_config` and `lora_config`: Training and fine-tuning parameters.
   - `logging_config`: Settings for logging. You can configure the logging level (for example, `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`). The default is `INFO`.
   - `llm_judge_config`: Large language model (LLM) as judge configuration. By default, the blueprint uses a self-hosted judge LLM, but you can switch to a remote LLM of your choice.

   **Example: Adding a New NIM**

   ```yaml
   nims:
     - model_name: "meta/llama-3.2-1b-instruct"
       context_length: 32768
       gpus: 1
       pvc_size: 25Gi
       tag: "1.8.3"
       customization_enabled: true
     - model_name: "meta/llama-3.1-8b-instruct"
       context_length: 32768
       gpus: 1
       pvc_size: 25Gi
       tag: "1.8.3"
   ```

   For more details, see the comments in the configuration file.

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

### 5. Load Data

You can feed data to the Flywheel in two ways:

1. **Manually:** For demo or short-lived environments, use the provided `load_test_data.py` script.
2. **Automatically:** For production environments where you deploy the blueprint to run continuously, use a [continuous log exportation flow](./01-architecture.md#how-production-logs-flow-into-the-flywheel).

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

> **Tip**:
> Review the [API spec](../openapi.json) for all available endpoints and request/response formats.

### Using Curl

#### Start Job

```bash
curl -X POST http://localhost:8000/api/jobs \
-H "Content-Type: application/json" \
-d '{"workload_id": "primary_assistant", "client_id": "aiva-1"}'
```

#### Create Job with Custom Data Split Configuration

You can customize the data split configuration by passing a `data_split_config` object in the POST request. This lets you override the default values for evaluation size, validation ratio, and other parameters at job creation time:

```bash
curl -X POST http://localhost:8000/api/jobs \
-H "Content-Type: application/json" \
-d '{
  "workload_id": "primary_assistant",
  "client_id": "aiva-1",
  "data_split_config": {
    "eval_size": 30,
    "val_ratio": 0.15,
    "min_total_records": 100,
    "random_seed": 42
  }
}'
```

The `data_split_config` is optional—if you don't provide it, the default values from the configuration file are used. You can also provide a partial configuration—any parameters you don't specify in the POST request use their default values. For example, if you only specify `eval_size`, all other parameters (`val_ratio`, `min_total_records`, and so on) use their default values from the configuration file. For detailed information about configuration options and their default values, see the [Data Split Configuration section](03-configuration.md#data-split-configuration) in the Configuration Guide.

#### Check Job Status and Results

```bash
curl -X GET http://localhost:8000/api/jobs/:job-id -H "Content-Type: application/json"
```

#### Cancel Job

If you need to stop a job that's currently running, you can cancel it using the cancel endpoint. This stops the job execution and marks it as cancelled:

```bash
curl -X POST http://localhost:8000/api/jobs/:job-id/cancel \
-H "Content-Type: application/json"
```

> **Important**: 
> - The job must be in a running state to be cancelled. Already finished jobs can't be cancelled.
> - If the job is already cancelled, the endpoint returns a message indicating the job is already cancelled.

##### Cancel Job Response Schema

When cancelling a job, you'll receive a JSON response with the following structure:

```json
{
  "id": "65f8a1b2c3d4e5f6a7b8c9d0",                 // Job identifier
  "message": "Job cancellation initiated successfully." // Confirmation message
}
```

##### Possible Error Responses

- **404 Not Found**: Job with the specified ID doesn't exist
- **400 Bad Request**: Job has already finished or invalid job ID format

> **Note**: To verify the cancellation status, use the GET `/api/jobs/{job_id}` endpoint to check the updated job status.

#### Delete Job and Resources

 To permanently remove a job and all its associated resources from the database, use the delete endpoint. This is useful for cleanup or removing jobs you no longer need:

 ```bash
 curl -X DELETE http://localhost:8000/api/jobs/:job-id \
 -H "Content-Type: application/json"
 ```

 > **Important**:
 >
 > - If the job is still running, you must cancel it first using the cancel endpoint before you can delete it.
 > - This is an asynchronous operation—the endpoint returns immediately while the deletion continues in the background.
 > - All associated resources, including datasets, evaluations, and customizations, are removed.

#### Delete Job Response Schema

When deleting a job, you'll receive a JSON response with the following structure:

```json
{
 "id": "65f8a1b2c3d4e5f6a7b8c9d0",                              // Job identifier
 "message": "Job deletion started. Resources will be cleaned up in the background." // Confirmation message
}
```

##### Possible Error Responses

- **404 Not Found**: Job with the specified ID doesn't exist
- **400 Bad Request**: Job is still running (must be cancelled first) or invalid job ID format
- **500 Internal Server Error**: Failed to initiate job deletion

> **Note**: Once a job is deleted, it's permanently removed from the database. Subsequent calls to GET `/api/jobs/{job_id}` return a 404 Not Found error.

#### Job Response Schema

When querying a job, you'll receive a JSON response with the following structure:

```json
{
 "id": "65f8a1b2c3d4e5f6a7b8c9d0",          // Unique job identifier
 "workload_id": "primary_assistant",               // Workload being processed
 "client_id": "aiva-1",                // Client identifier
 "status": "running",                        // Current job status
 "started_at": "2024-03-15T14:30:00Z",      // Job start timestamp
 "finished_at": "2024-03-15T15:30:00Z",      // Job completion timestamp (if finished)
 "num_records": 1000,                        // Number of processed records
 "llm_judge": { ... },                       // LLM Judge model status
 "nims": [ ... ],                           // List of NIMs and their evaluation results
 "datasets": [ ... ]                        // List of datasets used in the job
}
```

> **Note:**
> When a job starts, all NIMs specified in your configuration are immediately included in the `nims` list of the job response, each with a status of `"Pending"`. This is true even if NIMs are executed sequentially. The `Pending` status indicates that the NIM is scheduled for evaluation as part of the job. As the job progresses, each NIM's status will update (e.g., to `Running`, `Completed`, or `Error`) to reflect its current state. This approach provides a transparent and accurate view of the job's overall progress and planned evaluations.

### Using Notebooks

> **Note:** Make sure all services are running before accessing the Jupyter Lab interface.

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
4. Open the example notebook for running and monitoring jobs.

Follow the instructions in the Jupyter Lab notebook to interact with the Data Flywheel services.

## Evaluate Results

Refer to [Evaluation Types and Metrics Documentation](docs/06-evaluation-types-and-metrics.md) to learn more about how to evaluate results.

## Cleanup

### 1. Data Flywheel Services

When you're done using the services, you can stop them using the stop script:

```bash
./scripts/stop.sh
```

### 2. Resource Cleanup

The Data Flywheel Blueprint provides two types of resource cleanup:

#### Automatic Cleanup (During System Shutdown)

The system automatically cleans up running resources when workers are shut down gracefully. This happens automatically when:
- Docker containers are stopped (`docker compose down`)
- Celery workers receive shutdown signals
- The system is restarted

The automatic cleanup manager:
- Detects all running flywheel runs and NIMs
- Cancels active customization jobs
- Shuts down running deployments
- Marks all resources as cancelled in the database

For technical details about the automatic cleanup process, see the [Architecture Overview](01-architecture.md#automatic-resource-cleanup).

#### Manual Cleanup (For Maintenance)

If you need to manually clean up all running resources—flywheel runs, NIMs, evaluations, and customizations—use the cleanup script:

```bash
# Run the cleanup script
./scripts/cleanup_resources.sh
```

This script will:

- Find all running flywheel runs from MongoDB
- Shut down all running NIMs and LLM judge deployments
- Cancel running customization jobs and delete evaluation jobs
- Mark all resources as cancelled in the database

For detailed information about the cleanup process, safety features, and troubleshooting, see the [Scripts Documentation](scripts.md).

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
