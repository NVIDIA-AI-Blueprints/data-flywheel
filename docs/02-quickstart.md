# Getting Started with Data Flywheel Blueprint

Learn how to set up and deploy the Data Flywheel Blueprint using the steps in this guide.

This quickstart provides an initial [bug-triager dataset](../services/api/scripts/bug_triage_dataset.jsonl) and [notebook](../notebooks/monitor_job.ipynb) to help you get started working with the services.

## Prerequisites

### Hardware Requirements

- **Minimum GPU requirements**: 6x (NVIDIA H100 GPUs or NVIDIA A100 GPUs)
- **Cluster**: A single-node NVIDIA GPU cluster on a Linux host with cluster-admin level permissions
- **Disk space**: At least 200 GB of free disk space

### Software and Access Requirements

Before you begin, make sure you have:

- [Docker and Docker Compose](https://docs.docker.com/desktop) installed
- [Access to NVIDIA NGC](https://catalog.ngc.nvidia.com/) (for API key)

### Obtain an NGC API Key

You must [generate a personal API key](https://org.ngc.nvidia.com/setup/api-keys) with the `NGC catalog` and `Public API Endpoints` services selected. This enables you to:

- Complete deployment of NMP (NeMo Microservices Platform)
- Access NIM services
- Access models hosted in the NVIDIA API Catalog
- Download models on-premises

For detailed steps, see the official [NGC Private Registry User Guide](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#generating-personal-api-key).

---

## Set Up the Data Flywheel Blueprint

### 1. Deploy NMP

To deploy NMP, follow the [NeMo Microservices Platform Prerequisites](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html#beginner-tutorial-prerequisites) beginner tutorial. These instructions launch NMP using a local Minikube clsuter. You have two options:

- [Installing using deployment scripts](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html#nemo-ms-get-started-prerequisites-using-deployment-scripts)
- [Installing manually](https://docs.nvidia.com/nemo/microservices/latest/get-started/platform-prereq.html#installing-manually)

### 2. Configure Data Flywheel

1. Set up the required environment variables:

   ```bash
   export NVCF_API_KEY="<your-ngc-api-key>"
   ```

2. Clone the repository:

   ```bash
   git clone https://gitlab-master.nvidia.com/aire/microservices/data-flywheel-blueprint
   cd data-flywheel-blueprint
   git checkout main
   ```

3. Review and modify the [configuration file](../config/config.yaml) according to your requirements.

   **About the configuration file**

   The `config.yaml` file controls which models (NIMs) are deployed and how the system runs. The main sections are:

   - `nmp_config`: URLs and namespace for your NMP deployment.
   - `nims`: List of models to deploy. Each entry lets you set the model name, context length, GPU count, and other options. Uncomment or add entries to test different models.
   - `data_split_config`: How your data is split for training, validation, and evaluation.
   - `icl_config`: Settings for in-context learning (ICL) examples.
   - `training_config` and `lora_config`: Training and fine-tuning parameters.
   - `logging_config`: Settings got logging . You can configure the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is "INFO".

   **Example: Adding a new NIM**

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

   For more details, see the comments in the config file.

### 3. Start Services

You have several options to start the services:

1. **[Recommended]** Using the [launch script](../scripts/run.sh):

   ```bash
   ./scripts/run.sh
   ```


1. Using the [development script](../scripts/run-dev.sh):

   This script runs additional services for observability:

   - `flower`: A web UI for monitoring Celery tasks and workers
   - `kibana`: A visualization dashboard for exploring data stored in Elasticsearch

   ```bash
   ./scripts/run-dev.sh
   ```

1. Using Docker Compose directly:

   ```bash
   docker compose -f ./deploy/docker-compose.yaml up --build
   ```

### 4. Load Test Data

Load test data using the provided script:

```bash
uv run python src/scripts/load_test_data.py \
  --workload-id bug-triager \
  --client-id dev-notebook \
  --file bug_triage_dataset.jsonl
```

#### Supported Dataset Format

To submit your own dataset, make sure that you provide the loader with a file in [JSON Lines (JSONL)](https://jsonlines.org/) format, where each line is a JSON object with a `messages` field. The `messages` field should be a list of at least two objects, each with a `role` (e.g., `user` or `assistant`) and `content` (the text of the message).

**Example entry:**

```json
{"messages": [
  {"role": "user", "content": "Describe your issue here."},
  {"role": "assistant", "content": "Assistant's response goes here."}
]}
```

Each line in your dataset file should follow this structure.

---

## Run a Job

Now that you have the Data Flywheel running and loaded with data, you can start running jobs.

### Using curl

1. Start a new job:

   ```bash
   curl -X POST http://localhost:8000/api/jobs \
   -H "Content-Type: application/json" \
   -d '{"workload_id": "bug-triager", "client_id": "dev-notebook"}'
   ```

2. Check job status and results:

   ```bash
   curl -X GET http://localhost:8000/api/jobs/:job-id -H "Content-Type: application/json"
   ```

   #### Job Response Schema

   When querying a job, you'll receive a JSON response with the following structure:

   ```json
   {
     "id": "65f8a1b2c3d4e5f6a7b8c9d0",          // Unique job identifier
     "workload_id": "bug-triager",               // Workload being processed
     "client_id": "dev-notebook",                // Client identifier
     "status": "running",                        // Current job status
     "started_at": "2024-03-15T14:30:00Z",      // Job start timestamp
     "finished_at": "2024-03-15T15:30:00Z",      // Job completion timestamp (if finished)
     "num_records": 1000,                        // Number of processed records
     "llm_judge": { ... },                       // LLM Judge model status
     "nims": [ ... ],                           // List of NIMs and their evaluation results
     "datasets": [ ... ]                        // List of datasets used in the job
   }
   ```

### Using Notebooks

**Note**: Make sure all services are running before accessing the notebook interface.

1. Launch Jupyter Lab using uv:

   ```bash
   uv run jupyter lab \
     --allow-root \
     --ip=0.0.0.0 \
     --NotebookApp.token='' \
     --port=8889 \
     --no-browser
   ```

2. Access Jupyter Lab in your browser at `http://<your-host-ip>:8889`
3. Navigate to the `notebooks` directory
4. Open the example notebook for running and monitoring jobs

Follow the instructions in the notebook to interact with the Data Flywheel services.

## Cleanup

### 1. Data Flywheel

When you are done using the services, you can clean up using the [clear volumes script](../scripts/clear_all_volumes.sh):

```bash
./scripts/clear_all_volumes.sh
```

This script will clear all service volumes (Elasticsearch, Redis, and MongoDB).

### 2. NMP Cleanup

You can remove NMP when you are done using the platform by following the official [Uninstall NeMo Microservices Helm Chart](https://docs.nvidia.com/nemo/microservices/latest/set-up/deploy-as-platform/uninstall-platform-helm-chart.html) guide.

## Troubleshooting

If you encounter any issues:

1. Check that all environment variables are properly set
2. Ensure all prerequisites are installed and configured correctly
3. Verify that you have the necessary permissions and access to all required resources

## Additional Resources

- [Data Flywheel Blueprint Repository](https://gitlab-master.nvidia.com/aire/microservices/data-flywheel-blueprint)
