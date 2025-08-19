# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Consolidated Customization Integration Tests

Tests customization workflows with real database dependencies:
- Base customization workflow with NIM deployment
- Customization progress tracking and cancellation
- Error handling and rollback scenarios
- Model sync and cleanup workflows
- Only mocks external customization APIs, keeps internal logic real
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import (
    TaskResult,
)
from src.config import LLMJudgeConfig
from src.tasks.tasks import (
    cancel_job_resources,
    finalize_flywheel_run,
    initialize_workflow,
    shutdown_deployment,
    spin_up_nim,
    start_customization,
    wait_for_llm_as_judge,
)


def dict_to_task_result(data: dict[str, Any]) -> TaskResult:
    """Convert dictionary back to TaskResult object for testing."""
    return TaskResult(**data)


def ensure_task_result(result) -> TaskResult:
    """Ensure result is a TaskResult, converting from dict if needed."""
    return dict_to_task_result(result) if isinstance(result, dict) else result


@pytest.fixture(autouse=True)
def setup_db_manager(mongo_db):
    """Setup database manager for customization tests."""
    import src.tasks.tasks as tasks_module
    from src.api.db import init_db
    from src.api.db_manager import get_db_manager

    init_db()
    tasks_module.db_manager = get_db_manager()

    yield


def load_tool_calling_data():
    """Load first 30 records from aiva-test.jsonl for tool-calling tests."""
    try:
        with open("data/aiva-test.jsonl") as f:
            records = []
            for i, line in enumerate(f):
                if i >= 30:  # Load 30 records to get more unique records after deduplication
                    break
                records.append(json.loads(line.strip()))
            return records
    except FileNotFoundError:
        return []


@pytest.fixture(params=["generic", "tool_calling"])
def workload_type(request):
    """Simple workload type parameterization."""
    return request.param


@pytest.fixture(scope="module")
def load_customization_test_data():
    """Load test data once for all customization tests."""
    from src.lib.integration.es_client import ES_COLLECTION_NAME, get_es_client

    es_client = get_es_client()

    # Load tool-calling data once
    aiva_records = load_tool_calling_data()
    for i, aiva_record in enumerate(aiva_records):
        record = {
            "client_id": "test-client-tool_calling-customization",
            "workload_id": "test-tool_calling-customization",
            "request": aiva_record["request"],
            "response": aiva_record["response"],
            "timestamp": f"2023-01-01T{i//60:02d}:{i%60:02d}:00Z",
        }
        es_client.index(
            index=ES_COLLECTION_NAME,
            body=record,
            refresh=True,
        )

    # Load generic data once
    for i in range(20):
        record = {
            "client_id": "test-client-generic-customization",
            "workload_id": "test-generic-customization",
            "request": {
                "messages": [
                    {"role": "user", "content": f"Test question {i}?"},
                ]
            },
            "response": {
                "choices": [{"message": {"role": "assistant", "content": f"Test response {i}"}}]
            },
            "timestamp": f"2023-01-01T{i//60:02d}:{i%60:02d}:00Z",
        }
        es_client.index(
            index=ES_COLLECTION_NAME,
            body=record,
            refresh=True,
        )

    yield

    # Cleanup: Remove test data from Elasticsearch after all tests in module are completed
    try:
        es_client.delete_by_query(
            index=ES_COLLECTION_NAME,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {
                                "bool": {
                                    "must": [
                                        {
                                            "match": {
                                                "client_id": "test-client-toolcalling-customization"
                                            }
                                        },
                                        {
                                            "match": {
                                                "workload_id": "test-toolcalling-customization"
                                            }
                                        },
                                    ]
                                }
                            },
                            {
                                "bool": {
                                    "must": [
                                        {
                                            "match": {
                                                "client_id": "test-client-generic-customization"
                                            }
                                        },
                                        {"match": {"workload_id": "test-generic-customization"}},
                                    ]
                                }
                            },
                        ]
                    }
                }
            },
            refresh=True,
        )
    except Exception as e:
        print(f"Warning: Failed to cleanup ES data: {e}")


@pytest.fixture
def customization_environment(mongo_db, workload_type, load_customization_test_data):
    """Setup customization environment with parameterized workload types."""
    from src.tasks.tasks import create_datasets

    base_client_id = f"test-client-{workload_type}-customization"
    base_workload_id = f"test-{workload_type}-customization"

    # Setup comprehensive mocking for external services only
    with (
        patch("src.lib.nemo.customizer.requests") as mock_requests,
        patch("src.lib.nemo.data_uploader.DataUploader.upload_data") as mock_upload,
        patch("src.lib.nemo.data_uploader.DataUploader.get_file_uri") as mock_get_uri,
        patch("time.sleep") as mock_sleep,
        patch("src.lib.nemo.dms_client.DMSClient") as mock_dms,
        patch("src.lib.nemo.dms_client.requests") as mock_dms_requests,
        patch("src.lib.nemo.embedding.Embedding") as mock_embedding_class,
        patch("src.lib.flywheel.icl_selection.Embedding") as mock_embedding_class_icl,
        patch("src.lib.integration.es_client.index_embeddings_to_es") as mock_index_embeddings,
        patch("src.lib.integration.es_client.search_similar_embeddings") as mock_search_embeddings,
        patch("src.lib.integration.es_client.delete_embeddings_index") as mock_delete_embeddings,
        patch("src.lib.flywheel.icl_selection.index_embeddings_to_es") as mock_index_embeddings_icl,
        patch(
            "src.lib.flywheel.icl_selection.search_similar_embeddings"
        ) as mock_search_embeddings_icl,
    ):
        # Setup customization service mocking
        job_counter = 0

        def mock_post_request(url, **kwargs):
            nonlocal job_counter
            url_str = str(url)
            if "/v1/customization/configs" in url_str:
                return MagicMock(status_code=200, json=lambda: {"name": "test-config"})
            elif "/v1/customization/jobs" in url_str:
                job_counter += 1
                job_id = f"customization-job-{job_counter}"
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "id": job_id,
                        "output_model": f"customized-test-model-{job_counter}",
                    },
                )
            else:
                raise ValueError(f"Unexpected POST request: {url}")

        def mock_get_request(url, **kwargs):
            url_str = str(url)
            if "/v1/customization/jobs/" in url_str and "/status" in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "status": "completed",
                        "percentage_done": 100,
                        "epochs_completed": 3,
                        "steps_completed": 150,
                    },
                )
            elif "/v1/models" in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "data": [
                            {"id": "nim/test-model"},
                            {"id": "customized-test-model"},
                            {"id": "customized-test-model-1"},
                            {"id": "customized-test-model-2"},
                        ]
                    },
                )
            else:
                raise ValueError(f"Unexpected GET request: {url}")

        mock_requests.post.side_effect = mock_post_request
        mock_requests.get.side_effect = mock_get_request
        mock_sleep.return_value = None

        # Setup DMS client HTTP mocking
        def mock_dms_get_request(url, **kwargs):
            url_str = str(url)
            if "/v1/models" in url_str:
                # Return models list for model sync
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "data": [
                            {"id": "nim/test-model"},
                            {"id": "customized-test-model"},
                        ]
                    },
                )
            else:
                # Return deployment status for other endpoints
                return MagicMock(
                    status_code=200,
                    json=lambda: {"status": "ready", "status_details": {"status": "ready"}},
                )

        mock_dms_requests.post.return_value = MagicMock(
            status_code=200, json=lambda: {"id": "deployment-123"}
        )
        mock_dms_requests.get.side_effect = mock_dms_get_request
        mock_dms_requests.delete.return_value = MagicMock(
            status_code=200, json=lambda: {"status": "deleted"}
        )

        # Setup file upload mocking
        mock_upload.return_value = None
        mock_get_uri.return_value = f"test://dataset-uri-{base_workload_id}"

        # Setup DMS client mocking
        mock_dms_instance = mock_dms.return_value
        mock_dms_instance.is_deployed.return_value = False
        mock_dms_instance.deploy_model.return_value = None
        mock_dms_instance.wait_for_deployment.return_value = None
        mock_dms_instance.shutdown_deployment.return_value = None

        # Setup embedding mocking
        def mock_get_embedding(queries, input_type="query"):
            """Mock get_embedding to return appropriate number of embeddings"""
            if isinstance(queries, list):
                # Return one embedding per query
                return [[0.1] * 2048 for _ in queries]  # Mock 2048-dim embeddings
            else:
                # Single query
                return [[0.1] * 2048]  # Single mock 2048-dim embedding

        mock_embedding_instance = MagicMock()
        mock_embedding_instance.get_embedding.side_effect = mock_get_embedding
        mock_embedding_instance.get_embeddings_batch.side_effect = mock_get_embedding
        mock_embedding_class.return_value = mock_embedding_instance
        mock_embedding_class_icl.return_value = mock_embedding_instance

        # Setup Elasticsearch embeddings mocking
        mock_index_embeddings.return_value = "test-embeddings-index"
        mock_index_embeddings_icl.return_value = "test-embeddings-index"
        # Mock search_similar_embeddings to return list of (score, tool_name, record) tuples
        mock_record = {
            "request": {"messages": [{"role": "user", "content": "Test question"}]},
            "response": {"choices": [{"message": {"content": "Test response"}}]},
        }
        mock_search_embeddings.return_value = [(0.9, "no_tool", mock_record)]
        mock_search_embeddings_icl.return_value = [(0.9, "no_tool", mock_record)]
        mock_delete_embeddings.return_value = None

        # Create flywheel run record
        flywheel_run_id = ObjectId()
        mongo_db.flywheel_runs.insert_one(
            {
                "_id": flywheel_run_id,
                "workload_id": base_workload_id,
                "client_id": base_client_id,
                "started_at": datetime.utcnow(),
                "status": "pending",
                "num_records": 0,
            }
        )

        # Run real initialize_workflow
        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.LLMAsJudge") as mock_llm_judge,
        ):
            from src.config import NIMConfig

            mock_settings.nims = [
                NIMConfig(
                    model_name="nim/test-model",
                    context_length=8192,
                    customization_enabled=True,
                    customizer_configs={
                        "target": "test-target",
                        "gpus": 1,
                        "num_nodes": 1,
                        "tensor_parallel_size": 1,
                        "data_parallel_size": 1,
                        "use_sequence_parallel": False,
                        "micro_batch_size": 1,
                        "training_precision": "bf16",
                        "max_seq_length": 2048,
                    },
                )
            ]

            llm_judge_config = LLMJudgeConfig(
                deployment_type="remote",  # Use remote to skip deployment waiting
                model_name="test-judge",
                context_length=8192,
                url="http://test-judge-url/v1/chat/completions",
                api_key="test-api-key",
            )
            mock_llm_judge.return_value.config = llm_judge_config

            # Use base IDs for data lookup, but unique IDs for the flywheel run
            task_result = initialize_workflow(
                workload_id=base_workload_id,  # Use base ID for data lookup
                flywheel_run_id=str(flywheel_run_id),
                client_id=base_client_id,  # Use base ID for data lookup
            )

        # Ensure task_result is TaskResult object
        task_result = ensure_task_result(task_result)

        # Adjust validation requirements for tool-calling tests
        if workload_type == "tool_calling":
            # Use custom split config with lower requirements for tool-calling data
            from src.config import DataSplitConfig

            custom_split_config = DataSplitConfig(
                eval_size=10,  # Lower eval size to fit within 20 records
                val_ratio=0.1,
                min_total_records=10,  # Lower minimum for tool-calling due to deduplication
                random_seed=42,  # Set specific seed for consistent behavior
                limit=1000,
                parse_function_arguments=True,
            )
            task_result.data_split_config = custom_split_config
        else:
            # Use custom split config for generic data as well
            from src.config import DataSplitConfig

            custom_split_config = DataSplitConfig(
                eval_size=10,  # Lower eval size to fit within 20 records
                val_ratio=0.1,
                min_total_records=20,  # Lower minimum
                random_seed=42,  # Keep stratified splitting for generic data
                limit=1000,
                parse_function_arguments=True,
            )
            task_result.data_split_config = custom_split_config

        # Run real create_datasets (RecordExporter will find the indexed data)
        task_result = ensure_task_result(create_datasets(task_result))

        # Run real wait_for_llm_as_judge
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Get created NIM info from database
        nim_docs = list(mongo_db.nims.find({"flywheel_run_id": flywheel_run_id}))
        assert len(nim_docs) > 0, "No NIMs were created by initialize_workflow"

        # Add nim config to task result for customization
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={
                "target": "test-target",
                "gpus": 1,
                "num_nodes": 1,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "use_sequence_parallel": False,
                "micro_batch_size": 1,
                "training_precision": "bf16",
                "max_seq_length": 2048,
            },
        )

        # Return comprehensive environment info
        environment_info = {
            "flywheel_run_id": str(flywheel_run_id),
            "nim_id": nim_docs[0]["_id"],
            "model_name": "nim/test-model",
            "customization_enabled": True,
            "task_result": task_result,
            "test_records": [],  # Data is now pre-loaded, so no need to return it here
            "num_records": 0,  # Data is now pre-loaded, so no need to return it here
            "workload_type": workload_type,
            "workload_id": base_workload_id,
            "client_id": base_client_id,
            "mocks": {
                "requests": mock_requests,
                "sleep": mock_sleep,
                "dms_requests": mock_dms_requests,
                "upload": mock_upload,
                "dms": mock_dms,
                "embedding": mock_embedding_instance,
            },
        }

        yield environment_info

        mongo_db.customizations.delete_many({})
        mongo_db.nims.delete_many({})
        mongo_db.flywheel_runs.delete_many({})
        mongo_db.llm_judge_runs.delete_many({})


@pytest.mark.integration
@pytest.mark.customization
class TestCustomizationWorkflows:
    """Test customization workflows with real database dependencies."""

    def test_complete_customization_workflow(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test complete customization workflow with real data flow."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Execute spin up NIM
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))
        assert nim_result.error is None
        assert nim_result.nim is not None

        custom_result = ensure_task_result(start_customization(nim_result))
        assert custom_result.error is None
        assert custom_result.customization is not None
        assert custom_result.customization.model_name is not None
        assert custom_result.customization.percent_done == 100.0

        # Execute shutdown deployment
        shutdown_result = ensure_task_result(shutdown_deployment(custom_result))
        assert shutdown_result.error is None

        # Execute finalize
        finalize_result = ensure_task_result(finalize_flywheel_run(shutdown_result))
        assert finalize_result.error is None

        # Verify database state
        customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
        assert len(customizations) == 1

        customization_doc = customizations[0]
        assert customization_doc["progress"] == 100.0
        assert customization_doc["customized_model"] == "customized-test-model-1"
        assert customization_doc["job_id"] is not None

    def test_customization_progress_tracking(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test customization progress tracking and database updates."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Reuse existing mock_requests but add progress tracking behavior
        mock_requests = env_info["mocks"]["requests"]

        # Save original behavior
        original_get = mock_requests.get.side_effect

        # Define progress updates
        progress_updates = [
            {"status": "running", "percentage_done": 25, "epochs_completed": 1},
            {"status": "running", "percentage_done": 50, "epochs_completed": 2},
            {"status": "running", "percentage_done": 75, "epochs_completed": 2},
            {"status": "completed", "percentage_done": 100, "epochs_completed": 3},
        ]

        progress_counter = 0

        def mock_get_progress(url, **kwargs):
            nonlocal progress_counter
            url_str = str(url)
            if "/v1/customization/jobs/" in url_str and "/status" in url_str:
                # Return progress updates incrementally
                if progress_counter < len(progress_updates):
                    current_progress = progress_updates[progress_counter]
                    progress_counter += 1
                    return MagicMock(
                        status_code=200,
                        json=lambda: current_progress,
                    )
                else:
                    # Return final completed status
                    return MagicMock(
                        status_code=200,
                        json=lambda: progress_updates[-1],
                    )
            elif "/v1/models" in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "data": [
                            {"id": "nim/test-model"},
                            {"id": "customized-test-model"},
                            {"id": "customized-test-model-1"},
                            {"id": "customized-test-model-2"},
                        ]
                    },
                )
            else:
                # For any other requests, return a default response
                return MagicMock(
                    status_code=200,
                    json=lambda: {"status": "ok"},
                )

        mock_requests.get.side_effect = mock_get_progress

        try:
            # Execute spin up NIM
            nim_config = task_result.nim.model_dump()
            nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

            custom_result = ensure_task_result(start_customization(nim_result))
            assert custom_result.error is None
            assert custom_result.customization is not None

            # Verify progress was tracked in database
            customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
            assert len(customizations) == 1

            customization_doc = customizations[0]
            assert customization_doc["progress"] == 100.0
            assert customization_doc["epochs_completed"] == 3
        finally:
            # Restore original mock behavior
            mock_requests.get.side_effect = original_get

    def test_customization_cancellation_handling(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test customization cancellation handling and database state updates."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Cancel the flywheel run using the proper function
        cancel_job_resources(env_info["flywheel_run_id"])

        # Execute spin up NIM
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Execute customization (should be cancelled)
        result = ensure_task_result(start_customization(nim_result))
        assert result.error is not None
        assert "Flywheel run cancelled" in result.error

        # Verify no customization records created due to cancellation
        customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
        assert len(customizations) == 0

    def test_customization_skip_on_previous_error(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test customization skipping when previous task has error."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Execute spin up NIM
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Add previous error to task result after spin up (since spin_up_nim clears errors)
        nim_result.error = "Previous task failed"

        # Execute customization (should be skipped)
        result = ensure_task_result(start_customization(nim_result))
        assert result.error == "Previous task failed"

        # Verify no customization records created
        customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
        assert len(customizations) == 0

    def test_customization_disabled_skip(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test customization skipping when customization is disabled."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Disable customization for this NIM
        task_result.nim.customization_enabled = False

        # Execute spin up NIM
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Execute customization (should be skipped)
        result = ensure_task_result(start_customization(nim_result))
        assert result.error is None
        assert result.customization is None

        # Verify no customization records created
        customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
        assert len(customizations) == 0


@pytest.mark.integration
@pytest.mark.customization
class TestCustomizationErrorHandling:
    """Test customization error handling and rollback scenarios."""

    def test_customization_job_failure_handling(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test handling of customization job failures."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Reuse existing mock_requests but modify behavior to simulate failure
        mock_requests = env_info["mocks"]["requests"]

        # Temporarily modify the mock to return failure
        original_post = mock_requests.post.side_effect
        original_get = mock_requests.get.side_effect

        def mock_post_failure(url, **kwargs):
            url_str = str(url)
            if "/v1/customization/configs" in url_str:
                return MagicMock(status_code=200, json=lambda: {"name": "test-config"})
            elif "/v1/customization/jobs" in url_str:
                return MagicMock(
                    status_code=500,
                    text="Internal server error",
                )
            else:
                raise ValueError(f"Unexpected POST request: {url}")

        def mock_get_failure(url, **kwargs):
            url_str = str(url)
            if "/v1/customization/jobs/" in url_str and "/status" in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "status": "failed",
                        "percentage_done": 0,
                        "epochs_completed": 0,
                    },
                )
            elif "/v1/models" in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "data": [
                            {"id": "nim/test-model"},
                        ]
                    },
                )
            else:
                raise ValueError(f"Unexpected GET request: {url}")

        mock_requests.post.side_effect = mock_post_failure
        mock_requests.get.side_effect = mock_get_failure

        try:
            # Execute spin up NIM
            nim_config = task_result.nim.model_dump()
            nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

            # Execute customization (should fail)
            result = ensure_task_result(start_customization(nim_result))
            assert result.error is not None
            assert "Failed to start training job" in result.error

            # Verify error was recorded in database
            customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
            assert len(customizations) == 1

            customization_doc = customizations[0]
            assert customization_doc["progress"] == 0.0
            assert "error" in customization_doc
            assert customization_doc["finished_at"] is not None
        finally:
            # Restore original mock behavior
            mock_requests.post.side_effect = original_post
            mock_requests.get.side_effect = original_get

    def test_customization_timeout_handling(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test customization timeout handling."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Reuse existing mock_requests but modify behavior to simulate timeout
        mock_requests = env_info["mocks"]["requests"]

        # Temporarily modify the mock to return running status (timeout simulation)
        original_get = mock_requests.get.side_effect

        def mock_get_timeout(url, **kwargs):
            url_str = str(url)
            if "/v1/customization/jobs/" in url_str and "/status" in url_str:
                # Always return running status to simulate timeout
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "status": "running",
                        "percentage_done": 50,
                        "epochs_completed": 1,
                    },
                )
            elif "/v1/models" in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "data": [
                            {"id": "nim/test-model"},
                        ]
                    },
                )
            else:
                raise ValueError(f"Unexpected GET request: {url}")

        mock_requests.get.side_effect = mock_get_timeout

        try:
            # Execute spin up NIM
            nim_config = task_result.nim.model_dump()
            nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

            # Execute customization with timeout simulation
            with patch("src.lib.nemo.customizer.Customizer.wait_for_customization") as mock_wait:
                mock_wait.side_effect = TimeoutError("Customization timed out")

                result = ensure_task_result(start_customization(nim_result))
                assert result.error is not None
                assert "Customization timed out" in result.error
        finally:
            # Restore original mock behavior
            mock_requests.get.side_effect = original_get

    def test_customization_rollback_on_error(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test customization rollback when errors occur."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Reuse existing mock_requests but modify behavior to simulate rollback
        mock_requests = env_info["mocks"]["requests"]

        # Temporarily modify the mock to simulate rollback scenario
        original_post = mock_requests.post.side_effect
        original_get = mock_requests.get.side_effect

        job_created = False

        def mock_post_rollback(url, **kwargs):
            nonlocal job_created
            url_str = str(url)
            if "/v1/customization/configs" in url_str:
                return MagicMock(status_code=200, json=lambda: {"name": "test-config"})
            elif "/v1/customization/jobs" in url_str:
                job_created = True
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "id": "fail-job-123",
                        "output_model": "customized-test-model",
                    },
                )
            else:
                raise ValueError(f"Unexpected POST request: {url}")

        def mock_get_rollback(url, **kwargs):
            if job_created:
                raise Exception("Network error during customization")
            else:
                raise ValueError(f"Unexpected GET request: {url}")

        mock_requests.post.side_effect = mock_post_rollback
        mock_requests.get.side_effect = mock_get_rollback

        try:
            # Execute spin up NIM
            nim_config = task_result.nim.model_dump()
            nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

            # Execute customization (should fail and rollback)
            result = ensure_task_result(start_customization(nim_result))
            assert result.error is not None
            assert "Network error during customization" in result.error

            # Verify rollback occurred
            customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
            assert len(customizations) == 1

            customization_doc = customizations[0]
            assert customization_doc["progress"] == 0.0
            assert "error" in customization_doc
        finally:
            # Restore original mock behavior
            mock_requests.post.side_effect = original_post
            mock_requests.get.side_effect = original_get


@pytest.mark.integration
@pytest.mark.customization
class TestCustomizationCleanup:
    """Test customization cleanup and resource management."""

    def test_customization_cleanup_on_success(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test proper cleanup after successful customization."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Execute complete workflow
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))
        custom_result = ensure_task_result(start_customization(nim_result))
        shutdown_result = ensure_task_result(shutdown_deployment(custom_result))
        finalize_result = ensure_task_result(finalize_flywheel_run(shutdown_result))

        # Verify all components completed successfully
        assert finalize_result.error is None

        # Verify database state is clean
        flywheel_runs = list(
            mongo_db.flywheel_runs.find({"_id": ObjectId(env_info["flywheel_run_id"])})
        )
        assert len(flywheel_runs) == 1
        assert flywheel_runs[0]["status"] == "completed"

        customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
        assert len(customizations) == 1
        assert customizations[0]["progress"] == 100.0

    def test_customization_cleanup_on_failure(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test proper cleanup after customization failure."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Use existing mocks from environment
        mock_requests = env_info["mocks"]["requests"]
        mock_sleep = env_info["mocks"]["sleep"]

        # Mock customization to fail after job creation but before completion
        with patch("src.lib.nemo.customizer.Customizer.cancel_job") as mock_cancel_job:
            job_created = False

            def mock_post_request(url, **kwargs):
                nonlocal job_created
                url_str = str(url)
                if "/v1/customization/configs" in url_str:
                    return MagicMock(status_code=200, json=lambda: {"name": "test-config"})
                elif "/v1/customization/jobs" in url_str:
                    job_created = True
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            "id": "customization-job-123",
                            "output_model": "customized-test-model",
                        },
                    )
                else:
                    raise ValueError(f"Unexpected POST request: {url}")

            def mock_get_request(url, **kwargs):
                url_str = str(url)
                if "/v1/customization/jobs/" in url_str and "/status" in url_str:
                    if job_created:
                        # Simulate job failure after creation
                        raise Exception("Customization job failed during execution")
                    else:
                        # Job not created yet
                        return MagicMock(
                            status_code=404,
                            json=lambda: {"error": "Job not found"},
                        )
                elif "/v1/models" in url_str:
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            "data": [
                                {"id": "nim/test-model"},
                            ]
                        },
                    )
                else:
                    raise ValueError(f"Unexpected GET request: {url}")

            mock_requests.post.side_effect = mock_post_request
            mock_requests.get.side_effect = mock_get_request
            mock_sleep.return_value = None
            mock_cancel_job.return_value = None

            # Execute workflow with failure
            nim_config = task_result.nim.model_dump()
            nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))
            custom_result = ensure_task_result(start_customization(nim_result))
            shutdown_result = ensure_task_result(shutdown_deployment(custom_result))
            finalize_result = ensure_task_result(finalize_flywheel_run(shutdown_result))

            # Verify error was properly handled
            assert custom_result.error is not None
            assert "Customization job failed during execution" in custom_result.error
            assert finalize_result.error is not None  # Finalize should propagate the error

            # Verify cleanup operations were performed
            # 1. Verify the job was cancelled
            mock_cancel_job.assert_called_once_with("customization-job-123")

            # 2. Verify database state reflects the failure
            customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
            assert len(customizations) == 1
            assert customizations[0]["progress"] == 0.0
            assert "error" in customizations[0]
            assert "Customization job failed during execution" in customizations[0]["error"]

    def test_customization_cancellation_cleanup(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test cleanup when customization is cancelled."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Cancel the flywheel run using the proper function
        cancel_job_resources(env_info["flywheel_run_id"])

        # Execute workflow with cancellation
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))
        custom_result = ensure_task_result(start_customization(nim_result))
        shutdown_result = ensure_task_result(shutdown_deployment(custom_result))
        ensure_task_result(finalize_flywheel_run(shutdown_result))

        # Verify cancellation was handled properly
        assert custom_result.error is not None
        assert "Flywheel run cancelled" in custom_result.error

        # Verify no customization records were created
        customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
        assert len(customizations) == 0

    def test_customization_cancellation_during_model_sync(
        self,
        customization_environment,
        mongo_db,
    ):
        """Test customization cancellation during wait_for_model_sync phase."""
        env_info = customization_environment
        task_result = env_info["task_result"]

        # Use existing mocks from environment
        mock_requests = env_info["mocks"]["requests"]
        mock_sleep = env_info["mocks"]["sleep"]

        # Mock customization to succeed but then cancel during model sync
        with (
            patch("src.lib.nemo.customizer.Customizer.cancel_job") as mock_cancel_job,
            patch("src.lib.nemo.customizer.Customizer.wait_for_model_sync") as mock_wait_sync,
        ):

            def mock_post_request(url, **kwargs):
                url_str = str(url)
                if "/v1/customization/configs" in url_str:
                    return MagicMock(status_code=200, json=lambda: {"name": "test-config"})
                elif "/v1/customization/jobs" in url_str:
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            "id": "customization-job-456",
                            "output_model": "customized-test-model",
                        },
                    )
                else:
                    raise ValueError(f"Unexpected POST request: {url}")

            def mock_get_request(url, **kwargs):
                url_str = str(url)
                if "/v1/customization/jobs/" in url_str and "/status" in url_str:
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            "status": "completed",
                            "percentage_done": 100,
                            "epochs_completed": 3,
                        },
                    )
                elif "/v1/models" in url_str:
                    return MagicMock(
                        status_code=200,
                        json=lambda: {
                            "data": [
                                {"id": "nim/test-model"},
                                # Note: customized-test-model is NOT in the list to simulate sync timeout
                            ]
                        },
                    )
                else:
                    raise ValueError(f"Unexpected GET request: {url}")

            # Mock wait_for_model_sync to simulate cancellation during sync
            def mock_wait_for_model_sync(
                customized_model, flywheel_run_id, check_interval=30, timeout=3600
            ):
                # Simulate cancellation during model sync
                raise Exception(
                    "Flywheel run cancelled: Flywheel run was cancelled during model sync"
                )

            mock_requests.post.side_effect = mock_post_request
            mock_requests.get.side_effect = mock_get_request
            mock_sleep.return_value = None
            mock_cancel_job.return_value = None
            mock_wait_sync.side_effect = mock_wait_for_model_sync

            # Execute spin up NIM
            nim_config = task_result.nim.model_dump()
            nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

            # Execute customization (should fail during model sync)
            result = ensure_task_result(start_customization(nim_result))
            assert result.error is not None
            assert "Flywheel run cancelled" in result.error

            # Verify cleanup operations were performed
            # 1. Verify the job was cancelled
            mock_cancel_job.assert_called_once_with("customization-job-456")

            # 2. Verify database state reflects the cancellation
            customizations = list(mongo_db.customizations.find({"nim_id": env_info["nim_id"]}))
            assert len(customizations) == 1
            assert customizations[0]["progress"] == 0.0
            assert "error" in customizations[0]
            assert "Flywheel run cancelled" in customizations[0]["error"]
