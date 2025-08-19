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
Consolidated Cancellation Integration Tests

Tests cancellation workflows with real database dependencies:
- Task cancellation at every stage of the workflow
- Critical path cancellation (raise_error=True) that stops entire workflow
- Graceful skip cancellation (raise_error=False) that continues workflow
- Cancellation during waiting phases (DMS, Customizer, Evaluator)
- Database state verification and error propagation
- Uses real cancel_job_resources trigger mechanism
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
from src.lib.flywheel.cancellation import FlywheelCancelledError
from src.tasks.tasks import (
    cancel_job_resources,
    create_datasets,
    finalize_flywheel_run,
    initialize_workflow,
    run_base_eval,
    run_customization_eval,
    run_icl_eval,
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


def verify_subsequent_tasks_skipped(
    cancelled_result: TaskResult,
    env_info: dict,
    mongo_db,
    tasks_to_test: list[str] | None = None,
    verify_no_db_records: bool = True,
) -> None:
    """
    Helper function to verify that subsequent tasks are properly skipped after cancellation.

    Args:
        cancelled_result: The TaskResult from the cancelled task (should have error set)
        env_info: Environment info dictionary from the test fixture
        mongo_db: MongoDB connection for verification
        tasks_to_test: List of task names to test (defaults to all relevant tasks)
        verify_no_db_records: Whether to verify no database records exist (False for "during wait" cancellation tests)
    """
    if tasks_to_test is None:
        tasks_to_test = [
            "run_base_eval",
            "run_icl_eval",
            "start_customization",
            "run_customization_eval",
        ]

    # Test that specified subsequent tasks are properly skipped due to the error
    task_results = {}

    for task_name in tasks_to_test:
        if task_name == "run_base_eval":
            task_results[task_name] = ensure_task_result(run_base_eval(cancelled_result))
        elif task_name == "run_icl_eval":
            task_results[task_name] = ensure_task_result(run_icl_eval(cancelled_result))
        elif task_name == "start_customization":
            task_results[task_name] = ensure_task_result(start_customization(cancelled_result))
        elif task_name == "run_customization_eval":
            task_results[task_name] = ensure_task_result(run_customization_eval(cancelled_result))
        else:
            raise ValueError(f"Unsupported task: {task_name}")

    # Verify all tasks have errors and propagate the original error (with some flexibility)
    for task_name, result in task_results.items():
        assert result.error is not None, f"{task_name} should have an error"
        # For most tasks, the error should be exactly propagated
        # For customization_eval, it might have different error messages depending on state
        if task_name == "run_customization_eval":
            # More flexible error checking for customization eval
            expected_messages = [
                cancelled_result.error,
                "No customized model available for evaluation",
            ]
            assert any(
                msg in result.error for msg in expected_messages
            ), f"{task_name} error '{result.error}' doesn't match expected messages"
        else:
            assert (
                result.error == cancelled_result.error
            ), f"{task_name} should propagate the same error: expected '{cancelled_result.error}', got '{result.error}'"

    # Verify no spurious database records were created (only if requested)
    if verify_no_db_records:
        _verify_no_spurious_db_records(env_info, mongo_db)


def _verify_no_spurious_db_records(env_info: dict, mongo_db) -> None:
    """Verify no evaluation or customization records were created due to skipping."""
    # Verify no evaluation records created
    evaluations = list(
        mongo_db.evaluations.find({"flywheel_run_id": env_info["flywheel_run_id_obj"]})
    )
    assert len(evaluations) == 0, f"Expected no evaluations, found {len(evaluations)}"

    # Verify no customization records created
    nim_docs = list(mongo_db.nims.find({"flywheel_run_id": env_info["flywheel_run_id_obj"]}))
    if nim_docs:
        customizations = list(mongo_db.customizations.find({"nim_id": nim_docs[0]["_id"]}))
        assert len(customizations) == 0, f"Expected no customizations, found {len(customizations)}"


@pytest.fixture(autouse=True)
def setup_db_manager(mongo_db):
    """Setup database manager for cancellation tests."""
    import src.tasks.tasks as tasks_module
    from src.api.db import init_db
    from src.api.db_manager import get_db_manager

    init_db()
    tasks_module.db_manager = get_db_manager()

    yield


def load_tool_calling_data():
    """Load first 20 records from aiva-test.jsonl for tool-calling tests."""
    try:
        with open("data/aiva-test.jsonl") as f:
            records = []
            for i, line in enumerate(f):
                if i >= 20:  # Only first 20 records
                    break
                records.append(json.loads(line.strip()))
            return records
    except FileNotFoundError:
        return []


@pytest.fixture(scope="module")
def load_cancellation_test_data():
    """Load test data once for all cancellation tests."""
    from src.lib.integration.es_client import ES_COLLECTION_NAME, get_es_client

    es_client = get_es_client()

    # Load tool-calling data once
    aiva_records = load_tool_calling_data()
    for i, aiva_record in enumerate(aiva_records):
        record = {
            "client_id": "test-client-toolcalling-cancellation",
            "workload_id": "test-toolcalling-cancellation",
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
    for i in range(60):
        record = {
            "client_id": "test-client-generic-cancellation",
            "workload_id": "test-generic-cancellation",
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
                                                "client_id": "test-client-toolcalling-cancellation"
                                            }
                                        },
                                        {"match": {"workload_id": "test-toolcalling-cancellation"}},
                                    ]
                                }
                            },
                            {
                                "bool": {
                                    "must": [
                                        {
                                            "match": {
                                                "client_id": "test-client-generic-cancellation"
                                            }
                                        },
                                        {"match": {"workload_id": "test-generic-cancellation"}},
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


@pytest.fixture(params=["generic", "tool_calling"])
def workload_type(request):
    """Parameterized fixture for different workload types in cancellation tests."""
    return request.param


@pytest.fixture(scope="module")
def mock_embedding_services():
    """Mock embedding services once for all cancellation tests."""
    with (
        patch("src.lib.flywheel.icl_selection.Embedding") as mock_embedding_class,
        patch("src.lib.integration.es_client.index_embeddings_to_es") as mock_index_embeddings,
        patch("src.lib.integration.es_client.search_similar_embeddings") as mock_search_embeddings,
        patch("src.lib.integration.es_client.delete_embeddings_index") as mock_delete_embeddings,
        patch("src.lib.flywheel.icl_selection.index_embeddings_to_es") as mock_index_embeddings_icl,
        patch(
            "src.lib.flywheel.icl_selection.search_similar_embeddings"
        ) as mock_search_embeddings_icl,
    ):
        # Setup embedding service mocking
        mock_embedding_instance = mock_embedding_class.return_value

        def mock_get_embeddings_batch(queries, input_type="query"):
            # Return a unique, properly dimensioned vector for each query
            return [[0.1 + 0.01 * i] * 2048 for i in range(len(queries))]

        mock_embedding_instance.get_embeddings_batch.side_effect = mock_get_embeddings_batch

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

        yield {
            "embedding_class": mock_embedding_class,
            "embedding_instance": mock_embedding_instance,
            "index_embeddings": mock_index_embeddings,
            "search_embeddings": mock_search_embeddings,
            "delete_embeddings": mock_delete_embeddings,
            "index_embeddings_icl": mock_index_embeddings_icl,
            "search_embeddings_icl": mock_search_embeddings_icl,
        }


@pytest.fixture
def cancel_environment(
    mongo_db, workload_type, load_cancellation_test_data, mock_embedding_services
):
    """Setup cancellation environment with realistic cancellation triggers."""

    # Use the pre-loaded data instead of generating new data
    workload_name = (
        workload_type.replace("_", "") if workload_type == "tool_calling" else workload_type
    )
    client_id = f"test-client-{workload_name}-cancellation"
    workload_id = f"test-{workload_name}-cancellation"

    # Setup comprehensive mocking for external services only
    with (
        patch("src.lib.nemo.customizer.requests") as mock_requests,
        patch("src.lib.nemo.evaluator.requests") as mock_evaluator_requests,
        patch("src.lib.nemo.data_uploader.DataUploader.upload_data") as mock_upload,
        patch("src.lib.nemo.data_uploader.DataUploader.get_file_uri") as mock_get_uri,
        patch("time.sleep") as mock_sleep,
        patch("src.lib.nemo.dms_client.DMSClient") as mock_dms,
        patch("src.lib.nemo.dms_client.requests") as mock_dms_requests,
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

        # Setup evaluator service mocking
        eval_job_counter = 0

        def mock_evaluator_post_request(url, **kwargs):
            nonlocal eval_job_counter
            if "/v1/evaluation/jobs" in str(url):
                eval_job_counter += 1
                job_id = f"eval-job-{eval_job_counter}"
                return MagicMock(status_code=200, json=lambda: {"id": job_id})
            else:
                raise ValueError(f"Unexpected POST request: {url}")

        def mock_evaluator_get_request(url, **kwargs):
            url_str = str(url)
            if "/v1/evaluation/jobs/" in url_str and "/results" not in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {"status": "completed", "status_details": {"progress": 100}},
                )
            elif "/v1/evaluation/jobs/" in url_str and "/results" in url_str:
                return MagicMock(
                    status_code=200,
                    json=lambda: {
                        "tasks": {
                            "llm-as-judge": {
                                "metrics": {
                                    "llm-judge": {"scores": {"similarity": {"value": 0.85}}}
                                }
                            },
                            "custom-tool-calling": {
                                "metrics": {
                                    "tool-calling-accuracy": {
                                        "scores": {
                                            "function_name_accuracy": {"value": 0.90},
                                            "function_name_and_args_accuracy": {"value": 0.85},
                                        }
                                    },
                                    "correctness": {"scores": {"rating": {"value": 0.88}}},
                                }
                            },
                        }
                    },
                )
            else:
                raise ValueError(f"Unexpected GET request: {url}")

        mock_evaluator_requests.post.side_effect = mock_evaluator_post_request
        mock_evaluator_requests.get.side_effect = mock_evaluator_get_request

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
        mock_get_uri.return_value = f"test://dataset-uri-{workload_id}"

        # Setup DMS client mocking
        mock_dms_instance = mock_dms.return_value
        mock_dms_instance.is_deployed.return_value = False
        mock_dms_instance.deploy_model.return_value = None
        mock_dms_instance.wait_for_deployment.return_value = None
        mock_dms_instance.shutdown_deployment.return_value = None

        # Create flywheel run record
        flywheel_run_id = ObjectId()
        mongo_db.flywheel_runs.insert_one(
            {
                "_id": flywheel_run_id,
                "workload_id": workload_id,
                "client_id": client_id,
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
                workload_id=workload_id,  # Use base ID for data lookup
                flywheel_run_id=str(flywheel_run_id),
                client_id=client_id,  # Use base ID for data lookup
            )

        # Ensure task_result is TaskResult object
        task_result = ensure_task_result(task_result)

        # Adjust validation requirements for both workload types using minimal viable config
        from src.config import DataSplitConfig

        custom_split_config = DataSplitConfig(
            eval_size=1,  # Force random splitting by setting eval_size < num_classes
            val_ratio=0.1,
            min_total_records=10,  # Very low minimum to work with deduplicated tool-calling data
            random_seed=None,
            limit=100,  # Higher limit to accommodate larger generic dataset
            parse_function_arguments=True,
        )
        task_result.data_split_config = custom_split_config

        # Helper functions for realistic cancellation testing
        def trigger_cancellation(flywheel_run_id_str: str):
            """Trigger cancellation using the real production mechanism."""
            cancel_job_resources(flywheel_run_id_str)

        def verify_flywheel_cancelled(flywheel_run_id_str: str):
            """Verify flywheel run is marked as cancelled in database."""
            from src.api.db_manager import get_db_manager

            db_manager = get_db_manager()
            return db_manager.is_flywheel_run_cancelled(flywheel_run_id_str)

        def verify_nim_cancelled(nim_id: ObjectId):
            """Verify specific NIM is marked as cancelled."""
            nim_doc = mongo_db.nims.find_one({"_id": nim_id})
            return nim_doc and nim_doc.get("status") == "cancelled"

        def verify_all_nims_cancelled(flywheel_run_id_obj: ObjectId):
            """Verify all NIMs for this flywheel run are cancelled."""
            nims = list(mongo_db.nims.find({"flywheel_run_id": flywheel_run_id_obj}))
            return all(nim.get("status") == "cancelled" for nim in nims)

        # Return comprehensive environment info
        environment_info = {
            "flywheel_run_id": str(flywheel_run_id),
            "flywheel_run_id_obj": flywheel_run_id,
            "model_name": "nim/test-model",
            "customization_enabled": True,
            "task_result": task_result,
            "num_records": 0,
            "workload_type": workload_type,
            "workload_id": workload_id,
            "client_id": client_id,
            # Cancellation utilities
            "trigger_cancellation": trigger_cancellation,
            "verify_flywheel_cancelled": verify_flywheel_cancelled,
            "verify_nim_cancelled": verify_nim_cancelled,
            "verify_all_nims_cancelled": verify_all_nims_cancelled,
            "mocks": {
                "requests": mock_requests,
                "evaluator_requests": mock_evaluator_requests,
                "sleep": mock_sleep,
                "dms_requests": mock_dms_requests,
                "upload": mock_upload,
                "dms": mock_dms,
                "embedding": mock_embedding_services["embedding_class"],
            },
        }

        yield environment_info

        # Cleanup MongoDB collections (ES cleanup is handled by module-scoped fixture)
        mongo_db.customizations.delete_many({})
        mongo_db.nims.delete_many({})
        mongo_db.flywheel_runs.delete_many({})
        mongo_db.llm_judge_runs.delete_many({})


@pytest.mark.integration
@pytest.mark.cancellation
class TestCancellationCriticalPath:
    """Test cancellation in critical path tasks (raise_error=True) that stop entire workflow."""

    def test_initialize_workflow_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during initialize_workflow (critical path)."""
        env_info = cancel_environment

        # Trigger cancellation before initialize_workflow
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute initialize_workflow - should raise FlywheelCancelledError
        with pytest.raises(FlywheelCancelledError) as exc_info:
            initialize_workflow(
                workload_id=f"cancelled-{env_info['workload_id']}",
                flywheel_run_id=env_info["flywheel_run_id"],
                client_id=env_info["client_id"],
            )

        # Verify exception details
        assert env_info["flywheel_run_id"] in str(exc_info.value)

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Verify workflow stops completely (no NIMs created for new workflow)
        new_nims = list(mongo_db.nims.find({"workload_id": f"cancelled-{env_info['workload_id']}"}))
        assert len(new_nims) == 0

    def test_create_datasets_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during create_datasets (critical path)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Trigger cancellation before create_datasets
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute create_datasets - should raise FlywheelCancelledError
        with pytest.raises(FlywheelCancelledError):
            create_datasets(task_result)

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])
        assert env_info["verify_all_nims_cancelled"](env_info["flywheel_run_id_obj"])

    def test_wait_for_llm_as_judge_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during wait_for_llm_as_judge (critical path)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Run create_datasets first
        task_result = ensure_task_result(create_datasets(task_result))

        # Trigger cancellation before wait_for_llm_as_judge
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute wait_for_llm_as_judge - should raise FlywheelCancelledError
        with pytest.raises(ValueError) as exc_info:  # Wrapped in ValueError in the task
            wait_for_llm_as_judge(task_result)

        # Verify it's a cancellation error
        assert "Error waiting for LLM as judge" in str(exc_info.value)

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])
        assert env_info["verify_all_nims_cancelled"](env_info["flywheel_run_id_obj"])


@pytest.mark.integration
@pytest.mark.cancellation
class TestCancellationHybridBehavior:
    """Test cancellation in tasks with hybrid behavior (raise_error=True but graceful handling)."""

    def test_spin_up_nim_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during spin_up_nim (hybrid: raises exception but handles gracefully)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Run prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Trigger cancellation before spin_up_nim
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        nim_config = (
            task_result.nim.model_dump()
            if task_result.nim
            else {"model_name": "nim/test-model", "context_length": 8192}
        )

        # Mock the check_cancellation to verify exception-raising behavior
        with patch("src.tasks.tasks.check_cancellation") as mock_check:
            # Configure mock to raise FlywheelCancelledError as the real implementation would
            mock_check.side_effect = FlywheelCancelledError(
                env_info["flywheel_run_id"], "Flywheel run cancelled"
            )

            # Execute spin_up_nim - should catch FlywheelCancelledError internally and return TaskResult with error
            result = ensure_task_result(spin_up_nim(task_result, nim_config))

            # Verify that check_cancellation was called (confirming exception-raising path)
            mock_check.assert_called_once_with(env_info["flywheel_run_id"])

        # Verify graceful error handling - task returns TaskResult with error instead of raising
        assert result.error is not None
        assert "Flywheel run cancelled" in result.error

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Verify NIM is marked as cancelled in database
        # (NIM record was created earlier in initialize_workflow, not in spin_up_nim)
        nim_docs = list(mongo_db.nims.find({"flywheel_run_id": env_info["flywheel_run_id_obj"]}))
        if nim_docs:
            assert nim_docs[0].get("status") == "cancelled"

        # Test that all subsequent tasks are properly skipped due to the error
        verify_subsequent_tasks_skipped(result, env_info, mongo_db)


@pytest.mark.integration
@pytest.mark.cancellation
class TestCancellationGracefulSkip:
    """Test cancellation in graceful skip tasks (raise_error=False) that continue workflow."""

    def test_run_base_eval_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during run_base_eval (graceful skip)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(model_name="nim/test-model", context_length=8192)

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Trigger cancellation before run_base_eval
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute run_base_eval - should gracefully skip with error
        eval_result = ensure_task_result(run_base_eval(nim_result))

        # Verify graceful skip behavior
        assert eval_result.error is not None
        assert f"Task cancelled for flywheel run {env_info['flywheel_run_id']}" in eval_result.error

        # Verify database state - flywheel should be cancelled but task continues
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Verify no evaluation records created due to cancellation
        evaluations = list(
            mongo_db.evaluations.find({"flywheel_run_id": env_info["flywheel_run_id_obj"]})
        )
        assert len(evaluations) == 0

    def test_run_icl_eval_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during run_icl_eval (graceful skip)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={"target": "test-target", "gpus": 1},
        )

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Trigger cancellation before run_icl_eval
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute run_icl_eval - should gracefully skip with error
        eval_result = ensure_task_result(run_icl_eval(nim_result))

        # Verify graceful skip behavior
        assert eval_result.error is not None
        assert f"Task cancelled for flywheel run {env_info['flywheel_run_id']}" in eval_result.error

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Test that subsequent tasks are properly skipped due to the error
        verify_subsequent_tasks_skipped(
            eval_result,
            env_info,
            mongo_db,
            tasks_to_test=["start_customization", "run_customization_eval"],
        )

    def test_start_customization_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during start_customization (graceful skip)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={"target": "test-target", "gpus": 1},
        )

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Trigger cancellation before start_customization
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute start_customization - should gracefully skip with error
        custom_result = ensure_task_result(start_customization(nim_result))

        # Verify graceful skip behavior
        assert custom_result.error is not None
        assert (
            f"Task cancelled for flywheel run {env_info['flywheel_run_id']}" in custom_result.error
        )

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Test that subsequent task is properly skipped due to the error
        verify_subsequent_tasks_skipped(
            custom_result, env_info, mongo_db, tasks_to_test=["run_customization_eval"]
        )

    def test_run_customization_eval_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during run_customization_eval (graceful skip)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={"target": "test-target", "gpus": 1},
        )

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Trigger cancellation before run_customization_eval
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute run_customization_eval - should gracefully skip with error
        eval_result = ensure_task_result(run_customization_eval(nim_result))

        # Verify graceful skip behavior
        assert eval_result.error is not None
        # The error could be either cancellation or skip due to missing customization
        expected_messages = [
            f"Task cancelled for flywheel run {env_info['flywheel_run_id']}",
            "No customized model available for evaluation",
        ]
        assert any(msg in eval_result.error for msg in expected_messages)

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Verify no spurious database records were created
        _verify_no_spurious_db_records(env_info, mongo_db)

    def test_shutdown_deployment_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during shutdown_deployment (graceful skip)."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={"target": "test-target", "gpus": 1},
        )

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Trigger cancellation before shutdown_deployment
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute shutdown_deployment - should handle cancellation gracefully
        shutdown_result = ensure_task_result(shutdown_deployment(nim_result))

        # Verify the task completed (shutdown should still work even if cancelled)
        assert shutdown_result is not None

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Verify NIM is marked as cancelled
        nim_docs = list(mongo_db.nims.find({"flywheel_run_id": env_info["flywheel_run_id_obj"]}))
        if nim_docs:
            assert nim_docs[0].get("status") == "cancelled"

        # Note: shutdown_deployment is a cleanup task that doesn't use _should_skip_stage
        # and should run regardless of cancellation, so we don't test subsequent task skipping


@pytest.mark.integration
@pytest.mark.cancellation
class TestCancellationWaitingPhases:
    """Test cancellation during long-running waiting operations."""

    def test_dms_wait_for_deployment_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during DMSClient.wait_for_deployment."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
        )

        # Mock both check_cancellation import paths to simulate waiting and check cancellation
        with (
            patch("src.tasks.tasks.check_cancellation") as mock_check_cancel_tasks,
            patch("src.lib.nemo.dms_client.check_cancellation") as mock_check_cancel_dms,
        ):
            # First call (from tasks.py) succeeds, second call (from dms_client.py) raises cancellation
            mock_check_cancel_tasks.side_effect = [None]
            mock_check_cancel_dms.side_effect = [
                FlywheelCancelledError(
                    env_info["flywheel_run_id"], "Cancelled during deployment wait"
                )
            ]

            # Trigger cancellation
            env_info["trigger_cancellation"](env_info["flywheel_run_id"])

            # Execute spin_up_nim which will call wait_for_deployment
            nim_config = task_result.nim.model_dump()

            # spin_up_nim handles exceptions gracefully and returns TaskResult with error
            result = ensure_task_result(spin_up_nim(task_result, nim_config))

            # Verify cancellation was detected and handled gracefully
            assert result.error is not None
            assert (
                "Cancelled during deployment wait" in result.error
                or "Flywheel run cancelled" in result.error
            )

            # Verify check_cancellation was called
            assert mock_check_cancel_tasks.call_count >= 1
            assert mock_check_cancel_dms.call_count >= 1

        # Verify that subsequent tasks are properly skipped due to the error
        verify_subsequent_tasks_skipped(result, env_info, mongo_db)

    def test_dms_wait_for_model_sync_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during DMSClient.wait_for_model_sync."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
        )

        # Mock both check_cancellation import paths to simulate model sync cancellation
        with (
            patch("src.tasks.tasks.check_cancellation") as mock_check_cancel_tasks,
            patch("src.lib.nemo.dms_client.check_cancellation") as mock_check_cancel_dms,
        ):
            # First call (from tasks.py) succeeds, then DMS calls during wait_for_deployment and wait_for_model_sync
            mock_check_cancel_tasks.side_effect = [None]
            mock_check_cancel_dms.side_effect = [
                None,
                FlywheelCancelledError(env_info["flywheel_run_id"], "Cancelled during model sync"),
            ]

            # Trigger cancellation
            env_info["trigger_cancellation"](env_info["flywheel_run_id"])

            # Execute spin_up_nim which calls both wait_for_deployment and wait_for_model_sync
            nim_config = task_result.nim.model_dump()

            # spin_up_nim handles exceptions gracefully and returns TaskResult with error
            result = ensure_task_result(spin_up_nim(task_result, nim_config))

            # Verify cancellation was detected and handled gracefully
            assert result.error is not None
            assert (
                "Cancelled during model sync" in result.error
                or "Flywheel run cancelled" in result.error
            )

        # Verify that subsequent tasks are properly skipped due to the error
        verify_subsequent_tasks_skipped(result, env_info, mongo_db)

    def test_customizer_wait_for_customization_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during Customizer.wait_for_customization."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={
                "target": "test-target",
                "gpus": 1,
            },
        )

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Mock both check_cancellation import paths to simulate cancellation during waiting
        with (
            patch("src.tasks.tasks.check_cancellation") as mock_check_cancel_tasks,
            patch("src.lib.nemo.customizer.check_cancellation") as mock_check_cancel_customizer,
        ):
            # Task check succeeds, customizer check raises cancellation
            mock_check_cancel_tasks.side_effect = [None]
            mock_check_cancel_customizer.side_effect = [
                FlywheelCancelledError(
                    env_info["flywheel_run_id"], "Cancelled during customization wait"
                )
            ]

            # Trigger cancellation
            env_info["trigger_cancellation"](env_info["flywheel_run_id"])

            # Execute start_customization which calls wait_for_customization
            custom_result = ensure_task_result(start_customization(nim_result))

            # Verify cancellation was handled
            assert custom_result.error is not None
            assert (
                "Cancelled during customization wait" in custom_result.error
                or "Error starting customization" in custom_result.error
            )

        # Verify that subsequent task is properly skipped due to the error
        verify_subsequent_tasks_skipped(
            custom_result,
            env_info,
            mongo_db,
            tasks_to_test=["run_customization_eval"],
            verify_no_db_records=False,  # Customization record was legitimately created before cancellation
        )

    def test_customizer_wait_for_model_sync_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during Customizer.wait_for_model_sync."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={
                "target": "test-target",
                "gpus": 1,
            },
        )

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Mock both check_cancellation import paths to simulate cancellation during model sync
        with (
            patch("src.tasks.tasks.check_cancellation") as mock_check_cancel_tasks,
            patch("src.lib.nemo.customizer.check_cancellation") as mock_check_cancel_customizer,
            patch("src.lib.nemo.customizer.Customizer.cancel_job") as mock_cancel_job,
        ):
            # Task check succeeds, then customizer calls succeed initially, then cancellation during model sync
            mock_check_cancel_tasks.side_effect = [None]
            mock_check_cancel_customizer.side_effect = [
                None,
                FlywheelCancelledError(env_info["flywheel_run_id"], "Cancelled during model sync"),
            ]
            mock_cancel_job.return_value = None

            # Trigger cancellation
            env_info["trigger_cancellation"](env_info["flywheel_run_id"])

            # Execute start_customization
            custom_result = ensure_task_result(start_customization(nim_result))

            # Verify cancellation was handled
            assert custom_result.error is not None
            assert (
                "Cancelled during model sync" in custom_result.error
                or "Error starting customization" in custom_result.error
            )

            # Verify cleanup was attempted
            mock_cancel_job.assert_called()

        # Verify that subsequent task is properly skipped due to the error
        verify_subsequent_tasks_skipped(
            custom_result,
            env_info,
            mongo_db,
            tasks_to_test=["run_customization_eval"],
            verify_no_db_records=False,  # Customization record was legitimately created before cancellation
        )

    def test_evaluator_wait_for_evaluation_cancellation(self, cancel_environment, mongo_db):
        """Test cancellation during Evaluator.wait_for_evaluation."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Setup prerequisites
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
        )

        # Execute spin_up_nim first
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Mock both check_cancellation import paths to simulate cancellation during waiting
        with (
            patch("src.tasks.tasks.check_cancellation") as mock_check_cancel_tasks,
            patch("src.lib.nemo.evaluator.check_cancellation") as mock_check_cancel_evaluator,
        ):
            # Task check succeeds, evaluator check raises cancellation
            mock_check_cancel_tasks.side_effect = [None]
            mock_check_cancel_evaluator.side_effect = [
                FlywheelCancelledError(
                    env_info["flywheel_run_id"], "Cancelled during evaluation wait"
                )
            ]

            # Trigger cancellation
            env_info["trigger_cancellation"](env_info["flywheel_run_id"])

            # Execute run_base_eval which calls wait_for_evaluation
            eval_result = ensure_task_result(run_base_eval(nim_result))

            # Verify cancellation was handled
            assert eval_result.error is not None
            assert (
                "Cancelled during evaluation wait" in eval_result.error
                or "Error running base evaluation" in eval_result.error
                or f"Task cancelled for flywheel run {env_info['flywheel_run_id']}"
                in eval_result.error
            )

        # Verify that subsequent tasks are properly skipped due to the error
        verify_subsequent_tasks_skipped(
            eval_result,
            env_info,
            mongo_db,
            tasks_to_test=["run_icl_eval", "start_customization", "run_customization_eval"],
        )


@pytest.mark.integration
@pytest.mark.cancellation
class TestCancellationWorkflowIntegration:
    """Test end-to-end cancellation scenarios across entire workflow."""

    def test_full_workflow_cancellation_during_evaluation(self, cancel_environment, mongo_db):
        """Test cancellation during evaluation phase of full workflow."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Execute workflow up to evaluation
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
        )

        # Execute spin_up_nim
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Trigger cancellation during evaluation phase
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute evaluation stages
        base_eval_result = ensure_task_result(run_base_eval(nim_result))
        icl_eval_result = ensure_task_result(run_icl_eval(nim_result))

        # Both evaluations should be cancelled
        assert base_eval_result.error is not None
        assert icl_eval_result.error is not None

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Subsequent stages should also skip
        custom_result = ensure_task_result(start_customization(nim_result))
        assert custom_result.error is not None

        # Shutdown should still work
        shutdown_result = shutdown_deployment(custom_result)
        assert shutdown_result is not None

        # Finalize should still work
        finalize_result = finalize_flywheel_run(shutdown_result)
        assert finalize_result is not None

    def test_full_workflow_cancellation_during_customization(self, cancel_environment, mongo_db):
        """Test cancellation during customization phase of full workflow."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Execute workflow up to customization
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={
                "target": "test-target",
                "gpus": 1,
            },
        )

        # Execute workflow stages
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))
        _ = ensure_task_result(run_base_eval(nim_result))
        icl_eval_result = ensure_task_result(run_icl_eval(nim_result))

        # Trigger cancellation before customization
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute customization stages
        custom_result = ensure_task_result(start_customization(icl_eval_result))
        custom_eval_result = ensure_task_result(run_customization_eval(custom_result))

        # Both customization stages should be cancelled
        assert custom_result.error is not None
        assert custom_eval_result.error is not None

        # Verify database state
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Verify no customization records created
        nim_docs = list(mongo_db.nims.find({"flywheel_run_id": env_info["flywheel_run_id_obj"]}))
        if nim_docs:
            customizations = list(mongo_db.customizations.find({"nim_id": nim_docs[0]["_id"]}))
            assert len(customizations) == 0

        # Cleanup stages should still work
        shutdown_result = shutdown_deployment(custom_eval_result)
        finalize_result = finalize_flywheel_run(shutdown_result)
        assert shutdown_result is not None
        assert finalize_result is not None

    def test_partial_workflow_with_mixed_errors(self, cancel_environment, mongo_db):
        """Test workflow behavior with both cancellation and other errors."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Execute workflow up to evaluation
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={
                "target": "test-target",
                "gpus": 1,
            },
        )

        # Execute spin_up_nim
        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # First introduce a non-cancellation error
        nim_result.error = "Previous evaluation failed"

        # Then trigger cancellation
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute subsequent stages
        custom_result = ensure_task_result(start_customization(nim_result))

        # Should skip due to previous error, not cancellation
        assert custom_result.error == "Previous evaluation failed"

        # Verify database state shows cancellation
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

    def test_cancellation_cleanup_verification(self, cancel_environment, mongo_db):
        """Test that cancellation properly cleans up all resources."""
        env_info = cancel_environment
        task_result = env_info["task_result"]

        # Execute partial workflow
        task_result = ensure_task_result(create_datasets(task_result))
        task_result = ensure_task_result(wait_for_llm_as_judge(task_result))

        # Add nim config to task result
        from src.config import NIMConfig

        task_result.nim = NIMConfig(
            model_name="nim/test-model",
            context_length=8192,
            customization_enabled=True,
            customizer_configs={
                "target": "test-target",
                "gpus": 1,
            },
        )

        nim_config = task_result.nim.model_dump()
        nim_result = ensure_task_result(spin_up_nim(task_result, nim_config))

        # Trigger cancellation
        env_info["trigger_cancellation"](env_info["flywheel_run_id"])

        # Execute remaining workflow
        base_eval_result = ensure_task_result(run_base_eval(nim_result))
        custom_result = ensure_task_result(start_customization(nim_result))
        shutdown_result = ensure_task_result(shutdown_deployment(custom_result))
        finalize_result = ensure_task_result(finalize_flywheel_run(shutdown_result))

        # Verify comprehensive cleanup
        assert env_info["verify_flywheel_cancelled"](env_info["flywheel_run_id"])

        # Verify all tasks handled cancellation appropriately
        assert base_eval_result.error is not None
        assert custom_result.error is not None

        # Verify finalization still completed
        assert finalize_result is not None

        # Verify database consistency
        flywheel_run = mongo_db.flywheel_runs.find_one({"_id": env_info["flywheel_run_id_obj"]})
        assert flywheel_run["status"] == "cancelled"

        # Verify NIMs are properly marked
        nims = list(mongo_db.nims.find({"flywheel_run_id": env_info["flywheel_run_id_obj"]}))
        for nim in nims:
            assert nim["status"] in [
                "cancelled",
                "completed",
            ]  # May be completed if shutdown succeeded
