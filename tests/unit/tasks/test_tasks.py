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

from datetime import datetime
from unittest.mock import ANY, MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import (
    CustomizationResult,
    DatasetType,
    EvalType,
    EvaluationResult,
    NIMConfig,
    NIMRun,
    NIMRunStatus,
    TaskResult,
    ToolEvalType,
    WorkloadClassification,
)
from src.config import settings  # the singleton created in src.config
from src.tasks.tasks import (
    create_datasets,
    run_base_eval,
    run_customization_eval,
    run_icl_eval,
    shutdown_deployment,
    spin_up_nim,
    start_customization,
)


@pytest.fixture(name="mock_db")
def fixture_mock_task_db_manager():
    """Patch the *db_manager* instance used in tasks.py.

    After the recent refactor, Celery tasks no longer access raw pymongo
    collections; instead they delegate everything to the *TaskDBManager*
    helper stored as ``src.tasks.tasks.db_manager``.  Patch that singleton so
    each test can assert against the high-level helper methods.
    """
    with patch("src.tasks.tasks.db_manager") as mock:
        # Provide sensible defaults for the helper methods that return values.
        mock.create_nim_run.return_value = ObjectId()
        mock.find_nim_run.return_value = {"_id": ObjectId(), "model_name": "test-model"}
        mock.insert_evaluation.return_value = ObjectId()
        mock.insert_customization.return_value = ObjectId()
        yield mock


@pytest.fixture
def mock_init_db():
    """Fixture to mock database initialization."""
    with patch("src.tasks.tasks.init_db") as mock:
        yield mock


@pytest.fixture
def mock_data_uploader():
    """Fixture to mock DataUploader."""
    with patch("src.lib.integration.dataset_creator.DataUploader") as mock:
        mock_instance = MagicMock()
        # Ensure that `get_file_uri` (used when recording dataset metadata) returns a
        # plain string.  A raw ``MagicMock`` instance cannot be encoded by BSON and
        # causes an ``InvalidDocument`` error when the code under test attempts to
        # update MongoDB.
        mock_instance.get_file_uri.return_value = "nmp://test-namespace/datasets/dummy.jsonl"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_evaluator():
    """Fixture to mock Evaluator."""
    with patch("src.tasks.tasks.Evaluator") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_es_client():
    """Fixture to mock Elasticsearch client."""
    with patch("src.lib.integration.record_exporter.get_es_client") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


# This fixture automatically tweaks global settings for every test so that they
# use small, deterministic values and a test-specific namespace.  We patch
# *attributes* on the existing objects where they are mutable, but for frozen
# Pydantic models (e.g. `NMPConfig`) we replace the entire object with a
# modified copy generated via `model_copy(update={...})`.


@pytest.fixture(autouse=True)
def tweak_settings(monkeypatch):
    """Provide deterministic test configuration via the global `settings`."""

    # --- Data-split parameters (fields are *not* frozen) --------------------
    monkeypatch.setattr(settings.data_split_config, "min_total_records", 1, raising=False)
    monkeypatch.setattr(settings.data_split_config, "random_seed", 42, raising=False)
    monkeypatch.setattr(settings.data_split_config, "eval_size", 1, raising=False)
    monkeypatch.setattr(settings.data_split_config, "val_ratio", 0.25, raising=False)
    monkeypatch.setattr(settings.data_split_config, "limit", 100, raising=False)

    # --- NMP namespace (field *is* frozen, so create a new object) ----------
    new_nmp_cfg = settings.nmp_config.model_copy(update={"nmp_namespace": "test-namespace"})
    monkeypatch.setattr(settings, "nmp_config", new_nmp_cfg, raising=True)

    yield


# ---------------------------------------------------------------------------
# Compatibility: a few tests still request a ``mock_settings`` fixture.  Keep a
# thin wrapper so they receive the (already-tweaked) global ``settings``.
# ---------------------------------------------------------------------------


@pytest.fixture(name="mock_settings")
def fixture_mock_settings():
    """Return the globally patched `settings` instance used in the tests."""
    return settings


@pytest.fixture
def mock_dms_client():
    """Fixture to mock DMSClient."""
    with patch("src.tasks.tasks.DMSClient") as mock:
        mock_instance = MagicMock()
        # Configure the mock instance methods
        mock_instance.is_deployed.return_value = False
        mock_instance.deploy_model.return_value = None
        mock_instance.wait_for_deployment.return_value = None
        mock_instance.wait_for_model_sync.return_value = None
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def valid_nim_config():
    """Fixture to create a valid NIMConfig instance."""
    return NIMConfig(
        model_name="test-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
        registry_base="nvcr.io/nim",
        customization_enabled=True,
    )


@pytest.fixture
def valid_nim_run(valid_nim_config):
    """Fixture to create a valid NIMRun instance."""
    return NIMRun(
        flywheel_run_id=ObjectId(),
        model_name=valid_nim_config.model_name,
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        runtime_seconds=0,
        status=NIMRunStatus.RUNNING,
    )


def test_create_datasets(mock_es_client, mock_data_uploader, mock_db, mock_init_db, mock_settings):
    """Test creating datasets from Elasticsearch data."""
    workload_id = "test-workload"
    flywheel_run_id = str(ObjectId())
    client_id = "test-client"

    # Configure mock ES response with 5 records and proper message format
    mock_es_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": "What is machine learning?"},
                                {
                                    "role": "assistant",
                                    "content": "Machine learning is a subset of AI...",
                                },
                            ]
                        },
                        "response": {
                            "choices": [{"message": {"content": "Here's an explanation..."}}]
                        },
                    }
                },
                {
                    "_source": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": "Explain neural networks"},
                                {"role": "assistant", "content": "Neural networks are..."},
                            ]
                        },
                        "response": {
                            "choices": [{"message": {"content": "Neural networks explanation..."}}]
                        },
                    }
                },
                {
                    "_source": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": "What is deep learning?"},
                                {"role": "assistant", "content": "Deep learning is..."},
                            ]
                        },
                        "response": {
                            "choices": [{"message": {"content": "Deep learning details..."}}]
                        },
                    }
                },
                {
                    "_source": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": "Explain reinforcement learning"},
                                {"role": "assistant", "content": "Reinforcement learning is..."},
                            ]
                        },
                        "response": {"choices": [{"message": {"content": "RL explanation..."}}]},
                    }
                },
                {
                    "_source": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": "What are transformers?"},
                                {"role": "assistant", "content": "Transformers are..."},
                            ]
                        },
                        "response": {
                            "choices": [{"message": {"content": "Transformer architecture..."}}]
                        },
                    }
                },
            ]
        }
    }

    result = create_datasets(workload_id, flywheel_run_id, client_id)

    # Instead of checking the type directly, verify the dictionary has the expected structure
    assert "status" in result
    assert "workload_id" in result
    assert "client_id" in result
    assert "datasets" in result
    assert "base-dataset" in result["datasets"]
    assert "train-dataset" in result["datasets"]

    # Verify the specific values
    assert result["workload_id"] == workload_id
    assert result["client_id"] == client_id

    # Verify the function calls
    mock_es_client.search.assert_called_once()
    assert mock_data_uploader.upload_data.call_count >= 1


def test_create_datasets_empty_data(
    mock_es_client, mock_data_uploader, mock_db, mock_init_db, mock_settings
):
    """Test creating datasets with empty Elasticsearch response."""
    workload_id = "test-workload"
    flywheel_run_id = str(ObjectId())
    client_id = "test-client"

    # Configure mock ES response with no records
    mock_es_client.search.return_value = {
        "hits": {
            "hits": []  # Empty hits list
        }
    }

    # The function should raise an exception for empty dataset
    with pytest.raises(Exception) as exc_info:
        create_datasets(workload_id, flywheel_run_id, client_id)

    # Verify the error message
    assert "No records found" in str(exc_info.value)

    # Verify ES was called but data uploader was not
    mock_es_client.search.assert_called_once()
    mock_data_uploader.upload_data.assert_not_called()


def test_spin_up_nim(mock_db, mock_init_db, mock_dms_client, valid_nim_config):
    """Test spinning up NIM instance."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure TaskDBManager behaviour
    mock_db.create_nim_run.return_value = ObjectId()

    spin_up_nim(previous_result, valid_nim_config.model_dump())

    # Verify DMS client method calls
    mock_dms_client.is_deployed.assert_called_once()
    mock_dms_client.deploy_model.assert_called_once()
    mock_dms_client.wait_for_deployment.assert_called_once()
    mock_dms_client.wait_for_model_sync.assert_called_once()

    # Verify DB-helper interactions
    mock_db.create_nim_run.assert_called_once()
    assert mock_db.set_nim_status.call_count >= 2  # status transitions

    # No error should be present on the previous_result
    assert previous_result.error is None


def test_spin_up_nim_deployment_failure(mock_db, mock_init_db, mock_dms_client, valid_nim_config):
    """Test spinning up NIM instance when deployment fails."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure TaskDBManager behaviour
    mock_db.create_nim_run.return_value = ObjectId()

    # Configure DMS client to fail deployment
    mock_dms_client.deploy_model.side_effect = Exception("Deployment failed")

    # Call the function and ensure the error is captured on the returned TaskResult
    spin_up_nim(previous_result, valid_nim_config.model_dump())

    assert previous_result.error is not None
    assert "Deployment failed" in previous_result.error

    # Verify DMS client method calls
    mock_dms_client.is_deployed.assert_called_once()
    mock_dms_client.deploy_model.assert_called_once()
    mock_dms_client.wait_for_deployment.assert_not_called()
    mock_dms_client.wait_for_model_sync.assert_not_called()

    # Verify DB-helper interactions
    mock_db.create_nim_run.assert_called_once()
    assert mock_db.set_nim_status.call_count >= 2  # initial + error status


def test_spin_up_nim_already_deployed(mock_db, mock_init_db, mock_dms_client, valid_nim_config):
    """Test spinning up NIM instance when it's already deployed."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure TaskDBManager behaviour
    mock_db.create_nim_run.return_value = ObjectId()

    # Configure DMS client to indicate NIM is already deployed
    mock_dms_client.is_deployed.return_value = True

    spin_up_nim(previous_result, valid_nim_config.model_dump())

    # Verify DMS client method calls
    mock_dms_client.is_deployed.assert_called_once()
    mock_dms_client.deploy_model.assert_not_called()  # Should not try to deploy again
    mock_dms_client.wait_for_deployment.assert_called_once()
    mock_dms_client.wait_for_model_sync.assert_called_once()

    # Verify DB-helper interactions
    mock_db.create_nim_run.assert_called_once()
    assert mock_db.set_nim_status.call_count >= 2  # status updates


def test_run_base_eval(mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings):
    """Test running base evaluation."""
    nim_id = ObjectId()
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={DatasetType.BASE: "test-base-dataset"},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = ObjectId()

    # Configure mock evaluator
    mock_evaluator.run_evaluation.return_value = "job-123"
    mock_evaluator.get_job_uri.return_value = "http://test-uri"
    mock_evaluator.get_evaluation_results.return_value = {
        "tasks": {
            "llm-as-judge": {"metrics": {"llm-judge": {"scores": {"similarity": {"value": 0.95}}}}}
        }
    }

    run_base_eval(previous_result)

    # Verify evaluator calls
    mock_evaluator.run_evaluation.assert_called_once_with(
        namespace="test-namespace",
        dataset_name="test-base-dataset",
        workload_type=WorkloadClassification.GENERIC,
        target_model=valid_nim_config.target_model_for_evaluation(),
        test_file="eval_data.jsonl",
        tool_eval_type=None,
        limit=100,
    )
    mock_evaluator.wait_for_evaluation.assert_called_once()
    mock_evaluator.get_evaluation_results.assert_called_once_with("job-123")

    # Verify DB-helper interactions
    mock_db.find_nim_run.assert_called_once()
    mock_db.insert_evaluation.assert_called_once()
    assert mock_db.update_evaluation.call_count >= 2  # progress + final


def test_run_icl_eval(mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings):
    """Test running ICL evaluation."""
    nim_id = ObjectId()
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.TOOL_CALLING,
        datasets={DatasetType.ICL: "test-icl-dataset"},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = ObjectId()

    # Configure mock evaluator for tool calling evaluation
    mock_evaluator.run_evaluation.return_value = "job-123"
    mock_evaluator.get_job_uri.return_value = "http://test-uri"
    mock_evaluator.get_evaluation_results.return_value = {
        "tasks": {
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
            }
        }
    }

    run_icl_eval(previous_result)

    # Verify evaluator calls
    mock_evaluator.run_evaluation.assert_called_once_with(
        namespace="test-namespace",
        dataset_name="test-icl-dataset",
        workload_type=WorkloadClassification.TOOL_CALLING,
        target_model=valid_nim_config.target_model_for_evaluation(),
        test_file="eval_data.jsonl",
        tool_eval_type=ToolEvalType.TOOL_CALLING_METRIC,
        limit=100,
    )
    mock_evaluator.wait_for_evaluation.assert_called_once()
    mock_evaluator.get_evaluation_results.assert_called_once_with("job-123")

    # Verify DB-helper interactions
    mock_db.find_nim_run.assert_called_once()
    mock_db.insert_evaluation.assert_called_once()
    assert mock_db.update_evaluation.call_count >= 2  # progress + final


def test_run_base_eval_failure(
    mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings
):
    """Test running base evaluation when it fails."""
    nim_id = ObjectId()
    eval_id = ObjectId()
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={DatasetType.BASE: "test-base-dataset"},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = eval_id

    # Configure mock evaluator to fail
    mock_evaluator.run_evaluation.side_effect = Exception("Evaluation failed")

    run_base_eval(previous_result)

    # Verify error handling
    mock_db.update_evaluation.assert_called_with(
        ANY,
        {
            "error": "Error running base-eval evaluation: Evaluation failed",
            "finished_at": ANY,
            "progress": 0.0,
        },
    )


def test_run_base_eval_results_failure(
    mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings
):
    """Test running base evaluation when results retrieval fails."""
    nim_id = ObjectId()
    eval_id = ObjectId()
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={DatasetType.BASE: "test-base-dataset"},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = eval_id

    # Configure mock evaluator to fail during results retrieval
    mock_evaluator.run_evaluation.return_value = "job-123"
    mock_evaluator.get_job_uri.return_value = "http://test-uri"
    mock_evaluator.wait_for_evaluation.side_effect = Exception("Timeout waiting for evaluation")

    run_base_eval(previous_result)

    # Verify error handling
    mock_db.update_evaluation.assert_called_with(
        ANY,
        {
            "error": "Error running base-eval evaluation: Timeout waiting for evaluation",
            "finished_at": ANY,
            "progress": 0.0,
        },
    )


def test_run_icl_eval_failure(
    mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings
):
    """Test running ICL evaluation when it fails."""
    nim_id = ObjectId()
    eval_id = ObjectId()
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.TOOL_CALLING,
        datasets={DatasetType.ICL: "test-icl-dataset"},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = eval_id

    # Configure mock evaluator to fail
    mock_evaluator.run_evaluation.side_effect = Exception("Tool calling evaluation failed")

    run_icl_eval(previous_result)

    # Verify error handling
    mock_db.update_evaluation.assert_called_with(
        ANY,
        {
            "error": "Error running icl-eval evaluation: Tool calling evaluation failed",
            "finished_at": ANY,
            "progress": 0.0,
        },
    )


def test_run_icl_eval_results_failure(
    mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings
):
    """Test running ICL evaluation when results retrieval fails."""
    nim_id = ObjectId()
    eval_id = ObjectId()
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.TOOL_CALLING,
        datasets={DatasetType.ICL: "test-icl-dataset"},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = eval_id

    # Configure mock evaluator to fail during results retrieval
    mock_evaluator.run_evaluation.return_value = "job-123"
    mock_evaluator.get_job_uri.return_value = "http://test-uri"
    mock_evaluator.wait_for_evaluation.side_effect = Exception(
        "Timeout waiting for tool calling evaluation"
    )

    # Create a NIMEvaluation instance with a fixed ID
    with patch("src.api.models.ObjectId", return_value=eval_id):
        run_icl_eval(previous_result)

        # Verify error handling
        mock_db.update_evaluation.assert_called_with(
            ANY,
            {
                "error": "Error running icl-eval evaluation: Timeout waiting for tool calling evaluation",
                "finished_at": ANY,
                "progress": 0.0,
            },
        )


def test_start_customization(mock_db, mock_init_db, valid_nim_config):
    """Test starting customization process."""
    nim_id = ObjectId()
    customization_id = ObjectId()

    eval_result = EvaluationResult(
        job_id="test-job",
        scores={"accuracy": 0.95},
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        status="completed",
    )

    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={DatasetType.TRAIN: "test-train-dataset"},
        evaluations={EvalType.BASE: eval_result},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_customization.return_value = customization_id

    # Mock the settings
    with patch("src.tasks.tasks.settings") as mock_settings:
        mock_settings.nmp_config.nmp_namespace = "test-namespace"
        mock_settings.training_config = ANY  # Allow any training config

        # Mock the Customizer
        with patch("src.tasks.tasks.Customizer") as mock_customizer_class:
            mock_customizer = mock_customizer_class.return_value
            mock_customizer.start_training_job.return_value = ("job-123", "customized-test-model")
            mock_customizer.get_job_uri.return_value = "http://test-uri"

            start_customization(previous_result)

            # Verify Customizer calls
            mock_customizer.start_training_job.assert_called_once_with(
                namespace="test-namespace",
                name="customization-test-workload-test-model",
                base_model=valid_nim_config.model_name,
                output_model_name="customized-test-model",
                dataset_name="test-train-dataset",
                training_config=ANY,
            )
            mock_customizer.wait_for_customization.assert_called_once()
            mock_customizer.wait_for_model_sync.assert_called_once_with("customized-test-model")

            # Verify DB-helper interactions
            mock_db.find_nim_run.assert_called_once()
            mock_db.insert_customization.assert_called_once()

            # Verify customization document updates
            mock_db.update_customization.assert_any_call(ANY, {"nmp_uri": "http://test-uri"})
            mock_db.update_customization.assert_any_call(
                ANY, {"customized_model": "customized-test-model"}
            )
            mock_db.update_customization.assert_any_call(
                ANY,
                {
                    "finished_at": ANY,
                    "runtime_seconds": ANY,
                    "progress": 100.0,
                },
            )


def test_start_customization_failure(mock_db, mock_init_db, valid_nim_config):
    """Test starting customization process when it fails."""
    nim_id = ObjectId()
    customization_id = ObjectId()

    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={DatasetType.TRAIN: "test-train-dataset"},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_customization.return_value = customization_id

    # Mock the Customizer to fail
    with patch("src.tasks.tasks.Customizer") as mock_customizer_class:
        mock_customizer = mock_customizer_class.return_value
        mock_customizer.start_training_job.side_effect = Exception("Training job failed")

        start_customization(previous_result)

        # Verify error handling
        mock_db.update_customization.assert_called_with(
            ANY,
            {
                "error": "Error starting customization: Training job failed",
                "finished_at": ANY,
                "progress": 0.0,
            },
        )


def test_run_customization_eval(
    mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings
):
    """Test running customization evaluation."""
    nim_id = ObjectId()
    customization = CustomizationResult(
        job_id="test-job",
        model_name="test-model-custom",  # This is the customized model name
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        percent_done=100.0,
        epochs_completed=1,
        steps_completed=100,
    )

    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={
            DatasetType.BASE: "test-base-dataset"  # Need base dataset for evaluation
        },
        evaluations={},
        customization=customization,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = ObjectId()
    mock_db.find_customization.return_value = {
        "workload_id": "test-workload",
        "customized_model": "test-model-custom",
    }

    # Configure mock evaluator
    mock_evaluator.run_evaluation.return_value = "job-123"
    mock_evaluator.get_job_uri.return_value = "http://test-uri"
    mock_evaluator.get_evaluation_results.return_value = {
        "tasks": {
            "llm-as-judge": {"metrics": {"llm-judge": {"scores": {"similarity": {"value": 0.95}}}}}
        }
    }

    run_customization_eval(previous_result)

    # Verify evaluator calls
    mock_evaluator.run_evaluation.assert_called_once_with(
        namespace="test-namespace",
        dataset_name="test-base-dataset",
        workload_type=WorkloadClassification.GENERIC,
        target_model="test-model-custom",  # Should use the customized model
        test_file="eval_data.jsonl",
        tool_eval_type=None,
        limit=100,
    )
    mock_evaluator.wait_for_evaluation.assert_called_once()
    mock_evaluator.get_evaluation_results.assert_called_once_with("job-123")

    # Verify DB-helper interactions
    mock_db.find_nim_run.assert_called_once()
    mock_db.insert_evaluation.assert_called_once()
    assert mock_db.update_evaluation.call_count >= 2  # progress + final


def test_run_customization_eval_failure(
    mock_evaluator, mock_db, mock_init_db, valid_nim_config, mock_settings
):
    """Test running customization evaluation when it fails."""
    nim_id = ObjectId()
    eval_id = ObjectId()
    customization = CustomizationResult(
        job_id="test-job",
        model_name="test-model-custom",
        started_at=datetime.utcnow(),
        finished_at=datetime.utcnow(),
        percent_done=100.0,
        epochs_completed=1,
        steps_completed=100,
    )

    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={DatasetType.BASE: "test-base-dataset"},
        evaluations={},
        customization=customization,
        llm_judge_config=None,
    )

    # Configure DB-helper
    mock_db.find_nim_run.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.insert_evaluation.return_value = eval_id
    mock_db.find_customization.return_value = {
        "workload_id": "test-workload",
        "customized_model": "test-model-custom",
    }

    # Configure mock evaluator to fail
    mock_evaluator.run_evaluation.side_effect = Exception("Customization evaluation failed")

    run_customization_eval(previous_result)

    # Verify error handling
    mock_db.update_evaluation.assert_called_with(
        ANY,
        {
            "error": "Error running customized-eval evaluation: Customization evaluation failed",
            "finished_at": ANY,
            "progress": 0.0,
        },
    )


def test_shutdown_deployment(mock_db, mock_init_db, mock_dms_client, valid_nim_config):
    """Test shutting down NIM deployment."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    shutdown_deployment(previous_result)

    # Verify DMS client shutdown was called
    mock_dms_client.shutdown_deployment.assert_called_once()


def test_shutdown_deployment_with_group_results(
    mock_db, mock_init_db, mock_dms_client, valid_nim_config
):
    """Test shutting down NIM deployment with group results."""
    previous_results = [
        TaskResult(
            status="success",
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={},
            evaluations={},
            customization=None,
            llm_judge_config=None,
        ),
        TaskResult(
            status="success",
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=None,  # Second result without NIM config
            workload_type=WorkloadClassification.GENERIC,
            datasets={},
            evaluations={},
            customization=None,
            llm_judge_config=None,
        ),
    ]

    shutdown_deployment(previous_results)

    # Verify DMS client shutdown was called with the first result's NIM config
    mock_dms_client.shutdown_deployment.assert_called_once()


def test_shutdown_deployment_failure(mock_db, mock_init_db, mock_dms_client, valid_nim_config):
    """Test shutting down NIM deployment when it fails."""
    nim_id = ObjectId()
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        nim=valid_nim_config,
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure TaskDBManager behaviour
    mock_db.create_nim_run.return_value = ObjectId()
    mock_db.find_nim_run.return_value = {
        "_id": nim_id,
        "model_name": valid_nim_config.model_name,
    }

    # Configure shutdown to fail
    mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

    shutdown_deployment(previous_result)

    # Verify error handling in database
    mock_db.mark_nim_error.assert_called_with(
        nim_id,
        "Error shutting down NIM deployment: Shutdown failed",
    )
