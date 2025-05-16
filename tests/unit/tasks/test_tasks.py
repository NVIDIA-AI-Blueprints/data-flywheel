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
from src.api.schemas import DeploymentStatus
from src.tasks.tasks import (
    create_datasets,
    run_base_eval,
    run_customization_eval,
    run_icl_eval,
    shutdown_deployment,
    shutdown_llm_judge_deployment,
    spin_up_llm_judge,
    spin_up_nim,
    start_customization,
)


@pytest.fixture
def mock_db():
    """Fixture to mock database operations."""
    with patch("src.tasks.tasks.get_db") as mock:
        db_instance = MagicMock()
        mock.return_value = db_instance
        yield db_instance


@pytest.fixture
def mock_init_db():
    """Fixture to mock database initialization."""
    with patch("src.tasks.tasks.init_db") as mock:
        yield mock


@pytest.fixture
def mock_data_uploader():
    """Fixture to mock DataUploader."""
    with patch("src.tasks.tasks.DataUploader") as mock:
        mock_instance = MagicMock()
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
    with patch("src.tasks.tasks.get_es_client") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_settings():
    """Fixture to mock settings."""
    with patch("src.tasks.tasks.settings") as mock:
        mock.data_split_config.min_total_records = 1
        mock.data_split_config.random_seed = 42
        mock.data_split_config.eval_size = 1
        mock.data_split_config.val_ratio = 0.25
        mock.data_split_config.limit = 100
        mock.nmp_config.nmp_namespace = "test-namespace"
        yield mock


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

    # Configure mock database behavior
    mock_db.nims.insert_one.return_value.inserted_id = ObjectId()

    spin_up_nim(previous_result, valid_nim_config.model_dump())

    # Verify DMS client method calls
    mock_dms_client.is_deployed.assert_called_once()
    mock_dms_client.deploy_model.assert_called_once()
    mock_dms_client.wait_for_deployment.assert_called_once()
    mock_dms_client.wait_for_model_sync.assert_called_once()

    # Verify database operations
    mock_db.nims.insert_one.assert_called_once()
    assert mock_db.nims.update_one.call_count >= 2  # Called for status updates

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

    # Configure mock database behavior
    mock_db.nims.insert_one.return_value.inserted_id = ObjectId()

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

    # Verify database error status update
    mock_db.nims.insert_one.assert_called_once()
    assert mock_db.nims.update_one.call_count >= 2  # Initial status and error update


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

    # Configure mock database behavior
    mock_db.nims.insert_one.return_value.inserted_id = ObjectId()

    # Configure DMS client to indicate NIM is already deployed
    mock_dms_client.is_deployed.return_value = True

    spin_up_nim(previous_result, valid_nim_config.model_dump())

    # Verify DMS client method calls
    mock_dms_client.is_deployed.assert_called_once()
    mock_dms_client.deploy_model.assert_not_called()  # Should not try to deploy again
    mock_dms_client.wait_for_deployment.assert_called_once()
    mock_dms_client.wait_for_model_sync.assert_called_once()

    # Verify database operations
    mock_db.nims.insert_one.assert_called_once()
    assert mock_db.nims.update_one.call_count >= 2  # Status updates


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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = ObjectId()

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

    # Verify database operations
    mock_db.nims.find_one.assert_called_once()
    mock_db.evaluations.insert_one.assert_called_once()
    assert mock_db.evaluations.update_one.call_count >= 2  # Progress updates and final results


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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = ObjectId()

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

    # Verify database operations
    mock_db.nims.find_one.assert_called_once()
    mock_db.evaluations.insert_one.assert_called_once()
    assert mock_db.evaluations.update_one.call_count >= 2  # Progress updates and final results


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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = eval_id

    # Configure mock evaluator to fail
    mock_evaluator.run_evaluation.side_effect = Exception("Evaluation failed")

    run_base_eval(previous_result)

    # Verify error handling
    mock_db.evaluations.update_one.assert_called_with(
        {"_id": ANY},
        {
            "$set": {
                "error": "Error running base-eval evaluation: Evaluation failed",
                "finished_at": ANY,
                "progress": 0.0,
            }
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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = eval_id

    # Configure mock evaluator to fail during results retrieval
    mock_evaluator.run_evaluation.return_value = "job-123"
    mock_evaluator.get_job_uri.return_value = "http://test-uri"
    mock_evaluator.wait_for_evaluation.side_effect = Exception("Timeout waiting for evaluation")

    run_base_eval(previous_result)

    # Verify error handling
    mock_db.evaluations.update_one.assert_called_with(
        {"_id": ANY},
        {
            "$set": {
                "error": "Error running base-eval evaluation: Timeout waiting for evaluation",
                "finished_at": ANY,
                "progress": 0.0,
            }
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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = eval_id

    # Configure mock evaluator to fail
    mock_evaluator.run_evaluation.side_effect = Exception("Tool calling evaluation failed")

    run_icl_eval(previous_result)

    # Verify error handling
    mock_db.evaluations.update_one.assert_called_with(
        {"_id": ANY},
        {
            "$set": {
                "error": "Error running icl-eval evaluation: Tool calling evaluation failed",
                "finished_at": ANY,
                "progress": 0.0,
            }
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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = eval_id

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
        mock_db.evaluations.update_one.assert_called_with(
            {"_id": ANY},
            {
                "$set": {
                    "error": "Error running icl-eval evaluation: Timeout waiting for tool calling evaluation",
                    "finished_at": ANY,
                    "progress": 0.0,
                }
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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.customizations.insert_one.return_value.inserted_id = customization_id

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

            # Verify database operations
            mock_db.nims.find_one.assert_called_once()
            mock_db.customizations.insert_one.assert_called_once()

            # Verify customization document updates
            mock_db.customizations.update_one.assert_any_call(
                {"_id": ANY}, {"$set": {"nmp_uri": "http://test-uri"}}
            )
            mock_db.customizations.update_one.assert_any_call(
                {"_id": ANY}, {"$set": {"customized_model": "customized-test-model"}}
            )
            mock_db.customizations.update_one.assert_any_call(
                {"_id": ANY},
                {
                    "$set": {
                        "finished_at": ANY,
                        "runtime_seconds": ANY,
                        "progress": 100.0,
                    }
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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.customizations.insert_one.return_value.inserted_id = customization_id

    # Mock the Customizer to fail
    with patch("src.tasks.tasks.Customizer") as mock_customizer_class:
        mock_customizer = mock_customizer_class.return_value
        mock_customizer.start_training_job.side_effect = Exception("Training job failed")

        start_customization(previous_result)

        # Verify error handling
        mock_db.customizations.update_one.assert_called_with(
            {"_id": ANY},
            {
                "$set": {
                    "error": "Error starting customization: Training job failed",
                    "finished_at": ANY,
                    "progress": 0.0,
                }
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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = ObjectId()
    mock_db.customizations.find_one.return_value = {
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

    # Verify database operations
    mock_db.nims.find_one.assert_called_once()
    mock_db.evaluations.insert_one.assert_called_once()
    assert mock_db.evaluations.update_one.call_count >= 2  # Progress updates and final results


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

    # Configure mock database
    mock_db.nims.find_one.return_value = {"_id": nim_id, "model_name": valid_nim_config.model_name}
    mock_db.evaluations.insert_one.return_value.inserted_id = eval_id
    mock_db.customizations.find_one.return_value = {
        "workload_id": "test-workload",
        "customized_model": "test-model-custom",
    }

    # Configure mock evaluator to fail
    mock_evaluator.run_evaluation.side_effect = Exception("Customization evaluation failed")

    run_customization_eval(previous_result)

    # Verify error handling
    mock_db.evaluations.update_one.assert_called_with(
        {"_id": ANY},
        {
            "$set": {
                "error": "Error running customized-eval evaluation: Customization evaluation failed",
                "finished_at": ANY,
                "progress": 0.0,
            }
        },
    )


def test_spin_up_llm_judge(mock_db, mock_init_db, mock_dms_client, mock_settings):
    """Test spinning up LLM Judge instance."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure mock settings
    mock_settings.llm_judge_config.is_remote.return_value = False
    mock_settings.llm_judge_config.get_local_nim_config.return_value = NIMConfig(
        model_name="test-judge-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
    )

    # Configure mock database behavior
    mock_db.llm_judge_runs.insert_one.return_value.inserted_id = ObjectId()

    spin_up_llm_judge(previous_result)

    # Verify DMS client method calls
    mock_dms_client.is_deployed.assert_called_once()
    mock_dms_client.deploy_model.assert_called_once()
    mock_dms_client.wait_for_deployment.assert_called_once()
    mock_dms_client.wait_for_model_sync.assert_called_once()

    # Verify database operations
    mock_db.llm_judge_runs.insert_one.assert_called_once()


def test_spin_up_llm_judge_remote(mock_db, mock_init_db, mock_dms_client, mock_settings):
    """Test spinning up LLM Judge instance when using remote judge."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    spin_up_llm_judge(previous_result)

    # Verify no DMS client calls were made
    mock_dms_client.is_deployed.assert_not_called()
    mock_dms_client.deploy_model.assert_not_called()
    mock_dms_client.wait_for_deployment.assert_not_called()
    mock_dms_client.wait_for_model_sync.assert_not_called()


def test_spin_up_llm_judge_failure(mock_db, mock_init_db, mock_dms_client, mock_settings):
    """Test spinning up LLM Judge instance when deployment fails."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,
    )

    # Configure mock settings
    mock_settings.llm_judge_config.is_remote.return_value = False
    mock_settings.llm_judge_config.get_local_nim_config.return_value = NIMConfig(
        model_name="test-judge-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
    )

    # Configure mock database behavior
    mock_db.llm_judge_runs.insert_one.return_value.inserted_id = ObjectId()

    # Configure DMS client to fail deployment
    mock_dms_client.deploy_model.side_effect = Exception("Deployment failed")

    # The function should raise the deployment exception
    with pytest.raises(Exception) as exc_info:
        spin_up_llm_judge(previous_result)

    assert "Deployment failed" in str(exc_info.value)

    # Verify error handling in database
    mock_db.llm_judge_runs.update_one.assert_called_with(
        {"_id": ANY},
        {"$set": {"error": ANY, "deployment_status": DeploymentStatus.FAILED}},
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

    # Configure mock database with all required fields
    mock_db.nims.find_one.return_value = {
        "_id": nim_id,
        "model_name": valid_nim_config.model_name,
        "flywheel_run_id": ObjectId(),  # Add required field
        "started_at": datetime.utcnow(),  # Add required field
        "finished_at": datetime.utcnow(),  # Add required field
        "runtime_seconds": 0,  # Add required field
        "status": NIMRunStatus.RUNNING,  # Add required field
    }

    # Configure shutdown to fail
    mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

    shutdown_deployment(previous_result)

    # Verify error handling in database
    mock_db.nims.update_one.assert_called_with(
        {"_id": nim_id},
        {
            "$set": {
                "error": "Error shutting down NIM deployment: Shutdown failed",
                "status": NIMRunStatus.ERROR,
                "deployment_status": DeploymentStatus.FAILED,
            }
        },
    )


def test_shutdown_llm_judge_deployment(mock_db, mock_init_db, mock_dms_client, valid_nim_config):
    """Test shutting down LLM Judge deployment."""
    llm_judge_config = NIMConfig(
        model_name="test-judge-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
    )
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
        llm_judge_config=llm_judge_config,
    )

    shutdown_llm_judge_deployment(previous_result)

    # Verify DMS client shutdown was called
    mock_dms_client.shutdown_deployment.assert_called_once()

    # Verify flywheel run was updated
    mock_db.flywheel_runs.update_one.assert_called_with(
        {"_id": ObjectId(previous_result.flywheel_run_id)},
        {"$set": {"finished_at": ANY}},
    )


def test_shutdown_llm_judge_deployment_remote(mock_db, mock_init_db, mock_dms_client):
    """Test shutting down LLM Judge deployment when using remote judge."""
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=str(ObjectId()),
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=None,  # No local judge config
    )

    shutdown_llm_judge_deployment(previous_result)

    # Verify DMS client shutdown was not called
    mock_dms_client.shutdown_deployment.assert_not_called()

    # Verify flywheel run was still updated
    mock_db.flywheel_runs.update_one.assert_called_with(
        {"_id": ObjectId(previous_result.flywheel_run_id)},
        {"$set": {"finished_at": ANY}},
    )


def test_shutdown_llm_judge_deployment_failure(
    mock_db, mock_init_db, mock_dms_client, mock_settings
):
    """Test shutting down LLM Judge deployment when it fails."""
    llm_judge_run_id = ObjectId()
    flywheel_run_id = str(ObjectId())
    llm_judge_config = NIMConfig(
        model_name="test-judge-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
    )
    previous_result = TaskResult(
        status="success",
        workload_id="test-workload",
        client_id="test-client",
        flywheel_run_id=flywheel_run_id,
        workload_type=WorkloadClassification.GENERIC,
        datasets={},
        evaluations={},
        customization=None,
        llm_judge_config=llm_judge_config,
    )

    # Configure mock settings
    mock_settings.llm_judge_config.is_remote.return_value = False
    mock_settings.nmp_config = MagicMock()

    # Configure mock database with all required fields for LLMJudgeRun
    mock_db.llm_judge_runs.find_one.return_value = {
        "_id": llm_judge_run_id,
        "model_name": llm_judge_config.model_name,
        "flywheel_run_id": ObjectId(flywheel_run_id),
        "deployment_status": DeploymentStatus.RUNNING,
        # Add all required fields for LLMJudgeRun
        "started_at": datetime.utcnow(),
        "finished_at": None,
        "error": None,
    }

    # Configure shutdown to fail
    mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

    shutdown_llm_judge_deployment(previous_result)

    # Verify that find_one was called with correct parameters
    mock_db.llm_judge_runs.find_one.assert_called_with(
        {"flywheel_run_id": ObjectId(flywheel_run_id)}
    )

    # Verify error handling in database
    mock_db.llm_judge_runs.update_one.assert_called_with(
        {"_id": llm_judge_run_id},
        {
            "$set": {
                "error": "Error shutting down LLM Judge deployment: Shutdown failed",
                "deployment_status": DeploymentStatus.FAILED,
            }
        },
    )

    # Verify that flywheel run was updated
    mock_db.flywheel_runs.update_one.assert_called_with(
        {"_id": ObjectId(flywheel_run_id)},
        {"$set": {"finished_at": ANY}},
    )
