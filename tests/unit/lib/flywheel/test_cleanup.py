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
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import FlywheelRun, FlywheelRunStatus, NIMRunStatus
from src.lib.flywheel.cleanup_manager import CleanupManager
from src.lib.flywheel.job_manager import FlywheelJobManager


@pytest.fixture
def mock_db_manager():
    """Fixture to create a mock database manager."""
    mock = MagicMock()
    return mock


@pytest.fixture
def cleanup_manager(mock_db_manager):
    """Fixture to create a FlywheelJobCleanup instance with mocked dependencies."""
    with (
        patch("src.lib.nemo.customizer.Customizer") as mock_customizer_class,
        patch("src.lib.nemo.evaluator.Evaluator") as mock_evaluator_class,
    ):
        # Create and configure mock instances
        mock_customizer = MagicMock()
        mock_evaluator = MagicMock()

        # Set up the mock class to return our configured instances
        mock_customizer_class.return_value = mock_customizer
        mock_evaluator_class.return_value = mock_evaluator

        # Create the cleanup manager
        manager = FlywheelJobManager(mock_db_manager)

        # Replace the auto-created instances with our mocks
        manager.customizer = mock_customizer
        manager.evaluator = mock_evaluator

        return manager


@pytest.fixture
def cleanup_manager_instance(mock_db_manager):
    """Fixture to create a CleanupManager instance with mocked dependencies."""
    # Import the module to make sure it's loaded
    from src.lib.flywheel import cleanup_manager

    # Configure mock settings
    mock_nim_config = MagicMock()
    mock_nim_config.model_name = "test-model"

    mock_settings = MagicMock()
    mock_settings.nims = [mock_nim_config]
    mock_settings.nmp_config = MagicMock()
    mock_settings.llm_judge_config = MagicMock()
    mock_settings.llm_judge_config.is_remote = False
    mock_settings.llm_judge_config.model_name = "test-judge-model"

    # Create mock DMS client class
    mock_dms_client_class = MagicMock()

    # Create mock customizer class
    mock_customizer_class = MagicMock()
    mock_customizer = MagicMock()
    mock_customizer_class.return_value = mock_customizer

    with (
        patch.object(cleanup_manager, "settings", mock_settings),
        patch.object(cleanup_manager, "DMSClient", mock_dms_client_class),
        patch.object(cleanup_manager, "Customizer", mock_customizer_class),
    ):
        # Create the cleanup manager instance
        manager = CleanupManager(mock_db_manager)

        # Store references to mocks for test verification
        manager._mock_settings = mock_settings
        manager._mock_dms_client_class = mock_dms_client_class
        manager._mock_nim_config = mock_nim_config

        yield manager


def test_delete_job_success(cleanup_manager, mock_db_manager):
    """Test successful deletion of all job resources."""
    job_id = str(ObjectId())
    nim_id = ObjectId()

    # Mock flywheel run with datasets
    flywheel_run = FlywheelRun(
        workload_id="test-workload",
        client_id="test-client",
        started_at=datetime.utcnow(),
        datasets=[
            {"name": "test_dataset_1", "num_records": 100, "nmp_uri": "test_uri_1"},
            {"name": "test_dataset_2", "num_records": 100, "nmp_uri": "test_uri_2"},
        ],
    )

    # Configure mock DB responses
    mock_db_manager.get_flywheel_run.return_value = flywheel_run.to_mongo()
    mock_db_manager.find_nims_for_job.return_value = [{"_id": nim_id, "model_name": "test_model"}]
    mock_db_manager.find_customizations_for_nim.return_value = [
        {"customized_model": "test_model_custom_1"},
        {"customized_model": "test_model_custom_2"},
    ]
    mock_db_manager.find_evaluations_for_nim.return_value = [
        {"job_id": "eval_job_1"},
        {"job_id": "eval_job_2"},
    ]

    # Patch DataUploader at the module level where it's used
    with patch("src.lib.flywheel.job_manager.DataUploader") as mock_data_uploader_class:
        mock_data_uploader = MagicMock()
        mock_data_uploader_class.return_value = mock_data_uploader

        # Execute cleanup
        cleanup_manager.delete_job(job_id)

        # Verify DataUploader was called for each dataset
        assert mock_data_uploader_class.call_count == 2
        mock_data_uploader_class.assert_any_call(dataset_name="test_dataset_1")
        mock_data_uploader_class.assert_any_call(dataset_name="test_dataset_2")

        # Verify dataset deletion methods were called
        assert mock_data_uploader.delete_dataset.call_count == 2
        assert mock_data_uploader.unregister_dataset.call_count == 2

    # Verify customized models were deleted
    assert cleanup_manager.customizer.delete_customized_model.call_count == 2
    cleanup_manager.customizer.delete_customized_model.assert_any_call("test_model_custom_1")
    cleanup_manager.customizer.delete_customized_model.assert_any_call("test_model_custom_2")

    # Verify evaluation jobs were deleted
    assert cleanup_manager.evaluator.delete_evaluation_job.call_count == 2
    cleanup_manager.evaluator.delete_evaluation_job.assert_any_call("eval_job_1")
    cleanup_manager.evaluator.delete_evaluation_job.assert_any_call("eval_job_2")

    # Verify MongoDB cleanup
    mock_db_manager.delete_job_records.assert_called_once_with(ObjectId(job_id))


def test_delete_job_partial_failure(cleanup_manager, mock_db_manager):
    """Test deletion with some resources failing but overall task succeeding."""
    job_id = str(ObjectId())
    nim_id = ObjectId()

    # Mock flywheel run with datasets
    flywheel_run = FlywheelRun(
        workload_id="test-workload",
        client_id="test-client",
        started_at=datetime.utcnow(),
        datasets=[
            {"name": "test_dataset_1", "num_records": 100, "nmp_uri": "test_uri_1"},
        ],
    )

    # Configure mock DB responses
    mock_db_manager.get_flywheel_run.return_value = flywheel_run.to_mongo()
    mock_db_manager.find_nims_for_job.return_value = [{"_id": nim_id, "model_name": "test_model"}]
    mock_db_manager.find_customizations_for_nim.return_value = [
        {"customized_model": "test_model_custom_1"},
    ]
    mock_db_manager.find_evaluations_for_nim.return_value = [
        {"job_id": "eval_job_1"},
    ]

    # Configure mock instance to fail
    cleanup_manager.customizer.delete_customized_model.side_effect = Exception(
        "Failed to delete model"
    )

    # Patch DataUploader at the module level where it's used
    with patch("src.lib.flywheel.job_manager.DataUploader") as mock_data_uploader_class:
        mock_data_uploader = MagicMock()
        mock_data_uploader_class.return_value = mock_data_uploader

        # Execute cleanup
        cleanup_manager.delete_job(job_id)

        # Verify DataUploader was called for the dataset
        mock_data_uploader_class.assert_called_once_with(dataset_name="test_dataset_1")
        mock_data_uploader.delete_dataset.assert_called_once()
        mock_data_uploader.unregister_dataset.assert_called_once()

    # Verify the task continued despite the model deletion failure
    cleanup_manager.customizer.delete_customized_model.assert_called_once()
    cleanup_manager.evaluator.delete_evaluation_job.assert_called_once()

    # Verify MongoDB cleanup still happened
    mock_db_manager.delete_job_records.assert_called_once_with(ObjectId(job_id))


def test_delete_job_complete_failure(cleanup_manager, mock_db_manager):
    """Test complete failure of job deletion."""
    job_id = str(ObjectId())

    # Mock database to raise an exception
    mock_db_manager.get_flywheel_run.side_effect = Exception("Database connection failed")

    # Execute cleanup and verify it raises the exception
    with pytest.raises(Exception) as exc_info:
        cleanup_manager.delete_job(job_id)

    assert "Database connection failed" in str(exc_info.value)

    # Verify no further operations were attempted
    mock_db_manager.find_nims_for_job.assert_not_called()
    mock_db_manager.delete_job_records.assert_not_called()


def test_delete_job_invalid_job_id(cleanup_manager):
    """Test deletion with invalid job ID."""
    invalid_job_id = "invalid-id"

    # Execute cleanup and verify it raises an exception
    with pytest.raises(Exception) as exc_info:
        cleanup_manager.delete_job(invalid_job_id)

    # Verify the error message
    assert "ObjectId" in str(exc_info.value), "Should raise error about invalid ObjectId format"

    # Verify no database operations were attempted
    cleanup_manager.db_manager.get_flywheel_run.assert_not_called()
    cleanup_manager.db_manager.find_nims_for_job.assert_not_called()
    cleanup_manager.db_manager.delete_job_records.assert_not_called()


# New tests for CleanupManager


class TestCleanupManager:
    """Test cases for the CleanupManager class."""

    def test_find_running_flywheel_runs_success(self, cleanup_manager_instance, mock_db_manager):
        """Test successful retrieval of running flywheel runs."""
        # Mock database response with running flywheel runs
        expected_runs = [
            {"_id": ObjectId(), "status": FlywheelRunStatus.RUNNING.value, "workload_id": "test1"},
            {"_id": ObjectId(), "status": FlywheelRunStatus.PENDING.value, "workload_id": "test2"},
        ]
        mock_db_manager.find_running_flywheel_runs.return_value = expected_runs

        # Execute the method
        result = cleanup_manager_instance.find_running_flywheel_runs()

        # Verify results
        assert result == expected_runs
        assert len(result) == 2
        mock_db_manager.find_running_flywheel_runs.assert_called_once()

    def test_find_running_flywheel_runs_empty(self, cleanup_manager_instance, mock_db_manager):
        """Test finding running flywheel runs when none exist."""
        # Mock empty database response
        mock_db_manager.find_running_flywheel_runs.return_value = []

        # Execute the method
        result = cleanup_manager_instance.find_running_flywheel_runs()

        # Verify results
        assert result == []
        mock_db_manager.find_running_flywheel_runs.assert_called_once()

    def test_find_running_nims_success(self, cleanup_manager_instance, mock_db_manager):
        """Test successful retrieval of running NIMs for a flywheel run."""
        flywheel_run_id = ObjectId()
        expected_nims = [
            {"_id": ObjectId(), "status": NIMRunStatus.RUNNING.value, "model_name": "test-model-1"},
            {"_id": ObjectId(), "status": NIMRunStatus.PENDING.value, "model_name": "test-model-2"},
        ]
        mock_db_manager.find_running_nims_for_flywheel.return_value = expected_nims

        # Execute the method
        result = cleanup_manager_instance.find_running_nims(flywheel_run_id)

        # Verify results
        assert result == expected_nims
        assert len(result) == 2
        mock_db_manager.find_running_nims_for_flywheel.assert_called_once_with(flywheel_run_id)

    def test_find_customization_jobs_success(self, cleanup_manager_instance, mock_db_manager):
        """Test successful retrieval of customization jobs for a NIM."""
        nim_id = ObjectId()
        expected_customizations = [
            {"_id": ObjectId(), "job_id": "custom-job-1", "nim_id": nim_id},
            {"_id": ObjectId(), "job_id": "custom-job-2", "nim_id": nim_id},
        ]
        mock_db_manager.find_customizations_for_nim.return_value = expected_customizations

        # Execute the method
        result = cleanup_manager_instance.find_customization_jobs(nim_id)

        # Verify results
        assert result == expected_customizations
        assert len(result) == 2
        mock_db_manager.find_customizations_for_nim.assert_called_once_with(nim_id)

    def test_find_evaluation_jobs_success(self, cleanup_manager_instance, mock_db_manager):
        """Test successful retrieval of evaluation jobs for a NIM."""
        nim_id = ObjectId()
        expected_evaluations = [
            {"_id": ObjectId(), "job_id": "eval-job-1", "nim_id": nim_id},
            {"_id": ObjectId(), "job_id": "eval-job-2", "nim_id": nim_id},
        ]
        mock_db_manager.find_evaluations_for_nim.return_value = expected_evaluations

        # Execute the method
        result = cleanup_manager_instance.find_evaluation_jobs(nim_id)

        # Verify results
        assert result == expected_evaluations
        assert len(result) == 2
        mock_db_manager.find_evaluations_for_nim.assert_called_once_with(nim_id)

    def test_cancel_customization_jobs_success(self, cleanup_manager_instance):
        """Test successful cancellation of customization jobs."""
        customizations = [
            {"job_id": "custom-job-1"},
            {"job_id": "custom-job-2"},
        ]

        # Execute the method
        cleanup_manager_instance.cancel_customization_jobs(customizations)

        # Verify that cancel_job was called for each customization
        assert cleanup_manager_instance.customizer.cancel_job.call_count == 2
        cleanup_manager_instance.customizer.cancel_job.assert_any_call("custom-job-1")
        cleanup_manager_instance.customizer.cancel_job.assert_any_call("custom-job-2")

        # Verify no errors were recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cancel_customization_jobs_empty_list(self, cleanup_manager_instance):
        """Test cancellation with empty customization list."""
        # Execute the method with empty list
        cleanup_manager_instance.cancel_customization_jobs([])

        # Verify no calls were made
        cleanup_manager_instance.customizer.cancel_job.assert_not_called()
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cancel_customization_jobs_with_failures(self, cleanup_manager_instance):
        """Test cancellation of customization jobs with some failures."""
        customizations = [
            {"job_id": "custom-job-1"},
            {"job_id": "custom-job-2"},
            {"job_id": "custom-job-3"},
        ]

        # Configure one job to fail
        cleanup_manager_instance.customizer.cancel_job.side_effect = [
            None,  # First call succeeds
            Exception("Failed to cancel job"),  # Second call fails
            None,  # Third call succeeds
        ]

        # Execute the method
        cleanup_manager_instance.cancel_customization_jobs(customizations)

        # Verify all jobs were attempted
        assert cleanup_manager_instance.customizer.cancel_job.call_count == 3

        # Verify one error was recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert (
            "Failed to cancel customization job custom-job-2"
            in cleanup_manager_instance.cleanup_errors[0]
        )

    def test_shutdown_nim_success(self, cleanup_manager_instance):
        """Test successful NIM shutdown."""
        nim = {"model_name": "test-model", "_id": ObjectId()}

        # Mock DMS client
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.shutdown_nim(nim)

        # Verify DMS client was created and shutdown was called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once_with(
            nmp_config=cleanup_manager_instance._mock_settings.nmp_config,
            nim=cleanup_manager_instance._mock_nim_config,
        )
        mock_dms_client.shutdown_deployment.assert_called_once()
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_shutdown_nim_config_not_found(self, cleanup_manager_instance):
        """Test NIM shutdown when config is not found."""
        nim = {"model_name": "unknown-model", "_id": ObjectId()}

        # Mock empty NIM configs
        cleanup_manager_instance._mock_settings.nims = []

        # Execute the method
        cleanup_manager_instance.shutdown_nim(nim)

        # Verify no DMS client was created
        cleanup_manager_instance._mock_dms_client_class.assert_not_called()

    def test_shutdown_nim_failure(self, cleanup_manager_instance):
        """Test NIM shutdown with failure."""
        nim = {"model_name": "test-model", "_id": ObjectId()}

        # Mock DMS client that fails
        mock_dms_client = MagicMock()
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.shutdown_nim(nim)

        # Verify error was recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert "Failed to shutdown NIM test-model" in cleanup_manager_instance.cleanup_errors[0]

    def test_shutdown_llm_judge_local(self, cleanup_manager_instance):
        """Test shutdown of local LLM judge."""
        # Configure local LLM judge
        cleanup_manager_instance._mock_settings.llm_judge_config.is_remote = False

        # Mock DMS client
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.shutdown_llm_judge()

        # Verify DMS client was created and shutdown was called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once_with(
            nmp_config=cleanup_manager_instance._mock_settings.nmp_config,
            nim=cleanup_manager_instance._mock_settings.llm_judge_config,
        )
        mock_dms_client.shutdown_deployment.assert_called_once()
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_shutdown_llm_judge_remote(self, cleanup_manager_instance):
        """Test shutdown of remote LLM judge (should skip)."""
        # Configure remote LLM judge
        cleanup_manager_instance._mock_settings.llm_judge_config.is_remote = True

        # Execute the method
        cleanup_manager_instance.shutdown_llm_judge()

        # Verify no DMS client was created
        cleanup_manager_instance._mock_dms_client_class.assert_not_called()
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_shutdown_llm_judge_failure(self, cleanup_manager_instance):
        """Test LLM judge shutdown with failure."""
        # Configure local LLM judge
        cleanup_manager_instance._mock_settings.llm_judge_config.is_remote = False

        # Mock DMS client that fails
        mock_dms_client = MagicMock()
        mock_dms_client.shutdown_deployment.side_effect = Exception("Judge shutdown failed")
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.shutdown_llm_judge()

        # Verify error was recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert "Failed to shutdown LLM judge" in cleanup_manager_instance.cleanup_errors[0]

    def test_mark_resources_as_cancelled_success(self, cleanup_manager_instance, mock_db_manager):
        """Test successful marking of resources as cancelled."""
        flywheel_run_id = ObjectId()

        # Mock NIMs for the flywheel run
        mock_nims = [
            {"_id": ObjectId(), "model_name": "test-model-1"},
            {"_id": ObjectId(), "model_name": "test-model-2"},
        ]
        mock_db_manager.find_nims_for_job.return_value = mock_nims

        # Execute the method
        cleanup_manager_instance.mark_resources_as_cancelled(flywheel_run_id)

        # Verify database operations
        mock_db_manager.mark_flywheel_run_cancelled.assert_called_once_with(
            flywheel_run_id, error_msg="Cancelled by cleanup manager"
        )
        mock_db_manager.find_nims_for_job.assert_called_once_with(flywheel_run_id)

        # Verify each NIM was marked as cancelled
        assert mock_db_manager.mark_nim_cancelled.call_count == 2
        for nim in mock_nims:
            mock_db_manager.mark_nim_cancelled.assert_any_call(
                nim["_id"], error_msg="Cancelled by cleanup manager"
            )

        mock_db_manager.mark_llm_judge_cancelled.assert_called_once_with(
            flywheel_run_id, error_msg="Cancelled by cleanup manager"
        )

        # Verify no errors were recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_mark_resources_as_cancelled_failure(self, cleanup_manager_instance, mock_db_manager):
        """Test marking resources as cancelled with database failure."""
        flywheel_run_id = ObjectId()

        # Configure database to fail
        mock_db_manager.mark_flywheel_run_cancelled.side_effect = Exception("Database error")

        # Execute the method
        cleanup_manager_instance.mark_resources_as_cancelled(flywheel_run_id)

        # Verify error was recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert "Failed to mark resources as cancelled" in cleanup_manager_instance.cleanup_errors[0]
