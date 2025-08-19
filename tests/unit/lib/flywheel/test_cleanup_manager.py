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

from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.lib.flywheel.cleanup_manager import CleanupManager


@pytest.fixture
def mock_db_manager():
    """Fixture to create a mock database manager."""
    mock = MagicMock()
    return mock


@pytest.fixture
def cleanup_manager_instance(mock_db_manager):
    """Fixture to create a CleanupManager instance with mocked dependencies."""
    # Import the module to make sure it's loaded
    from src.lib.flywheel import cleanup_manager

    # Configure mock settings with multiple NIM configs
    mock_nim_config_1 = MagicMock()
    mock_nim_config_1.model_name = "test-namespace/test-model-1"
    mock_nim_config_1.customizer_configs = MagicMock()
    mock_nim_config_1.customizer_configs.target = "test-model-1@2.0"

    mock_nim_config_2 = MagicMock()
    mock_nim_config_2.model_name = "test-namespace/test-model-2"
    mock_nim_config_2.customizer_configs = MagicMock()
    mock_nim_config_2.customizer_configs.target = "test-model-2@2.0"

    mock_nim_config_3 = MagicMock()
    mock_nim_config_3.model_name = "test-namespace/test-model-3"
    mock_nim_config_3.customizer_configs = None  # No customization configs

    mock_settings = MagicMock()
    mock_settings.nims = [mock_nim_config_1, mock_nim_config_2, mock_nim_config_3]
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
        manager._mock_nim_configs = [mock_nim_config_1, mock_nim_config_2, mock_nim_config_3]

        yield manager


class TestCleanupManager:
    """Test cases for the CleanupManager class."""

    def test_init(self, mock_db_manager):
        """Test initialization of CleanupManager."""
        with patch("src.lib.flywheel.cleanup_manager.Customizer") as mock_customizer_class:
            manager = CleanupManager(mock_db_manager)

            assert manager.db_manager is mock_db_manager
            assert manager.cleanup_errors == []
            mock_customizer_class.assert_called_once()

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

    def test_cancel_customization_jobs_without_job_id(self, cleanup_manager_instance):
        """Test cancellation with customizations missing job_id field."""
        customizations = [
            {"job_id": "custom-job-1"},
            {"customization_id": "custom-2"},  # Missing job_id
            {"job_id": "custom-job-3"},
        ]

        # Execute the method
        cleanup_manager_instance.cancel_customization_jobs(customizations)

        # Verify only jobs with job_id were called
        assert cleanup_manager_instance.customizer.cancel_job.call_count == 2
        cleanup_manager_instance.customizer.cancel_job.assert_any_call("custom-job-1")
        cleanup_manager_instance.customizer.cancel_job.assert_any_call("custom-job-3")

        # Verify no errors were recorded
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
        nim = {"model_name": "test-namespace/test-model-1", "_id": ObjectId()}

        # Mock DMS client
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Find the expected NIM config
        expected_nim_config = next(
            cfg
            for cfg in cleanup_manager_instance._mock_nim_configs
            if cfg.model_name == nim["model_name"]
        )

        # Execute the method
        cleanup_manager_instance.shutdown_nim(nim)

        # Verify DMS client was created and shutdown was called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once_with(
            nmp_config=cleanup_manager_instance._mock_settings.nmp_config,
            nim=expected_nim_config,
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
        nim = {"model_name": "test-namespace/test-model-1", "_id": ObjectId()}

        # Mock DMS client that fails
        mock_dms_client = MagicMock()
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Find the expected NIM config
        expected_nim_config = next(
            cfg
            for cfg in cleanup_manager_instance._mock_nim_configs
            if cfg.model_name == nim["model_name"]
        )

        # Execute the method
        cleanup_manager_instance.shutdown_nim(nim)

        # Verify DMS client was created and shutdown was called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once_with(
            nmp_config=cleanup_manager_instance._mock_settings.nmp_config,
            nim=expected_nim_config,
        )
        mock_dms_client.shutdown_deployment.assert_called_once()
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert "Failed to shutdown NIM" in cleanup_manager_instance.cleanup_errors[0]

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
            {"_id": ObjectId(), "model_name": "test-namespace/test-model-1"},
            {"_id": ObjectId(), "model_name": "test-namespace/test-model-2"},
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

    def test_cleanup_flywheel_run_success(self, cleanup_manager_instance, mock_db_manager):
        """Test successful cleanup of a flywheel run."""
        flywheel_run_id = ObjectId()
        flywheel_run = {"_id": flywheel_run_id, "workload_id": "test-workload"}

        # Mock running NIMs
        mock_nims = [
            {"_id": ObjectId(), "model_name": "test-namespace/test-model-1"},
            {"_id": ObjectId(), "model_name": "test-namespace/test-model-2"},
        ]
        mock_db_manager.find_running_nims_for_flywheel.return_value = mock_nims

        # Mock customizations for each NIM
        mock_customizations = [
            {"job_id": "custom-job-1"},
            {"job_id": "custom-job-2"},
        ]
        mock_db_manager.find_customizations_for_nim.return_value = mock_customizations

        # Mock NIMs for marking as cancelled
        mock_db_manager.find_nims_for_job.return_value = mock_nims

        # Mock NIM configs to match the model names
        mock_nim_config_1 = MagicMock()
        mock_nim_config_1.model_name = "test-namespace/test-model-1"
        mock_nim_config_2 = MagicMock()
        mock_nim_config_2.model_name = "test-namespace/test-model-2"
        cleanup_manager_instance._mock_settings.nims = [mock_nim_config_1, mock_nim_config_2]

        # Mock DMS client
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.cleanup_flywheel_run(flywheel_run)

        # Verify database calls
        mock_db_manager.find_running_nims_for_flywheel.assert_called_once_with(flywheel_run_id)
        assert mock_db_manager.find_customizations_for_nim.call_count == 2

        # Verify customization jobs were cancelled
        assert (
            cleanup_manager_instance.customizer.cancel_job.call_count == 4
        )  # 2 customizations per NIM

        # Verify NIMs were shutdown
        assert cleanup_manager_instance._mock_dms_client_class.call_count == 2
        assert mock_dms_client.shutdown_deployment.call_count == 2

        # Verify resources were marked as cancelled
        mock_db_manager.mark_flywheel_run_cancelled.assert_called_once()
        mock_db_manager.mark_llm_judge_cancelled.assert_called_once()

        # Verify no errors
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cleanup_flywheel_run_with_errors(self, cleanup_manager_instance, mock_db_manager):
        """Test cleanup of flywheel run with errors in individual operations."""
        flywheel_run_id = ObjectId()
        flywheel_run = {"_id": flywheel_run_id, "workload_id": "test-workload"}

        # Mock running NIMs - use the same model name as in settings
        mock_nims = [
            {
                "_id": ObjectId(),
                "model_name": "test-namespace/test-model-1",
            },  # Use model name that exists in settings
        ]
        mock_db_manager.find_running_nims_for_flywheel.return_value = mock_nims

        # Mock customizations
        mock_customizations = [
            {"job_id": "custom-job-1"},
        ]
        mock_db_manager.find_customizations_for_nim.return_value = mock_customizations

        # Mock NIMs for marking as cancelled
        mock_db_manager.find_nims_for_job.return_value = mock_nims

        # Configure customizer to fail
        cleanup_manager_instance.customizer.cancel_job.side_effect = Exception("Cancel failed")

        # Configure NIM shutdown to fail
        mock_dms_client = MagicMock()
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Configure marking as cancelled to fail
        mock_db_manager.mark_flywheel_run_cancelled.side_effect = Exception("Mark failed")

        # Execute the method
        cleanup_manager_instance.shutdown_nim(mock_nims[0])  # Ensure shutdown_nim is called
        cleanup_manager_instance.cleanup_flywheel_run(flywheel_run)

        # Verify all operations were attempted despite failures
        mock_db_manager.find_running_nims_for_flywheel.assert_called_with(flywheel_run_id)
        cleanup_manager_instance.customizer.cancel_job.assert_called()
        mock_dms_client.shutdown_deployment.assert_called()

        # Verify errors were recorded
        assert (
            len(cleanup_manager_instance.cleanup_errors) >= 2
        )  # At least cancel and shutdown errors

    def test_cleanup_flywheel_run_no_running_nims(self, cleanup_manager_instance, mock_db_manager):
        """Test cleanup of flywheel run with no running NIMs."""
        flywheel_run_id = ObjectId()
        flywheel_run = {"_id": flywheel_run_id, "workload_id": "test-workload"}

        # Mock empty running NIMs
        mock_db_manager.find_running_nims_for_flywheel.return_value = []
        mock_db_manager.find_nims_for_job.return_value = []

        # Execute the method
        cleanup_manager_instance.cleanup_flywheel_run(flywheel_run)

        # Verify database calls
        mock_db_manager.find_running_nims_for_flywheel.assert_called_once_with(flywheel_run_id)

        # Verify no customization or shutdown operations
        cleanup_manager_instance.customizer.cancel_job.assert_not_called()
        cleanup_manager_instance._mock_dms_client_class.assert_not_called()

        # Verify resources were marked as cancelled
        mock_db_manager.mark_flywheel_run_cancelled.assert_called_once()

        # Verify no errors
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cleanup_all_running_resources_success(self, cleanup_manager_instance, mock_db_manager):
        """Test successful cleanup of all running resources."""
        # Mock running flywheel runs
        flywheel_run_1 = {"_id": ObjectId(), "workload_id": "test-1"}
        flywheel_run_2 = {"_id": ObjectId(), "workload_id": "test-2"}
        mock_db_manager.find_running_flywheel_runs.return_value = [flywheel_run_1, flywheel_run_2]

        # Mock empty NIMs for simplicity
        mock_db_manager.find_running_nims_for_flywheel.return_value = []
        mock_db_manager.find_nims_for_job.return_value = []

        # Mock DMS client for LLM judge
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.cleanup_all_running_resources()

        # Verify database calls
        mock_db_manager.find_running_flywheel_runs.assert_called_once()
        assert mock_db_manager.find_running_nims_for_flywheel.call_count == 2

        # Verify LLM judge shutdown was called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once()
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify no errors
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cleanup_all_running_resources_no_runs(self, cleanup_manager_instance, mock_db_manager):
        """Test cleanup when no running resources exist."""
        # Mock empty database response
        mock_db_manager.find_running_flywheel_runs.return_value = []

        # Mock DMS client for LLM judge
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.cleanup_all_running_resources()

        # Verify database was queried
        mock_db_manager.find_running_flywheel_runs.assert_called_once()

        # Verify no flywheel cleanup was attempted
        mock_db_manager.find_running_nims_for_flywheel.assert_not_called()

        # Verify LLM judge shutdown was still called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once()
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify no errors
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cleanup_all_running_resources_with_errors(
        self, cleanup_manager_instance, mock_db_manager
    ):
        """Test cleanup with errors in individual flywheel runs."""
        # Mock running flywheel runs
        flywheel_run_1 = {"_id": ObjectId(), "workload_id": "test-1"}
        flywheel_run_2 = {"_id": ObjectId(), "workload_id": "test-2"}
        mock_db_manager.find_running_flywheel_runs.return_value = [flywheel_run_1, flywheel_run_2]

        # Configure first run to fail
        mock_db_manager.find_running_nims_for_flywheel.side_effect = [
            Exception("Database error"),
            [],  # Second call succeeds with empty result
        ]
        mock_db_manager.find_nims_for_job.return_value = []

        # Mock DMS client for LLM judge
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.cleanup_all_running_resources()

        # Verify database calls
        mock_db_manager.find_running_flywheel_runs.assert_called_once()
        assert mock_db_manager.find_running_nims_for_flywheel.call_count == 2

        # Verify LLM judge shutdown was still called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once()
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify error was recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert "Failed to clean up flywheel run" in cleanup_manager_instance.cleanup_errors[0]

    def test_cleanup_all_running_resources_database_error(
        self, cleanup_manager_instance, mock_db_manager
    ):
        """Test cleanup with database error when finding running resources."""
        # Configure database to fail
        mock_db_manager.find_running_flywheel_runs.side_effect = Exception(
            "Database connection failed"
        )

        # Execute the method and verify it raises the exception
        with pytest.raises(Exception) as exc_info:
            cleanup_manager_instance.cleanup_all_running_resources()

        assert "Database connection failed" in str(exc_info.value)

        # Verify no further operations were attempted
        cleanup_manager_instance._mock_dms_client_class.assert_not_called()

    def test_cleanup_all_running_resources_llm_judge_error(
        self, cleanup_manager_instance, mock_db_manager
    ):
        """Test cleanup with LLM judge shutdown error."""
        # Mock empty database response
        mock_db_manager.find_running_flywheel_runs.return_value = []

        # Mock DMS client for LLM judge that fails
        mock_dms_client = MagicMock()
        mock_dms_client.shutdown_deployment.side_effect = Exception("LLM judge shutdown failed")
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.cleanup_all_running_resources()

        # Verify error was recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert "Failed to shutdown LLM judge" in cleanup_manager_instance.cleanup_errors[0]

    def test_cleanup_flywheel_run_no_running_nims_all_have_customization_configs(
        self, cleanup_manager_instance, mock_db_manager
    ):
        """Test Case 1: No running NIMs, all NIMs have customization configs."""
        flywheel_run_id = ObjectId()
        flywheel_run = {"_id": flywheel_run_id, "workload_id": "test-workload"}

        # Mock no running NIMs but all NIMs have customization configs
        mock_db_manager.find_running_nims_for_flywheel.return_value = []
        mock_db_manager.find_nims_for_job.return_value = []

        # Execute the method
        cleanup_manager_instance.cleanup_flywheel_run(flywheel_run)

        # Verify database calls
        mock_db_manager.find_running_nims_for_flywheel.assert_called_once_with(flywheel_run_id)

        # Verify no job cancellation or shutdown operations
        cleanup_manager_instance.customizer.cancel_job.assert_not_called()
        cleanup_manager_instance._mock_dms_client_class.assert_not_called()

        # Verify resources were marked as cancelled
        mock_db_manager.mark_flywheel_run_cancelled.assert_called_once()

        # Verify no errors
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cleanup_flywheel_run_mixed_nim_statuses(
        self, cleanup_manager_instance, mock_db_manager
    ):
        """Test Case 2: Mixed NIM statuses - some running, some completed."""
        flywheel_run_id = ObjectId()
        flywheel_run = {"_id": flywheel_run_id, "workload_id": "test-workload"}

        # Mock 2 running NIMs and 1 completed NIM
        running_nims = [
            {"_id": ObjectId(), "model_name": "test-namespace/test-model-1"},
            {"_id": ObjectId(), "model_name": "test-namespace/test-model-2"},
        ]
        mock_db_manager.find_running_nims_for_flywheel.return_value = running_nims

        # Mock customizations for running NIMs
        mock_customizations = [{"job_id": "custom-job-1"}]
        mock_db_manager.find_customizations_for_nim.return_value = mock_customizations

        # Mock DMS client for shutdown
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.cleanup_flywheel_run(flywheel_run)

        # Verify database calls
        mock_db_manager.find_running_nims_for_flywheel.assert_called_once_with(flywheel_run_id)

        # Verify job cancellation for running NIMs (2 NIMs x 1 customization each)
        assert cleanup_manager_instance.customizer.cancel_job.call_count == 2
        cleanup_manager_instance.customizer.cancel_job.assert_any_call("custom-job-1")

        # Verify NIM shutdown for running NIMs
        assert cleanup_manager_instance._mock_dms_client_class.call_count == 2
        assert mock_dms_client.shutdown_deployment.call_count == 2

        # Verify resources were marked as cancelled
        mock_db_manager.mark_flywheel_run_cancelled.assert_called_once()

        # Verify no errors
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cleanup_all_running_resources_customization_config_cleanup(
        self, cleanup_manager_instance, mock_db_manager
    ):
        """Test Case 3: Customization config cleanup in cleanup_all_running_resources."""
        # Mock no running flywheel runs
        mock_db_manager.find_running_flywheel_runs.return_value = []

        # Mock DMS client for LLM judge
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Execute the method
        cleanup_manager_instance.cleanup_all_running_resources()

        # Verify database calls
        mock_db_manager.find_running_flywheel_runs.assert_called_once()

        # Verify LLM judge shutdown was called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once()
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify customization config cleanup for NIMs with configs
        # Should be called for 2 NIMs that have customization configs (test-namespace/test-model-1 and test-namespace/test-model-2)
        assert cleanup_manager_instance.customizer.delete_customization_config.call_count == 2
        cleanup_manager_instance.customizer.delete_customization_config.assert_any_call(
            "test-model-1@v1.0.0+dfw"
        )
        cleanup_manager_instance.customizer.delete_customization_config.assert_any_call(
            "test-model-2@v1.0.0+dfw"
        )

        # Verify no errors
        assert len(cleanup_manager_instance.cleanup_errors) == 0

    def test_cleanup_all_running_resources_customization_config_error(
        self, cleanup_manager_instance, mock_db_manager
    ):
        """Test Case 3: Error handling in customization config cleanup."""
        # Mock no running flywheel runs
        mock_db_manager.find_running_flywheel_runs.return_value = []

        # Mock DMS client for LLM judge
        mock_dms_client = MagicMock()
        cleanup_manager_instance._mock_dms_client_class.return_value = mock_dms_client

        # Configure customizer to fail on second config deletion
        cleanup_manager_instance.customizer.delete_customization_config.side_effect = [
            None,  # First call succeeds
            Exception("Config deletion failed"),  # Second call fails
            None,  # Third call succeeds
        ]

        # Execute the method
        cleanup_manager_instance.cleanup_all_running_resources()

        # Verify database calls
        mock_db_manager.find_running_flywheel_runs.assert_called_once()

        # Verify LLM judge shutdown was called
        cleanup_manager_instance._mock_dms_client_class.assert_called_once()
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify all config deletions were attempted (only for NIMs with configs)
        assert cleanup_manager_instance.customizer.delete_customization_config.call_count == 2

        # Verify error was recorded
        assert len(cleanup_manager_instance.cleanup_errors) == 1
        assert "Failed to delete customization config" in cleanup_manager_instance.cleanup_errors[0]

    def test_delete_nim_customization_config_invalid_model_format(self, cleanup_manager_instance):
        """Test delete_nim_customization_config with invalid model format."""
        # Create a NIM with invalid model name format that will cause ValueError
        nim = {"model_name": "invalid-format"}  # Missing namespace

        # Add a NIM config with customizer configs to the mock settings
        mock_nim_config_with_customizer = MagicMock()
        mock_nim_config_with_customizer.model_name = "invalid-format"
        mock_nim_config_with_customizer.customizer_configs = MagicMock()  # Has customizer configs
        cleanup_manager_instance._mock_settings.nims.append(mock_nim_config_with_customizer)

        # Mock NIMConfig.generate_config_name to raise ValueError
        with patch(
            "src.lib.flywheel.cleanup_manager.NIMConfig.generate_config_name"
        ) as mock_generate:
            mock_generate.side_effect = ValueError("Invalid base model format")

            # Execute the method
            cleanup_manager_instance.delete_nim_customization_config(nim)

            # Verify generate_config_name was called
            mock_generate.assert_called_once_with("invalid-format")

        # Verify no customizer calls were made due to ValueError
        cleanup_manager_instance.customizer.delete_customization_config.assert_not_called()

        # Verify no errors were recorded (ValueError is handled gracefully)
        assert len(cleanup_manager_instance.cleanup_errors) == 0
