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

"""Tests for NIM deployment tasks."""

from datetime import datetime
from unittest.mock import patch

import pytest
from bson import ObjectId

from src.api.models import (
    LLMJudgeConfig,
    NIMConfig,
    NIMRunStatus,
    TaskResult,
    WorkloadClassification,
)
from src.lib.flywheel.cancellation import FlywheelCancelledError
from src.tasks.tasks import (
    shutdown_deployment,
    spin_up_nim,
    wait_for_llm_as_judge,
)
from tests.unit.tasks.conftest import convert_result_to_task_result


class TestSpinUpNIM:
    """Tests for NIM spin up functionality."""

    def test_spin_up_nim(self, mock_task_db, mock_dms_client, valid_nim_config):
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

        spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Verify DMS client method calls
        mock_dms_client.is_deployed.assert_called_once()
        mock_dms_client.deploy_model.assert_called_once()
        mock_dms_client.wait_for_deployment.assert_called_once()
        mock_dms_client.wait_for_model_sync.assert_called_once()
        assert mock_task_db.set_nim_status.call_count >= 1  # status transitions

        # No error should be present on the previous_result
        assert previous_result.error is None

    def test_spin_up_nim_deployment_failure(
        self, mock_task_db, mock_dms_client, valid_nim_config, make_llm_as_judge_config
    ):
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure TaskDBManager behaviour
        mock_task_db.create_nim_run.return_value = ObjectId()

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
        assert mock_task_db.mark_nim_error.call_count >= 1  # error status

    def test_spin_up_nim_already_deployed(
        self, mock_task_db, mock_dms_client, valid_nim_config, make_llm_as_judge_config
    ):
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure TaskDBManager behaviour
        mock_task_db.create_nim_run.return_value = ObjectId()

        # Configure DMS client to indicate NIM is already deployed
        mock_dms_client.is_deployed.return_value = True

        spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Verify DMS client method calls
        mock_dms_client.is_deployed.assert_called_once()
        mock_dms_client.deploy_model.assert_not_called()  # Should not try to deploy again
        mock_dms_client.wait_for_deployment.assert_called_once()
        mock_dms_client.wait_for_model_sync.assert_called_once()

        # Verify DB-helper interactions
        assert mock_task_db.set_nim_status.call_count >= 1  # status updates

    def test_spin_up_nim_progress_callback(self, mock_task_db, mock_dms_client, valid_nim_config):
        """Test spin_up_nim progress callback functionality."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.create_nim_run.return_value = nim_run_id

        # Mock wait_for_deployment to call progress callback with valid status
        def mock_wait_for_deployment(flywheel_run_id, progress_callback=None):
            if progress_callback:
                # Use valid DeploymentStatus values
                progress_callback({"status": "pending", "progress": 50})
            return True

        mock_dms_client.wait_for_deployment.side_effect = mock_wait_for_deployment

        spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Verify progress callback was used
        mock_dms_client.wait_for_deployment.assert_called_once()
        # Verify DB status updates occurred
        assert mock_task_db.update_nim_deployment_status.call_count >= 1

    def test_spin_up_nim_dms_client_shutdown_error(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test spin_up_nim when DMS client shutdown fails."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": valid_nim_config.model_name,
            "started_at": datetime.utcnow(),
            "finished_at": None,
            "runtime_seconds": 0,
            "status": NIMRunStatus.RUNNING,
        }

        # Configure DMS client to fail during wait_for_deployment (which goes to exception handler)
        mock_dms_client.wait_for_deployment.side_effect = Exception("Deployment wait failed")
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

        spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Verify error is captured
        assert previous_result.error is not None
        assert "Deployment wait failed" in previous_result.error

        # Verify shutdown was attempted despite failure
        mock_dms_client.shutdown_deployment.assert_called_once()

    def test_spin_up_nim_dms_client_shutdown_in_exception_handler(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test spin_up_nim DMS client shutdown in exception handler."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.create_nim_run.return_value = nim_run_id

        # Configure DMS client to fail during wait_for_deployment
        mock_dms_client.wait_for_deployment.side_effect = Exception("Wait failed")

        spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Verify error is captured
        assert previous_result.error is not None
        assert "Wait failed" in previous_result.error

        # Verify shutdown was called in exception handler
        mock_dms_client.shutdown_deployment.assert_called_once()

    def test_spin_up_nim_shutdown_also_fails_in_exception_handler(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test spin_up_nim when both deployment and shutdown fail."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.create_nim_run.return_value = nim_run_id

        # Configure DMS client to fail during wait_for_model_sync and shutdown
        mock_dms_client.wait_for_model_sync.side_effect = Exception("Model sync failed")
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

        spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Verify original error is preserved
        assert previous_result.error is not None
        assert "Model sync failed" in previous_result.error

        # Verify shutdown was attempted
        mock_dms_client.shutdown_deployment.assert_called_once()


class TestSpinUpNIMCancellation:
    """Tests for NIM spin up cancellation scenarios."""

    def test_spin_up_nim_cancellation(self, mock_task_db, mock_dms_client, valid_nim_config):
        """Test spin_up_nim when job is cancelled at the start."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager with all required fields for NIMRun
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": valid_nim_config.model_name,
            "started_at": datetime.utcnow(),
            "finished_at": None,
            "runtime_seconds": 0,
            "status": NIMRunStatus.RUNNING,
        }

        with patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation:
            # Configure cancellation check to raise FlywheelCancelledError
            mock_check_cancellation.side_effect = FlywheelCancelledError(
                flywheel_run_id, "Flywheel run was cancelled"
            )

            result = spin_up_nim(previous_result, valid_nim_config.model_dump())

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Verify cancellation was checked
            mock_check_cancellation.assert_called_once_with(flywheel_run_id, raise_error=True)

            # Verify NIM was marked as cancelled
            mock_task_db.mark_nim_cancelled.assert_called_once_with(
                nim_run_id,
                error_msg="Flywheel run cancelled",
            )

            # Verify error message in result
            assert result.error is not None
            assert "Flywheel run cancelled" in result.error

            # Verify no deployment operations occurred
            mock_dms_client.deploy_model.assert_not_called()
            mock_dms_client.wait_for_deployment.assert_not_called()

    def test_spin_up_nim_cancellation_during_deployment_wait(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test spin_up_nim when cancellation occurs during wait_for_deployment."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.create_nim_run.return_value = nim_run_id
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": valid_nim_config.model_name,
            "started_at": datetime.utcnow(),
            "finished_at": None,
            "runtime_seconds": 0,
            "status": NIMRunStatus.RUNNING,
        }

        # Configure DMS client to raise cancellation during wait_for_deployment
        mock_dms_client.wait_for_deployment.side_effect = FlywheelCancelledError(
            flywheel_run_id, "Flywheel run was cancelled"
        )

        result = spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # Verify NIM was marked as cancelled
        mock_task_db.mark_nim_cancelled.assert_called_once_with(
            nim_run_id,
            error_msg="Flywheel run cancelled",
        )

        # Verify error message in result
        assert result.error is not None
        assert "Flywheel run cancelled" in result.error

    def test_spin_up_nim_cancellation_during_model_sync_wait(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test spin_up_nim when cancellation occurs during wait_for_model_sync."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.create_nim_run.return_value = nim_run_id
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": valid_nim_config.model_name,
            "started_at": datetime.utcnow(),
            "finished_at": None,
            "runtime_seconds": 0,
            "status": NIMRunStatus.RUNNING,
        }

        # Configure DMS client to raise cancellation during wait_for_model_sync
        mock_dms_client.wait_for_model_sync.side_effect = FlywheelCancelledError(
            flywheel_run_id, "Flywheel run was cancelled"
        )

        result = spin_up_nim(previous_result, valid_nim_config.model_dump())

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # Verify NIM was marked as cancelled
        mock_task_db.mark_nim_cancelled.assert_called_once_with(
            nim_run_id,
            error_msg="Flywheel run cancelled",
        )

        # Verify error message in result
        assert result.error is not None
        assert "Flywheel run cancelled" in result.error


class TestWaitForLLMAsJudge:
    """Tests for LLM-as-Judge waiting functionality."""

    def test_wait_for_llm_as_judge_cancellation(self, mock_task_db, mock_dms_client):
        """Test wait_for_llm_as_judge when job is cancelled."""
        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            llm_judge_config=LLMJudgeConfig(
                deployment_type="local",
                model_name="test-judge-model",
                context_length=2048,
                gpus=1,
                pvc_size="10Gi",
                tag="latest",
                registry_base="nvcr.io/nim",
                customization_enabled=False,
            ),
        )

        # Mock the find_llm_judge_run to return required fields
        mock_task_db.find_llm_judge_run.return_value = {
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": "test-judge-model",
            "deployment_type": "local",
        }

        with patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation:
            # Configure cancellation check to raise FlywheelCancelledError
            mock_check_cancellation.side_effect = FlywheelCancelledError(
                flywheel_run_id, "Flywheel run was cancelled"
            )

            with pytest.raises(ValueError) as exc_info:
                wait_for_llm_as_judge(previous_result)

            # Verify cancellation was checked
            mock_check_cancellation.assert_called_once_with(flywheel_run_id, raise_error=True)

            # Verify error message contains cancellation info
            assert "Error waiting for LLM as judge" in str(exc_info.value)

    def test_wait_for_llm_as_judge_remote_config(self, mock_task_db, mock_dms_client):
        """Test wait_for_llm_as_judge with remote LLM judge configuration."""
        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            llm_judge_config=LLMJudgeConfig(
                deployment_type="remote",
                model_name="test-judge-model",
                url="http://remote-url",
                api_key="test-key",
            ),
        )

        # Mock the find_llm_judge_run to return required fields
        mock_task_db.find_llm_judge_run.return_value = {
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": "test-judge-model",
            "deployment_type": "remote",
        }

        result = wait_for_llm_as_judge(previous_result)

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # For remote config, should return immediately without DMS operations
        mock_dms_client.is_deployed.assert_not_called()
        mock_dms_client.deploy_model.assert_not_called()

        # Should not have error
        assert result.error is None

    def test_wait_for_llm_as_judge_cancellation_during_deployment_wait(
        self, mock_task_db, mock_dms_client
    ):
        """Test wait_for_llm_as_judge when cancellation occurs during deployment wait."""
        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            llm_judge_config=LLMJudgeConfig(
                deployment_type="local",
                model_name="test-judge-model",
                context_length=2048,
                gpus=1,
                pvc_size="10Gi",
                tag="latest",
                registry_base="nvcr.io/nim",
                customization_enabled=False,
            ),
        )

        # Mock the find_llm_judge_run to return required fields
        mock_task_db.find_llm_judge_run.return_value = {
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": "test-judge-model",
            "deployment_type": "local",
        }

        # Configure DMS client to raise cancellation during wait_for_deployment
        mock_dms_client.wait_for_deployment.side_effect = FlywheelCancelledError(
            flywheel_run_id, "Flywheel run was cancelled"
        )

        with pytest.raises(ValueError) as exc_info:
            wait_for_llm_as_judge(previous_result)

        # Verify error message contains cancellation info
        assert "Error waiting for LLM as judge" in str(exc_info.value)

    def test_wait_for_llm_as_judge_cancellation_during_model_sync_wait(
        self, mock_task_db, mock_dms_client
    ):
        """Test wait_for_llm_as_judge when cancellation occurs during model sync wait."""
        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            llm_judge_config=LLMJudgeConfig(
                deployment_type="local",
                model_name="test-judge-model",
                context_length=2048,
                gpus=1,
                pvc_size="10Gi",
                tag="latest",
                registry_base="nvcr.io/nim",
                customization_enabled=False,
            ),
        )

        # Mock the find_llm_judge_run to return required fields
        mock_task_db.find_llm_judge_run.return_value = {
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": "test-judge-model",
            "deployment_type": "local",
        }

        # Configure DMS client to raise cancellation during wait_for_model_sync
        mock_dms_client.wait_for_model_sync.side_effect = FlywheelCancelledError(
            flywheel_run_id, "Flywheel run was cancelled"
        )

        with pytest.raises(ValueError) as exc_info:
            wait_for_llm_as_judge(previous_result)

        # Verify error message contains cancellation info
        assert "Error waiting for LLM as judge" in str(exc_info.value)

    def test_wait_for_llm_as_judge_progress_callback(self, mock_task_db, mock_dms_client):
        """Test wait_for_llm_as_judge progress callback functionality."""
        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            llm_judge_config=LLMJudgeConfig(
                deployment_type="local",
                model_name="test-judge-model",
                context_length=2048,
                gpus=1,
                pvc_size="10Gi",
                tag="latest",
                registry_base="nvcr.io/nim",
                customization_enabled=False,
            ),
        )

        # Mock the find_llm_judge_run to return required fields
        mock_task_db.find_llm_judge_run.return_value = {
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": "test-judge-model",
            "deployment_type": "local",
        }

        # Mock wait_for_deployment to call progress callback with valid status
        def mock_wait_for_deployment(flywheel_run_id, progress_callback=None):
            if progress_callback:
                # Use valid DeploymentStatus values
                progress_callback({"status": "pending", "progress": 75})
            return True

        mock_dms_client.wait_for_deployment.side_effect = mock_wait_for_deployment

        wait_for_llm_as_judge(previous_result)

        # Verify progress callback was used
        mock_dms_client.wait_for_deployment.assert_called_once()


class TestShutdownDeployment:
    """Tests for deployment shutdown functionality."""

    def test_shutdown_deployment(
        self, mock_task_db, mock_dms_client, valid_nim_config, make_llm_as_judge_config
    ):
        """Test shutting down deployment."""
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
            llm_judge_config=make_llm_as_judge_config,
        )

        result = shutdown_deployment(previous_result)

        # Verify DMS client shutdown was called
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify result is returned
        assert result is not None

    def test_shutdown_deployment_with_group_results(
        self, mock_task_db, mock_dms_client, valid_nim_config, make_llm_as_judge_config
    ):
        """Test shutting down deployment with group results."""
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
                llm_judge_config=make_llm_as_judge_config,
            )
        ]

        result = shutdown_deployment(previous_results)

        # Verify DMS client shutdown was called
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify result is returned
        assert result is not None

    def test_shutdown_deployment_failure(
        self, mock_task_db, mock_dms_client, valid_nim_config, make_llm_as_judge_config
    ):
        """Test shutdown deployment when shutdown fails."""
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DMS client to fail shutdown
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

        result = shutdown_deployment(previous_result)

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # Verify shutdown was attempted
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Verify error is captured in result
        assert result.error is not None
        assert "Shutdown failed" in result.error

    def test_shutdown_deployment_cancellation(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test shutdown_deployment when job is cancelled."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": valid_nim_config.model_name,
            "started_at": datetime.utcnow(),
            "finished_at": None,
            "runtime_seconds": 0,
            "status": NIMRunStatus.RUNNING,
        }

        with patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation:
            # Configure cancellation check to return True (cancelled)
            mock_check_cancellation.return_value = True

            shutdown_deployment(previous_result)

            # Verify cancellation was checked with raise_error=False (actual implementation)
            mock_check_cancellation.assert_called_once_with(flywheel_run_id, raise_error=False)

            # Verify NIM was marked as cancelled
            mock_task_db.mark_nim_cancelled.assert_called_once_with(
                nim_run_id,
                error_msg="Flywheel run cancelled",
            )

            # Should still call shutdown
            mock_dms_client.shutdown_deployment.assert_called_once()

    def test_shutdown_deployment_same_model_as_llm_judge(self, mock_task_db, mock_dms_client):
        """Test shutdown_deployment when NIM model is same as LLM judge."""
        flywheel_run_id = str(ObjectId())

        # Create NIM config and LLM judge config with same model
        nim_config = NIMConfig(
            model_name="shared-model",
            context_length=2048,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
            registry_base="nvcr.io/nim",
            customization_enabled=False,
        )

        llm_judge_config = LLMJudgeConfig(
            deployment_type="local",
            model_name="shared-model",  # Same model name
            context_length=2048,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
            registry_base="nvcr.io/nim",
            customization_enabled=False,
        )

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=nim_config,
            llm_judge_config=llm_judge_config,
        )

        result = shutdown_deployment(previous_result)

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # When NIM and LLM judge use same model, shutdown should not be called
        mock_dms_client.shutdown_deployment.assert_not_called()

        # Result should not have error
        assert result.error is None

    def test_shutdown_deployment_with_error_but_valid_nim(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test shutdown_deployment with error but valid NIM config."""
        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=valid_nim_config,
            error="Previous stage failed",
        )

        result = shutdown_deployment(previous_result)

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # Even with error, shutdown should be called if NIM config is valid
        mock_dms_client.shutdown_deployment.assert_called_once()

        # Original error should be preserved
        assert result.error == "Previous stage failed"

    def test_shutdown_deployment_extract_previous_result_failure(
        self, mock_task_db, mock_dms_client
    ):
        """Test shutdown_deployment when extracting previous result fails."""
        # Pass invalid input that will cause _extract_previous_result to fail
        invalid_input = []

        result = shutdown_deployment(invalid_input)

        # Should not call shutdown
        mock_dms_client.shutdown_deployment.assert_not_called()

        # Result should be None when extraction fails completely
        assert result is None

    def test_shutdown_deployment_update_nim_status_exception(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test shutdown_deployment when updating NIM status raises exception."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager to raise exception when updating status
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": valid_nim_config.model_name,
            "started_at": datetime.utcnow(),
            "finished_at": None,
            "runtime_seconds": 0,
            "status": NIMRunStatus.RUNNING,
        }
        mock_task_db.mark_nim_completed.side_effect = Exception("DB update failed")

        result = shutdown_deployment(previous_result)

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # Shutdown should still be called
        mock_dms_client.shutdown_deployment.assert_called_once()

        # No error should be captured since the exception is caught and logged
        assert result.error is None

    def test_shutdown_deployment_nim_run_not_found_in_error_handler(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test shutdown_deployment when NIM run is not found in error handler."""
        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager to return None for find_nim_run
        mock_task_db.find_nim_run.return_value = None
        # Configure DMS client to fail
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

        result = shutdown_deployment(previous_result)

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # Shutdown should be attempted
        mock_dms_client.shutdown_deployment.assert_called_once()

    def test_shutdown_deployment_error_handler_no_previous_result(
        self, mock_task_db, mock_dms_client
    ):
        """Test shutdown_deployment error handler when no previous result available."""
        # Configure DMS client to fail
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

        # Pass empty list to trigger error
        result = shutdown_deployment([])

        # Result should be None when no previous result can be extracted
        assert result is None

    def test_shutdown_deployment_dms_client_error_with_nim_run_found(
        self, mock_task_db, mock_dms_client, valid_nim_config
    ):
        """Test shutdown_deployment when DMS client fails but NIM run is found."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "flywheel_run_id": ObjectId(flywheel_run_id),
            "model_name": valid_nim_config.model_name,
            "started_at": datetime.utcnow(),
            "finished_at": None,
            "runtime_seconds": 0,
            "status": NIMRunStatus.RUNNING,
        }

        # Configure DMS client to fail
        mock_dms_client.shutdown_deployment.side_effect = Exception("Shutdown failed")

        result = shutdown_deployment(previous_result)

        # Convert result to TaskResult if it's a dict (Celery serialization behavior)
        result = convert_result_to_task_result(result)

        # Shutdown should be attempted
        mock_dms_client.shutdown_deployment.assert_called_once()

        # NIM should be marked as error
        mock_task_db.mark_nim_error.assert_called_once()

        # Error should be captured
        assert result.error is not None
        assert "Shutdown failed" in result.error
