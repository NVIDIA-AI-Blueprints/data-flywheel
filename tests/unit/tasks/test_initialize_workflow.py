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
"""Unit tests for initialize_workflow.

These tests focus on the different code-paths involved when initializing a workflow.
The scenarios covered are:

Parameter flow test cases:
1. Successful parameter population in TaskResult
2. Parameter validation and error handling
3. Integration with downstream tasks

llm_as_judge_config Setup:
1. Local judge configuration - no outbound request should be issued.
2. Remote judge happy-path - successful health-check call.
3. Remote judge with missing critical configuration values.
4. validation of mandatory arguments

NIMs setup:
1. Single NIM
2. Multiple NIMs
3. No NIMs
In each of these cases verify the result parameters are setup correctly.
"""

import unittest
import unittest.mock
from unittest.mock import patch

import pytest
from bson import ObjectId

from src.api.models import TaskResult
from src.api.schemas import DeploymentStatus, NIMRunStatus
from src.config import LLMJudgeConfig, NIMConfig
from src.lib.nemo.llm_as_judge import LLMAsJudge
from src.tasks.tasks import initialize_workflow


class TestInitializeWorkflow(unittest.TestCase):
    def setUp(self):
        # Common setup for all tests
        self.flywheel_run_id = str(ObjectId())
        self.workload_id = "test-workload"
        self.client_id = "test-client"

        # Mock the LLMAsJudge class and its config
        self.mock_llm_judge_config_local = LLMJudgeConfig(
            model_name="local-judge-model",
            type="local",
            context_length=8192,
            tag="1.0.0",
            gpus=1,
            pvc_size="10Gi",
            customization_enabled=False,
        )

        self.mock_llm_judge_config_remote = LLMJudgeConfig(
            model_name="remote-judge-model",
            type="remote",
            url="https://api.example.com/v1/chat/completions",
            api_key="test-api-key",
        )

        # Set up NIM configs for testing
        self.single_nim_config = [
            NIMConfig(
                model_name="single-nim-model",
                context_length=8192,
                tag="1.0.0",
                gpus=1,
                pvc_size="10Gi",
                customization_enabled=True,
            )
        ]

        self.multiple_nim_configs = [
            NIMConfig(
                model_name="nim-model-1",
                context_length=8192,
                tag="1.0.0",
                gpus=1,
                pvc_size="10Gi",
                customization_enabled=True,
            ),
            NIMConfig(
                model_name="nim-model-2",
                context_length=4096,
                tag="2.0.0",
                gpus=2,
                pvc_size="20Gi",
                customization_enabled=False,
            ),
        ]

        self.no_nim_configs = []

    def _convert_result_to_task_result(self, result):
        """Helper method to convert result to TaskResult if it's a dictionary."""
        if isinstance(result, dict):
            return TaskResult(**result)
        return result

    @patch("src.tasks.tasks.LLMAsJudge")
    @patch("src.tasks.tasks.db_manager")
    @patch("src.tasks.tasks.settings")
    def test_parameter_population_success(self, mock_settings, mock_db_manager, mock_llm_as_judge):
        """Test that initialize_workflow properly populates parameters in TaskResult."""
        mock_llm_as_judge.return_value.config = self.mock_llm_judge_config_local
        mock_settings.nims = self.single_nim_config
        mock_db_manager.create_llm_judge_run.return_value = ObjectId()
        mock_db_manager.create_nim_run.return_value = ObjectId()

        result = initialize_workflow(
            workload_id=self.workload_id,
            flywheel_run_id=self.flywheel_run_id,
            client_id=self.client_id,
        )
        result = self._convert_result_to_task_result(result)

        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.workload_id, self.workload_id)
        self.assertEqual(result.flywheel_run_id, self.flywheel_run_id)
        self.assertEqual(result.client_id, self.client_id)
        self.assertIsNone(result.error)
        self.assertEqual(result.llm_judge_config, self.mock_llm_judge_config_local)
        self.assertEqual(result.datasets, {})

        # Verify database operations
        mock_db_manager.create_llm_judge_run.assert_called_once()
        mock_db_manager.create_nim_run.assert_called_once()

    @patch("src.tasks.tasks.LLMAsJudge")
    @patch("src.tasks.tasks.db_manager")
    @patch("src.tasks.tasks.settings")
    def test_success_case(self, mock_settings, mock_db_manager, mock_llm_as_judge):
        """Test successful initialization with valid previous results."""
        mock_llm_as_judge.return_value.config = self.mock_llm_judge_config_local
        mock_settings.nims = self.single_nim_config
        mock_db_manager.create_llm_judge_run.return_value = ObjectId()
        mock_db_manager.create_nim_run.return_value = ObjectId()

        result = initialize_workflow(
            workload_id=self.workload_id,
            flywheel_run_id=self.flywheel_run_id,
            client_id=self.client_id,
        )
        result = self._convert_result_to_task_result(result)

        self.assertTrue(hasattr(result, "workload_id"))
        self.assertTrue(hasattr(result, "flywheel_run_id"))
        self.assertTrue(hasattr(result, "client_id"))
        self.assertIsNotNone(result.workload_id)
        self.assertIsNotNone(result.flywheel_run_id)
        self.assertIsNotNone(result.client_id)

    @patch("src.tasks.tasks.LLMAsJudge")
    @patch("src.tasks.tasks.db_manager")
    @patch("src.tasks.tasks.settings")
    def test_local_judge_configuration(self, mock_settings, mock_db_manager, mock_llm_as_judge):
        """Test with local judge configuration (no outbound request)."""
        mock_llm_as_judge.return_value.config = self.mock_llm_judge_config_local
        mock_settings.nims = self.single_nim_config
        mock_db_manager.create_llm_judge_run.return_value = ObjectId()
        mock_db_manager.create_nim_run.return_value = ObjectId()

        result = initialize_workflow(
            workload_id=self.workload_id,
            flywheel_run_id=self.flywheel_run_id,
            client_id=self.client_id,
        )
        result = self._convert_result_to_task_result(result)

        self.assertIsNotNone(result)
        self.assertEqual(result.llm_judge_config, self.mock_llm_judge_config_local)
        mock_db_manager.create_llm_judge_run.assert_called_once_with(unittest.mock.ANY)
        # Extract the LLMJudgeRun from the call arguments
        llm_judge_run = mock_db_manager.create_llm_judge_run.call_args[0][0]
        self.assertEqual(llm_judge_run.deployment_status, DeploymentStatus.CREATED)
        self.assertEqual(llm_judge_run.model_name, "local-judge-model")
        self.assertEqual(llm_judge_run.type, "local")

    @patch("src.tasks.tasks.db_manager")
    def test_remote_judge_configuration(self, mock_db_manager):
        """Test with remote judge configuration."""
        self.mock_llm_judge_config_remote.url = None
        mock_db_manager.create_llm_judge_run.return_value = ObjectId()
        mock_db_manager.create_nim_run.return_value = ObjectId()

        with pytest.raises(Exception) as e:
            llm_as_judge = LLMAsJudge()
            llm_as_judge.config = self.mock_llm_judge_config_remote
            llm_as_judge.validate_llm_judge_availability()
            result = initialize_workflow(
                workload_id=self.workload_id,
                flywheel_run_id=self.flywheel_run_id,
                client_id=self.client_id,
            )

            assert llm_as_judge.config == self.mock_llm_judge_config_remote
            assert result.llm_judge_config == self.mock_llm_judge_config_remote

        assert "missing 'url'" in str(e.value)

    @patch("src.tasks.tasks.LLMAsJudge")
    @patch("src.tasks.tasks.db_manager")
    @patch("src.tasks.tasks.settings")
    def test_single_nim(self, mock_settings, mock_db_manager, mock_llm_as_judge):
        """Test with a single NIM configuration."""
        mock_llm_as_judge.return_value.config = self.mock_llm_judge_config_local
        mock_settings.nims = self.single_nim_config
        llm_judge_run_id = ObjectId()
        nim_run_id = ObjectId()
        mock_db_manager.create_llm_judge_run.return_value = llm_judge_run_id
        mock_db_manager.create_nim_run.return_value = nim_run_id

        result = initialize_workflow(
            workload_id=self.workload_id,
            flywheel_run_id=self.flywheel_run_id,
            client_id=self.client_id,
        )

        self.assertIsNotNone(result)
        mock_db_manager.create_llm_judge_run.assert_called_once()
        # Should create exactly one NIM run
        mock_db_manager.create_nim_run.assert_called_once()
        # Verify NIM run details
        nim_run = mock_db_manager.create_nim_run.call_args[0][0]
        self.assertEqual(nim_run.model_name, "single-nim-model")
        self.assertEqual(nim_run.flywheel_run_id, ObjectId(self.flywheel_run_id))
        self.assertEqual(nim_run.status, NIMRunStatus.PENDING)

    @patch("src.tasks.tasks.LLMAsJudge")
    @patch("src.tasks.tasks.db_manager")
    @patch("src.tasks.tasks.settings")
    def test_multiple_nims(self, mock_settings, mock_db_manager, mock_llm_as_judge):
        """Test with multiple NIM configurations."""
        mock_llm_as_judge.return_value.config = self.mock_llm_judge_config_local
        mock_settings.nims = self.multiple_nim_configs
        llm_judge_run_id = ObjectId()
        nim_run_id1 = ObjectId()
        nim_run_id2 = ObjectId()
        mock_db_manager.create_llm_judge_run.return_value = llm_judge_run_id
        mock_db_manager.create_nim_run.side_effect = [nim_run_id1, nim_run_id2]

        result = initialize_workflow(
            workload_id=self.workload_id,
            flywheel_run_id=self.flywheel_run_id,
            client_id=self.client_id,
        )

        self.assertIsNotNone(result)
        mock_db_manager.create_llm_judge_run.assert_called_once()
        # Should create two NIM runs
        self.assertEqual(mock_db_manager.create_nim_run.call_count, 2)

        # Verify first NIM run details
        first_nim_run = mock_db_manager.create_nim_run.call_args_list[0][0][0]
        self.assertEqual(first_nim_run.model_name, "nim-model-1")
        self.assertEqual(first_nim_run.flywheel_run_id, ObjectId(self.flywheel_run_id))
        self.assertEqual(first_nim_run.status, NIMRunStatus.PENDING)

        # Verify second NIM run details
        second_nim_run = mock_db_manager.create_nim_run.call_args_list[1][0][0]
        self.assertEqual(second_nim_run.model_name, "nim-model-2")
        self.assertEqual(second_nim_run.flywheel_run_id, ObjectId(self.flywheel_run_id))
        self.assertEqual(second_nim_run.status, NIMRunStatus.PENDING)

    @patch("src.tasks.tasks.LLMAsJudge")
    @patch("src.tasks.tasks.db_manager")
    @patch("src.tasks.tasks.settings")
    def test_no_nims(self, mock_settings, mock_db_manager, mock_llm_as_judge):
        """Test with no NIM configurations."""
        mock_llm_as_judge.return_value.config = self.mock_llm_judge_config_local
        mock_settings.nims = self.no_nim_configs
        llm_judge_run_id = ObjectId()
        mock_db_manager.create_llm_judge_run.return_value = llm_judge_run_id

        result = initialize_workflow(
            workload_id=self.workload_id,
            flywheel_run_id=self.flywheel_run_id,
            client_id=self.client_id,
        )

        self.assertIsNotNone(result)
        mock_db_manager.create_llm_judge_run.assert_called_once()
        # Should not create any NIM runs
        mock_db_manager.create_nim_run.assert_not_called()
