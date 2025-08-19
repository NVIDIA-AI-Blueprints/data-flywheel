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

"""Tests for workflow management tasks."""

from unittest.mock import ANY, MagicMock, call, patch

import pytest
from bson import ObjectId

from src.api.models import (
    LLMJudgeConfig,
    NIMConfig,
    TaskResult,
)
from src.api.schemas import FlywheelRunStatus
from src.config import EmbeddingConfig, ICLConfig, SimilarityConfig
from src.lib.flywheel.cancellation import FlywheelCancelledError
from src.tasks.tasks import (
    finalize_flywheel_run,
    initialize_workflow,
    run_nim_workflow_dag,
)
from tests.unit.tasks.conftest import convert_result_to_task_result


class TestInitializeWorkflow:
    """Tests for workflow initialization."""

    @pytest.mark.parametrize(
        "nim_configs, llm_as_judge_config",
        [
            [
                [
                    NIMConfig(
                        model_name="test-model",
                        context_length=2048,
                        gpus=1,
                        pvc_size="10Gi",
                        tag="latest",
                        registry_base="nvcr.io/nim",
                        customization_enabled=False,
                    )
                ],
                LLMJudgeConfig(
                    deployment_type="remote",
                    url="http://test-remote-url/v1/chat/completions",
                    model_name="remote-model-id",
                    api_key="test-api-key",
                ),
            ],
            [
                [
                    NIMConfig(
                        model_name="test-model1",
                        context_length=2048,
                        gpus=1,
                        pvc_size="10Gi",
                        tag="latest",
                        registry_base="nvcr.io/nim",
                        customization_enabled=False,
                    ),
                    NIMConfig(
                        model_name="test-model2",
                        context_length=2048,
                        gpus=1,
                        pvc_size="10Gi",
                        tag="latest",
                        registry_base="nvcr.io/nim",
                        customization_enabled=False,
                    ),
                ],
                LLMJudgeConfig(
                    deployment_type="local",
                    model_name="test-model-id",
                    context_length=2048,
                    gpus=1,
                    pvc_size="10Gi",
                    tag="latest",
                    registry_base="nvcr.io/nim",
                    customization_enabled=False,
                ),
            ],
        ],
    )
    def test_initialize_workflow(
        self, mock_task_db, mock_dms_client, nim_configs, llm_as_judge_config
    ):
        """Test initializing workflow."""
        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.LLMAsJudge", autospec=True) as mock_llm_class,
        ):
            # Set up the LLMAsJudge mock
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.config = llm_as_judge_config
            mock_settings.nims = nim_configs

            result = initialize_workflow(
                workload_id=workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

            result = convert_result_to_task_result(result)

            assert isinstance(result, TaskResult)
            assert result.llm_judge_config == llm_as_judge_config
            assert result.workload_id == workload_id
            assert result.flywheel_run_id == flywheel_run_id
            assert result.client_id == client_id

            # Verify DB interactions
            assert mock_task_db.create_nim_run.call_count == len(nim_configs)
            mock_task_db.create_llm_judge_run.assert_called_once()

            # Verify that the LLMAsJudge was called
            mock_llm_class.assert_called_once()

    def test_initialize_workflow_cancellation_success(self, mock_task_db, mock_dms_client):
        """Test initialize_workflow when cancellation check passes (not cancelled)."""
        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        nim_config = NIMConfig(
            model_name="test-model",
            context_length=2048,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
            registry_base="nvcr.io/nim",
            customization_enabled=False,
        )

        llm_as_judge_config = LLMJudgeConfig(
            deployment_type="remote",
            url="http://test-remote-url/v1/chat/completions",
            model_name="remote-model-id",
            api_key="test-api-key",
        )

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.LLMAsJudge", autospec=True) as mock_llm_class,
            patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation,
        ):
            # Set up the LLMAsJudge mock
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.config = llm_as_judge_config
            mock_settings.nims = [nim_config]

            # Configure cancellation check to pass (not cancelled)
            mock_check_cancellation.return_value = None  # No exception raised

            result = initialize_workflow(
                workload_id=workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

            result = convert_result_to_task_result(result)

            # Verify cancellation was checked
            mock_check_cancellation.assert_called_once_with(flywheel_run_id, raise_error=True)

            # Verify normal initialization proceeded
            assert isinstance(result, TaskResult)
            assert result.llm_judge_config == llm_as_judge_config
            assert result.workload_id == workload_id
            assert result.flywheel_run_id == flywheel_run_id
            assert result.client_id == client_id

            # Verify DB interactions
            mock_task_db.update_flywheel_run_status.assert_called_once()
            mock_task_db.create_nim_run.assert_called_once()
            mock_task_db.create_llm_judge_run.assert_called_once()

            # Verify that the LLMAsJudge was called
            mock_llm_class.assert_called_once()

    def test_initialize_workflow_cancellation_failure(self, mock_task_db, mock_dms_client):
        """Test initialize_workflow when job is cancelled."""
        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        nim_config = NIMConfig(
            model_name="test-model",
            context_length=2048,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
            registry_base="nvcr.io/nim",
            customization_enabled=False,
        )

        llm_as_judge_config = LLMJudgeConfig(
            deployment_type="remote",
            url="http://test-remote-url/v1/chat/completions",
            model_name="remote-model-id",
            api_key="test-api-key",
        )

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.LLMAsJudge", autospec=True) as mock_llm_class,
            patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation,
        ):
            # Set up the LLMAsJudge mock
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.config = llm_as_judge_config
            mock_settings.nims = [nim_config]

            # Configure cancellation check to raise FlywheelCancelledError
            mock_check_cancellation.side_effect = FlywheelCancelledError(
                flywheel_run_id, "Flywheel run has been cancelled"
            )

            # Verify that FlywheelCancelledError is raised
            with pytest.raises(FlywheelCancelledError) as exc_info:
                initialize_workflow(
                    workload_id=workload_id,
                    flywheel_run_id=flywheel_run_id,
                    client_id=client_id,
                )

            # Verify cancellation was checked
            mock_check_cancellation.assert_called_once_with(flywheel_run_id, raise_error=True)

            # Verify the exception details
            assert "Flywheel run has been cancelled" in str(exc_info.value)
            assert exc_info.value.flywheel_run_id == flywheel_run_id

            # Verify that initialization steps after cancellation check were not executed
            mock_task_db.update_flywheel_run_status.assert_not_called()
            mock_task_db.create_nim_run.assert_not_called()
            mock_task_db.create_llm_judge_run.assert_not_called()
            mock_llm_class.assert_not_called()

    def test_initialize_workflow_with_data_split_config(self, mock_task_db, mock_dms_client):
        """Test initialize_workflow with custom data split config."""
        from src.config import DataSplitConfig

        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"
        data_split_config = {
            "min_total_records": 10,
            "random_seed": 123,
            "eval_size": 5,
            "val_ratio": 0.2,
            "limit": 50,
        }

        nim_config = NIMConfig(
            model_name="test-model",
            context_length=2048,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
            registry_base="nvcr.io/nim",
            customization_enabled=False,
        )

        llm_as_judge_config = LLMJudgeConfig(
            deployment_type="remote",
            url="http://test-remote-url/v1/chat/completions",
            model_name="remote-model-id",
            api_key="test-api-key",
        )

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.LLMAsJudge", autospec=True) as mock_llm_class,
            patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation,
        ):
            # Set up the LLMAsJudge mock
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.config = llm_as_judge_config
            mock_settings.nims = [nim_config]

            # Configure cancellation check to pass (not cancelled)
            mock_check_cancellation.return_value = None  # No exception raised

            result = initialize_workflow(
                workload_id=workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
                data_split_config=data_split_config,
            )

            result = convert_result_to_task_result(result)

            # Verify normal initialization proceeded
            assert isinstance(result, TaskResult)
            assert isinstance(result.data_split_config, DataSplitConfig)
            assert result.data_split_config.limit == 50

    def test_initialize_workflow_no_data_split_config(self, mock_task_db, mock_dms_client):
        """Test initialize_workflow without custom data split config."""
        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        nim_config = NIMConfig(
            model_name="test-model",
            context_length=2048,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
            registry_base="nvcr.io/nim",
            customization_enabled=False,
        )

        llm_as_judge_config = LLMJudgeConfig(
            deployment_type="remote",
            url="http://test-remote-url/v1/chat/completions",
            model_name="remote-model-id",
            api_key="test-api-key",
        )

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.LLMAsJudge", autospec=True) as mock_llm_class,
            patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation,
        ):
            # Set up the LLMAsJudge mock
            mock_llm_instance = mock_llm_class.return_value
            mock_llm_instance.config = llm_as_judge_config
            mock_settings.nims = [nim_config]

            # Configure cancellation check to pass (not cancelled)
            mock_check_cancellation.return_value = None  # No exception raised

            result = initialize_workflow(
                workload_id=workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
                data_split_config=None,
            )

            result = convert_result_to_task_result(result)

            # Verify normal initialization proceeded
            assert isinstance(result, TaskResult)
            assert result.data_split_config is None


class TestRunNimWorkflowDag:
    """Tests for running NIM workflow DAG."""

    def test_run_nim_workflow_dag_flywheel_not_found(self, mock_task_db):
        """Test run_nim_workflow_dag when flywheel run not found."""
        flywheel_run_id = str(ObjectId())

        # Configure DB manager to return None (not found)
        mock_task_db.get_flywheel_run.return_value = None

        with pytest.raises(ValueError) as exc_info:
            run_nim_workflow_dag(
                workload_id="test-workload",
                flywheel_run_id=flywheel_run_id,
                client_id="test-client",
            )

        assert f"FlywheelRun {flywheel_run_id} not found in database" in str(exc_info.value)

    def test_run_nim_workflow_dag_already_running(self, mock_task_db):
        """Test run_nim_workflow_dag when flywheel run is already running."""
        from src.api.schemas import FlywheelRunStatus

        flywheel_run_id = str(ObjectId())

        # Configure DB manager to return a running flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.RUNNING,
            "_id": ObjectId(flywheel_run_id),
        }

        result = run_nim_workflow_dag(
            workload_id="test-workload", flywheel_run_id=flywheel_run_id, client_id="test-client"
        )

        assert result["status"] == "skipped"
        assert result["reason"] == f"already_{FlywheelRunStatus.RUNNING}"
        assert result["flywheel_run_id"] == flywheel_run_id

    def test_run_nim_workflow_dag_already_completed(self, mock_task_db):
        """Test run_nim_workflow_dag when flywheel run is already completed."""
        from src.api.schemas import FlywheelRunStatus

        flywheel_run_id = str(ObjectId())

        # Configure DB manager to return a completed flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.COMPLETED,
            "_id": ObjectId(flywheel_run_id),
        }

        result = run_nim_workflow_dag(
            workload_id="test-workload", flywheel_run_id=flywheel_run_id, client_id="test-client"
        )

        assert result["status"] == "skipped"
        assert result["reason"] == f"already_{FlywheelRunStatus.COMPLETED}"
        assert result["flywheel_run_id"] == flywheel_run_id

    def test_run_nim_workflow_dag_success(self, mock_task_db, valid_nim_config):
        """Test successful run_nim_workflow_dag execution."""
        from src.api.schemas import FlywheelRunStatus

        flywheel_run_id = str(ObjectId())

        # Configure DB manager to return a pending flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.PENDING,
            "_id": ObjectId(flywheel_run_id),
        }

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.chain") as mock_chain,
        ):
            mock_settings.nims = [valid_nim_config]

            # Mock the workflow chain
            mock_workflow = MagicMock()
            mock_chain.return_value = mock_workflow

            # Mock the async result
            mock_async_result = MagicMock()
            mock_workflow.apply_async.return_value = mock_async_result
            mock_async_result.get.return_value = {"status": "completed"}

            result = run_nim_workflow_dag(
                workload_id="test-workload",
                flywheel_run_id=flywheel_run_id,
                client_id="test-client",
            )

            assert result["status"] == "completed"
            mock_workflow.apply_async.assert_called_once()
            mock_async_result.get.assert_called_once_with(disable_sync_subtasks=False)

    def test_run_nim_workflow_dag_with_data_split_config(self, mock_task_db, valid_nim_config):
        """Test run_nim_workflow_dag with custom data split config."""
        from src.api.schemas import FlywheelRunStatus

        flywheel_run_id = str(ObjectId())
        data_split_config = {
            "min_total_records": 10,
            "random_seed": 123,
            "eval_size": 5,
            "val_ratio": 0.2,
            "limit": 50,
        }

        # Configure DB manager to return a pending flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.PENDING,
            "_id": ObjectId(flywheel_run_id),
        }

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.chain") as mock_chain,
            patch("src.tasks.tasks.DataSplitConfig") as mock_data_split_config_class,
        ):
            mock_settings.nims = [valid_nim_config]

            # Mock DataSplitConfig
            mock_split_config = MagicMock()
            mock_data_split_config_class.return_value = mock_split_config
            mock_split_config.model_dump.return_value = data_split_config

            # Mock the workflow chain
            mock_workflow = MagicMock()
            mock_chain.return_value = mock_workflow

            # Mock the async result
            mock_async_result = MagicMock()
            mock_workflow.apply_async.return_value = mock_async_result
            mock_async_result.get.return_value = {"status": "completed"}

            result = run_nim_workflow_dag(
                workload_id="test-workload",
                flywheel_run_id=flywheel_run_id,
                client_id="test-client",
                data_split_config=data_split_config,
            )

            assert result["status"] == "completed"
            mock_data_split_config_class.assert_called_once_with(**data_split_config)
            mock_workflow.apply_async.assert_called_once()

    def test_run_nim_workflow_dag_skip_due_to_previous_error(self, mock_task_db):
        """Test run_nim_workflow_dag when flywheel run not found."""
        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        # Configure DB manager to return None (not found)
        mock_task_db.get_flywheel_run.return_value = None

        with pytest.raises(ValueError) as exc_info:
            run_nim_workflow_dag(
                workload_id=workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

        # The function should fail because flywheel run is not found
        assert f"FlywheelRun {flywheel_run_id} not found in database" in str(exc_info.value)

    def test_run_nim_workflow_dag_database_error(self, mock_task_db):
        """Test run_nim_workflow_dag when database get fails."""

        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        # Make database get fail
        mock_task_db.get_flywheel_run.side_effect = Exception("Database get failed")

        with pytest.raises(Exception) as exc_info:
            run_nim_workflow_dag(
                workload_id=workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

        # Verify the database error was raised
        assert "Database get failed" in str(exc_info.value)

    def test_run_nim_workflow_dag_with_chain_execution(self, mock_task_db, valid_nim_config):
        """Test run_nim_workflow_dag with chain execution."""
        from src.api.schemas import FlywheelRunStatus

        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        # Configure DB manager to return a pending flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.PENDING,
            "_id": ObjectId(flywheel_run_id),
        }

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.chain") as mock_chain,
        ):
            mock_settings.nims = [valid_nim_config]

            mock_chain_instance = MagicMock()
            mock_chain.return_value = mock_chain_instance
            mock_async_result = MagicMock()
            mock_chain_instance.apply_async.return_value = mock_async_result
            mock_async_result.get.return_value = {"status": "completed"}

            result = run_nim_workflow_dag(
                workload_id=workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

            # Verify chain was created and executed (multiple chains are created for the workflow)
            assert mock_chain.call_count >= 1
            mock_chain_instance.apply_async.assert_called_once()
            mock_async_result.get.assert_called_once_with(disable_sync_subtasks=False)

            # Should return the completed result
            assert result["status"] == "completed"

    def test_run_nim_workflow_dag_chain_execution_error(self, mock_task_db, valid_nim_config):
        """Test run_nim_workflow_dag when chain execution fails."""
        from src.api.schemas import FlywheelRunStatus

        workload_id = "test-workload"
        flywheel_run_id = str(ObjectId())
        client_id = "test-client"

        # Configure DB manager to return a pending flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.PENDING,
            "_id": ObjectId(flywheel_run_id),
        }

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.chain") as mock_chain,
        ):
            mock_settings.nims = [valid_nim_config]

            mock_chain_instance = MagicMock()
            mock_chain.return_value = mock_chain_instance
            mock_async_result = MagicMock()
            mock_chain_instance.apply_async.return_value = mock_async_result
            mock_async_result.get.side_effect = Exception("Chain execution failed")

            with pytest.raises(Exception) as exc_info:
                run_nim_workflow_dag(
                    workload_id=workload_id,
                    flywheel_run_id=flywheel_run_id,
                    client_id=client_id,
                )

            # Verify chain execution was attempted
            mock_chain_instance.apply_async.assert_called_once()
            mock_async_result.get.assert_called_once_with(disable_sync_subtasks=False)

            # Verify the error was raised
            assert "Chain execution failed" in str(exc_info.value)

    def test_run_nim_workflow_dag_with_uniform_distribution(self, mock_task_db, valid_nim_config):
        """Test run_nim_workflow_dag with uniform distribution ICL configuration."""

        flywheel_run_id = str(ObjectId())

        # Configure DB manager to return a pending flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.PENDING,
            "_id": ObjectId(flywheel_run_id),
        }

        # Configure ICL for uniform distribution
        uniform_icl_config = ICLConfig(
            example_selection="uniform_distribution",
            similarity_config=None,
        )

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.chain") as mock_chain,
            patch("src.tasks.tasks.DataSplitConfig"),
        ):
            mock_settings.icl_config = uniform_icl_config
            mock_settings.nims = [valid_nim_config]

            # Track chain calls to verify workflow structure
            chain_calls = []

            def track_chain(*args):
                chain_calls.append([str(arg) for arg in args])
                mock_workflow = MagicMock()
                mock_async_result = MagicMock()
                mock_workflow.apply_async.return_value = mock_async_result
                mock_async_result.get.return_value = {"status": "completed"}
                return mock_workflow

            mock_chain.side_effect = track_chain

            # Execute workflow
            result = run_nim_workflow_dag(
                workload_id="test-workload",
                flywheel_run_id=flywheel_run_id,
                client_id="test-client",
            )

            # Verify workflow completed
            assert result["status"] == "completed"

            # Verify that no embedding workflow exists (key distinction for uniform_distribution)
            # Look for chain calls that have both spin_up_nim and create_datasets
            # For uniform_distribution, create_datasets should be in main workflow, not a separate embedding chain
            embedding_workflow_found = False

            for chain_call in chain_calls:
                if (
                    len(chain_call) == 2
                    and any("spin_up_nim" in task and "embedding" in task for task in chain_call)
                    and any("create_datasets" in task for task in chain_call)
                ):
                    embedding_workflow_found = True
                    break

            assert (
                not embedding_workflow_found
            ), "Unexpected embedding workflow found for uniform_distribution"

    def test_run_nim_workflow_dag_with_semantic_similarity_local(
        self, mock_task_db, valid_nim_config
    ):
        """Test run_nim_workflow_dag with semantic similarity local embedding configuration."""

        flywheel_run_id = str(ObjectId())

        # Configure DB manager to return a pending flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.PENDING,
            "_id": ObjectId(flywheel_run_id),
        }

        # Configure ICL for semantic similarity with LOCAL embedding
        embedding_config = EmbeddingConfig(
            deployment_type="local",
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            context_length=32768,
            gpus=1,
            pvc_size="25Gi",
            tag="1.9.0",
        )

        similarity_config = SimilarityConfig(
            relevance_ratio=0.7,
            embedding_nim_config=embedding_config,
        )

        semantic_icl_config = ICLConfig(
            example_selection="semantic_similarity",
            similarity_config=similarity_config,
        )

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.chain") as mock_chain,
            patch("src.tasks.tasks.DataSplitConfig"),
        ):
            mock_settings.icl_config = semantic_icl_config
            mock_settings.nims = [valid_nim_config]

            # Track chain calls to verify workflow structure
            chain_calls = []

            def track_chain(*args):
                chain_calls.append([str(arg) for arg in args])
                mock_workflow = MagicMock()
                mock_async_result = MagicMock()
                mock_workflow.apply_async.return_value = mock_async_result
                mock_async_result.get.return_value = {"status": "completed"}
                return mock_workflow

            mock_chain.side_effect = track_chain

            # Execute workflow
            result = run_nim_workflow_dag(
                workload_id="test-workload",
                flywheel_run_id=flywheel_run_id,
                client_id="test-client",
            )

            # Verify workflow completed
            assert result["status"] == "completed"

            # Verify embedding workflow exists for semantic similarity + local
            # Should be: spin_up_nim.s(embedding_config) -> create_datasets.s()
            embedding_workflow_found = False

            for chain_call in chain_calls:
                # Look for 2-task chain with embedding spin_up_nim -> create_datasets
                if (
                    len(chain_call) == 2
                    and any("spin_up_nim" in task and "embedding" in task for task in chain_call)
                    and any("create_datasets" in task for task in chain_call)
                ):
                    embedding_workflow_found = True
                    break

            assert embedding_workflow_found, "Expected embedding workflow (spin_up_nim -> create_datasets) for semantic_similarity + local"

    def test_run_nim_workflow_dag_with_semantic_similarity_remote(
        self, mock_task_db, valid_nim_config
    ):
        """Test run_nim_workflow_dag with semantic similarity remote embedding configuration."""
        flywheel_run_id = str(ObjectId())

        # Configure DB manager to return a pending flywheel run
        mock_task_db.get_flywheel_run.return_value = {
            "status": FlywheelRunStatus.PENDING,
            "_id": ObjectId(flywheel_run_id),
        }

        # Configure ICL for semantic similarity with REMOTE embedding
        embedding_config = EmbeddingConfig(
            deployment_type="remote",
            url="http://mock-embedding-service:9022/v1/embeddings",
            model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
            api_key_env="NGC_API_KEY",
        )

        similarity_config = SimilarityConfig(
            relevance_ratio=0.7,
            embedding_nim_config=embedding_config,
        )

        semantic_icl_config = ICLConfig(
            example_selection="semantic_similarity",
            similarity_config=similarity_config,
        )

        with (
            patch("src.tasks.tasks.settings") as mock_settings,
            patch("src.tasks.tasks.chain") as mock_chain,
            patch("src.tasks.tasks.DataSplitConfig"),
        ):
            mock_settings.icl_config = semantic_icl_config
            mock_settings.nims = [valid_nim_config]

            # Track chain calls to verify workflow structure
            chain_calls = []

            def track_chain(*args):
                chain_calls.append([str(arg) for arg in args])
                mock_workflow = MagicMock()
                mock_async_result = MagicMock()
                mock_workflow.apply_async.return_value = mock_async_result
                mock_async_result.get.return_value = {"status": "completed"}
                return mock_workflow

            mock_chain.side_effect = track_chain

            # Execute workflow
            result = run_nim_workflow_dag(
                workload_id="test-workload",
                flywheel_run_id=flywheel_run_id,
                client_id="test-client",
            )

            # Verify workflow completed
            assert result["status"] == "completed"

            # Verify NO embedding workflow for semantic similarity + remote
            # For remote embedding, create_datasets should be in main workflow like uniform_distribution
            embedding_workflow_found = False

            for chain_call in chain_calls:
                if (
                    len(chain_call) == 2
                    and any("spin_up_nim" in task and "embedding" in task for task in chain_call)
                    and any("create_datasets" in task for task in chain_call)
                ):
                    embedding_workflow_found = True
                    break

            assert (
                not embedding_workflow_found
            ), "Unexpected embedding workflow found for semantic_similarity + remote"


class TestFinalizeFlywheelRun:
    """Tests for finalizing flywheel runs."""

    def test_finalize_flywheel_run_success(self, mock_task_db):
        """Test successful finalize_flywheel_run."""
        flywheel_run_id = str(ObjectId())
        previous_result = TaskResult(
            workload_id="test-workload", flywheel_run_id=flywheel_run_id, client_id="test-client"
        )

        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            result = finalize_flywheel_run(previous_result)

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Verify sleep was called for cleanup delay
            mock_sleep.assert_called_once_with(60)

            # Verify flywheel run was marked as completed
            mock_task_db.mark_flywheel_run_completed.assert_called_once_with(flywheel_run_id, ANY)

            assert result.flywheel_run_id == flywheel_run_id
            assert result.workload_id == "test-workload"

    def test_finalize_flywheel_run_with_list(self, mock_task_db):
        """Test finalize_flywheel_run with list of results."""
        flywheel_run_id = str(ObjectId())
        previous_results = [
            TaskResult(
                workload_id="test-workload",
                flywheel_run_id=flywheel_run_id,
                client_id="test-client",
            ),
            TaskResult(
                workload_id="test-workload-2",
                flywheel_run_id=str(ObjectId()),
                client_id="test-client",
            ),
        ]

        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            result = finalize_flywheel_run(previous_results)

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Verify sleep was called for cleanup delay
            mock_sleep.assert_called_once_with(60)

            # Should use the last result's flywheel_run_id
            mock_task_db.mark_flywheel_run_completed.assert_called_once_with(
                previous_results[-1].flywheel_run_id, ANY
            )

            assert result.flywheel_run_id == previous_results[-1].flywheel_run_id
            assert result.workload_id == "test-workload-2"

    def test_finalize_flywheel_run_error_with_previous_result(self, mock_task_db):
        """Test finalize_flywheel_run error handling when previous_result exists."""
        flywheel_run_id = str(ObjectId())
        previous_result = TaskResult(
            workload_id="test-workload", flywheel_run_id=flywheel_run_id, client_id="test-client"
        )

        # Make mark_flywheel_run_completed raise an exception
        mock_task_db.mark_flywheel_run_completed.side_effect = Exception("Database error")

        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            result = finalize_flywheel_run(previous_result)

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Verify sleep was called twice - once in normal path, once in error path
            assert mock_sleep.call_count == 2
            mock_sleep.assert_has_calls([call(60), call(60)])

            # Verify error was set on result
            assert result.error is not None
            assert "Error finalizing Flywheel run" in result.error

    def test_finalize_flywheel_run_error_no_previous_result(self, mock_task_db):
        """Test finalize_flywheel_run error handling when no valid previous_result."""
        # Pass invalid data that will cause _extract_previous_result to fail
        invalid_results = ["invalid", 123]

        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            # The function now handles this case properly and returns a TaskResult with error
            result = finalize_flywheel_run(invalid_results)

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Sleep is NOT called when previous_result is None - function returns early
            mock_sleep.assert_not_called()

            # Verify error was set on result
            assert result.error is not None
            assert "Error finalizing Flywheel run" in result.error

    def test_finalize_flywheel_run_extract_previous_result_exception_handling(self, mock_task_db):
        """Test finalize_flywheel_run error handling when _extract_previous_result fails."""
        # Pass invalid data that will cause _extract_previous_result to fail
        invalid_results = ["invalid", 123]

        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            # The function now handles this case properly and returns a TaskResult with error
            result = finalize_flywheel_run(invalid_results)

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Sleep is NOT called when previous_result is None - function returns early
            mock_sleep.assert_not_called()

            # Verify error was set on result
            assert result.error is not None
            assert "Error finalizing Flywheel run" in result.error

    def test_finalize_flywheel_run_construct_minimal_result_on_error(self, mock_task_db):
        """Test finalize_flywheel_run constructs minimal result when no previous_result available."""
        # This test covers the error path where a minimal TaskResult is constructed
        with (
            patch("src.tasks.tasks._extract_previous_result") as mock_extract,
            patch("src.tasks.tasks.time.sleep") as mock_sleep,
        ):
            # Make _extract_previous_result raise an exception
            mock_extract.side_effect = ValueError("No valid TaskResult found")

            # The function now handles this case properly and returns a TaskResult with error
            result = finalize_flywheel_run(["invalid"])

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Sleep is NOT called when previous_result is None - function returns early
            mock_sleep.assert_not_called()

            # Verify error was set on result
            assert result.error is not None
            assert "Error finalizing Flywheel run" in result.error
            assert "No valid TaskResult found" in result.error

    def test_finalize_flywheel_run_mark_completed_failure_with_previous_result(self, mock_task_db):
        """Test finalize_flywheel_run when mark_flywheel_run_completed fails but previous_result exists."""
        flywheel_run_id = str(ObjectId())
        previous_result = TaskResult(
            workload_id="test-workload", flywheel_run_id=flywheel_run_id, client_id="test-client"
        )

        # Make mark_flywheel_run_completed raise an exception
        mock_task_db.mark_flywheel_run_completed.side_effect = Exception("Database error")

        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            result = finalize_flywheel_run(previous_result)

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Verify sleep was called twice - once in normal path, once in error path
            assert mock_sleep.call_count == 2
            mock_sleep.assert_has_calls([call(60), call(60)])

            # Verify error was set on result
            assert result.error is not None
            assert "Error finalizing Flywheel run" in result.error
            assert "Database error" in result.error

    def test_finalize_flywheel_run_skip_due_to_previous_error(self, mock_task_db):
        """Test finalize_flywheel_run with previous task error (still executes but with error)."""
        flywheel_run_id = str(ObjectId())

        # Create previous result with an error
        previous_result_with_error = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            error="Previous task failed with error",
        )

        # finalize_flywheel_run doesn't skip on errors - it still executes
        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            result = finalize_flywheel_run(previous_result_with_error)

            # Convert result to TaskResult if it's a dict
            result = convert_result_to_task_result(result)

            # Should still execute and return the result with the error preserved
            assert result.flywheel_run_id == flywheel_run_id
            assert result.error == "Previous task failed with error"

            # Verify sleep was called for cleanup delay
            mock_sleep.assert_called_once_with(60)

            # Verify finalization operations occurred
            mock_task_db.mark_flywheel_run_completed.assert_called_once_with(flywheel_run_id, ANY)

    def test_finalize_flywheel_run_database_error(self, mock_task_db):
        """Test finalize_flywheel_run when database update fails."""
        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
        )

        # Make database update fail
        mock_task_db.mark_flywheel_run_completed.side_effect = Exception("Database update failed")

        with patch("src.tasks.tasks.time.sleep") as mock_sleep:
            mock_sleep.return_value = None
            result = finalize_flywheel_run(previous_result)

            # Convert result to TaskResult if it's a dict
            result = convert_result_to_task_result(result)

            # Verify database update was attempted with correct parameters
            mock_task_db.mark_flywheel_run_completed.assert_called_once_with(flywheel_run_id, ANY)

            # Verify error was set on the result
            assert result.error is not None
            assert "Error finalizing Flywheel run" in result.error
            assert "Database update failed" in result.error
