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

"""Tests for evaluation tasks."""

from datetime import datetime
from unittest.mock import ANY, patch

import pytest
from bson import ObjectId

from src.api.models import (
    CustomizationResult,
    DatasetType,
    EvalType,
    LLMJudgeConfig,
    TaskResult,
    ToolEvalType,
    WorkloadClassification,
)
from src.tasks.tasks import (
    run_base_eval,
    run_customization_eval,
    run_icl_eval,
)
from tests.unit.tasks.conftest import convert_result_to_task_result


class TestBaseEvaluation:
    """Tests for base evaluation functionality."""

    def test_run_base_eval(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
    ):
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = ObjectId()

        # Configure mock evaluator
        mock_evaluator.run_evaluation.return_value = "job-123"
        mock_evaluator.get_job_uri.return_value = "http://test-uri"
        mock_evaluator.get_evaluation_results.return_value = {
            "tasks": {
                "llm-as-judge": {
                    "metrics": {"llm-judge": {"scores": {"similarity": {"value": 0.95}}}}
                }
            }
        }

        run_base_eval(previous_result)

        # Verify evaluator calls
        mock_evaluator.run_evaluation.assert_called_once_with(
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
        mock_task_db.find_nim_run.assert_called_once()
        mock_task_db.insert_evaluation.assert_called_once()
        assert mock_task_db.update_evaluation.call_count >= 2  # progress + final

    def test_run_base_eval_failure(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id

        # Configure mock evaluator to fail
        mock_evaluator.run_evaluation.side_effect = Exception("Evaluation failed")

        run_base_eval(previous_result)

        # Verify error handling
        mock_task_db.update_evaluation.assert_called_with(
            ANY,
            {
                "error": "Error running base-eval evaluation: Evaluation failed",
                "finished_at": ANY,
                "progress": 0.0,
            },
        )

    def test_run_base_eval_results_failure(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id

        # Configure mock evaluator to fail during results retrieval
        mock_evaluator.run_evaluation.return_value = "job-123"
        mock_evaluator.get_job_uri.return_value = "http://test-uri"
        mock_evaluator.wait_for_evaluation.side_effect = Exception("Timeout waiting for evaluation")

        run_base_eval(previous_result)

        # Verify error handling
        mock_task_db.update_evaluation.assert_called_with(
            ANY,
            {
                "error": "Error running base-eval evaluation: Timeout waiting for evaluation",
                "finished_at": ANY,
                "progress": 0.0,
            },
        )


class TestIclEvaluation:
    """Tests for ICL evaluation functionality."""

    def test_run_icl_eval(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
    ):
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = ObjectId()

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
        mock_task_db.find_nim_run.assert_called_once()
        mock_task_db.insert_evaluation.assert_called_once()
        assert mock_task_db.update_evaluation.call_count >= 2  # progress + final

    def test_run_icl_eval_failure(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id

        # Configure mock evaluator to fail
        mock_evaluator.run_evaluation.side_effect = Exception("Tool calling evaluation failed")

        run_icl_eval(previous_result)

        # Verify error handling
        mock_task_db.update_evaluation.assert_called_with(
            ANY,
            {
                "error": "Error running icl-eval evaluation: Tool calling evaluation failed",
                "finished_at": ANY,
                "progress": 0.0,
            },
        )

    def test_run_icl_eval_results_failure(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id

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
            mock_task_db.update_evaluation.assert_called_with(
                ANY,
                {
                    "error": "Error running icl-eval evaluation: Timeout waiting for tool calling evaluation",
                    "finished_at": ANY,
                    "progress": 0.0,
                },
            )


class TestCustomizationEvaluation:
    """Tests for customization evaluation functionality."""

    def test_run_customization_eval(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = ObjectId()
        mock_task_db.find_customization.return_value = {
            "workload_id": "test-workload",
            "customized_model": "test-model-custom",
        }

        # Configure mock evaluator
        mock_evaluator.run_evaluation.return_value = "job-123"
        mock_evaluator.get_job_uri.return_value = "http://test-uri"
        mock_evaluator.get_evaluation_results.return_value = {
            "tasks": {
                "llm-as-judge": {
                    "metrics": {"llm-judge": {"scores": {"similarity": {"value": 0.95}}}}
                }
            }
        }

        run_customization_eval(previous_result)

        # Verify evaluator calls
        mock_evaluator.run_evaluation.assert_called_once_with(
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
        mock_task_db.find_nim_run.assert_called_once()
        mock_task_db.insert_evaluation.assert_called_once()
        assert mock_task_db.update_evaluation.call_count >= 2  # progress + final

    def test_run_customization_eval_failure(
        self,
        mock_evaluator,
        mock_task_db,
        valid_nim_config,
        mock_settings,
        make_llm_as_judge_config,
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
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB-helper
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id
        mock_task_db.find_customization.return_value = {
            "workload_id": "test-workload",
            "customized_model": "test-model-custom",
        }

        # Configure mock evaluator to fail
        mock_evaluator.run_evaluation.side_effect = Exception("Customization evaluation failed")

        run_customization_eval(previous_result)

        # Verify error handling
        mock_task_db.update_evaluation.assert_called_with(
            ANY,
            {
                "error": "Error running customized-eval evaluation: Customization evaluation failed",
                "finished_at": ANY,
                "progress": 0.0,
            },
        )

    def test_run_customization_eval_customization_disabled(
        self, mock_evaluator, mock_task_db, make_llm_as_judge_config
    ):
        """Test run_customization_eval when customization is disabled (lines 808-809)."""
        nim_id = ObjectId()

        # Create NIM config with customization disabled
        from src.api.models import NIMConfig

        nim_config_disabled = NIMConfig(
            model_name="external-nim-model",
            context_length=2048,
            gpus=1,
            pvc_size="10Gi",
            tag="latest",
            registry_base="nvcr.io/nim",
            customization_enabled=False,  # This triggers lines 808-809
        )

        # Create customization result (even though customization is disabled)
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
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=nim_config_disabled,  # Customization disabled
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            evaluations={},
            customization=customization,
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": nim_config_disabled.model_name,
        }

        result = run_customization_eval(previous_result)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Should return the same result without attempting evaluation (lines 808-809)
        assert result == previous_result
        assert result.nim.customization_enabled is False

        # Verify no evaluation operations occurred
        mock_evaluator.run_evaluation.assert_not_called()
        mock_task_db.insert_evaluation.assert_not_called()
        mock_task_db.find_customization.assert_not_called()

    def test_run_customization_eval_no_customization_model(
        self, mock_evaluator, mock_task_db, valid_nim_config, make_llm_as_judge_config
    ):
        """Test run_customization_eval when no customization model available."""
        nim_id = ObjectId()

        # Create customization without model_name
        customization_no_model = CustomizationResult(
            job_id="test-job",
            model_name=None,  # No model name - will trigger error
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            percent_done=100.0,
            epochs_completed=1,
            steps_completed=100,
        )

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            evaluations={},
            customization=customization_no_model,
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }

        result = run_customization_eval(previous_result)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Verify error was set on the result
        assert result.error is not None
        assert "Error running customization evaluation" in result.error
        assert "No customized model available for evaluation" in result.error

        # Verify no evaluation operations occurred
        mock_evaluator.run_evaluation.assert_not_called()
        mock_task_db.insert_evaluation.assert_not_called()
        mock_task_db.find_customization.assert_not_called()

    def test_run_customization_eval_find_customization_failure(
        self, mock_evaluator, mock_task_db, valid_nim_config, make_llm_as_judge_config
    ):
        """Test run_customization_eval when find_customization fails."""
        nim_id = ObjectId()
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
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            evaluations={},
            customization=customization,
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        # Make find_customization fail
        mock_task_db.find_customization.side_effect = Exception(
            "Database error finding customization"
        )

        result = run_customization_eval(previous_result)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Verify error was set on the result (not via update_evaluation)
        assert result.error is not None
        assert "Error running customization evaluation" in result.error
        assert "Database error finding customization" in result.error

        # Verify find_customization was called
        mock_task_db.find_customization.assert_called_once()

    def test_run_customization_eval_customization_not_found(
        self, mock_evaluator, mock_task_db, valid_nim_config, make_llm_as_judge_config
    ):
        """Test run_customization_eval when customization document not found."""
        nim_id = ObjectId()
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
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            evaluations={},
            customization=customization,
            llm_judge_config=make_llm_as_judge_config,
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_id,
            "model_name": valid_nim_config.model_name,
        }
        # Make find_customization return None (not found)
        mock_task_db.find_customization.return_value = None

        result = run_customization_eval(previous_result)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Verify error was set on the result
        assert result.error is not None
        assert "Error running customization evaluation" in result.error
        assert "No customization found for model test-model-custom" in result.error

        # Verify find_customization was called
        mock_task_db.find_customization.assert_called_once()

        # Verify no evaluation operations occurred
        mock_evaluator.run_evaluation.assert_not_called()
        mock_task_db.insert_evaluation.assert_not_called()

    def test_run_customization_eval_skip_due_to_previous_error(
        self, mock_evaluator, mock_task_db, valid_nim_config, make_llm_as_judge_config
    ):
        """Test run_customization_eval skips execution when previous task has error."""
        customization = CustomizationResult(
            job_id="test-job",
            model_name="test-model-custom",
            started_at=datetime.utcnow(),
            finished_at=datetime.utcnow(),
            percent_done=100.0,
            epochs_completed=1,
            steps_completed=100,
        )

        # Create previous result with an error
        previous_result_with_error = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=str(ObjectId()),
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            evaluations={},
            customization=customization,
            llm_judge_config=make_llm_as_judge_config,
            error="Previous task failed with error",  # This will cause the stage to be skipped
        )

        result = run_customization_eval(previous_result_with_error)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Should return the same result without attempting evaluation
        assert result == previous_result_with_error
        assert result.error == "Previous task failed with error"

        # Verify no evaluation operations occurred
        mock_evaluator.run_evaluation.assert_not_called()
        mock_task_db.insert_evaluation.assert_not_called()
        mock_task_db.find_nim_run.assert_not_called()
        mock_task_db.find_customization.assert_not_called()


class TestGenericEvaluation:
    """Tests for generic evaluation functionality."""

    def test_run_generic_eval_cancellation(self, mock_evaluator, mock_task_db, valid_nim_config):
        """Test run_generic_eval when job is cancelled at the start."""
        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            llm_judge_config=LLMJudgeConfig(deployment_type="local", model_name="test-judge"),
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "model_name": valid_nim_config.model_name,
        }

        with patch("src.tasks.tasks._check_cancellation") as mock_check_cancellation:
            # Configure cancellation check to return True (cancelled)
            mock_check_cancellation.return_value = True

            # Import the function to test
            from src.tasks.tasks import run_generic_eval

            result = run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)

            # Convert result to TaskResult if it's a dict (Celery serialization behavior)
            result = convert_result_to_task_result(result)

            # Verify cancellation was checked
            mock_check_cancellation.assert_called_once_with(flywheel_run_id, raise_error=False)

            # Verify error message in result
            assert result.error is not None
            assert "Task cancelled for flywheel run" in result.error

            # Verify no evaluation operations occurred
            mock_evaluator.run_evaluation.assert_not_called()
            mock_task_db.insert_evaluation.assert_not_called()

    def test_run_generic_eval_missing_dataset(self, mock_evaluator, mock_task_db, valid_nim_config):
        """Test run_generic_eval when required dataset is missing."""
        from src.tasks.tasks import run_generic_eval

        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={},  # Missing BASE dataset
            llm_judge_config=LLMJudgeConfig(deployment_type="local", model_name="test-judge"),
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "model_name": valid_nim_config.model_name,
        }

        result = run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Should have error due to missing dataset - the actual error message includes the DatasetType enum
        assert result.error is not None
        assert "base-eval evaluation" in result.error.lower()
        assert "datasettype.base" in result.error.lower()

        # Verify no evaluation was attempted
        mock_evaluator.run_evaluation.assert_not_called()

    def test_run_generic_eval_progress_callback(
        self, mock_evaluator, mock_task_db, valid_nim_config
    ):
        """Test run_generic_eval progress callback functionality."""
        from src.tasks.tasks import run_generic_eval

        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()
        eval_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            llm_judge_config=LLMJudgeConfig(deployment_type="local", model_name="test-judge"),
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id

        # Configure mock evaluator
        mock_evaluator.run_evaluation.return_value = "job-123"
        mock_evaluator.get_job_uri.return_value = "http://test-uri"
        mock_evaluator.get_evaluation_results.return_value = {
            "tasks": {
                "llm-as-judge": {
                    "metrics": {"llm-judge": {"scores": {"similarity": {"value": 0.95}}}}
                }
            }
        }

        # Mock the wait_for_evaluation to call progress callback
        def mock_wait_for_evaluation(job_id, progress_callback=None):
            if progress_callback:
                # Simulate progress updates
                progress_callback({"progress": 0.5, "status": "running"})
                progress_callback({"progress": 1.0, "status": "completed"})

        mock_evaluator.wait_for_evaluation.side_effect = mock_wait_for_evaluation

        run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)

        # Verify progress callback was used and database was updated
        assert mock_task_db.update_evaluation.call_count >= 2  # At least progress + final updates

    def test_run_generic_eval_nim_run_not_found(
        self, mock_evaluator, mock_task_db, valid_nim_config
    ):
        """Test run_generic_eval when NIM run is not found (lines 492-494)."""
        from src.tasks.tasks import run_generic_eval

        flywheel_run_id = str(ObjectId())

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            llm_judge_config=LLMJudgeConfig(deployment_type="local", model_name="test-judge"),
        )

        # Configure DB manager to return None (NIM run not found) - lines 492-494
        mock_task_db.find_nim_run.return_value = None

        with pytest.raises(ValueError) as exc_info:
            run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)

        # Verify the specific error message from lines 492-494
        assert f"No NIM run found for model {valid_nim_config.model_name}" in str(exc_info.value)

        # Verify find_nim_run was called
        mock_task_db.find_nim_run.assert_called_once()

        # Verify no evaluation operations occurred
        mock_evaluator.run_evaluation.assert_not_called()

    def test_run_generic_eval_skip_due_to_previous_error(
        self, mock_evaluator, mock_task_db, valid_nim_config
    ):
        """Test run_generic_eval skips execution when previous task has error."""
        from src.tasks.tasks import run_generic_eval

        flywheel_run_id = str(ObjectId())

        # Create previous result with an error
        previous_result_with_error = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            llm_judge_config=LLMJudgeConfig(deployment_type="local", model_name="test-judge"),
            error="Previous task failed with error",  # This will cause the stage to be skipped
        )

        result = run_generic_eval(previous_result_with_error, EvalType.BASE, DatasetType.BASE)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Should return the same result without attempting evaluation
        assert result == previous_result_with_error
        assert result.error == "Previous task failed with error"

        # Verify no evaluation operations occurred
        mock_evaluator.run_evaluation.assert_not_called()
        mock_task_db.insert_evaluation.assert_not_called()
        mock_task_db.find_nim_run.assert_not_called()

    def test_run_generic_eval_wait_for_evaluation_exception(
        self, mock_evaluator, mock_task_db, valid_nim_config
    ):
        """Test run_generic_eval when wait_for_evaluation raises exception."""
        from src.tasks.tasks import run_generic_eval

        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()
        eval_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            llm_judge_config=LLMJudgeConfig(deployment_type="local", model_name="test-judge"),
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id

        # Configure mock evaluator to succeed until wait_for_evaluation
        mock_evaluator.run_evaluation.return_value = "job-123"
        mock_evaluator.get_job_uri.return_value = "http://test-uri"
        mock_evaluator.wait_for_evaluation.side_effect = Exception("Wait for evaluation failed")

        result = run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Verify error handling
        mock_task_db.update_evaluation.assert_called_with(
            ANY,
            {
                "error": ANY,
                "finished_at": ANY,
                "progress": 0.0,
            },
        )

        # Verify the error message contains the expected text
        error_call = mock_task_db.update_evaluation.call_args_list[-1]
        assert "Error running base-eval evaluation" in error_call[0][1]["error"]
        assert "Wait for evaluation failed" in error_call[0][1]["error"]

        # Verify error was set on previous_result
        assert result.error is not None
        assert "Wait for evaluation failed" in result.error

    def test_run_generic_eval_evaluator_get_results_failure(
        self, mock_evaluator, mock_task_db, valid_nim_config
    ):
        """Test run_generic_eval when get_evaluation_results fails."""
        from src.tasks.tasks import run_generic_eval

        flywheel_run_id = str(ObjectId())
        nim_run_id = ObjectId()
        eval_id = ObjectId()

        previous_result = TaskResult(
            workload_id="test-workload",
            client_id="test-client",
            flywheel_run_id=flywheel_run_id,
            nim=valid_nim_config,
            workload_type=WorkloadClassification.GENERIC,
            datasets={DatasetType.BASE: "test-base-dataset"},
            llm_judge_config=LLMJudgeConfig(deployment_type="local", model_name="test-judge"),
        )

        # Configure DB manager
        mock_task_db.find_nim_run.return_value = {
            "_id": nim_run_id,
            "model_name": valid_nim_config.model_name,
        }
        mock_task_db.insert_evaluation.return_value = eval_id

        # Configure mock evaluator to succeed until get_evaluation_results
        mock_evaluator.run_evaluation.return_value = "job-123"
        mock_evaluator.get_job_uri.return_value = "http://test-uri"
        mock_evaluator.wait_for_evaluation.return_value = None
        mock_evaluator.get_evaluation_results.side_effect = Exception("Failed to get results")

        result = run_generic_eval(previous_result, EvalType.BASE, DatasetType.BASE)

        # Convert result to TaskResult if it's a dict
        result = convert_result_to_task_result(result)

        # Verify error handling
        mock_task_db.update_evaluation.assert_called_with(
            ANY,
            {
                "error": ANY,
                "finished_at": ANY,
                "progress": 0.0,
            },
        )

        # Verify the error message contains the expected text
        error_call = mock_task_db.update_evaluation.call_args_list[-1]
        assert "Error running base-eval evaluation" in error_call[0][1]["error"]
        assert "Failed to get results" in error_call[0][1]["error"]
