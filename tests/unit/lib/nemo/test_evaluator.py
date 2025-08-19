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

from src.api.models import EvalType, NIMEvaluation, ToolEvalType, WorkloadClassification
from src.lib.nemo.evaluator import Evaluator

# Mock time.sleep globally for all tests in this module
pytestmark = pytest.mark.usefixtures("mock_sleep_globally", "mock_settings_globally")


@pytest.fixture(autouse=True)
def mock_sleep_globally():
    """Automatically mock time.sleep for all tests in this module."""
    with patch("time.sleep"):
        yield


@pytest.fixture(autouse=True)
def mock_settings_globally():
    """Automatically mock settings for all tests in this module."""
    with patch("src.lib.nemo.evaluator.settings") as mock_settings:
        # Default mock settings configuration
        mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
        mock_settings.nmp_config.nmp_namespace = "test-namespace"

        # Mock judge config
        mock_judge_cfg = MagicMock()
        mock_judge_cfg.is_remote.return_value = False
        mock_local_nim_cfg = MagicMock()
        mock_local_nim_cfg.model_name = "test-judge-model"
        mock_judge_cfg.get_local_nim_config.return_value = mock_local_nim_cfg
        mock_settings.judge_model_config = mock_judge_cfg

        yield mock_settings


@pytest.fixture
def evaluator() -> Evaluator:
    """Standard evaluator instance with default test configuration."""
    return Evaluator(judge_model_config={"model": "test-judge"})


@pytest.fixture
def mock_evaluation() -> NIMEvaluation:
    return NIMEvaluation(
        nim_id=ObjectId(),  # Generate a new ObjectId
        eval_type=EvalType.BASE,
        scores={"base": 0.0},
        started_at=datetime.utcnow(),
        finished_at=None,
        runtime_seconds=0.0,
        progress=0.0,
    )


@pytest.fixture
def sample_flywheel_run_id():
    """Fixture to provide a valid ObjectId string for tests."""
    return str(ObjectId())


class TestWaitForEvaluation:
    """Test class for wait_for_evaluation method and related functionality."""

    def test_wait_for_evaluation_created_state(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test handling of created state in wait_for_evaluation"""
        job_id = "test-job-id"

        # Mock the job status response for created state
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "created",
            "status_details": {"progress": 0},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(TimeoutError):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=0.001,  # Very short timeout to trigger immediately
                    )

    def test_wait_for_evaluation_created_state_with_callback(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test handling of created state with progress callback in wait_for_evaluation"""
        job_id = "test-job-id"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        # Mock the job status response for created state
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "created",
            "status_details": {"progress": 0},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(TimeoutError):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=0.001,  # Very short timeout to trigger immediately
                        progress_callback=progress_callback,
                    )

                # Should have progress callback for created state
                assert len(progress_updates) >= 2  # At least one created + one timeout error
                assert progress_updates[0]["progress"] == 0.0  # Created state progress

    def test_wait_for_evaluation_running_state(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test handling of running state in wait_for_evaluation"""
        job_id = "test-job-id"

        # Mock the job status response for running state
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "running",
            "status_details": {"progress": 50},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(TimeoutError):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=0.001,  # Very short timeout to trigger immediately
                    )

    def test_wait_for_evaluation_completed_state(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test handling of completed state in wait_for_evaluation"""
        job_id = "test-job-id"

        # Mock the job status response for completed state
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "completed",
            "status_details": {"progress": 100},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                result = evaluator.wait_for_evaluation(
                    job_id=job_id,
                    flywheel_run_id=sample_flywheel_run_id,
                    polling_interval=1,
                    timeout=1,
                )
                assert result["status"] == "completed"

    def test_wait_for_evaluation_error_state(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test handling of error state in wait_for_evaluation"""
        job_id = "test-job-id"

        # Mock the job status response for error state
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "failed",
            "status_details": {"error": "Test error"},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(Exception) as exc_info:
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=1,
                    )
                assert "Test error" in str(exc_info.value)

    def test_wait_for_evaluation_timeout(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test timeout in wait_for_evaluation"""
        job_id = "test-job-id"

        # Mock the job status response for running state (never completes)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "running",
            "status_details": {"progress": 50},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(TimeoutError):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=0.001,  # Very short timeout to trigger immediately
                    )

    def test_wait_for_evaluation_none_progress(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test handling of None progress in wait_for_evaluation"""
        job_id = "test-job-id"

        # Mock the job status response with None progress
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "running",
            "status_details": {"progress": None},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(TimeoutError):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=0.001,  # Very short timeout to trigger immediately
                    )

    def test_wait_for_evaluation_unknown_status(
        self, evaluator: Evaluator, sample_flywheel_run_id
    ) -> None:
        """Test handling of unknown status in wait_for_evaluation"""
        job_id = "test-job-id"

        # Mock the job status response with unknown status
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "unknown_status",
            "status_details": {"some_detail": "value"},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(Exception, match="Job status: unknown_status"):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=10,
                    )


class TestJudgeConfiguration:
    """Test class for judge model configuration functionality."""

    def make_remote_judge_config(self):
        from src.config import LLMJudgeConfig

        return LLMJudgeConfig(
            deployment_type="remote",
            url="http://test-remote-url/v1/chat/completions",
            model_name="remote-model-id",
            api_key_env="TEST_API_KEY_ENV",
            api_key="test-api-key",
        )

    def make_local_judge_config(self):
        from src.config import LLMJudgeConfig

        return LLMJudgeConfig(
            deployment_type="local",
            model_name="local-model-name",
            tag="test-tag",
            context_length=1234,
            gpus=1,
            pvc_size="10Gi",
            registry_base="test-registry",
            customization_enabled=False,
        )

    def test_evaluator_uses_remote_judge_config(self, monkeypatch):
        from src.lib.nemo.evaluator import Evaluator

        remote_cfg = self.make_remote_judge_config()
        judge_model_config = remote_cfg.judge_model_config()
        monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)
        evaluator = Evaluator(judge_model_config=judge_model_config)
        # Should use the remote config dict
        assert isinstance(judge_model_config, dict)
        assert evaluator.judge_model_config["api_endpoint"]["url"] == remote_cfg.url
        assert evaluator.judge_model_config["api_endpoint"]["model_id"] == remote_cfg.model_name
        assert evaluator.judge_model_config["api_endpoint"]["api_key"] == remote_cfg.api_key

    def test_evaluator_uses_local_judge_config(self, monkeypatch):
        from src.lib.nemo.evaluator import Evaluator

        local_cfg = self.make_local_judge_config()
        judge_model_config = local_cfg.judge_model_config()
        monkeypatch.setattr("src.config.settings.llm_judge_config", local_cfg)
        evaluator = Evaluator(judge_model_config=judge_model_config)
        # Should use the local model name
        assert evaluator.judge_model_config == local_cfg.model_name

    def test_evaluator_prefers_explicit_llm_judge_config(self, monkeypatch):
        from src.lib.nemo.evaluator import Evaluator

        remote_cfg = self.make_remote_judge_config()
        judge_model_config = remote_cfg.judge_model_config()
        monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)

        # If you pass an explicit NIMConfig, it should use that model_name
        model_name = "explicit-model"
        judge_model_config["api_endpoint"]["model_id"] = model_name
        evaluator = Evaluator(judge_model_config=judge_model_config)
        assert evaluator.judge_model_config["api_endpoint"]["model_id"] == "explicit-model"


@pytest.fixture
def evaluator_instance():
    """
    Provides an instance of the Evaluator with default mocked settings.
    """
    return Evaluator()


@pytest.fixture
def mock_response():
    """Provides a mock successful response for requests.post."""
    response = MagicMock()
    response.status_code = 201
    response.json.return_value = {"id": "mock-job-id"}
    return response


class TestEvaluator:
    def test_init_without_nemo_url(self, mock_settings_globally):
        """Test initialization fails without nemo_base_url."""
        mock_settings_globally.nmp_config.nemo_base_url = None
        with pytest.raises(AssertionError, match="nemo_base_url must be set in config"):
            Evaluator()

    def test_init_with_all_params(self):
        """Test initialization with all parameters."""
        judge_config = {"model": "test-judge"}

        evaluator = Evaluator(
            judge_model_config=judge_config,
            include_tools=True,
            include_tool_choice=True,
            include_nvext=True,
        )

        assert evaluator.nemo_url == "http://test-nemo-url"
        assert evaluator.namespace == "test-namespace"
        assert evaluator.judge_model_config == judge_config
        assert evaluator.include_tools is True
        assert evaluator.include_tool_choice is True
        assert evaluator.include_nvext is True


class TestEvaluatorBasicMethods:
    def test_get_job_uri(self, evaluator):
        """Test get_job_uri method."""
        job_id = "test-job-123"
        expected_uri = "http://test-nemo-url/v1/evaluation/jobs/test-job-123"
        assert evaluator.get_job_uri(job_id) == expected_uri

    def test_get_evaluation_status_running(self, evaluator):
        """Test get_evaluation_status with running status."""
        job_id = "test-job"
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "running", "status_details": {"progress": 75}}

        with patch("requests.get", return_value=mock_response):
            result = evaluator.get_evaluation_status(job_id)
            assert result["status"] == "running"
            assert result["status_details"]["progress"] == 75

    def test_get_evaluation_status_completed(self, evaluator):
        """Test get_evaluation_status with completed status."""
        job_id = "test-job"
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "completed",
            "status_details": {"progress": 100},
        }

        with patch("requests.get", return_value=mock_response):
            result = evaluator.get_evaluation_status(job_id)
            assert result["status"] == "completed"

    def test_get_evaluation_results_success(self, evaluator):
        """Test successful get_evaluation_results."""
        job_id = "test-job"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": "test-results"}

        with patch("requests.get", return_value=mock_response):
            result = evaluator.get_evaluation_results(job_id)
            assert result == {"results": "test-results"}

    def test_get_evaluation_results_failure(self, evaluator):
        """Test get_evaluation_results with failure."""
        job_id = "test-job"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(AssertionError, match="Failed to get evaluation results"):
                evaluator.get_evaluation_results(job_id)

    def test_delete_evaluation_job_success(self, evaluator):
        """Test successful delete_evaluation_job."""
        job_id = "test-job"
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response):
            evaluator.delete_evaluation_job(job_id)  # Should not raise

    def test_delete_evaluation_job_success_204(self, evaluator):
        """Test successful delete_evaluation_job with 204 status."""
        job_id = "test-job"
        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("requests.delete", return_value=mock_response):
            evaluator.delete_evaluation_job(job_id)  # Should not raise

    def test_delete_evaluation_job_failure(self, evaluator):
        """Test delete_evaluation_job with failure."""
        job_id = "test-job"
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        with patch("requests.delete", return_value=mock_response):
            with pytest.raises(Exception, match="Failed to delete evaluation job"):
                evaluator.delete_evaluation_job(job_id)


class TestWaitForEvaluationCallbacks:
    def test_wait_for_evaluation_with_progress_callback(self, evaluator, sample_flywheel_run_id):
        """Test wait_for_evaluation with progress callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        # Mock running then completed
        call_count = 0

        def mock_get_response(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "status": "running" if call_count == 1 else "completed",
                "status_details": {"progress": 50 if call_count == 1 else 100},
            }
            return mock_response

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", side_effect=mock_get_response):
                result = evaluator.wait_for_evaluation(
                    job_id=job_id,
                    flywheel_run_id=sample_flywheel_run_id,
                    polling_interval=1,
                    timeout=10,
                    progress_callback=progress_callback,
                )

                assert result["status"] == "completed"
                assert len(progress_updates) == 2
                assert progress_updates[0]["progress"] == 50.0
                assert progress_updates[1]["progress"] == 100.0

    def test_wait_for_evaluation_timeout_with_callback(self, evaluator, sample_flywheel_run_id):
        """Test wait_for_evaluation timeout with progress callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "running", "status_details": {"progress": 25}}

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(TimeoutError):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=0.001,  # Very short timeout
                        progress_callback=progress_callback,
                    )

                # Should have progress updates including error
                assert len(progress_updates) >= 2
                # Last update should contain error
                assert "error" in progress_updates[-1]
                assert progress_updates[-1]["progress"] == 0.0

    def test_wait_for_evaluation_error_with_callback(self, evaluator, sample_flywheel_run_id):
        """Test wait_for_evaluation error with progress callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "status": "failed",
            "status_details": {"error": "Test error message"},
        }

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(Exception, match="Test error message"):
                    evaluator.wait_for_evaluation(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        polling_interval=1,
                        timeout=10,
                        progress_callback=progress_callback,
                    )

                # Should have error callback
                assert len(progress_updates) == 1
                assert "error" in progress_updates[0]
                assert progress_updates[0]["progress"] == 0.0

    def test_wait_for_evaluation_cancellation(self, evaluator, sample_flywheel_run_id):
        """Test wait_for_evaluation with cancellation."""
        job_id = "test-job"

        # Mock cancellation to raise an exception immediately
        with patch("src.lib.nemo.evaluator.check_cancellation") as mock_check_cancellation:
            mock_check_cancellation.side_effect = Exception("Evaluation cancelled")
            with pytest.raises(Exception, match="Evaluation cancelled"):
                evaluator.wait_for_evaluation(
                    job_id=job_id,
                    flywheel_run_id=sample_flywheel_run_id,
                    polling_interval=1,
                    timeout=10,
                )


class TestEvaluatorTemplates:
    def test_get_template_with_all_options(self):
        """Test get_template with all options enabled."""
        evaluator_with_options = Evaluator(
            judge_model_config={"model": "test-judge"},
            include_tools=True,
            include_tool_choice=True,
            include_nvext=True,
        )
        template = evaluator_with_options.get_template(tool_call=False)

        assert "messages" in template["template"]
        assert "tools" in template["template"]
        assert "tool_choice" in template["template"]
        assert "nvext" in template["template"]

    def test_get_template_tool_call_true(self):
        """Test get_template with tool_call=True."""
        evaluator_with_options = Evaluator(
            judge_model_config={"model": "test-judge"},
            include_tools=True,
            include_tool_choice=True,
            include_nvext=True,
        )
        template = evaluator_with_options.get_template(tool_call=True)

        assert "messages" in template["template"]
        assert "tools" in template["template"]
        assert template["template"]["tool_choice"] == "required"

    def test_get_template_minimal(self):
        """Test get_template with minimal options."""
        evaluator = Evaluator(
            judge_model_config={"model": "test-judge"},
            include_tools=False,
            include_tool_choice=False,
            include_nvext=False,
        )

        template = evaluator.get_template(tool_call=False)

        assert "messages" in template["template"]
        assert "tools" not in template["template"]
        assert "tool_choice" not in template["template"]
        assert "nvext" not in template["template"]


class TestRunEvaluationErrorHandling:
    def test_run_evaluation_request_failure(self, evaluator):
        """Test run_evaluation with request failure."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(AssertionError, match="Failed to launch evaluation job"):
                evaluator.run_evaluation(
                    dataset_name="test-dataset",
                    workload_type=WorkloadClassification.GENERIC,
                    target_model="test-model",
                    test_file="test.jsonl",
                )

    def test_run_evaluation_tool_calling_judge(self, evaluator):
        """Test run_evaluation with tool calling judge."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "test-job-id"}

        with patch("requests.post", return_value=mock_response):
            job_id = evaluator.run_evaluation(
                dataset_name="test-dataset",
                workload_type=WorkloadClassification.TOOL_CALLING,
                target_model="test-model",
                test_file="test.jsonl",
                tool_eval_type=ToolEvalType.TOOL_CALLING_JUDGE,
                limit=50,
            )

            assert job_id == "test-job-id"


class TestRunEvaluation:
    @pytest.mark.parametrize(
        "test_params",
        [
            # Test case 1: GENERIC workload
            {
                "workload_type": WorkloadClassification.GENERIC,
                "tool_eval_type": None,
                "dataset_name": "test-dataset",
                "target_model": "meta/llama-3.3-70b-instruct",
                "test_file": "test.jsonl",
                "limit": 75,
                "expected_config_method": "get_llm_as_judge_config",
                "should_raise_error": False,
            },
            # Test case 2: TOOL_CALLING workload with missing tool_eval_type
            {
                "workload_type": WorkloadClassification.TOOL_CALLING,
                "tool_eval_type": None,
                "dataset_name": "test-dataset",
                "target_model": "meta/llama-3.3-70b-instruct",
                "test_file": "test-tools.jsonl",
                "limit": 50,
                "expected_error": ValueError,
                "expected_error_msg": "tool_eval_type must be provided for tool calling workload",
                "should_raise_error": True,
            },
            # Test case 3: TOOL_CALLING workload with TOOL_CALLING_METRIC
            {
                "workload_type": WorkloadClassification.TOOL_CALLING,
                "tool_eval_type": ToolEvalType.TOOL_CALLING_METRIC,
                "dataset_name": "test-tool-dataset",
                "target_model": "custom/tool-model",
                "test_file": "tools-data.jsonl",
                "limit": 99,
                "expected_config_method": "get_tool_calling_config",
                "should_raise_error": False,
            },
        ],
    )
    def test_run_evaluation_scenarios(self, evaluator_instance, test_params):
        # Mock the specific evaluation jobs endpoint
        evaluation_jobs_endpoint = f"{evaluator_instance.nemo_url}/v1/evaluation/jobs"

        with patch("src.lib.nemo.evaluator.requests.post") as mock_post:
            # Configure mock response for the specific endpoint
            mock_response = MagicMock()
            mock_response.status_code = 201
            mock_response.json.return_value = {
                "id": "mock-job-id",
                "status": "created",
                "created_at": "2024-03-20T10:00:00Z",
            }
            mock_post.return_value = mock_response

            if test_params["should_raise_error"]:
                with pytest.raises(
                    test_params["expected_error"], match=test_params["expected_error_msg"]
                ):
                    evaluator_instance.run_evaluation(
                        dataset_name=test_params["dataset_name"],
                        workload_type=test_params["workload_type"],
                        target_model=test_params["target_model"],
                        test_file=test_params["test_file"],
                        tool_eval_type=test_params["tool_eval_type"],
                        limit=test_params["limit"],
                    )
                return

            config_method = getattr(evaluator_instance, test_params["expected_config_method"])
            expected_config_payload = config_method(
                dataset_name=test_params["dataset_name"],
                test_file=test_params["test_file"],
                limit=test_params["limit"],
            )

            job_id = evaluator_instance.run_evaluation(
                dataset_name=test_params["dataset_name"],
                workload_type=test_params["workload_type"],
                tool_eval_type=test_params["tool_eval_type"],
                target_model=test_params["target_model"],
                test_file=test_params["test_file"],
                limit=test_params["limit"],
            )

            assert job_id == "mock-job-id"

            # Verify the POST request was made to the correct endpoint
            mock_post.assert_called_once_with(
                evaluation_jobs_endpoint,
                json={
                    "config": expected_config_payload,
                    "target": {"type": "model", "model": test_params["target_model"]},
                },
            )


@pytest.mark.parametrize(
    "limit",
    [
        75,  # Normal positive limit
        0,  # Zero limit
        None,  # No limit
    ],
)
def test_run_evaluation_limit_propagation(evaluator_instance, mock_response, limit):
    """Test that run_evaluation correctly passes limit to config methods."""
    # Mock the network request
    with patch("src.lib.nemo.evaluator.requests.post", return_value=mock_response):
        # Mock the config method
        with patch.object(evaluator_instance, "get_llm_as_judge_config") as mock_config_method:
            # Set up mock return value
            mock_config_method.return_value = {"type": "mock-config"}

            # Call run_evaluation with limit
            evaluator_instance.run_evaluation(
                dataset_name="test-dataset",
                workload_type=WorkloadClassification.GENERIC,
                target_model="test-model",
                test_file="test.jsonl",
                limit=limit,
            )

            # Verify config method was called with correct limit
            mock_config_method.assert_called_once_with(
                dataset_name="test-dataset",
                test_file="test.jsonl",
                limit=limit,
            )
