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
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import EvalType, NIMEvaluation
from src.config import settings
from src.lib.nemo.evaluator import Evaluator


@pytest.fixture
def evaluator() -> Evaluator:
    with patch.dict("os.environ", {"NEMO_URL": "http://test-nemo-url"}):
        return Evaluator(llm_judge_config=settings.llm_judge_config)


@pytest.fixture
def mock_evaluation() -> NIMEvaluation:
    return NIMEvaluation(
        nim_id=ObjectId(),  # Generate a new ObjectId
        eval_type=EvalType.BASE,
        scores={"base": 0.0},
        started_at=datetime.now(timezone.utc),
        finished_at=None,
        runtime_seconds=0.0,
        progress=0.0,
    )


def test_wait_for_evaluation_created_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of 'created' state in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response for 'created' state
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "created",
        "status_details": {"message": None, "task_status": {}, "progress": None},
    }

    with patch("requests.get", return_value=mock_response):
        # This should not raise an exception and should continue polling
        evaluator.wait_for_evaluation(
            job_id=job_id,
            evaluation=mock_evaluation,
            polling_interval=1,
            timeout=1,
            progress_callback=progress_callback,
        )

        # Verify progress callback was called with 0 progress
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 0.0


def test_wait_for_evaluation_running_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of 'running' state in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response for 'running' state with progress
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "running", "status_details": {"progress": 50}}

    with patch("requests.get", return_value=mock_response):
        evaluator.wait_for_evaluation(
            job_id=job_id,
            evaluation=mock_evaluation,
            polling_interval=1,
            timeout=1,
            progress_callback=progress_callback,
        )

        # Verify progress callback was called with correct progress
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 50.0


def test_wait_for_evaluation_completed_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of 'completed' state in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response for 'completed' state
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "completed", "status_details": {"progress": 100}}

    with patch("requests.get", return_value=mock_response):
        result = evaluator.wait_for_evaluation(
            job_id=job_id,
            evaluation=mock_evaluation,
            polling_interval=1,
            timeout=1,
            progress_callback=progress_callback,
        )

        # Verify progress callback was called with 100% progress
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 100.0
        # Verify the job data was returned
        assert result["status"] == "completed"


def test_wait_for_evaluation_error_state(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of error state in wait_for_evaluation"""
    job_id = "test-job-id"

    # Mock the job status response for error state
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "status": "failed",
        "status_details": {"error": "Test error"},
    }

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(Exception) as exc_info:
            evaluator.wait_for_evaluation(
                job_id=job_id, evaluation=mock_evaluation, polling_interval=1, timeout=1
            )

        assert "Job status: failed" in str(exc_info.value)


def test_wait_for_evaluation_timeout(evaluator: Evaluator, mock_evaluation: NIMEvaluation) -> None:
    """Test timeout handling in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response to always return 'running'
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "running", "status_details": {"progress": 50}}

    with patch("requests.get", return_value=mock_response):
        result = evaluator.wait_for_evaluation(
            job_id=job_id,
            evaluation=mock_evaluation,
            polling_interval=1,
            timeout=1,
            progress_callback=progress_callback,
        )

        # Verify timeout status was returned
        assert result["status"] == "timeout"
        # Verify progress callback was called with 0 progress on timeout
        assert len(progress_updates) > 0
        assert progress_updates[-1]["progress"] == 0.0


def test_wait_for_evaluation_none_progress(
    evaluator: Evaluator, mock_evaluation: NIMEvaluation
) -> None:
    """Test handling of None progress value in wait_for_evaluation"""
    job_id = "test-job-id"
    progress_updates: list[dict[str, Any]] = []

    def progress_callback(update_data: dict[str, Any]) -> None:
        progress_updates.append(update_data)

    # Mock the job status response with None progress
    mock_response = MagicMock()
    mock_response.json.return_value = {"status": "running", "status_details": {"progress": None}}

    with patch("requests.get", return_value=mock_response):
        evaluator.wait_for_evaluation(
            job_id=job_id,
            evaluation=mock_evaluation,
            polling_interval=1,
            timeout=1,
            progress_callback=progress_callback,
        )

        # Verify progress callback was called with 0 progress when progress is None
        assert len(progress_updates) > 0
        assert progress_updates[0]["progress"] == 0.0


def make_remote_judge_config():
    from src.config import LLMJudgeConfig

    return LLMJudgeConfig(
        type="remote",
        url="http://test-remote-url/v1/chat/completions",
        model_id="remote-model-id",
        api_key_env="TEST_API_KEY_ENV",
        api_key="test-api-key",
    )


def make_local_judge_config():
    from src.config import LLMJudgeConfig

    return LLMJudgeConfig(
        type="local",
        model_name="local-model-name",
        tag="test-tag",
        context_length=1234,
        gpus=1,
        pvc_size="10Gi",
        registry_base="test-registry",
        customization_enabled=True,
    )


def test_evaluator_uses_remote_judge_config(monkeypatch):
    from src.lib.nemo.evaluator import Evaluator

    remote_cfg = make_remote_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)
    evaluator = Evaluator()
    # Should use the remote config dict
    assert isinstance(evaluator.judge_model_config, dict)
    assert evaluator.judge_model_config["api_endpoint"]["url"] == remote_cfg.url
    assert evaluator.judge_model_config["api_endpoint"]["model_id"] == remote_cfg.model_id
    assert evaluator.judge_model_config["api_endpoint"]["api_key"] == remote_cfg.api_key


def test_evaluator_uses_local_judge_config(monkeypatch):
    from src.lib.nemo.evaluator import Evaluator

    local_cfg = make_local_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", local_cfg)
    evaluator = Evaluator()
    # Should use the local model name
    assert evaluator.judge_model_config == local_cfg.model_name


def test_evaluator_prefers_explicit_llm_judge_config(monkeypatch):
    from src.lib.nemo.evaluator import Evaluator

    remote_cfg = make_remote_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)

    # If you pass an explicit NIMConfig, it should use that model_name
    class DummyNIMConfig:
        model_name = "explicit-model"

    evaluator = Evaluator(llm_judge_config=DummyNIMConfig())
    assert evaluator.judge_model_config == "explicit-model"
