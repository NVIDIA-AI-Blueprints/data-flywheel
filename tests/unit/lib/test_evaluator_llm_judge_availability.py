# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Evaluator.validate_llm_judge_availability.

These tests focus on the different code-paths involved when verifying that the
configured LLM judge can be reached.  The scenarios covered are:

1. Local judge configuration - no outbound request should be issued.
2. Remote judge happy-path - successful health-check call.
3. Remote judge with missing critical configuration values.
4. Remote judge that responds with a non-200 HTTP status code.
5. Remote judge that responds with 200 OK but with an unexpected payload.
"""

from unittest.mock import MagicMock

import pytest

from src.config import LLMJudgeConfig
from src.lib.nemo.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def make_remote_judge_config(**overrides):
    """Return a fully-populated *remote* LLMJudgeConfig that can be tweaked via
    keyword overrides in the individual test-cases."""

    cfg_dict = {
        "type": "remote",
        "url": "http://remote-judge.test/v1/chat/completions",
        "model_id": "remote-model-id",
        "api_key_env": "TEST_API_KEY_ENV",
        "api_key": "super-secret-key",
    }
    cfg_dict.update(overrides)
    return LLMJudgeConfig(**cfg_dict)


def make_local_judge_config():
    """Return a *local* LLMJudgeConfig - used to ensure the remote check is skipped."""

    return LLMJudgeConfig(
        type="local",
        model_name="local-judge-model",
        tag="test-tag",
        context_length=4096,
        gpus=1,
        pvc_size="10Gi",
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_validate_llm_judge_availability_local_skips_remote_call(monkeypatch):
    """When the judge is local no HTTP request should be made."""

    # Arrange - patch the global settings with a *local* judge config
    local_cfg = make_local_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", local_cfg)

    # Patch requests.post so we can assert it is *never* invoked
    mock_post = MagicMock()
    monkeypatch.setattr("requests.post", mock_post)

    # Patch Evaluator.spin_up_llm_judge to avoid external DMS interactions
    mock_spin_up = MagicMock(return_value=True)
    monkeypatch.setattr(Evaluator, "spin_up_llm_judge", mock_spin_up)

    # Act - should return True and avoid any HTTP calls
    result = Evaluator().validate_llm_judge_availability()

    # Assert
    mock_post.assert_not_called()
    mock_spin_up.assert_called_once()
    assert result is True


def test_validate_llm_judge_availability_remote_happy_path(monkeypatch):
    """A correctly configured remote judge that returns 200/JSON containing
    a ``choices`` field should pass the health-check without raising."""

    remote_cfg = make_remote_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)

    # Craft a successful mock response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "hi"}}]}

    mock_post = MagicMock(return_value=mock_response)
    monkeypatch.setattr("requests.post", mock_post)

    # Act - should return True indicating the judge is reachable
    result = Evaluator().validate_llm_judge_availability()

    # Assert - verify the outbound request was built correctly and a successful result is propagated
    mock_post.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == remote_cfg.model_id
    assert kwargs["headers"]["Authorization"] == f"Bearer {remote_cfg.api_key}"
    assert result is True


@pytest.mark.parametrize(
    "url,model_id,missing_field",
    [
        (None, "some-model", "url"),
        ("http://remote-judge.test/v1/chat/completions", None, "model_id"),
    ],
)
def test_validate_llm_judge_availability_remote_missing_config(
    monkeypatch, url, model_id, missing_field
):
    """Missing *url* or *model_id* values should raise a RuntimeError before any HTTP call."""

    remote_cfg = make_remote_judge_config(url=url, model_id=model_id)
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)

    mock_post = MagicMock()
    monkeypatch.setattr("requests.post", mock_post)

    with pytest.raises(RuntimeError, match="missing 'url' or 'model_id'"):
        Evaluator().validate_llm_judge_availability()

    mock_post.assert_not_called()


@pytest.mark.parametrize("status_code", [400, 500])
def test_validate_llm_judge_availability_remote_http_error(monkeypatch, status_code):
    """Non-200 status codes returned by the judge endpoint should result in a health-check failure (False)."""

    remote_cfg = make_remote_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)

    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.text = "error"
    mock_post = MagicMock(return_value=mock_response)
    monkeypatch.setattr("requests.post", mock_post)

    # Act
    result = Evaluator().validate_llm_judge_availability()

    # Assert - should return False indicating the judge is not reachable
    assert result is False


def test_validate_llm_judge_availability_remote_unexpected_payload(monkeypatch):
    """If the response JSON lacks a ``choices`` key the health-check should return False."""

    remote_cfg = make_remote_judge_config()
    monkeypatch.setattr("src.config.settings.llm_judge_config", remote_cfg)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"unexpected": "structure"}

    mock_post = MagicMock(return_value=mock_response)
    monkeypatch.setattr("requests.post", mock_post)

    # Act
    result = Evaluator().validate_llm_judge_availability()

    # Assert - should return False due to unexpected payload structure
    assert result is False
