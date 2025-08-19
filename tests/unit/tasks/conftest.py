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

"""Shared fixtures and utilities for task tests."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import (
    DeploymentStatus,
    LLMJudgeConfig,
    NIMConfig,
    NIMRun,
    NIMRunStatus,
    TaskResult,
)
from src.config import settings  # the singleton created in src.config


@pytest.fixture(name="mock_task_db", autouse=True)
def fixture_mock_task_db_manager():
    """Patch the *db_manager* instance used in tasks.py.

    After the recent refactor, Celery tasks no longer access raw pymongo
    collections; instead they delegate everything to the *TaskDBManager*
    helper stored as ``src.tasks.tasks.db_manager``.  Patch that singleton so
    each test can assert against the high-level helper methods.
    """
    with patch("src.tasks.tasks.db_manager") as mock_task_db_manager:
        # Setup default behavior
        mock_task_db_manager.create_nim_run.return_value = ObjectId()
        mock_task_db_manager.insert_evaluation.return_value = ObjectId()
        mock_task_db_manager.insert_customization.return_value = ObjectId()

        # Configure find_nim_run with a valid response
        mock_task_db_manager.find_nim_run.return_value = {
            "flywheel_run_id": ObjectId(),
            "_id": ObjectId(),
            "model_name": "test-model",
            "started_at": datetime.utcnow(),
            "finished_at": datetime.utcnow(),
            "runtime_seconds": 0,
            "deployment_status": DeploymentStatus.PENDING,
        }

        # Collections mocked as attributes
        for collection in [
            "flywheel_runs",
            "nims",
            "evaluations",
            "llm_judge_runs",
            "customizations",
        ]:
            setattr(mock_task_db_manager, collection, MagicMock())

        yield mock_task_db_manager


@pytest.fixture(name="mock_init_db")
def fixture_mock_init_db():
    """Mock the database initialization function to avoid real database interactions."""
    with patch("src.tasks.tasks.init_db") as mock_init_db:
        yield mock_init_db


@pytest.fixture
def mock_evaluator():
    """Fixture to mock Evaluator."""
    with patch("src.tasks.tasks.Evaluator") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture(autouse=True)
def tweak_settings(monkeypatch):
    """Provide deterministic test configuration via the global `settings`."""

    # --- Data-split parameters (fields are *not* frozen) --------------------
    monkeypatch.setattr(settings.data_split_config, "min_total_records", 1, raising=False)
    monkeypatch.setattr(settings.data_split_config, "random_seed", 42, raising=False)
    monkeypatch.setattr(settings.data_split_config, "eval_size", 1, raising=False)
    monkeypatch.setattr(settings.data_split_config, "val_ratio", 0.25, raising=False)
    monkeypatch.setattr(settings.data_split_config, "limit", 100, raising=False)

    # --- NMP namespace (field *is* frozen, so create a new object) ----------
    new_nmp_cfg = settings.nmp_config.model_copy(update={"nmp_namespace": "test-namespace"})
    monkeypatch.setattr(settings, "nmp_config", new_nmp_cfg, raising=True)

    yield


@pytest.fixture(name="mock_settings")
def fixture_mock_settings():
    """Return the globally patched `settings` instance used in the tests."""
    return settings


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
def sample_customizer_config():
    """Fixture to create a sample CustomizerConfig instance."""
    from src.config import CustomizerConfig

    return CustomizerConfig(
        target="test-model@v1.0.0",
        gpus=1,
        num_nodes=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        use_sequence_parallel=False,
        micro_batch_size=1,
        training_precision="bf16-mixed",
        max_seq_length=2048,
    )


@pytest.fixture
def valid_nim_config(sample_customizer_config):
    """Fixture to create a valid NIMConfig instance."""
    return NIMConfig(
        model_name="test-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
        registry_base="nvcr.io/nim",
        customization_enabled=True,
        customizer_configs=sample_customizer_config,
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


@pytest.fixture
def make_remote_judge_config():
    return LLMJudgeConfig(
        type="remote",
        url="http://test-remote-url/v1/chat/completions",
        model_name="remote-model-id",
        api_key_env="TEST_API_KEY_ENV",
        api_key="test-api-key",
    )


@pytest.fixture
def make_llm_as_judge_config():
    return LLMJudgeConfig(
        deployment_type="local",
        model_name="test-judge-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
        registry_base="nvcr.io/nim",
        customization_enabled=False,
    )


def convert_result_to_task_result(result):
    """Helper method to convert result to TaskResult if it's a dictionary."""
    if isinstance(result, dict):
        return TaskResult(**result)
    return result
