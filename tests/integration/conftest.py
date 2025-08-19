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
import os
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import FlywheelRun, TaskResult
from src.config import settings

os.environ["ELASTICSEARCH_URL"] = "http://localhost:9200"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
os.environ["ES_COLLECTION_NAME"] = "flywheel-test"
os.environ["MONGODB_DB"] = "flywheel-test"


@pytest.fixture(scope="session")
def mongo_db():
    """Fixture to provide a database connection for each test."""
    from src.api.db import get_db, init_db

    init_db()
    db = get_db()
    yield db


# === Data Validation Test Fixtures ===


@pytest.fixture
def create_flywheel_run_generic(test_workload_id: str, client_id: str):
    """Helper fixture to create a flywheel run document."""
    flywheel_run = FlywheelRun(
        workload_id=test_workload_id,
        client_id=client_id,
        started_at=datetime.utcnow(),
        num_records=0,
        nims=[],
    )
    from src.api.db import get_db, init_db

    init_db()
    mongo_db = get_db()
    result = mongo_db.flywheel_runs.insert_one(flywheel_run.to_mongo())
    return str(result.inserted_id), mongo_db


@pytest.fixture
def mock_external_services_validation() -> Generator[dict[str, MagicMock], None, None]:
    """Mock external service responses for data validation tests"""
    with (
        patch("src.lib.nemo.data_uploader.DataUploader.upload_data") as mock_upload_data,
        patch("src.lib.nemo.data_uploader.DataUploader.get_file_uri") as mock_get_file_uri,
    ):
        mock_get_file_uri.return_value = "test_uri"
        mock_upload_data.side_effect = lambda data, file_path: file_path

        yield {
            "upload_data": mock_upload_data,
            "get_file_uri": mock_get_file_uri,
        }


@pytest.fixture
def mock_embedding_nim() -> Generator[MagicMock, None, None]:
    """Mock the Embedding NIM to simulate embedding generation."""
    with patch("src.lib.flywheel.icl_selection.Embedding") as mock_embedding_class:
        mock_embedding_instance = MagicMock()

        def mock_get_embeddings_batch(queries, input_type="query"):
            # Return a unique, properly dimensioned vector for each query
            return [[0.1 + 0.01 * i] * 2048 for i in range(len(queries))]

        mock_embedding_instance.get_embeddings_batch.side_effect = mock_get_embeddings_batch
        mock_embedding_class.return_value = mock_embedding_instance
        yield mock_embedding_instance


@pytest.fixture
def icl_test_setup(
    test_workload_id,
    client_id,
    mongo_db,
    index_test_data,
    monkeypatch,
):
    """A comprehensive fixture for setting up ICL integration tests."""
    # Create the flywheel run
    flywheel_run = FlywheelRun(
        workload_id=test_workload_id,
        client_id=client_id,
        started_at=datetime.utcnow(),
    )
    result = mongo_db.flywheel_runs.insert_one(flywheel_run.to_mongo())
    flywheel_run_id = str(result.inserted_id)

    # Common settings adjustments
    monkeypatch.setattr(settings.data_split_config, "min_total_records", 20)
    monkeypatch.setattr(settings.data_split_config, "eval_size", 5)

    # Index standard test data
    test_data = [
        {
            "workload_id": test_workload_id,
            "request": {"messages": [{"role": "user", "content": f"Question {i}"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": f"Answer {i}"}}]},
        }
        for i in range(20)
    ]
    index_test_data(test_data, test_workload_id, client_id)

    # Provide the initial TaskResult
    init_result = TaskResult(
        workload_id=test_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    return init_result, monkeypatch


@pytest.fixture
def validation_test_settings(monkeypatch):
    """Configure settings for validation tests with deterministic values."""
    # Data-split parameters
    monkeypatch.setattr(settings.data_split_config, "min_total_records", 1, raising=False)
    monkeypatch.setattr(settings.data_split_config, "random_seed", 42, raising=False)
    monkeypatch.setattr(settings.data_split_config, "eval_size", 2, raising=False)
    monkeypatch.setattr(settings.data_split_config, "val_ratio", 0.25, raising=False)
    monkeypatch.setattr(settings.data_split_config, "limit", 10, raising=False)
    monkeypatch.setattr(settings.data_split_config, "parse_function_arguments", True, raising=False)

    # NMP namespace
    new_nmp_cfg = settings.nmp_config.model_copy(update={"nmp_namespace": "test-namespace"})
    monkeypatch.setattr(settings, "nmp_config", new_nmp_cfg, raising=True)

    # ICL config - use uniform_distribution to avoid embedding API issues
    monkeypatch.setattr(
        settings.icl_config, "example_selection", "uniform_distribution", raising=False
    )

    yield


@pytest.fixture
def valid_generic_records() -> list[dict[str, Any]]:
    """Create valid records for generic workload testing."""
    return [
        {
            "request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Question {i}: What is the capital of France?"},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "The capital of France is Paris.",
                        }
                    }
                ]
            },
            "timestamp": "2023-01-01T00:00:00Z",
            "client_id": "test-client",
            "workload_id": "test-workload",
        }
        for i in range(15)  # Enough for testing
    ]


@pytest.fixture
def valid_tool_calling_records() -> list[dict[str, Any]]:
    """Create valid records for tool calling workload testing."""
    return [
        {
            "request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant with tool access."},
                    {"role": "user", "content": f"Get weather for city {i}"},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": f"call_{i}",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": f'{{"location": "City {i}"}}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            "timestamp": "2023-01-01T00:00:00Z",
            "client_id": "test-client",
            "workload_id": "test-workload",
        }
        for i in range(15)  # Enough for testing
    ]


@pytest.fixture
def invalid_format_records() -> list[dict[str, Any]]:
    """Create records with invalid OpenAI format."""
    return [
        # Missing request field
        {
            "response": {"choices": [{"message": {"content": "Response without request"}}]},
            "timestamp": "2023-01-01T00:00:00Z",
        },
        # Missing response field
        {
            "request": {"messages": [{"role": "user", "content": "Request without response"}]},
            "timestamp": "2023-01-01T00:00:00Z",
        },
        # Invalid messages format
        {
            "request": {"messages": "not a list"},
            "response": {"choices": [{"message": {"content": "Invalid messages"}}]},
            "timestamp": "2023-01-01T00:00:00Z",
        },
        # Empty choices
        {
            "request": {"messages": [{"role": "user", "content": "Empty choices"}]},
            "response": {"choices": []},
            "timestamp": "2023-01-01T00:00:00Z",
        },
        # Missing choices field
        {
            "request": {"messages": [{"role": "user", "content": "Missing choices"}]},
            "response": {},
            "timestamp": "2023-01-01T00:00:00Z",
        },
    ]


@pytest.fixture
def duplicate_query_records() -> list[dict[str, Any]]:
    """Create records with duplicate user queries."""
    return [
        {
            "request": {"messages": [{"role": "user", "content": "What is AI?"}]},
            "response": {"choices": [{"message": {"content": "AI is artificial intelligence."}}]},
            "timestamp": "2023-01-01T00:00:00Z",
        },
        {
            "request": {"messages": [{"role": "user", "content": "What is AI?"}]},  # Duplicate
            "response": {"choices": [{"message": {"content": "Different response about AI."}}]},
            "timestamp": "2023-01-01T00:00:01Z",
        },
        {
            "request": {"messages": [{"role": "user", "content": "What is ML?"}]},
            "response": {"choices": [{"message": {"content": "ML is machine learning."}}]},
            "timestamp": "2023-01-01T00:00:02Z",
        },
    ]


@pytest.fixture
def mixed_quality_tool_records() -> list[dict[str, Any]]:
    """Create mixed records for tool calling quality filter testing."""
    return [
        # Record without tool calls (should pass quality filter for tool calling)
        {
            "request": {"messages": [{"role": "user", "content": "Simple question"}]},
            "response": {"choices": [{"message": {"content": "Simple answer"}}]},
            "timestamp": "2023-01-01T00:00:00Z",
        },
        # Record with tool calls (should fail quality filter for tool calling)
        {
            "request": {"messages": [{"role": "user", "content": "Get weather"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "NYC"}',
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
            "timestamp": "2023-01-01T00:00:01Z",
        },
    ]


@pytest.fixture
def mock_record_exporter():
    """Mock the RecordExporter to control data returned."""
    with patch("src.tasks.tasks.RecordExporter") as mock_exporter_class:
        mock_exporter = MagicMock()
        mock_exporter_class.return_value = mock_exporter
        yield mock_exporter


@pytest.fixture
def mock_db_manager():
    """Mock the database manager for error handling tests."""
    with patch("src.tasks.tasks.db_manager") as mock_manager:
        yield mock_manager
