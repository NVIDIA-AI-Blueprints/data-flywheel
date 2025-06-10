from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from bson.objectid import ObjectId

from src.api.models import LLMJudgeConfig, TaskResult
from src.api.schemas import DeploymentStatus, FlywheelRunStatus
from src.config import settings
from src.lib.nemo.llm_as_judge import LLMAsJudge


@pytest.fixture(scope="session", autouse=True)
def mock_db():
    """Fixture to provide a database connection for each test."""
    # Import here after patching to avoid real MongoDB connections

    # Create a TaskDBManager mock
    mock_db_manager = MagicMock()

    # Configure default return values for common methods
    mock_db_manager.create_nim_run.return_value = MagicMock()
    mock_db_manager.find_nim_run.return_value = {}
    mock_db_manager.insert_evaluation.return_value = MagicMock()
    mock_db_manager.insert_customization.return_value = MagicMock()

    # Patch TaskDBManager
    with patch("src.api.db_manager.TaskDBManager", return_value=mock_db_manager):
        yield mock_db_manager


@pytest.fixture(scope="function", autouse=True)
def mock_db_functions(mock_db):
    """
    Mock all database-related functions to prevent direct DB access during tests.

    This fixture is automatically applied to all tests and ensures that:
    1. No actual database connections are made
    2. All db.init_db() and db.get_db() calls are mocked
    3. All functions in different modules that use these database
       functions are properly mocked
    """
    # Configure the mock database manager to return False for cancellation checks by default
    mock_db.is_flywheel_run_cancelled.return_value = False

    with (
        patch("src.api.db.get_db", return_value=mock_db),
        patch("src.api.db.init_db", return_value=mock_db),
        patch("src.api.job_service.get_db", return_value=mock_db),
        patch("src.api.db_manager.get_db_manager", return_value=mock_db),
        patch("src.tasks.tasks.db_manager", mock_db),
        patch("src.lib.integration.dataset_creator.get_db", return_value=mock_db),
    ):
        yield


@pytest.fixture
def remote_llm_judge_config():
    """Create a remote LLM judge configuration for testing."""
    llm_as_judge = LLMAsJudge()
    llm_as_judge.config = LLMJudgeConfig.from_json(
        {
            "type": "remote",
            "url": "http://test-remote-url/v1/chat/completions",
            "model_name": "remote-model-id",
            "api_key": "test-api-key",
            "api_key_env": "TEST_API_KEY_ENV",
        }
    )
    yield llm_as_judge.config


@pytest.fixture
def local_llm_judge_config():
    """Create a local LLM judge configuration for testing."""

    return LLMJudgeConfig.from_json(
        {
            "type": "local",
            "model_name": "test-model-id",
            "context_length": 2048,
            "gpus": 1,
            "pvc_size": "10Gi",
            "tag": "latest",
            "registry_base": "nvcr.io/nim",
            "customization_enabled": True,
        }
    )


@pytest.fixture()
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


@pytest.fixture
def mock_data_uploader():
    """Fixture to mock DataUploader."""
    with patch("src.lib.integration.dataset_creator.DataUploader") as mock:
        mock_instance = MagicMock()
        # Ensure that `get_file_uri` (used when recording dataset metadata) returns a
        # plain string.  A raw ``MagicMock`` instance cannot be encoded by BSON and
        # causes an ``InvalidDocument`` error when the code under test attempts to
        # update MongoDB.
        mock_instance.get_file_uri.return_value = "nmp://test-namespace/datasets/dummy.jsonl"
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_es_client():
    """Fixture to mock Elasticsearch client."""
    with patch("src.lib.integration.record_exporter.get_es_client") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def workflow_params():
    """Setup parameters for workflow initialization tests"""
    return {
        "workload_id": "test_workload",
        "flywheel_run_id": str(ObjectId()),
        "client_id": "test_client",
    }


@pytest.fixture
def task_result_setup():
    """Setup TaskResult for dataset creation tests"""
    return TaskResult(
        workload_id="test_workload",
        flywheel_run_id=str(ObjectId()),
        client_id="test_client",
    )


@pytest.fixture
def workflow_setup():
    """Setup fixture for workflow initialization tests"""
    return {
        "nims_config": [
            {"model_name": "model_1", "deployment_type": "local", "gpus": 1, "tag": "latest"},
            {"model_name": "model_2", "deployment_type": "local", "gpus": 1, "tag": "latest"},
        ],
        "llm_judge_config": {
            "type": "remote",
            "url": "http://llm-judge-endpoint.com",
            "model_name": "llm_judge_model",
            "api_key": "test_api_key",
        },
    }


@pytest.fixture
def dataset_setup():
    """Setup fixture for dataset creation tests"""
    return {
        "workload_id": "test_workload",
        "flywheel_run_id": str(ObjectId()),
        "client_id": "test_client",
        "num_records": 100,
        "output_dataset_prefix": "test_prefix",
    }


@pytest.fixture
def common_workflow_params():
    """Common workflow parameters used across multiple test files"""
    return {
        "workload_id": "test-workload-123",
        "flywheel_run_id": str(ObjectId()),
        "client_id": "test-client-456",
        "output_dataset_prefix": "",
    }


@pytest.fixture
def sample_es_data():
    """Sample Elasticsearch data for testing dataset creation"""
    return {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "request": {
                            "messages": [
                                {"role": "user", "content": f"Question {i}"},
                                {"role": "assistant", "content": f"Answer {i}"},
                            ]
                        },
                        "response": {"choices": [{"message": {"content": f"Response {i}"}}]},
                    }
                }
                for i in range(30)
            ]
        }
    }


@pytest.fixture
def empty_es_data():
    """Empty Elasticsearch data for testing error scenarios"""
    return {"hits": {"hits": []}}


@pytest.fixture
def test_db(mock_db):
    """Fixture to set up test database with basic sample data"""
    # Create test flywheel run
    flywheel_run_id = ObjectId()
    mock_db.flywheel_runs.insert_one.return_value = {"inserted_id": flywheel_run_id}

    # Create test NIMs
    nim1_id = ObjectId()
    nim2_id = ObjectId()

    # Create test LLM Judge
    llm_judge_id = ObjectId()

    # Mock flywheel run data
    mock_db.flywheel_runs.find_one.return_value = {
        "_id": flywheel_run_id,
        "status": FlywheelRunStatus.PENDING,
        "workload_id": "test_workload",
        "client_id": "test_client",
        "started_at": datetime.utcnow(),
        "num_records": 0,
        "nims": [],
        "datasets": [],
    }

    # Mock NIMs data
    mock_db.nims.find.return_value = []

    mock_db.nims.insert_many.return_value = {"inserted_ids": [nim1_id, nim2_id]}

    # Mock evaluations data
    mock_db.evaluations.find.return_value = []
    # Mock customizations data
    mock_db.customizations.find.return_value = []

    mock_db.llm_judge_runs.insert_one.return_value = {"inserted_id": llm_judge_id}

    # Mock LLM Judge data
    mock_db.llm_judge_runs.find_one.return_value = None

    return {
        "flywheel_run_id": str(flywheel_run_id),
        "workload_id": "test_workload",
        "client_id": "test_client",
        "nim1_id": str(nim1_id),
        "nim2_id": str(nim2_id),
        "llm_judge_id": str(llm_judge_id),
    }


@pytest.fixture
def test_db_success(mock_db):
    """Fixture to set up test database with complete successful data"""
    # Create test flywheel run
    flywheel_run_id = ObjectId()
    mock_db.flywheel_runs.insert_one.return_value = {"inserted_id": flywheel_run_id}

    # Create test NIMs
    nim1_id = ObjectId()
    nim2_id = ObjectId()

    # Create test LLM Judge
    llm_judge_id = ObjectId()

    # Mock flywheel run data
    mock_db.flywheel_runs.find_one.return_value = {
        "_id": flywheel_run_id,
        "workload_id": "test_workload",
        "client_id": "test_client",
        "started_at": datetime.utcnow(),
        "num_records": 100,
        "nims": [],
        "datasets": [
            {
                "name": f"test_dataset_{i+1}",
                "num_records": 100,
                "nmp_uri": f"test_uri_{i+1}",
            }
            for i in range(4)
        ],
    }

    # Mock NIMs data
    mock_db.nims.find.return_value = [
        {
            "_id": nim1_id,
            "model_name": "test_model_1",
            "flywheel_run_id": flywheel_run_id,
            "deployment_status": DeploymentStatus.COMPLETED,
            "runtime_seconds": 120.0,
        },
        {
            "_id": nim2_id,
            "model_name": "test_model_2",
            "flywheel_run_id": flywheel_run_id,
            "deployment_status": DeploymentStatus.PENDING,
            "runtime_seconds": 60.0,
        },
    ]

    mock_db.nims.insert_many.return_value = {"inserted_ids": [nim1_id, nim2_id]}

    # Mock evaluations data
    mock_db.evaluations.find.return_value = [
        {
            "nim_id": nim1_id,
            "eval_type": "accuracy",
            "scores": {"function_name": 0.95},
            "started_at": datetime.utcnow(),
            "finished_at": datetime.utcnow(),
            "runtime_seconds": 15.0,
            "progress": 100,
            "nmp_uri": "test_uri_eval_1",
        },
        {
            "nim_id": nim2_id,
            "eval_type": "accuracy",
            "scores": {"function_name": 0.85},
            "started_at": datetime.utcnow(),
            "finished_at": datetime.utcnow(),
            "runtime_seconds": 15.0,
            "progress": 100,
            "nmp_uri": "test_uri_eval_2",
        },
    ]

    # Mock customizations data
    mock_db.customizations.find.return_value = [
        {
            "nim_id": nim1_id,
            "started_at": datetime.utcnow(),
            "finished_at": datetime.utcnow(),
            "runtime_seconds": 30.0,
            "progress": 100,
            "epochs_completed": 10,
            "steps_completed": 1000,
            "nmp_uri": "test_uri_custom_1",
        },
        {
            "nim_id": nim2_id,
            "started_at": datetime.utcnow(),
            "finished_at": datetime.utcnow(),
            "runtime_seconds": 30.0,
            "progress": 100,
            "epochs_completed": 8,
            "steps_completed": 800,
            "nmp_uri": "test_uri_custom_2",
        },
    ]

    mock_db.llm_judge_runs.insert_one.return_value = {"inserted_id": llm_judge_id}

    # Mock LLM Judge data
    mock_db.llm_judge_runs.find_one.return_value = {
        "_id": llm_judge_id,
        "flywheel_run_id": flywheel_run_id,
        "model_name": "test-llm-judge",
        "type": "remote",
        "deployment_status": DeploymentStatus.READY,
        "url": "http://test-llm-judge.com",
    }

    return {
        "flywheel_run_id": str(flywheel_run_id),
        "workload_id": "test_workload",
        "client_id": "test_client",
        "nim1_id": str(nim1_id),
        "nim2_id": str(nim2_id),
        "llm_judge_id": str(llm_judge_id),
    }


@pytest.fixture
def valid_openai_record():
    """Valid OpenAI chat completion format record."""
    return {
        "request": {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
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
    }


@pytest.fixture
def openai_record_with_tool_calls():
    """OpenAI record with tool calls."""
    return {
        "request": {"messages": [{"role": "user", "content": "What's the weather in New York?"}]},
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "I'll check the weather for you.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "New York", "unit": "fahrenheit"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    }


@pytest.fixture
def openai_records_batch():
    """Batch of various OpenAI format records for testing."""
    return [
        # Simple conversation
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            "response": {"choices": [{"message": {"content": "Hi there!"}}]},
        },
        # With system message
        {
            "request": {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Tell me a joke"},
                ]
            },
            "response": {
                "choices": [{"message": {"content": "Why did the chicken cross the road?"}}]
            },
        },
        # Multi-turn conversation
        {
            "request": {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 4"},
                    {"role": "user", "content": "What about 3+3?"},
                ]
            },
            "response": {"choices": [{"message": {"content": "3+3 equals 6"}}]},
        },
    ]


@pytest.fixture
def invalid_openai_records():
    """Collection of invalid OpenAI format records."""
    return [
        # Missing request
        {"response": {"choices": []}},
        # Missing response
        {"request": {"messages": []}},
        # Invalid request type
        {"request": "not a dict", "response": {"choices": []}},
        # Missing messages
        {"request": {}, "response": {"choices": []}},
        # Invalid messages type
        {"request": {"messages": "not a list"}, "response": {"choices": []}},
        # Missing choices
        {"request": {"messages": []}, "response": {}},
        # Invalid choices type
        {"request": {"messages": []}, "response": {"choices": "not a list"}},
    ]
