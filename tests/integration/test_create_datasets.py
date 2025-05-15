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
from collections.abc import Generator
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from bson import ObjectId

from src.api.models import DatasetType, FlywheelRun, TaskResult, WorkloadClassification  # E402
from src.config import settings
from src.lib.flywheel.util import DataSplitConfig  # E402
from src.tasks.tasks import create_datasets  # E402


@pytest.fixture
def mock_external_services() -> Generator[dict[str, MagicMock], None, None]:
    """Mock external service responses"""
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
def create_flywheel_run(test_workload_id: str, client_id: str):
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


@pytest.mark.integration
def test_create_datasets(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test the create_datasets task with default split configuration."""
    # Create default split configuration
    settings.data_split_config = DataSplitConfig(
        eval_size=1,  # Small eval set for testing
        val_ratio=0.5,  # 50% validation split
    )

    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id

    flywheel_run_id, mongo_db = create_flywheel_run

    # Run the task
    result = create_datasets(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Convert result to TaskResult model
    task_result = TaskResult.model_validate(result)

    # Verify result structure
    assert task_result is not None
    assert task_result.workload_id == current_workload_id
    assert task_result.flywheel_run_id == flywheel_run_id

    # Verify datasets were created with correct types
    assert task_result.datasets is not None
    assert DatasetType.BASE in task_result.datasets
    assert DatasetType.ICL in task_result.datasets
    assert DatasetType.TRAIN in task_result.datasets

    # Verify dataset names follow expected format
    for dataset_name in task_result.datasets.values():
        assert dataset_name.startswith("flywheel-")
        assert current_workload_id in dataset_name

    # Verify MongoDB document structure
    db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
    assert db_doc is not None
    assert "datasets" in db_doc
    assert len(db_doc["datasets"]) == 3  # BASE, ICL, and TRAIN datasets

    # Verify each dataset document structure
    for dataset in db_doc["datasets"]:
        assert "name" in dataset
        assert "num_records" in dataset
        assert "nmp_uri" in dataset
        assert dataset["nmp_uri"] == "test_uri"  # From our mock
        assert dataset["name"].startswith("flywheel-")
        assert current_workload_id in dataset["name"]

    # Verify mock calls
    mock_upload_data = mock_external_services["upload_data"]
    mock_get_file_uri = mock_external_services["get_file_uri"]

    # Should be called 4 times: eval data, icl data, train data, and val data
    assert mock_upload_data.call_count == 4
    assert mock_get_file_uri.call_count == 3  # Called once for each dataset type


@pytest.mark.integration
def test_create_datasets_workload_classification(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test that workload type is correctly identified and stored."""
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id
    flywheel_run_id, _ = create_flywheel_run

    result = create_datasets(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )
    task_result = TaskResult.model_validate(result)

    # Verify workload type is set
    assert task_result.workload_type is not None
    assert task_result.workload_type in [
        WorkloadClassification.GENERIC,
        WorkloadClassification.TOOL_CALLING,
    ]


@pytest.mark.integration
def test_create_datasets_different_split_configs(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test dataset creation with different split configurations."""
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id
    flywheel_run_id, mongo_db = create_flywheel_run

    # Test with different split configurations
    test_configs = [
        DataSplitConfig(eval_size=2, val_ratio=0.3),
        DataSplitConfig(eval_size=3, val_ratio=0.4),
    ]

    for config in test_configs:
        settings.data_split_config = config
        result = create_datasets(
            workload_id=current_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )
        TaskResult.model_validate(result)

        # Verify datasets reflect the split configuration
        db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        datasets = db_doc["datasets"]
        eval_dataset = next(d for d in datasets if "eval" in d["name"])
        assert eval_dataset["num_records"] == config.eval_size


@pytest.mark.integration
def test_create_datasets_upload_validation(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test that training data is properly formatted."""
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id

    flywheel_run_id, mongo_db = create_flywheel_run

    create_datasets(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Verify the upload_data calls contain properly formatted data
    mock_upload_data = mock_external_services["upload_data"]
    calls = mock_upload_data.call_args_list

    eval_call, icl_call, train_call, val_call = calls
    eval_data = eval_call[0][0]
    icl_data = icl_call[0][0]
    train_data = train_call[0][0]
    val_data = val_call[0][0]

    # Verify data is not None
    assert eval_data is not None
    assert icl_data is not None
    assert train_data is not None
    assert val_data is not None


@pytest.mark.integration
def test_create_datasets_with_prefix(
    test_workload_id: str, flywheel_run_id: str, client_id: str, mock_external_services
) -> None:
    """Test the create_datasets task with a custom dataset prefix."""
    # Create default split configuration
    settings.data_split_config = DataSplitConfig(eval_size=1, val_ratio=0.5)

    # Run the task with prefix
    prefix = "test-prefix"
    result = create_datasets(
        workload_id=test_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
        output_dataset_prefix=prefix,
    )

    # Convert result to TaskResult model
    task_result = TaskResult.model_validate(result)

    # Verify dataset names include the prefix
    assert task_result.datasets is not None
    for dataset_name in task_result.datasets.values():
        assert prefix in dataset_name


@pytest.mark.integration
def test_create_datasets_no_records_error(
    client_id: str,
    create_flywheel_run,
    mock_external_services,
) -> None:
    """Test that create_datasets properly handles and logs errors when no records are found."""
    # Create a non-existent workload ID
    non_existent_workload_id = "non-existent-workload-id"

    # Get flywheel run ID from fixture
    flywheel_run_id, mongo_db = create_flywheel_run

    # The function should raise ValueError for empty dataset
    with pytest.raises(ValueError) as exc_info:
        create_datasets(
            workload_id=non_existent_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

    # Verify the error message
    assert "No records found" in str(exc_info.value)

    # Verify the error was saved to the database
    db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
    assert db_doc is not None
    assert "error" in db_doc
    assert "No records found" in db_doc["error"]
    assert db_doc["status"] == "error"


@pytest.mark.integration
def test_create_datasets_not_enough_records_error(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    mock_external_services,
) -> None:
    """Test that create_datasets properly handles and logs errors when not enough records are found."""
    flywheel_run_id, mongo_db = create_flywheel_run

    with patch("src.tasks.tasks.get_es_client") as mock_get_es_client:
        mock_es_client = MagicMock()
        mock_get_es_client.return_value = mock_es_client

        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "client_id": client_id,
                            "workload_id": test_workload_id,
                            "request": {"messages": [{"role": "user", "content": "test"}]},
                            "response": {
                                "choices": [
                                    {"message": {"role": "assistant", "content": "test response"}}
                                ]
                            },
                            "timestamp": "2023-01-01T00:00:00Z",
                        }
                    }
                ]
            }
        }

        with patch("src.tasks.tasks.settings") as mock_settings:
            mock_settings.data_split_config.min_total_records = 10  # Set high minimum
            mock_settings.data_split_config.limit = 100
            mock_settings.data_split_config.eval_size = 1
            mock_settings.data_split_config.val_ratio = 0.2

            with pytest.raises(ValueError) as exc_info:
                create_datasets(
                    workload_id=test_workload_id,
                    flywheel_run_id=flywheel_run_id,
                    client_id=client_id,
                )

            assert "Not enough records found" in str(exc_info.value)
            assert "minimum of 10 records is required" in str(exc_info.value)
            assert "only 1 were found" in str(exc_info.value)

            db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
            assert db_doc is not None
            assert "error" in db_doc
            assert "Not enough records found" in db_doc["error"]
            assert db_doc["status"] == "error"


@pytest.mark.integration
@pytest.mark.parametrize(
    "malformed_case, expected_error",
    [
        (
            # Case 1: Empty choices in response
            {
                "client_id": "test-client",
                "workload_id": "test-workload",
                "request": {"messages": [{"role": "user", "content": "malformed request"}]},
                "response": {"choices": []},  # Empty choices list
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "No choices found in response",
        ),
        (
            # Case 2: Missing messages in request
            {
                "client_id": "test-client",
                "workload_id": "test-workload",
                "request": {},  # Missing messages
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": "response"}}]
                },
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "Error processing record: 'messages'",
        ),
        (
            # Case 3: Malformed message structure in response
            {
                "client_id": "test-client",
                "workload_id": "test-workload",
                "request": {"messages": [{"role": "user", "content": "malformed response"}]},
                "response": {"choices": [{}]},  # Missing message field in choice
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "Error processing record: 'message'",
        ),
        (
            # Case 4: Missing response field
            {
                "client_id": "test-client",
                "workload_id": "test-workload",
                "request": {"messages": [{"role": "user", "content": "missing response"}]},
                # Missing response field entirely
                "timestamp": "2023-01-01T00:00:00Z",
            },
            "Error processing record: 'response'",
        ),
    ],
)
def test_create_datasets_specific_malformed_records(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    mock_external_services,
    malformed_case: dict,
    expected_error: str,
) -> None:
    """Test that create_datasets properly handles specific types of malformed records.
    The malformed record is skipped and not included in any dataset."""
    # Create local copies to avoid modifying the parameters
    malformed_case = malformed_case.copy()
    malformed_case["client_id"] = client_id
    malformed_case["workload_id"] = test_workload_id

    flywheel_run_id, mongo_db = create_flywheel_run

    with patch("src.tasks.tasks.get_es_client") as mock_get_es_client:
        mock_es_client = MagicMock()
        mock_get_es_client.return_value = mock_es_client

        # Number of valid records to add
        num_valid_records = 10

        # Create a dataset with one malformed record and enough valid records to meet minimum
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    # The specific malformed record being tested
                    {"_source": malformed_case},
                    # Valid records to meet the minimum requirement
                    *[
                        {
                            "_source": {
                                "client_id": client_id,
                                "workload_id": test_workload_id,
                                "request": {
                                    "messages": [{"role": "user", "content": f"valid request {i}"}]
                                },
                                "response": {
                                    "choices": [
                                        {
                                            "message": {
                                                "role": "assistant",
                                                "content": f"valid response {i}",
                                            }
                                        }
                                    ]
                                },
                                "timestamp": "2023-01-01T00:00:00Z",
                            }
                        }
                        for i in range(num_valid_records)
                    ],
                ]
            }
        }

        with patch("src.tasks.tasks.settings") as mock_settings:
            # Configure settings
            mock_settings.data_split_config.min_total_records = 5
            mock_settings.data_split_config.limit = 100
            mock_settings.data_split_config.eval_size = 2  # Fixed eval size
            mock_settings.data_split_config.val_ratio = 0.2  # 20% validation
            mock_settings.data_split_config.random_seed = 42  # Fixed random seed

            # Capture logging messages
            with patch("src.lib.flywheel.util.logger") as mock_logger:
                # Run the dataset creation function
                result = create_datasets(
                    workload_id=test_workload_id,
                    flywheel_run_id=flywheel_run_id,
                    client_id=client_id,
                )

                # Verify the expected error was logged
                error_logged = False
                for call_args in mock_logger.error.call_args_list:
                    if expected_error in call_args[0][0]:
                        error_logged = True
                        break

                assert (
                    error_logged
                ), f"Expected error message containing '{expected_error}' was not logged"

            # Verify task still completed successfully
            task_result = TaskResult.model_validate(result)
            assert task_result is not ModuleNotFoundError

            # Verify MongoDB document structure
            db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
            assert db_doc is not None
            assert "datasets" in db_doc
            assert len(db_doc["datasets"]) == 3

            # Get the uploaded data to check record counts
            mock_upload_data = mock_external_services["upload_data"]
            calls = mock_upload_data.call_args_list

            # Should be 4 uploads: eval_data, icl_data, train_data, val_data
            assert len(calls) == 4
            eval_call, icl_call, train_call, val_call = calls

            # Extract the data from each call
            eval_data = eval_call[0][0]
            icl_data = icl_call[0][0]
            train_data = train_call[0][0]
            val_data = val_call[0][0]

            # # Convert to list if not already
            eval_data = (
                [eval_data]
                if isinstance(eval_data, dict)
                else eval_data.split("\n")
                if isinstance(eval_data, str)
                else eval_data
            )
            icl_data = (
                [icl_data]
                if isinstance(icl_data, dict)
                else icl_data.split("\n")
                if isinstance(icl_data, str)
                else icl_data
            )
            train_data = (
                [train_data]
                if isinstance(train_data, dict)
                else train_data.split("\n")
                if isinstance(train_data, str)
                else train_data
            )
            val_data = (
                [val_data]
                if isinstance(val_data, dict)
                else val_data.split("\n")
                if isinstance(val_data, str)
                else val_data
            )

            expected_eval_count, expected_val_count, expected_train_count = 2, 2, 6

            # # Check record counts in each dataset
            # # Counts may vary slightly due to rounding in the split calculation
            db_datasets = db_doc["datasets"]
            eval_dataset = next((d for d in db_datasets if "eval" in d["name"]), None)
            train_dataset = next((d for d in db_datasets if "train" in d["name"]), None)

            # Verify that the malformed record was excluded from all datasets
            assert len(eval_data) == expected_eval_count == eval_dataset["num_records"]
            assert len(train_data) == expected_train_count == train_dataset["num_records"]
            assert len(val_data) == expected_val_count
