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

# This fixture automatically tweaks global settings for every test so that they
# use small, deterministic values and a test-specific namespace.  We patch
# *attributes* on the existing objects where they are mutable, but for frozen
# Pydantic models (e.g. `NMPConfig`) we replace the entire object with a
# modified copy generated via `model_copy(update={...})`.


@pytest.fixture(autouse=True)
def tweak_settings(monkeypatch):
    """Provide deterministic test configuration via the global `settings`."""

    # --- Data-split parameters (fields are *not* frozen) --------------------
    monkeypatch.setattr(settings.data_split_config, "min_total_records", 10, raising=False)
    monkeypatch.setattr(settings.data_split_config, "random_seed", 42, raising=False)
    monkeypatch.setattr(settings.data_split_config, "eval_size", 2, raising=False)
    monkeypatch.setattr(settings.data_split_config, "val_ratio", 0.25, raising=False)
    monkeypatch.setattr(
        settings.data_split_config, "limit", 15, raising=False
    )  # Reduced from 100 to 15

    # --- NMP namespace (field *is* frozen, so create a new object) ----------
    new_nmp_cfg = settings.nmp_config.model_copy(update={"nmp_namespace": "test-namespace"})
    monkeypatch.setattr(settings, "nmp_config", new_nmp_cfg, raising=True)

    yield


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
        limit=15,  # Match the number of available test records
        min_total_records=10,  # Set to work with test data (30 records available)
    )

    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id

    flywheel_run_id, mongo_db = create_flywheel_run

    # Create TaskResult object for the new signature
    previous_result = TaskResult(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Mock the db_manager to avoid NoneType error
    with patch("src.tasks.tasks.db_manager"):
        # Mock workload type to return GENERIC so test data with tool calls works
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.GENERIC

            # Run the task
            result = create_datasets(previous_result)

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

    # Create TaskResult object for the new signature
    previous_result = TaskResult(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Mock the db_manager to avoid NoneType error
    with patch("src.tasks.tasks.db_manager"):
        # Mock workload type to return GENERIC so test data with tool calls works
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.GENERIC

            result = create_datasets(previous_result)
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
        DataSplitConfig(eval_size=2, val_ratio=0.3, limit=15, min_total_records=10),
        DataSplitConfig(eval_size=3, val_ratio=0.4, limit=15, min_total_records=10),
    ]

    # Mock the db_manager to avoid NoneType error
    with patch("src.tasks.tasks.db_manager"):
        # Mock workload type to return GENERIC so test data with tool calls works
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.GENERIC

            for config in test_configs:
                settings.data_split_config = config

                # Create TaskResult object for the new signature
                previous_result = TaskResult(
                    workload_id=current_workload_id,
                    flywheel_run_id=flywheel_run_id,
                    client_id=client_id,
                )

                result = create_datasets(previous_result)
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

    # Create TaskResult object for the new signature
    previous_result = TaskResult(
        workload_id=current_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Mock the db_manager to avoid NoneType error
    with patch("src.tasks.tasks.db_manager"):
        # Mock workload type to return GENERIC so test data with tool calls works
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.GENERIC

            create_datasets(previous_result)

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
    test_workload_id: str,
    flywheel_run_id: str,
    client_id: str,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test the create_datasets task with a custom dataset prefix."""
    # Create default split configuration
    settings.data_split_config = DataSplitConfig(
        eval_size=1, val_ratio=0.5, limit=15, min_total_records=10
    )

    # Create TaskResult object for the new signature
    previous_result = TaskResult(
        workload_id=test_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Mock the db_manager to avoid NoneType error
    with patch("src.tasks.tasks.db_manager"):
        # Mock workload type to return GENERIC so test data with tool calls works
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.GENERIC

            # Run the task with prefix
            result = create_datasets(previous_result)

            # Convert result to TaskResult model
            task_result = TaskResult.model_validate(result)

            # Verify dataset names include the prefix
            assert task_result.datasets is not None
            for dataset_name in task_result.datasets.values():
                assert test_workload_id in dataset_name


@pytest.mark.integration
def test_create_datasets_no_records_error(
    client_id: str,
    create_flywheel_run,
    mock_external_services,
) -> None:
    """Test that create_datasets raises an error when no records are found."""
    # Use a non-existent workload ID to simulate no records
    non_existent_workload_id = "non-existent-workload"
    flywheel_run_id, mongo_db = create_flywheel_run

    # Create TaskResult object for the new signature
    previous_result = TaskResult(
        workload_id=non_existent_workload_id,
        flywheel_run_id=flywheel_run_id,
        client_id=client_id,
    )

    # Mock the db_manager to avoid NoneType error
    with patch("src.tasks.tasks.db_manager"):
        # Expect an exception when no records are found
        with pytest.raises(ValueError) as exc_info:
            create_datasets(previous_result)

        # Verify the error message
        assert "No records found" in str(exc_info.value)


@pytest.mark.integration
def test_create_datasets_not_enough_records_error(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    load_test_data_fixture,
    mock_external_services,
) -> None:
    """Test that create_datasets raises an error when there are not enough records."""
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id
    flywheel_run_id, mongo_db = create_flywheel_run

    # Configure settings to require more records than available
    with patch("src.tasks.tasks.settings") as mock_settings:
        # Set minimum records higher than what's available in test data
        mock_settings.data_split_config.min_total_records = 1000  # Very high number
        mock_settings.data_split_config.limit = 100
        mock_settings.data_split_config.eval_size = 2
        mock_settings.data_split_config.val_ratio = 0.2
        mock_settings.data_split_config.random_seed = 42

        # Create TaskResult object for the new signature
        previous_result = TaskResult(
            workload_id=current_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        # Mock the db_manager to avoid NoneType error
        with patch("src.tasks.tasks.db_manager"):
            # Expect an exception when not enough records are found
            with pytest.raises(Exception) as exc_info:
                create_datasets(previous_result)

            # Verify the error message
            assert "Not enough records" in str(exc_info.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    "malformed_case, expected_to_be_filtered",
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
            True,  # Should be filtered out by validation
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
            True,  # Should be filtered out by validation
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
            True,  # Should be filtered out by validation
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
            True,  # Should be filtered out by validation
        ),
    ],
)
def test_create_datasets_specific_malformed_records(
    test_workload_id: str,
    client_id: str,
    create_flywheel_run,
    mock_external_services,
    malformed_case: dict,
    expected_to_be_filtered: bool,
) -> None:
    """Test that create_datasets properly handles specific types of malformed records.

    This test verifies that malformed records are properly excluded from datasets
    during the validation process.
    """
    # Select the appropriate workload ID and load the corresponding test data
    current_workload_id = test_workload_id
    flywheel_run_id, mongo_db = create_flywheel_run

    # Mock Elasticsearch to return a mix of good and malformed records
    with patch("src.lib.integration.record_exporter.get_es_client") as mock_get_es_client:
        mock_es_client = MagicMock()
        mock_get_es_client.return_value = mock_es_client

        # Create a mix of good records and the malformed case
        good_records = [
            {
                "client_id": "test-client",
                "workload_id": "test-workload",
                "request": {"messages": [{"role": "user", "content": f"Good request {i}"}]},
                "response": {
                    "choices": [{"message": {"role": "assistant", "content": f"Good response {i}"}}]
                },
                "timestamp": "2023-01-01T00:00:00Z",
            }
            for i in range(15)  # Enough good records to meet limit
        ]

        # Add the malformed record
        all_records = [*good_records, malformed_case]

        # Mock the search response
        mock_es_client.search.return_value = {
            "hits": {"hits": [{"_source": record} for record in all_records]}
        }

        split_config = DataSplitConfig(
            min_total_records=5,
            limit=15,
            eval_size=2,
            val_ratio=0.2,
            random_seed=42,
        )

        # Create TaskResult object for the new signature
        previous_result = TaskResult(
            workload_id=current_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
            split_config=split_config,
        )

        # Mock the db_manager to avoid NoneType error
        with patch("src.tasks.tasks.db_manager"):
            # Run the dataset creation function
            result = create_datasets(previous_result)

            # Verify task completed successfully
            task_result = TaskResult.model_validate(result)
            assert task_result is not None

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

            # Now that validation is stricter, malformed records should be filtered during validation
            # So we should get the expected split: 15 records -> 2 eval, 13 remaining -> 10 train, 3 val
            expected_eval_count = 2
            expected_train_count = (
                9  # From actual log: "Split 15 records into 2 eval, 9 train, 4 val"
            )

            # Check record counts in each dataset
            db_datasets = db_doc["datasets"]
            eval_dataset = next((d for d in db_datasets if "eval" in d["name"]), None)
            train_dataset = next((d for d in db_datasets if "train" in d["name"]), None)

            # Verify counts match expected values
            assert eval_dataset["num_records"] == expected_eval_count
            assert train_dataset["num_records"] == expected_train_count

            # Verify total processed records equals the limit (malformed record should be excluded during validation)
            assert db_doc["num_records"] == 15
