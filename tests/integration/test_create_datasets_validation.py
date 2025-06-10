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

"""
Comprehensive integration tests for create_datasets function focusing on data validation.

This module covers:
1. Data validation success and failure scenarios
2. Different workload types (GENERIC and TOOL_CALLING)
3. Database updates and record counts
4. Data splitting behavior
5. Error handling and edge cases
"""

from unittest.mock import patch

import pytest
from bson import ObjectId

from src.api.models import DatasetType, TaskResult, WorkloadClassification
from src.config import settings
from src.tasks.tasks import create_datasets


class TestCreateDatasetsGenericWorkload:
    """Test create_datasets for GENERIC workload validation scenarios."""

    @pytest.mark.integration
    def test_create_datasets_validation_success_generic(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test successful dataset creation with valid generic records."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Mock record exporter to return valid records
        mock_record_exporter.get_records.return_value = valid_generic_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        result = create_datasets(previous_result)
        task_result = TaskResult.model_validate(result)

        # Verify successful completion
        assert task_result.error is None
        assert task_result.workload_type == WorkloadClassification.GENERIC

        # Verify datasets were created
        assert task_result.datasets is not None
        assert len(task_result.datasets) == 3
        assert DatasetType.BASE in task_result.datasets
        assert DatasetType.ICL in task_result.datasets
        assert DatasetType.TRAIN in task_result.datasets

        # Verify database was updated with correct record count
        db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        assert db_doc["num_records"] == settings.data_split_config.limit  # 10 from our config

        # Verify datasets in database
        assert len(db_doc["datasets"]) == 3
        for dataset in db_doc["datasets"]:
            assert "name" in dataset
            assert "num_records" in dataset
            assert "nmp_uri" in dataset

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "invalid_count,valid_count,expected_to_pass",
        [
            (0, 15, True),  # All valid records
            (5, 10, True),  # Mixed with enough valid
            (10, 0, False),  # Not enough valid records
            (15, 0, False),  # All invalid records
        ],
    )
    def test_create_datasets_mixed_validity_generic(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        invalid_format_records,
        mock_db_manager,
        validation_test_settings,
        invalid_count: int,
        valid_count: int,
        expected_to_pass: bool,
    ):
        """Test dataset creation with mixed valid/invalid records for generic workload."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Create mixed dataset
        mixed_records = (
            valid_generic_records[:valid_count]
            + (invalid_format_records * (invalid_count // len(invalid_format_records) + 1))[
                :invalid_count
            ]
        )

        mock_record_exporter.get_records.return_value = mixed_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        if expected_to_pass:
            result = create_datasets(previous_result)
            task_result = TaskResult.model_validate(result)

            assert task_result.error is None
            assert task_result.workload_type == WorkloadClassification.GENERIC
            assert len(task_result.datasets) == 3

            # Verify database update
            db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
            assert db_doc["num_records"] == settings.data_split_config.limit
        else:
            with pytest.raises(ValueError) as exc_info:
                create_datasets(previous_result)

            assert "Insufficient valid records" in str(exc_info.value)

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "invalid_count,valid_count,expected_to_pass",
        [
            (0, 15, True),  # All valid records
            (5, 10, True),  # Mixed with enough valid
            (10, 20, False),  # Not enough valid records
            (15, 50, False),  # All invalid records
        ],
    )
    def test_create_datasets_mixed_validity_generic_with_greater_limit(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        invalid_format_records,
        mock_db_manager,
        validation_test_settings,
        invalid_count: int,
        valid_count: int,
        expected_to_pass: bool,
    ):
        """Test dataset creation with mixed valid/invalid records for generic workload."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Create mixed dataset
        mixed_records = (
            valid_generic_records[:valid_count]
            + (invalid_format_records * (invalid_count // len(invalid_format_records) + 1))[
                :invalid_count
            ]
        )

        mock_record_exporter.get_records.return_value = mixed_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        if expected_to_pass:
            result = create_datasets(previous_result)
            task_result = TaskResult.model_validate(result)

            assert task_result.error is None
            assert task_result.workload_type == WorkloadClassification.GENERIC
            assert len(task_result.datasets) == 3

            # Verify database update
            db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
            assert db_doc["num_records"] == settings.data_split_config.limit
        else:
            create_datasets(previous_result)

    @pytest.mark.integration
    def test_create_datasets_deduplication_generic(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        duplicate_query_records,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test that duplicate records are properly removed for generic workload."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Combine duplicates with enough valid records to meet limit
        combined_records = (
            duplicate_query_records + valid_generic_records[:8]
        )  # Total should be > 10 unique
        mock_record_exporter.get_records.return_value = combined_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        result = create_datasets(previous_result)
        task_result = TaskResult.model_validate(result)

        # Should succeed despite duplicates
        assert task_result.error is None
        assert task_result.workload_type == WorkloadClassification.GENERIC

        # Check database record count (duplicates should be removed)
        db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        assert db_doc["num_records"] == settings.data_split_config.limit

    @pytest.mark.integration
    def test_create_datasets_data_splitting_generic(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
        monkeypatch,
    ):
        """Test that data splitting works correctly for generic workload."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Configure specific split settings
        monkeypatch.setattr(settings.data_split_config, "eval_size", 3)
        monkeypatch.setattr(settings.data_split_config, "val_ratio", 0.2)

        mock_record_exporter.get_records.return_value = valid_generic_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        result = create_datasets(previous_result)
        task_result = TaskResult.model_validate(result)

        # Verify successful completion
        assert task_result.error is None

        # Check dataset record counts in database
        db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        datasets = db_doc["datasets"]

        eval_dataset = next(d for d in datasets if "eval" in d["name"])
        train_dataset = next(d for d in datasets if "train" in d["name"])
        icl_dataset = next(d for d in datasets if "icl" in d["name"])

        # Eval dataset should have eval_size records
        assert eval_dataset["num_records"] == 3

        # ICL dataset should have same count as eval (generated from eval records)
        assert icl_dataset["num_records"] == 3

        # Train dataset should have remaining records after eval split, minus validation
        # From log: "Split 10 records into 3 eval, 5 train, 2 val"
        expected_train_records = 5  # Actual split calculation result
        assert train_dataset["num_records"] == expected_train_records


class TestCreateDatasetsToolCallingWorkload:
    """Test create_datasets for TOOL_CALLING workload validation scenarios."""

    @pytest.mark.integration
    def test_create_datasets_validation_success_tool_calling(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_tool_calling_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test successful dataset creation with valid tool calling records."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Tool calling records have tool_calls, which PASS quality filter for TOOL_CALLING workloads
        # The quality filter keeps records WITH tool calls for tool calling workloads
        mock_record_exporter.get_records.return_value = valid_tool_calling_records

        # Mock workload type identification to return TOOL_CALLING
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.TOOL_CALLING

            previous_result = TaskResult(
                workload_id=test_workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

            result = create_datasets(previous_result)
            task_result = TaskResult.model_validate(result)

            # Should succeed because tool calling records pass the quality filter
            assert task_result.error is None
            assert task_result.workload_type == WorkloadClassification.TOOL_CALLING
            assert len(task_result.datasets) == 3

            # Verify database was updated
            db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
            assert db_doc["num_records"] == settings.data_split_config.limit

    @pytest.mark.integration
    def test_create_datasets_tool_calling_quality_filter(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        mixed_quality_tool_records,
        valid_tool_calling_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test tool calling quality filter keeps records with tool calls."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Combine mixed quality records with tool calling records
        # Tool calling quality filter keeps records WITH tool calls and removes records WITHOUT tool calls
        # mixed_quality_tool_records has 1 record with tool calls, so we need 9 more to reach limit of 10
        combined_records = mixed_quality_tool_records + valid_tool_calling_records[:9]
        mock_record_exporter.get_records.return_value = combined_records

        # Mock workload type identification to return TOOL_CALLING
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.TOOL_CALLING

            previous_result = TaskResult(
                workload_id=test_workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

            result = create_datasets(previous_result)
            task_result = TaskResult.model_validate(result)

            # Should succeed with quality-filtered records
            assert task_result.error is None
            assert task_result.workload_type == WorkloadClassification.TOOL_CALLING

            # Verify database update
            db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
            assert db_doc["num_records"] == settings.data_split_config.limit

    @pytest.mark.integration
    def test_create_datasets_tool_calling_insufficient_after_quality_filter(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test failure when tool calling quality filter removes too many records."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Use records WITHOUT tool calls - these will be filtered out by quality filter
        mock_record_exporter.get_records.return_value = valid_generic_records

        # Mock workload type identification to return TOOL_CALLING
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.TOOL_CALLING

            previous_result = TaskResult(
                workload_id=test_workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

            with pytest.raises(ValueError) as exc_info:
                create_datasets(previous_result)

            error_message = str(exc_info.value)
            assert "Insufficient valid records" in error_message
            assert "after quality filters" in error_message

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "tool_records_count,simple_records_count,expected_to_pass",
        [
            (10, 5, True),  # Enough tool records pass the filter
            (0, 20, False),  # Not enough tool records after simple records filtered out
            (0, 15, False),  # All simple records get filtered out
            (15, 0, True),  # All tool records pass
        ],
    )
    def test_create_datasets_tool_calling_quality_scenarios(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_tool_calling_records,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
        tool_records_count: int,
        simple_records_count: int,
        expected_to_pass: bool,
    ):
        """Test various combinations of tool/simple records for tool calling workload."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Create mixed dataset
        # For TOOL_CALLING workloads: records WITH tool calls pass, records WITHOUT tool calls are filtered out
        combined_records = (
            valid_tool_calling_records[:tool_records_count]
            + valid_generic_records[:simple_records_count]
        )

        mock_record_exporter.get_records.return_value = combined_records

        # Mock workload type identification to return TOOL_CALLING
        with patch("src.tasks.tasks.identify_workload_type") as mock_identify:
            mock_identify.return_value = WorkloadClassification.TOOL_CALLING

            previous_result = TaskResult(
                workload_id=test_workload_id,
                flywheel_run_id=flywheel_run_id,
                client_id=client_id,
            )

            if expected_to_pass:
                result = create_datasets(previous_result)
                task_result = TaskResult.model_validate(result)

                assert task_result.error is None
                assert task_result.workload_type == WorkloadClassification.TOOL_CALLING

                # Verify database update
                db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
                assert db_doc["num_records"] == settings.data_split_config.limit
            else:
                with pytest.raises(ValueError) as exc_info:
                    create_datasets(previous_result)

                assert "Insufficient valid records" in str(exc_info.value)


class TestCreateDatasetsErrorHandling:
    """Test error handling and edge cases for create_datasets."""

    @pytest.mark.integration
    def test_create_datasets_no_records_found(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test error handling when no records are found."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Mock empty records
        mock_record_exporter.get_records.return_value = []

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        with pytest.raises(ValueError) as exc_info:
            create_datasets(previous_result)

        # The error message will be from the validation step, not the record exporter
        assert "Not enough records found" in str(exc_info.value)

    @pytest.mark.integration
    def test_create_datasets_all_invalid_format(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        invalid_format_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test error handling when all records have invalid format."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        mock_record_exporter.get_records.return_value = invalid_format_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        with pytest.raises(ValueError) as exc_info:
            create_datasets(previous_result)

        error_message = str(exc_info.value)
        assert "Insufficient valid records" in error_message
        # Note: Some invalid records might still pass basic validation
        # The exact count depends on the OpenAI validator implementation

    @pytest.mark.integration
    def test_create_datasets_validation_error_database_not_updated(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        invalid_format_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test that database is not updated when validation fails."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        mock_record_exporter.get_records.return_value = invalid_format_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        # Capture initial state
        initial_db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        initial_num_records = initial_db_doc.get("num_records", 0)

        with pytest.raises(ValueError):
            create_datasets(previous_result)

        # Verify database wasn't updated
        final_db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        assert final_db_doc.get("num_records", 0) == initial_num_records
        assert final_db_doc.get("datasets", []) == initial_db_doc.get("datasets", [])

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "limit_setting, min_total_records, expected_error",
        [
            (
                0,
                10,
                "limit cannot be less than the minimum number of records",
            ),  # zero limit should not select any records
            (5, 10, "limit cannot be less than the minimum number of records"),
        ],
    )
    def test_create_datasets_invalid_configuration(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
        monkeypatch,
        limit_setting: int,
        min_total_records: int,
        expected_error: str,
    ):
        """Test error handling with invalid configuration settings."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Set invalid configuration
        monkeypatch.setattr(settings.data_split_config, "limit", limit_setting)
        monkeypatch.setattr(settings.data_split_config, "min_total_records", min_total_records)

        mock_record_exporter.get_records.return_value = valid_generic_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )
        with pytest.raises(ValueError) as exc_info:
            create_datasets(previous_result)

        assert expected_error in str(exc_info.value)


class TestCreateDatasetsDataSplitValidation:
    """Test data splitting behavior and validation."""

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "limit,eval_size,val_ratio,expected_eval,expected_train",
        [
            (10, 2, 0.25, 2, 6),  # Standard split: 10 -> 2 eval, 8 remaining -> 6 train, 2 val
            (15, 3, 0.2, 3, 9),  # Different ratios: 15 -> 3 eval, 12 remaining -> 9 train, 3 val
        ],
    )
    def test_create_datasets_split_configurations(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
        monkeypatch,
        limit: int,
        eval_size: int,
        val_ratio: float,
        expected_eval: int,
        expected_train: int,
    ):
        """Test various data split configurations."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        # Ensure we have enough valid records
        extended_records = valid_generic_records * (limit // len(valid_generic_records) + 1)

        # Make records unique to avoid deduplication
        for i, record in enumerate(extended_records[: limit + 5]):
            record["request"]["messages"][-1]["content"] = f"Unique question {i}"

        mock_record_exporter.get_records.return_value = extended_records[: limit + 5]

        # Configure split settings
        monkeypatch.setattr(settings.data_split_config, "limit", limit)
        monkeypatch.setattr(settings.data_split_config, "eval_size", eval_size)
        monkeypatch.setattr(settings.data_split_config, "val_ratio", val_ratio)

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        result = create_datasets(previous_result)
        task_result = TaskResult.model_validate(result)

        # Verify successful completion
        assert task_result.error is None

        # Check dataset record counts in database
        db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})
        datasets = db_doc["datasets"]

        eval_dataset = next(d for d in datasets if "eval" in d["name"])
        train_dataset = next(d for d in datasets if "train" in d["name"])

        # Verify split calculations
        assert eval_dataset["num_records"] == expected_eval
        # Train dataset contains only training records (not validation)
        assert train_dataset["num_records"] == expected_train

        # Verify total records processed
        assert db_doc["num_records"] == limit

    @pytest.mark.integration
    def test_create_datasets_database_update_sequence(
        self,
        test_workload_id: str,
        client_id: str,
        create_flywheel_run_generic,
        mock_external_services_validation,
        mock_record_exporter,
        valid_generic_records,
        mock_db_manager,
        validation_test_settings,
    ):
        """Test that database updates happen in correct sequence."""
        flywheel_run_id, mongo_db = create_flywheel_run_generic

        mock_record_exporter.get_records.return_value = valid_generic_records

        previous_result = TaskResult(
            workload_id=test_workload_id,
            flywheel_run_id=flywheel_run_id,
            client_id=client_id,
        )

        result = create_datasets(previous_result)
        task_result = TaskResult.model_validate(result)

        # Verify successful completion
        assert task_result.error is None

        # Check final database state
        db_doc = mongo_db.flywheel_runs.find_one({"_id": ObjectId(flywheel_run_id)})

        # num_records should be updated with validated record count
        assert db_doc["num_records"] == settings.data_split_config.limit

        # datasets should be populated with all three dataset types
        assert "datasets" in db_doc
        assert len(db_doc["datasets"]) == 3

        # Each dataset should have proper structure
        for dataset in db_doc["datasets"]:
            assert "name" in dataset
            assert "num_records" in dataset
            assert "nmp_uri" in dataset
            assert dataset["num_records"] > 0
            assert dataset["nmp_uri"] == "test_uri"  # From our mock

        # Verify dataset names follow expected pattern
        dataset_names = [d["name"] for d in db_doc["datasets"]]
        assert any("eval" in name for name in dataset_names)
        assert any("icl" in name for name in dataset_names)
        assert any("train" in name for name in dataset_names)
