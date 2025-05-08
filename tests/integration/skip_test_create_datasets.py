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
from uuid import uuid4

import pytest
from bson import ObjectId

os.environ["ELASTICSEARCH_URL"] = "http://localhost:9200"
os.environ["ES_COLLECTION_NAME"] = f"flywheel-test-{int(datetime.now().timestamp())}"
os.environ["MONGODB_DB"] = "flywheel-test-" + str(uuid4())
os.environ["NVCF_API_KEY"] = "test"

from src.api.db import init_db  # noqa E402
from src.lib.integration.es_client import get_es_client  # noqa E402
from src.lib.flywheel.util import DataSplitConfig  # noqa E402
from src.api.models import TaskResult  # noqa E402
from src.tasks.tasks import create_datasets  # noqa E402

from src.scripts.load_test_data import load_data_to_elasticsearch  # noqa E402

ES_CLIENT = get_es_client()


@pytest.fixture
def workload_id() -> str:
    """Generate a unique workload ID for each test."""
    return f"test-workload-{uuid4()}"


@pytest.fixture
def flywheel_run_id() -> str:
    """Generate a unique flywheel run ID for each test."""
    return str(ObjectId())


@pytest.fixture(autouse=True)
def setup_teardown(workload_id: str) -> Generator[None, None, None]:
    """Setup and teardown for each test."""
    # Setup
    init_db()
    load_data_to_elasticsearch(workload_id)

    yield  # This is where the test runs

    # Teardown
    ES_CLIENT.indices.delete(index=os.environ["ES_COLLECTION_NAME"], ignore_unavailable=True)


@pytest.mark.integration
def test_create_datasets(workload_id: str, flywheel_run_id: str) -> None:
    """Test the create_datasets task with default split configuration."""
    # Create default split configuration
    split_config = DataSplitConfig(
        eval_size=1,  # Small eval set for testing
        val_ratio=0.5,  # 50% validation split
    )

    # Run the task
    result = create_datasets(
        workload_id=workload_id, flywheel_run_id=flywheel_run_id, data_split_config=split_config
    )

    # Convert result to TaskResult model
    task_result = TaskResult.model_validate(result)

    # Verify result structure
    assert task_result is not None
    assert task_result.workload_id == workload_id
    assert task_result.flywheel_run_id == flywheel_run_id

    # Verify datasets were created
    assert task_result.datasets is not None
    assert "train" in task_result.datasets
    assert "val" in task_result.datasets
    assert "test" in task_result.datasets
    assert "icl_eval" in task_result.datasets

    # Verify dataset names follow expected format
    for dataset_name in task_result.datasets.values():
        assert dataset_name.startswith("flywheel-")
        assert workload_id in dataset_name


@pytest.mark.integration
def test_create_datasets_with_prefix(workload_id: str, flywheel_run_id: str) -> None:
    """Test the create_datasets task with a custom dataset prefix."""
    # Create default split configuration
    split_config = DataSplitConfig(eval_size=1, val_ratio=0.5)

    # Run the task with prefix
    prefix = "test-prefix"
    result = create_datasets(
        workload_id=workload_id,
        flywheel_run_id=flywheel_run_id,
        data_split_config=split_config,
        output_dataset_prefix=prefix,
    )

    # Convert result to TaskResult model
    task_result = TaskResult.model_validate(result)

    # Verify dataset names include the prefix
    assert task_result.datasets is not None
    for dataset_name in task_result.datasets.values():
        assert prefix in dataset_name


@pytest.mark.integration
def test_create_datasets_with_empty_workload(flywheel_run_id: str) -> None:
    """Test the create_datasets task with a non-existent workload."""
    non_existent_workload = f"non-existent-{uuid4()}"

    # Verify task raises ValueError for non-existent workload
    with pytest.raises(ValueError):
        create_datasets(workload_id=non_existent_workload, flywheel_run_id=flywheel_run_id)
