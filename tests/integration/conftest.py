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
import uuid
from collections.abc import Generator

import pytest
from bson import ObjectId

os.environ["ELASTICSEARCH_URL"] = "http://localhost:9200"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["MONGODB_URL"] = "mongodb://localhost:27017"
os.environ["ES_COLLECTION_NAME"] = "flywheel-test"
os.environ["MONGODB_DB"] = "flywheel-test"


@pytest.fixture(scope="session")
def test_workload_id() -> str:
    """Generate a unique workload ID for each test."""
    return f"test-workload-{uuid.uuid4()}"


@pytest.fixture(scope="session")
def client_id() -> str:
    """Generate a unique client ID for each test."""
    return f"test-client-{uuid.uuid4()}"


@pytest.fixture(scope="session")
def flywheel_run_id() -> str:
    """Generate a unique flywheel run ID for each test."""
    return str(ObjectId())


@pytest.fixture(scope="session")
def mongo_db():
    """Fixture to provide a database connection for each test."""
    from src.api.db import get_db, init_db

    init_db()
    db = get_db()
    yield db


@pytest.fixture(autouse=True)
def load_test_data_fixture(test_workload_id: str, client_id: str) -> Generator:
    """Fixture to provide the load_test_data function."""
    from src.scripts.load_test_data_es import load_data_to_elasticsearch

    # Use the canonical integration-test dataset instead of the old placeholder.
    # The file lives under data/aiva-final.jsonl relative to the project root.
    yield load_data_to_elasticsearch(test_workload_id, client_id, file_path="aiva-test.jsonl")
