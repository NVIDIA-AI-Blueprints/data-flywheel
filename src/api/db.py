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

from pymongo import MongoClient
from pymongo.database import Database

# Global database connection
_client: MongoClient | None = None
_db: Database | None = None


def get_db() -> Database:
    """Get the MongoDB database instance."""
    global _db
    if _db is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _db


def init_db():
    """Initialize MongoDB connection."""
    global _client, _db

    # Return existing connection if available
    if _client is not None and _db is not None:
        try:
            # Verify connection is alive
            _client.admin.command("ping")
            return _db
        except Exception:
            # Connection dead, clean up and recreate
            close_db()

    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_db = os.getenv("MONGODB_DB", "flywheel")

    _client = MongoClient(mongodb_url)
    _db = _client[mongodb_db]

    # Create indexes
    _db.flywheel_runs.create_index("workload_id")
    _db.flywheel_runs.create_index("started_at")

    return _db


def close_db():
    """Close the MongoDB connection."""
    global _client
    if _client:
        _client.close()
        _client = None
