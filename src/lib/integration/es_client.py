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
# Initialize Elasticsearch client
import os
import time

from elasticsearch import Elasticsearch

from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.es_client")

ES_COLLECTION_NAME = os.getenv("ES_COLLECTION_NAME", "flywheel")
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")


ES_INDEX_SETTINGS = {
    "settings": {
        "mapping": {
            "total_fields": {
                "limit": 1000  # Keep the default limit
            }
        },
        "index": {
            "mapping": {
                "ignore_malformed": True  # Ignore malformed fields
            }
        },
    },
    "mappings": {
        "dynamic": "strict",  # Only allow explicitly defined fields
        "properties": {
            "workload_id": {"type": "keyword"},
            "client_id": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "request": {
                "type": "object",
                "dynamic": False,  # Don't map any fields in request
                "properties": {},  # No properties to map
            },
            "response": {
                "type": "object",
                "dynamic": False,  # Don't map any fields in response
                "properties": {},  # No properties to map
            },
        },
    },
}


def get_es_client():
    """Get a working Elasticsearch client, retrying if needed."""
    for attempt in range(30):  # Try for up to 30 seconds
        try:
            client = Elasticsearch(hosts=[ES_URL])
            if client.ping():
                health = client.cluster.health()
                if health["status"] in ["yellow", "green"]:
                    logger.info(f"Elasticsearch is ready! Status: {health['status']}")
                    # Create index if it doesn't exist

                    client.indices.refresh()
                    if not client.indices.exists(index=ES_COLLECTION_NAME):
                        logger.info("Creating index...")
                        # Define the index settings with field mappings
                        client.indices.create(index=ES_COLLECTION_NAME, body=ES_INDEX_SETTINGS)
                    else:
                        logger.info("Index already exists")

                    return client
                else:
                    logger.info(
                        f"Waiting for Elasticsearch to be healthy (status: {health['status']})..."
                    )
            time.sleep(1)
        except ConnectionError as err:
            if attempt == 29:
                msg = "Could not connect to Elasticsearch"
                logger.error(msg)
                raise RuntimeError(msg) from err
            time.sleep(1)

    msg = "Elasticsearch did not become healthy in time"
    logger.error(msg)
    raise RuntimeError(msg)
