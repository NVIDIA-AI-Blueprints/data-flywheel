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
import time
from datetime import datetime
from typing import Any

from elasticsearch import ConnectionError, Elasticsearch
from elasticsearch.helpers import bulk

from src.lib.flywheel.util import extract_user_query
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.es_client")

# Global Elasticsearch client connection
_es_client: Elasticsearch | None = None

# Primary collection settings
ES_COLLECTION_NAME = os.getenv("ES_COLLECTION_NAME", "flywheel")
ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

# Embeddings index settings
ES_EMBEDDINGS_INDEX_NAME = os.getenv("ES_EMBEDDINGS_INDEX_NAME", "flywheel_embeddings_index")
EMBEDDING_DIMS = 2048

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

EMBEDDINGS_INDEX_SETTINGS = {
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine",
            },
            "tool_name": {"type": "keyword"},
            "query_text": {"type": "text"},
            "record_id": {"type": "keyword"},
            "timestamp": {"type": "date"},
            "record": {
                "type": "object",
                "dynamic": False,  # Don't map any fields in record
                "properties": {},  # No properties to map
            },
        },
    },
}


def get_es_client() -> Elasticsearch:
    """Get a working Elasticsearch client, retrying if needed."""
    global _es_client

    # Return existing connection if available
    if _es_client is not None:
        try:
            # Verify connection is alive
            if _es_client.ping():
                health = _es_client.cluster.health()
                if health["status"] in ["yellow", "green"]:
                    return _es_client
            # Connection dead, clean up and recreate
            close_es_client()
        except Exception:
            # Connection dead, clean up and recreate
            close_es_client()

    # Create new connection with retry logic
    for attempt in range(30):  # Try for up to 30 seconds
        try:
            client = Elasticsearch(hosts=[ES_URL])
            if client.ping():
                health = client.cluster.health()
                if health["status"] in ["yellow", "green"]:
                    logger.info(f"Elasticsearch is ready! Status: {health['status']}")
                    _es_client = client

                    # Create primary index if it doesn't exist
                    client.indices.refresh()  # Refresh before checking existence
                    if not client.indices.exists(index=ES_COLLECTION_NAME):
                        logger.info(f"Creating primary index: {ES_COLLECTION_NAME}...")
                        client.indices.create(index=ES_COLLECTION_NAME, body=ES_INDEX_SETTINGS)
                    else:
                        logger.info(f"Primary index '{ES_COLLECTION_NAME}' already exists.")

                    return _es_client
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


def close_es_client():
    """Close the Elasticsearch connection."""
    global _es_client
    if _es_client:
        try:
            _es_client.close()
        except Exception as e:
            logger.warning(f"Error closing Elasticsearch client: {e}")
        _es_client = None


def ensure_embeddings_index(client: Elasticsearch, index_name: str):
    """Ensure the dedicated embeddings index exists with the correct mapping."""
    client.indices.refresh()  # Refresh before checking existence
    if not client.indices.exists(index=index_name):
        logger.info(f"Creating embeddings index: {index_name}...")
        client.indices.create(index=index_name, body=EMBEDDINGS_INDEX_SETTINGS)
    else:
        logger.info(f"Embeddings index '{index_name}' already exists.")


def index_embeddings_to_es(
    client: Elasticsearch,
    binned_data: dict[str, list[tuple[list[float], dict[str, Any]]]],
    workload_id: str,
    client_id: str,
) -> str:
    """Index embeddings to Elasticsearch."""
    ts = int(datetime.utcnow().timestamp())
    index_name = f"{ES_EMBEDDINGS_INDEX_NAME}_{workload_id}_{client_id}_{ts}"
    ensure_embeddings_index(client, index_name)

    actions = []
    for tool_name, examples in binned_data.items():
        for embedding_vector, record in examples:
            user_query = extract_user_query(record)
            doc_id = f"{tool_name}_{hash(user_query)}_{record.get('timestamp', time.time())}"

            actions.append(
                {
                    "_op_type": "index",
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": {
                        "embedding": embedding_vector,
                        "tool_name": tool_name,
                        "query_text": user_query,
                        "record_id": record.get("workload_id"),
                        "timestamp": record.get("timestamp"),
                        "record": record,
                    },
                }
            )

    if actions:
        success, failed = bulk(
            client, actions, chunk_size=1000, request_timeout=60, refresh="wait_for"
        )
        if failed:
            logger.error(f"Failed to index {len(failed)} documents")
        else:
            logger.info(f"Successfully indexed {success} documents")

    return index_name


def search_similar_embeddings(
    client: Elasticsearch,
    query_embedding: list[float],
    index_name: str,
    max_candidates: int = 50,  # Remove None option, provide default
) -> list[tuple[float, str, dict[str, Any]]]:  # Fix return type
    """Search for similar embeddings in Elasticsearch."""

    search_request = {
        "knn": {
            "field": "embedding",
            "k": max_candidates,
            "num_candidates": max(50, max_candidates * 2),
            "query_vector": query_embedding,
        },
        "_source": ["query_text", "tool_name", "record_id", "timestamp", "record"],
    }

    try:
        response = client.search(index=index_name, body=search_request)
        hits = response.get("hits", {}).get("hits", [])

        candidates = []  # Fix: Use list, not dict
        for hit in hits:
            score = hit.get("_score", 0.0)
            source = hit.get("_source", {})
            tool_name = source.get("tool_name")
            record = source.get("record")

            candidates.append((score, tool_name, record))  # Fix: Correct format

        return candidates

    except Exception as e:
        logger.error(f"Error during Elasticsearch k-NN search: {e}")
        return []


def delete_embeddings_index(client: Elasticsearch, index_name: str = ES_EMBEDDINGS_INDEX_NAME):
    """Delete the embeddings index to clean up resources."""
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            logger.info(f"Deleted embeddings index: {index_name}")
        else:
            logger.info(f"Embeddings index '{index_name}' does not exist, nothing to delete")
    except Exception as e:
        logger.error(f"Error deleting embeddings index: {e}")
