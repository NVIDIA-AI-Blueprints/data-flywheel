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
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

# Add both the project root and src directory to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

from lib.integration.es_client import ES_COLLECTION_NAME, get_es_client  # noqa: E402

from src.scripts.utils import validate_path  # noqa: E402

ES_CLIENT = get_es_client()


def create_openai_request_response(data: dict[str, Any]) -> dict[str, Any]:
    """Transform the data into an OpenAI-style request/response pair."""
    # Create a timestamp for the request
    timestamp = int(datetime.utcnow().timestamp())

    # Create the request structure
    request = {
        "model": "not-a-model",
        "messages": data["messages"][:-1],
        "temperature": 0.7,
        "max_tokens": 1000,
    }

    if data.get("tools"):
        request["tools"] = data["tools"]

    # Create the response structure
    response = {
        "id": f"chatcmpl-{timestamp}",
        "object": "chat.completion",
        "created": timestamp,
        "model": "not-a-model",
        "choices": [
            {
                "index": 0,
                "message": data["messages"][-1],
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(data["messages"][0]["content"].split()),
            "completion_tokens": len(data["messages"][1]["content"].split()),
            "total_tokens": len(data["messages"][0]["content"].split())
            + len(data["messages"][1]["content"].split()),
        },
    }

    return {"timestamp": timestamp, "request": request, "response": response}


def load_data_to_elasticsearch(
    workload_id: str = "",
    client_id: str = "",
    file_path: str = "aiva-final.jsonl",
    index_name: str = ES_COLLECTION_NAME,
):
    """Load test data from JSON file into Elasticsearch."""
    # Initialize Elasticsearch client
    es = ES_CLIENT

    # Validate and get the safe path
    safe_path = validate_path(file_path, is_input=True, data_dir="data")

    # Read the test data
    with open(safe_path) as f:
        test_data = [json.loads(line) for line in f]

    if test_data and test_data[0].get("workload_id"):
        # Document is already in the correct log format. However, for repeatable
        # integration tests we want the ability to override the `workload_id`
        # and `client_id` so that search queries scoped to those dynamic values
        # will find the freshly-loaded records. When callers provide non-empty
        # workload_id/client_id arguments we overwrite the existing values.

        print("Document is already in the log format. Loading with overrides.")

        for doc in test_data:
            # Ensure we do not mutate the original dict across iterations
            indexed_doc = dict(doc)

            # Override identifiers if provided by caller. This allows the
            # integration tests to generate unique IDs while reusing a static
            # JSONL fixture on disk.
            if workload_id:
                indexed_doc["workload_id"] = workload_id
            if client_id:
                indexed_doc["client_id"] = client_id

            es.index(index=index_name, document=indexed_doc)
    else:
        # Document is not in the correct format, so we need to transform it
        for item in test_data:
            # Create OpenAI-style request/response pair
            doc = create_openai_request_response(item)

            doc["workload_id"] = workload_id

            if client_id:
                doc["client_id"] = client_id

            # Index the document
            es.index(index=index_name, document=doc)

    # Flush the index to disk
    es.indices.flush(index=index_name)

    # Refresh the index to make all operations performed since the last refresh available for search
    es.indices.refresh(index=index_name)

    print("Data loaded successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load test data into Elasticsearch with specified parameters."
    )
    parser.add_argument("--workload-id", help="Unique identifier for the workload")
    parser.add_argument("--file", help="Input JSONL file path (defaults based on workload-type)")
    parser.add_argument(
        "--client-id", default="load_test_data_script", help="Optional client identifier"
    )
    parser.add_argument("--index-name", default=ES_COLLECTION_NAME, help="Optional index name")

    args = parser.parse_args()

    load_data_to_elasticsearch(
        workload_id=args.workload_id,
        client_id=args.client_id,
        file_path=args.file,
        index_name=args.index_name,
    )
