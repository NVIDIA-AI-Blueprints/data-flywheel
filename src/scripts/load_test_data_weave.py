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

import weave
from weave.trace.context.weave_client_context import (
     get_weave_client,
 )
from src.scripts.utils import validate_path  # noqa: E402
from src.config import settings

@weave.op
def call_chat(messages: list[dict], model: str = "not-a-model", temperature: float = 0.7, max_tokens: int = 1000, workload_id: str = "", client_id: str = "") -> dict:
    """Proxy function to make chat calls with Weave tracing."""
    # Create a timestamp for the request
    timestamp = int(datetime.utcnow().timestamp())
    
    # Create the response structure
    response = {
        "id": f"chatcmpl-{timestamp}",
        "object": "chat.completion",
        "created": timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": messages[-1],  # Use the last message as the response
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": len(messages[0]["content"].split()),
            "completion_tokens": len(messages[-1]["content"].split()),
            "total_tokens": len(messages[0]["content"].split()) + len(messages[-1]["content"].split()),
        },
    }
    
    return {
        "timestamp": timestamp,
        "request": {
            "model": model,
            "messages": messages[:-1],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        "response": response,
        "workload_id": workload_id,
        "client_id": client_id
    }

def load_data_to_weave(
    workload_id: str = "",
    client_id: str = "",
    file_path: str = "aiva-final.jsonl",
):
    client = get_weave_client()
    if not client:
        client = weave.init(project_name=settings.wandb_config.project)

    """Load test data from JSON file into Weave."""
    # Validate and get the safe path
    safe_path = validate_path(file_path, is_input=True, data_dir="data")

    # Read the test data
    with open(safe_path) as f:
        test_data = [json.loads(line) for line in f]

    if test_data and test_data[0].get("workload_id"):
        # Document is already in the correct log format
        print("Document is already in the log format. Loading with overrides.")

        for doc in test_data:
            # Ensure we do not mutate the original dict across iterations
            indexed_doc = dict(doc)

            # Override identifiers if provided by caller
            if workload_id:
                indexed_doc["workload_id"] = workload_id
            if client_id:
                indexed_doc["client_id"] = client_id

            # Call the weavified function to log the interaction
            call_chat(
                messages=indexed_doc["request"]["messages"],
                model=indexed_doc["request"]["model"],
                temperature=indexed_doc["request"].get("temperature", 0.7),
                max_tokens=indexed_doc["request"].get("max_tokens", 1000),
                workload_id=indexed_doc["workload_id"],
                client_id=indexed_doc["client_id"]
            )
    else:
        # Document is not in the correct format, so we need to transform it
        for item in test_data:
            # Call the weavified function to log the interaction
            call_chat(
                messages=item["messages"],
                workload_id=workload_id,
                client_id=client_id
            )

    print("Data loaded successfully to Weave.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load test data into Weave with specified parameters."
    )
    parser.add_argument("--workload-id", help="Unique identifier for the workload")
    parser.add_argument("--file", help="Input JSONL file path (defaults based on workload-type)")
    parser.add_argument(
        "--client-id", default="load_test_data_script", help="Optional client identifier"
    )

    args = parser.parse_args()

    load_data_to_weave(
        workload_id=args.workload_id,
        client_id=args.client_id,
        file_path=args.file,
    )
