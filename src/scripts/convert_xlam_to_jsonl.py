#!/usr/bin/env python3

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

import json
import sys
from typing import Any


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a single XLAM record to OpenAI chat completion format.
    """
    # Parse the answers and tools from string to JSON
    answers = json.loads(record["answers"])
    tools = json.loads(record["tools"])

    # Create the messages array
    messages = [
        {"role": "user", "content": record["query"]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": answer["name"], "arguments": answer["arguments"]},
                }
                for answer in answers
            ],
        },
    ]

    # Create the tools array
    converted_tools = [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]

    return {"messages": messages, "tools": converted_tools}


def convert_xlam_to_jsonl(input_file: str, output_file: str) -> None:
    """
    Convert XLAM JSON file to OpenAI chat completion JSONL format.
    """
    try:
        # Read input JSON file
        with open(input_file) as f:
            data = json.load(f)

        # Convert each record and write to output file
        with open(output_file, "w") as f:
            for record in data:
                converted_record = convert_record(record)
                f.write(json.dumps(converted_record) + "\n")

        print(f"Successfully converted {len(data)} records to {output_file}")

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_xlam_to_jsonl.py <input_file.json> <output_file.jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_xlam_to_jsonl(input_file, output_file)
