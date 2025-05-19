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
import random
from copy import deepcopy
from typing import Any, TypedDict

import tiktoken

from src.api.models import WorkloadClassification
from src.config import DataSplitConfig, ICLConfig, settings
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.flywheel.util")


class Message(TypedDict):
    role: str
    content: str


class ToolCall(TypedDict):
    id: str
    type: str
    function: dict[str, Any]


class Request(TypedDict):
    messages: list[Message]


class Choice(TypedDict):
    role: str
    content: str
    tool_calls: list[ToolCall] | None
    message: Message | None


class Response(TypedDict):
    choices: list[Choice]


class Record(TypedDict):
    request: Request
    response: Response


DEFAULT_SYSTEM_MESSAGE = """You are a helpful assistant that can answer questions and help with tasks.
Here are some examples of how you should respond to different types of requests:"""


def estimate_tokens(text: str, buffer_percent: int = 20) -> int:
    """Estimate tokens in text with a safety buffer."""
    if not text:
        return 0

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
    except Exception:
        # Fallback: Count words as a simple approximation of tokens
        token_count = len(text.split())

    # Add buffer percentage
    buffer_tokens = (token_count * buffer_percent) // 100
    return token_count + buffer_tokens


def format_example(record: Record) -> tuple[str, int]:
    """Format a record into an example string and estimate its token count."""
    request_messages = "".join(
        [f"{msg['role']}: {msg['content']}\n\n" for msg in record["request"]["messages"]]
    )
    resp = record["response"]["choices"][0]["message"]
    response_content = f"{resp.get('role', 'unknown')}: {resp.get('content', 'unknown')}"

    # Add tool calls if present
    if resp.get("tool_calls"):
        tool_calls_str = json.dumps(resp["tool_calls"], indent=2)
        response_content += f"\nTool calls:\n{tool_calls_str}"

    example_str = f"For example, if the conversation looks like this:\n{request_messages}\nThen you'll respond with:\n{response_content}"
    token_count = estimate_tokens(example_str)

    return example_str, token_count


def generate_icl_records(
    records: list[Record], config: ICLConfig = settings.icl_config
) -> list[Record]:
    """
    Generate ICL records from the base records with token awareness.

    Logic:
    - Pick shortest max_examples: By sorting all records by token count (ascending) and selecting the first max_examples (shortest ones)
    - Inject ICL examples into each target record: Fit as many examples as possible into the record, starting with all and reducing if needed (while checking if it exceeds context limits)
    - Skip ICL injection for records that would exceed context limits
    """

    if not records:
        return []

    # Step 1: Pick examples with non-empty content
    potential_examples: list[tuple[Record, str, int]] = []

    for record in records:
        # Skip if we already have enough examples
        if len(potential_examples) >= config.max_examples:
            break

        # Get example and check if content is non-empty
        example_str, token_count = format_example(record)

        if example_str:
            potential_examples.append((record, example_str, token_count))

    # Step 2: Inject ICL examples into each target record
    result = deepcopy(records)
    for record in result:
        # Calculate the token size of the current record
        record_tokens = estimate_tokens(json.dumps(record))
        available_tokens = config.max_context_length - config.reserved_tokens - record_tokens

        if available_tokens <= 0:
            # skip ICL injection
            continue

        # Try to fit examples, starting with all and reducing if needed
        for num_examples in range(len(potential_examples), config.min_examples - 1, -1):
            examples_subset: list[tuple[Record, str, int]] = potential_examples[:num_examples]
            example_strings: list[str] = [ex[1] for ex in examples_subset]
            examples_tokens = sum(ex[2] for ex in examples_subset)

            # This subset fits, use it
            if examples_tokens <= available_tokens:
                concatenated_string = "\n\n".join(example_strings)

                first_message = record["request"]["messages"][0]
                if first_message["role"] == "system":
                    # Prepend DEFAULT_SYSTEM_MESSAGE if not already present
                    first_message["content"] = (
                        f"{DEFAULT_SYSTEM_MESSAGE.strip()}\n\n"
                        f"{concatenated_string}\n\n"
                        f"{first_message['content']}"
                    )

                else:
                    # If there is no system message, add one with the default message
                    system_message: Message = {
                        "role": "system",
                        "content": f"{DEFAULT_SYSTEM_MESSAGE.strip()}\n\n{concatenated_string}",
                    }
                    record["request"]["messages"].insert(0, system_message)

                break

    return result


def identify_workload_type(records: list[Record]) -> WorkloadClassification:
    """
    Identify the type of workload from the response.
    """
    tool_records = [record for record in records if record.get("request", {}).get("tools", None)]
    if len(tool_records) > 0:
        return WorkloadClassification.TOOL_CALLING
    return WorkloadClassification.GENERIC


def validate_records(
    records: list[Record], workload_id: str, split_config: DataSplitConfig
) -> None:
    if len(records) < split_config.min_total_records:
        msg = (
            f"Not enough records found for the given workload_id: {workload_id}. "
            + f"A minimum of {split_config.min_total_records} records is required, "
            + f"but only {len(records)} were found."
        )
        logger.error(msg)
        raise ValueError(msg)

    return


def split_records(
    records: list[Record], split_config: DataSplitConfig
) -> tuple[list[Record], list[Record], list[Record]]:
    """Split records into eval, train and validation sets."""

    if split_config.random_seed is not None:
        random.seed(split_config.random_seed)

    # Create indices list once and shuffle it
    indices = list(range(len(records)))
    random.shuffle(indices)

    # Split indices directly into three parts
    eval_end = split_config.eval_size
    train_end = eval_end + int((len(indices) - eval_end) * (1 - split_config.val_ratio))

    # Use the shuffled indices to get the records in one pass
    eval_records = [records[i] for i in indices[:eval_end]]
    train_records = [records[i] for i in indices[eval_end:train_end]]
    val_records = [records[i] for i in indices[train_end:]]

    return eval_records, train_records, val_records


def format_training_data(records: list[Record]) -> list[dict[str, Any]]:
    """Format training data for the model.
    Args:
        records: List of conversation records containing request and response data
    Returns:
        List of message sequences where each sequence contains the conversation
        history followed by the model's response
    Raises:
        KeyError: If required fields are missing from the record structure
        IndexError: If response choices are empty
    """
    training_data = []

    for record in records:
        try:
            # Deep copy to avoid modifying original data
            messages = deepcopy(record["request"]["messages"])

            # Validate response structure
            if not record["response"]["choices"]:
                raise IndexError(f"No choices found in response: {record}")

            response_message = record["response"]["choices"][0]["message"]
            messages.append(response_message)

            rec = {"messages": messages}

            if "tools" in record["request"]:
                rec["tools"] = record["request"]["tools"]

            training_data.append(rec)

        except (KeyError, IndexError) as e:
            # Log error but continue processing other records
            logger.error(f"Error processing record: {e}")
            continue

    return training_data
