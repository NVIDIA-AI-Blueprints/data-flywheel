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


def uniform_bins(max_records: int, num_tools: int) -> list[int]:
    """Calculate uniform distribution of records across tools."""
    base = max_records // num_tools
    remainder = max_records % num_tools
    return [base + 1 if i < remainder else base for i in range(num_tools)]


def get_tool_name(record: Record) -> str:
    """Get the tool name from the record."""
    tool_calls = record["response"]["choices"][0]["message"].get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        return tool_calls[0]["function"]["name"]
    return "no_tool"


def select_icl_examples(
    source_records: list[Record], config: ICLConfig, workload_type: WorkloadClassification
) -> dict[str, list[tuple[Record, str, int]]]:
    """
    Select and organize ICL examples by tool groups with uniform binning for tool_calling records,
    or simple max_records selection for normal records.
    Returns binned tool groups for later round-robin fitting per record.
    """
    if not source_records:
        return {}

    # Step 1: Group records by tools and format examples
    tool_groups: dict[str, list[tuple[Record, str, int]]] = {}

    for record in source_records:
        example_str, token_count = format_example(record)
        if not example_str:
            continue

        tool_name = get_tool_name(record)

        if tool_name not in tool_groups:
            tool_groups[tool_name] = []
        tool_groups[tool_name].append((record, example_str, token_count))

    # Step 2: Sort each tool group by token count (shortest first)
    for examples in tool_groups.values():
        examples.sort(key=lambda x: x[2])

    # Step 3: Apply different selection logic based on workflow type
    if workload_type == WorkloadClassification.TOOL_CALLING:
        # Tool calling workflow: Apply uniform binning to limit examples per tool
        if tool_groups:
            num_tools = len(tool_groups)
            bins = uniform_bins(config.max_examples, num_tools)

            # Limit each tool group to its allocated bin size
            tool_names = list(tool_groups.keys())
            for i, tool_name in enumerate(tool_names):
                allocated_size = bins[i]
                tool_groups[tool_name] = tool_groups[tool_name][:allocated_size]
    else:
        # Normal workflow: Simple max_records selection after sorting
        if tool_groups:
            # Combine all examples from all groups and sort by token count
            all_examples = []
            for examples in tool_groups.values():
                all_examples.extend(examples)
            all_examples.sort(key=lambda x: x[2])
            selected_examples = all_examples[: config.max_examples]
            tool_groups = {"generic_examples": selected_examples}

    return tool_groups


def fit_examples_for_record(
    tool_groups: dict[str, list[tuple[Record, str, int]]],
    available_tokens: int,
) -> list[tuple[Record, str, int]]:
    """
    Fit examples for a single record using round-robin selection with token checking.
    Selected examples:
        1. Grouped by tool name,
        2. Each group is already sorted by token count

    For Generic workload: `no_tool` is the only group.
    For Tool Calling workload:
            follow round-robin approach to pick smallest examples from each tool group.
            we try to fit as many examples as possible for each record, but we don't want to exceed the max context length.
    For Generic workload:
            we try to fit as many examples as possible for each record, but we don't want to exceed the max examples.
    """
    if not tool_groups or available_tokens <= 0:
        return []

    tool_names = list(tool_groups.keys())

    # Round-robin selection with token checking
    selected_examples: list[tuple[Record, str, int]] = []
    tool_indices = {tool: 0 for tool in tool_names}
    total_tokens = 0

    # Calculate maximum possible iterations based on available examples
    max_examples_per_tool = (
        max(len(examples) for examples in tool_groups.values()) if tool_groups else 0
    )

    for _ in range(max_examples_per_tool):
        for tool_name in tool_names:
            # Check if this tool has more examples available
            if tool_indices[tool_name] < len(tool_groups[tool_name]):
                example = tool_groups[tool_name][tool_indices[tool_name]]

                # Check if adding this example exceeds token limit
                if total_tokens + example[2] <= available_tokens:
                    selected_examples.append(example)
                    total_tokens += example[2]
                    tool_indices[tool_name] += 1
                else:
                    # Stop if tokens run out
                    return selected_examples

    return selected_examples


def generate_icl_records(
    records: list[Record],
    config: ICLConfig = settings.icl_config,
    selected_examples: dict[str, list[tuple[Record, str, int]]] | None = None,
) -> list[Record]:
    """Generate ICL records with per-record round-robin fitting and token checking."""
    if not records:
        return []

    # If selected_examples is None, select examples from the same records
    if selected_examples is None:
        workload_type = identify_workload_type(records)
        selected_examples = select_icl_examples(records, config, workload_type)

    # Inject ICL examples into each target record with per-record fitting
    result = deepcopy(records)
    remaining_cnts = []

    for record in result:
        # Calculate available tokens for this specific record
        record_tokens = estimate_tokens(json.dumps(record))
        available_tokens = config.max_context_length - config.reserved_tokens - record_tokens

        if available_tokens <= 0:
            continue  # Skip if there are NOT enough tokens

        # Fit examples for this record using round-robin with token checking
        fitted_examples = fit_examples_for_record(selected_examples, available_tokens)
        example_tokens = sum(ex[2] for ex in fitted_examples)
        remaining_cnts.append(
            (
                example_tokens,
                len(fitted_examples),
                available_tokens - example_tokens,
            )
        )
        if not fitted_examples:
            continue  # No examples fit

        # Create system message with fitted examples
        example_strings = [ex[1] for ex in fitted_examples]
        concatenated_string = "\n\n".join(example_strings)

        if example_tokens <= available_tokens:
            first_message = record["request"]["messages"][0]
            if first_message["role"] == "system":
                first_message["content"] = (
                    f"{DEFAULT_SYSTEM_MESSAGE.strip()}\n\n"
                    f"{concatenated_string}\n\n"
                    f"{first_message['content']}"
                )
            else:
                system_message: Message = {
                    "role": "system",
                    "content": f"{DEFAULT_SYSTEM_MESSAGE.strip()}\n\n{concatenated_string}",
                }
                record["request"]["messages"].insert(0, system_message)
    logger.info("ICL Injection Done")
    logger.info("-------------------------------------------------")
    logger.info(f"Total ICL Eval Dataset Size: {len(result)}.")
    logger.info(f"Total Max Context Length: {config.max_context_length}.")
    logger.info(f"Total Reserved Tokens: {config.reserved_tokens}.")
    logger.info(f"Tried to fit max_examples={config.max_examples} examples per record")
    if len(remaining_cnts) > 0:
        logger.info(
            f"On Average Injected {sum(cnt[1] for cnt in remaining_cnts) / len(remaining_cnts)} examples per record"
        )
        logger.info(
            f"On Average Used {sum(cnt[0] for cnt in remaining_cnts) / len(remaining_cnts)} tokens per record"
        )
        logger.info(
            f"On Average Remaining {sum(cnt[2] for cnt in remaining_cnts) / len(remaining_cnts)} tokens per record"
        )
    else:
        logger.info("No examples were injected")

    logger.info("-------------------------------------------------")
    return result


def identify_workload_type(records: list[Record]) -> WorkloadClassification:
    """
    Identify the type of workload from the response.
    """
    # Check for tool calls in response messages
    for record in records:
        try:
            tool_calls = record["response"]["choices"][0]["message"].get("tool_calls")
            if tool_calls and len(tool_calls) > 0:
                return WorkloadClassification.TOOL_CALLING
        except (KeyError, IndexError):
            continue

    return WorkloadClassification.GENERIC


def format_evaluator(records: list[Record]) -> list[Record]:
    """
    Format records specifically for evaluation by converting tool call function arguments
    to JSON strings in the request section only.

    This ensures OpenAI API compatibility during evaluation while preserving the original
    data structure for other purposes.

    Args:
        records: List of records to format

    Returns:
        List of formatted records with tool call arguments as JSON strings
    """
    formatted_records = []

    for record in records:
        # Create a deep copy to avoid modifying the original record
        formatted_record = deepcopy(record)

        # Only process request messages for tool call argument formatting
        if "request" in formatted_record and "messages" in formatted_record["request"]:
            for message in formatted_record["request"]["messages"]:
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call and "arguments" in tool_call["function"]:
                            arguments = tool_call["function"]["arguments"]
                            # Convert arguments to JSON string if they're currently an object
                            if isinstance(arguments, dict):
                                try:
                                    tool_call["function"]["arguments"] = json.dumps(arguments)
                                except (TypeError, ValueError) as e:
                                    logger.warning(
                                        f"Failed to serialize tool call arguments: {arguments}, error: {e}"
                                    )
                                    # Keep original value if serialization fails

        formatted_records.append(formatted_record)

    return formatted_records


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


def format_training_data(
    records: list[Record], workload_type: WorkloadClassification
) -> list[dict[str, Any]]:
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
            # Current customizer expects non-empty content for assistant messages
            # workaround to convert None to ""
            # TODO: remove this once customizer is updated
            # for tool-calling workloads, convert response content to ""
            rec = {}

            for message in messages:
                if message["role"] == "assistant" and message["content"] is None:
                    message["content"] = ""

            response_message = record["response"]["choices"][0]["message"]
            if workload_type == WorkloadClassification.TOOL_CALLING:
                response_message["content"] = ""
                rec["tools"] = record["request"]["tools"]

            messages.append(response_message)
            rec["messages"] = messages

            training_data.append(rec)

        except (KeyError, IndexError) as e:
            # Log error but continue processing other records
            logger.error(f"Error processing record: {e}")
            continue

    return training_data
