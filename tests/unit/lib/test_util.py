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
import random

import pytest

from src.config import DataSplitConfig, settings
from src.lib.flywheel.util import (
    DEFAULT_SYSTEM_MESSAGE,
    Record,
    format_evaluator,
    generate_icl_records,
    identify_workload_type,
    select_icl_examples,
    split_records,
)
from src.lib.integration.data_validator import DataValidator

validator = DataValidator()


def test_generate_icl_records_basic():
    """Test basic functionality with a single record without tool calls."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
            "response": {
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi there!", "tool_calls": None}}
                ]
            },
        }
    ]
    workload_type = identify_workload_type(records)
    selected_examples = select_icl_examples(records, settings.icl_config, workload_type)
    result = generate_icl_records(records, settings.icl_config, selected_examples)
    assert len(result) == 1
    assert len(result[0]["request"]["messages"]) == 3
    assert result[0]["request"]["messages"][0]["role"] == "system"
    assert "For example" in result[0]["request"]["messages"][0]["content"]
    assert "tool calls" not in result[0]["request"]["messages"][0]["content"].lower()


def test_generate_icl_records_with_tool_calls():
    """Test with a record containing tool calls."""
    records: list[Record] = [
        {
            "request": {
                "messages": [{"role": "user", "content": "What is the weather in New York?"}]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        }
    ]

    workload_type = identify_workload_type(records)
    selected_examples = select_icl_examples(records, settings.icl_config, workload_type)
    result = generate_icl_records(records, settings.icl_config, selected_examples)
    assert len(result) == 1
    assert len(result[0]["request"]["messages"]) == 2
    assert result[0]["request"]["messages"][0]["role"] == "system"
    system_message = result[0]["request"]["messages"][0]["content"]
    assert "tool calls" in system_message.lower()
    assert "get_weather" in system_message
    assert "New York" in system_message


def test_generate_icl_records_multiple():
    """Test with multiple records, some with tool calls."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ]
            },
            "response": {
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi there!", "tool_calls": None}}
                ]
            },
        },
        {
            "request": {
                "messages": [{"role": "user", "content": "What is the weather in New York?"}]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        },
    ]

    workload_type = identify_workload_type(records)
    selected_examples = select_icl_examples(records, settings.icl_config, workload_type)
    result = generate_icl_records(records, settings.icl_config, selected_examples)
    assert len(result) == 2

    assert len(result[0]["request"]["messages"]) == 3
    assert len(result[1]["request"]["messages"]) == 2

    # Check that each record has a system message with examples
    for record in result:
        assert record["request"]["messages"][0]["role"] == "system"
        system_message = record["request"]["messages"][0]["content"]
        assert "For example" in system_message
        assert "tool calls" in system_message.lower()
        assert "get_weather" in system_message


def test_generate_icl_records_with_existing_system_message():
    """Test with records that already have a system message."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is the weather in New York?"},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I will check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        }
    ]

    result = generate_icl_records(records)
    assert len(result) == 1
    assert len(result[0]["request"]["messages"]) == 2
    assert result[0]["request"]["messages"][0]["role"] == "system"
    system_message = result[0]["request"]["messages"][0]["content"]
    assert "You are a helpful assistant." in system_message
    assert "For example" in system_message
    assert "tool calls" in system_message.lower()
    assert "get_weather" in system_message


def test_generate_icl_records_empty():
    """Test with empty records list."""
    records: list[Record] = []
    result = generate_icl_records(records)
    assert result == []


def test_generate_icl_records_invalid_structure():
    """Test with invalid record structure."""
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            "response": {
                "choices": []  # Empty choices array
            },
        }
    ]

    with pytest.raises(IndexError):
        generate_icl_records(records)


def test_generate_icl_records_sampling():
    """Test that the function samples records correctly."""
    # Create 10 records
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": f"Message {i}"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Response {i}",
                            "tool_calls": None,
                        }
                    }
                ]
            },
        }
        for i in range(10)
    ]

    # Set a fixed random seed for reproducibility
    random.seed(42)
    result = generate_icl_records(records)

    # Check that the system message contains examples from sampled records
    system_message = result[0]["request"]["messages"][0]["content"]
    sampled_messages = [f"Message {i}" for i in range(10)]
    sampled_responses = [f"Response {i}" for i in range(10)]

    # Ensure there are three unique "Message X" strings in the system message
    unique_messages = set()
    unique_responses = set()
    for message in sampled_messages:
        if message in system_message:
            unique_messages.add(message)
    for response in sampled_responses:
        if response in system_message:
            unique_responses.add(response)

    assert len(unique_messages) == 3, "There should be three unique 'Message X' strings"
    assert len(unique_responses) == 3, "There should be three unique 'Response X' strings"


def test_generate_icl_records_tool_call_formatting():
    """Test proper formatting of tool calls in examples."""
    records: list[Record] = [
        {
            "request": {
                "messages": [{"role": "user", "content": "Get weather for multiple locations"}]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "London"}',
                                    },
                                },
                            ],
                        }
                    }
                ]
            },
        }
    ]

    result = generate_icl_records(records)
    system_message = result[0]["request"]["messages"][0]["content"]

    # Check that both tool calls are properly formatted in the example
    assert "get_weather" in system_message
    assert "New York" in system_message
    assert "London" in system_message
    assert "Tool calls:" in system_message
    assert "call_1" in system_message
    assert "call_2" in system_message


def test_generate_icl_records_mixed_tool_calls():
    """Test with records containing both tool calls and regular responses."""
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": "Get weather for New York"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check the weather.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "New York"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        },
        {
            "request": {"messages": [{"role": "user", "content": "Say hello"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Hello!",
                            "tool_calls": None,
                        }
                    }
                ]
            },
        },
    ]

    result = generate_icl_records(records)
    assert len(result) == 2

    # Check that both types of responses are properly represented in the examples
    system_message = result[0]["request"]["messages"][0]["content"]
    assert "get_weather" in system_message
    assert "New York" in system_message
    assert "Hello!" in system_message
    assert "Tool calls:" in system_message


def test_validate_records_valid():
    # Test with valid number of records
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": f"Message {i}"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Response {i}",
                            "tool_calls": None,
                        }
                    }
                ]
            },
        }
        for i in range(25)
    ]

    config = DataSplitConfig(eval_size=5, val_ratio=0.1, min_total_records=20)
    validator.validate_records(records, "test_workload", config)

    with pytest.raises(ValueError) as exc_info:
        validator.validate_records(records[:10], "test_workload", config)
    assert "Not enough records found" in str(exc_info.value)


@pytest.mark.parametrize(
    "total_records,config,expected_sizes",
    [
        # Standard case (100 records, 5 eval, 10% val)
        (
            100,
            DataSplitConfig(eval_size=5, val_ratio=0.1, min_total_records=20, random_seed=42),
            {"eval": 5, "val": 10, "train": 85},
        ),
        # Larger eval set (200 records, 20 eval, 15% val)
        (
            200,
            DataSplitConfig(eval_size=20, val_ratio=0.15, min_total_records=50, random_seed=42),
            {"eval": 20, "val": 27, "train": 153},
        ),
        # Minimum case (20 records, 2 eval, 20% val)
        (
            20,
            DataSplitConfig(eval_size=2, val_ratio=0.2, min_total_records=20, random_seed=42),
            {"eval": 2, "val": 4, "train": 14},
        ),
    ],
)
def test_split_records_parameterized(total_records, config, expected_sizes):
    # Test split_records with different configurations.
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": f"Message {i}"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Response {i}",
                            "tool_calls": None,
                        }
                    }
                ]
            },
        }
        for i in range(total_records)
    ]
    validator.validate_records(records, "test_workload", config)
    eval_records, train_records, val_records = split_records(records, config)
    assert len(eval_records) == expected_sizes["eval"], "Eval set size mismatch"
    assert len(val_records) == expected_sizes["val"], "Validation set size mismatch"
    assert len(train_records) == expected_sizes["train"], "Training set size mismatch"
    assert len(eval_records) + len(train_records) + len(val_records) == total_records


def test_generate_icl_records_default_system_message():
    """Test that the default system message is used when one isn't present."""
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            "response": {
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi there!", "tool_calls": None}}
                ]
            },
        }
    ]

    result = generate_icl_records(records)
    assert len(result) == 1
    assert len(result[0]["request"]["messages"]) == 2
    assert result[0]["request"]["messages"][0]["role"] == "system"
    system_message = result[0]["request"]["messages"][0]["content"]

    # Check that the default system message is used
    print(DEFAULT_SYSTEM_MESSAGE, system_message)
    assert DEFAULT_SYSTEM_MESSAGE in system_message
    assert "Here are some examples" in system_message
    assert "For example" in system_message
    assert "tool calls" not in system_message.lower()


def test_generate_icl_records_prepends_default_system_message():
    """Test that DEFAULT_SYSTEM_MESSAGE is prepended if not present in an existing system message."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {"role": "system", "content": "Custom system message."},
                    {"role": "user", "content": "Hello"},
                ]
            },
            "response": {
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi!", "tool_calls": None}}
                ]
            },
        }
    ]
    result = generate_icl_records(records)
    system_message = result[0]["request"]["messages"][0]["content"]
    assert DEFAULT_SYSTEM_MESSAGE.strip() in system_message
    assert system_message.startswith(DEFAULT_SYSTEM_MESSAGE.strip())
    assert "Custom system message." in system_message


def test_format_evaluator():
    """Test that format_evaluator converts tool call arguments to JSON strings in request messages only."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {
                                        "location": "New York",
                                        "unit": "celsius",
                                    },  # Object format
                                },
                            }
                        ],
                    }
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check the weather for you.",
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_time",
                                        "arguments": {"timezone": "EST"},  # Object format
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        }
    ]

    result = format_evaluator(records)

    # Verify that request tool call arguments are converted to strings
    request_tool_calls = result[0]["request"]["messages"][0]["tool_calls"]
    assert (
        request_tool_calls[0]["function"]["arguments"]
        == '{"location": "New York", "unit": "celsius"}'
    )

    # Verify that response tool call arguments remain as objects (unchanged)
    response_tool_calls = result[0]["response"]["choices"][0]["message"]["tool_calls"]
    assert response_tool_calls[0]["function"]["arguments"] == {"timezone": "EST"}


def test_format_evaluator_already_strings():
    """Test that format_evaluator doesn't modify arguments that are already strings."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',  # Already a string
                                }
                            }
                        ],
                    }
                ]
            },
            "response": {"choices": [{"message": {"content": "Weather info"}}]},
        }
    ]

    result = format_evaluator(records)

    # Arguments should remain unchanged
    request_tool_calls = result[0]["request"]["messages"][0]["tool_calls"]
    assert request_tool_calls[0]["function"]["arguments"] == '{"location": "NYC"}'


def test_format_evaluator_no_tool_calls():
    """Test that format_evaluator handles records without tool calls."""
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            "response": {"choices": [{"message": {"content": "Hi there!"}}]},
        }
    ]

    result = format_evaluator(records)

    # Record should be unchanged
    assert result == records
