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
from collections import Counter

import pytest

from src.api.models import WorkloadClassification
from src.config import DataSplitConfig
from src.lib.flywheel.util import (
    Record,
    format_evaluator,
    format_training_data,
    get_tool_name,
    identify_workload_type,
    split_records,
)
from src.lib.integration.data_validator import DataValidator

validator = DataValidator()


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
        (
            100,
            DataSplitConfig(eval_size=5, val_ratio=0.1, min_total_records=20, random_seed=42),
            {"eval": 5, "val": 10, "train": 85},
        ),
        (
            200,
            DataSplitConfig(eval_size=20, val_ratio=0.15, min_total_records=50, random_seed=42),
            {"eval": 20, "val": 27, "train": 153},
        ),
        (
            20,
            DataSplitConfig(eval_size=2, val_ratio=0.2, min_total_records=20, random_seed=42),
            {"eval": 2, "val": 4, "train": 14},
        ),
    ],
)
def test_split_records_parameterized(total_records, config, expected_sizes):
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

    assert len(eval_records) == expected_sizes["eval"]
    assert len(train_records) == expected_sizes["train"]
    assert len(val_records) == expected_sizes["val"]
    assert len(eval_records) + len(train_records) + len(val_records) == total_records


def test_split_records_class_aware_stratification():
    """Test that split_records preserves class distribution across splits."""
    # Create records with different tool types to test stratification
    records: list[Record] = []

    # Create 30 records with tool_a, 20 with tool_b, 10 with no_tool
    for i in range(30):
        records.append(
            {
                "request": {"messages": [{"role": "user", "content": f"Message {i}"}]},
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Response {i}",
                                "tool_calls": [{"function": {"name": "tool_a"}}],
                            }
                        }
                    ]
                },
            }
        )

    for i in range(30, 50):
        records.append(
            {
                "request": {"messages": [{"role": "user", "content": f"Message {i}"}]},
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Response {i}",
                                "tool_calls": [{"function": {"name": "tool_b"}}],
                            }
                        }
                    ]
                },
            }
        )

    for i in range(50, 60):
        records.append(
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
        )

    config = DataSplitConfig(eval_size=6, val_ratio=0.2, random_seed=42)

    eval_records, train_records, val_records = split_records(records, config)

    # Check that each split preserves class distribution
    eval_labels = [get_tool_name(record) for record in eval_records]
    train_labels = [get_tool_name(record) for record in train_records]
    val_labels = [get_tool_name(record) for record in val_records]

    eval_distribution = Counter(eval_labels)
    train_distribution = Counter(train_labels)
    val_distribution = Counter(val_labels)

    # Assert all classes are represented in each split (if the split is large enough)
    assert len(eval_distribution) >= 2, "Eval set should contain multiple classes"
    assert len(train_distribution) >= 2, "Train set should contain multiple classes"
    assert len(val_distribution) >= 2, "Validation set should contain multiple classes"


def test_split_records_single_class_no_stratification():
    """Test that split_records handles single class correctly (no stratification)."""
    # Create records with only one tool type
    records: list[Record] = []
    for i in range(60):
        records.append(
            {
                "request": {"messages": [{"role": "user", "content": f"Message {i}"}]},
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Response {i}",
                                "tool_calls": [{"function": {"name": "single_tool"}}],
                            }
                        }
                    ]
                },
            }
        )

    config = DataSplitConfig(eval_size=6, val_ratio=0.2, random_seed=42)

    # This should work without issues even though there's only one class
    eval_records, train_records, val_records = split_records(records, config)

    # Verify correct sizes
    assert len(eval_records) == 6
    # Verify that the validation set size is approximately 20% of remaining
    remaining_after_eval = 60 - 6
    expected_val_range = range(
        int(remaining_after_eval * 0.15), int(remaining_after_eval * 0.25) + 1
    )
    assert (
        len(val_records) in expected_val_range
    ), f"Validation set size {len(val_records)} not in expected range {expected_val_range}"
    assert len(train_records) == 60 - 6 - len(val_records)


def test_split_records_no_tool():
    """Test stratification with mix of tool calls and regular responses."""
    records: list[Record] = []

    # 60 records without tool calls
    for i in range(60):
        records.append(
            {
                "request": {"messages": [{"role": "user", "content": f"Regular message {i}"}]},
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Regular response {i}",
                                "tool_calls": None,
                            }
                        }
                    ]
                },
            }
        )

    config = DataSplitConfig(eval_size=6, val_ratio=0.25, random_seed=42)
    eval_records, train_records, val_records = split_records(records, config)

    # Check that both tool and no-tool records are represented in splits
    eval_labels = Counter([get_tool_name(r) for r in eval_records])
    train_labels = Counter([get_tool_name(r) for r in train_records])
    val_labels = Counter([get_tool_name(r) for r in val_records])

    # Both "no_tool" should appear in each split
    assert "no_tool" in eval_labels
    assert "no_tool" in train_labels
    assert "no_tool" in val_labels


def test_split_records_different_seeds_produce_different_results():
    """Test that different random seeds produce different splits."""
    records: list[Record] = []

    for i in range(50):
        records.append(
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
        )

    config1 = DataSplitConfig(eval_size=5, val_ratio=0.2, random_seed=42)
    config2 = DataSplitConfig(eval_size=5, val_ratio=0.2, random_seed=123)

    eval1, train1, val1 = split_records(records, config1)
    eval2, train2, val2 = split_records(records, config2)

    # With different seeds, at least one split should be different
    # (extremely unlikely to be the same by chance)
    assert eval1 != eval2 or train1 != train2 or val1 != val2


def test_split_records_insufficient_samples_per_class_fallback():
    """Test graceful fallback when stratification fails due to insufficient samples per class."""
    records: list[Record] = []

    # Create records where some classes have only 1 sample
    # This will cause sklearn to fail with stratification
    for i in range(18):
        records.append(
            {
                "request": {"messages": [{"role": "user", "content": f"Message {i}"}]},
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": f"Response {i}",
                                "tool_calls": [{"function": {"name": "frequent_tool"}}],
                            }
                        }
                    ]
                },
            }
        )

    # Add one record with a different tool (only 1 sample - will cause stratification to fail)
    records.append(
        {
            "request": {"messages": [{"role": "user", "content": "Rare message"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Rare response",
                            "tool_calls": [{"function": {"name": "rare_tool"}}],
                        }
                    }
                ]
            },
        }
    )

    # Add one more record with another different tool (only 1 sample)
    records.append(
        {
            "request": {"messages": [{"role": "user", "content": "Another rare message"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Another rare response",
                            "tool_calls": [{"function": {"name": "another_rare_tool"}}],
                        }
                    }
                ]
            },
        }
    )

    config = DataSplitConfig(eval_size=5, val_ratio=0.2, random_seed=42)

    # This should work by falling back to non-stratified splitting
    eval_records, train_records, val_records = split_records(records, config)

    # Basic size checks
    assert len(eval_records) == 5
    assert len(eval_records) + len(train_records) + len(val_records) == 20

    # Verify all records are accounted for
    all_records = eval_records + train_records + val_records
    assert len(all_records) == 20


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


def test_identify_workload_type_with_keyerror():
    """Test identify_workload_type with malformed records that cause KeyError."""
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            "response": {"choices": []},  # Empty choices will cause IndexError
        },
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            # Missing response will cause KeyError
        },
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            "response": {
                "choices": [{"message": {"tool_calls": [{"function": {"name": "test_tool"}}]}}]
            },
        },
    ]

    result = identify_workload_type(records)
    # Should still identify as tool calling despite errors in first records
    assert result == WorkloadClassification.TOOL_CALLING


def test_format_training_data_basic():
    """Test basic functionality of format_training_data."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {"role": "user", "content": "Hello"},
                ]
            },
            "response": {"choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]},
        }
    ]

    result = format_training_data(records, WorkloadClassification.GENERIC)

    assert len(result) == 1
    assert len(result[0]["messages"]) == 2
    assert result[0]["messages"][0]["role"] == "user"
    assert result[0]["messages"][0]["content"] == "Hello"
    assert result[0]["messages"][1]["role"] == "assistant"
    assert result[0]["messages"][1]["content"] == "Hi there!"


def test_format_training_data_tool_calling():
    """Test format_training_data with tool calling workload."""
    records: list[Record] = [
        {
            "request": {
                "messages": [{"role": "user", "content": "Get weather"}],
                "tools": [{"name": "get_weather"}],
            },
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "I'll check the weather",
                            "tool_calls": [{"function": {"name": "get_weather"}}],
                        }
                    }
                ]
            },
        }
    ]

    result = format_training_data(records, WorkloadClassification.TOOL_CALLING)

    assert len(result) == 1
    assert result[0]["tools"] == [{"name": "get_weather"}]
    assert result[0]["messages"][1]["content"] == ""  # Content set to empty for tool calling


def test_format_training_data_with_none_content():
    """Test format_training_data handles None content in assistant messages."""
    records: list[Record] = [
        {
            "request": {
                "messages": [
                    {
                        "role": "assistant",
                        "content": None,
                    },  # None content should be converted to ""
                    {"role": "user", "content": "Hello"},
                ]
            },
            "response": {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]},
        }
    ]

    result = format_training_data(records, WorkloadClassification.GENERIC)

    assert result[0]["messages"][0]["content"] == ""  # None converted to empty string


def test_format_training_data_empty_choices():
    """Test format_training_data with empty choices array."""
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            "response": {"choices": []},  # Empty choices should be skipped
        },
        {
            "request": {"messages": [{"role": "user", "content": "Hello again"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]},
        },
    ]

    result = format_training_data(records, WorkloadClassification.GENERIC)

    # Should only return the valid record
    assert len(result) == 1
    assert result[0]["messages"][0]["content"] == "Hello again"


def test_format_training_data_missing_keys():
    """Test format_training_data with missing required keys."""
    records: list[Record] = [
        {
            "request": {"messages": [{"role": "user", "content": "Hello"}]},
            # Missing response key
        },
        {
            # Missing request key
            "response": {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]},
        },
        {
            "request": {"messages": [{"role": "user", "content": "Valid"}]},
            "response": {
                "choices": [{"message": {"role": "assistant", "content": "Valid response"}}]
            },
        },
    ]

    result = format_training_data(records, WorkloadClassification.GENERIC)

    # Should only return the valid record
    assert len(result) == 1
    assert result[0]["messages"][0]["content"] == "Valid"


def test_split_records_stratification_constraints():
    """Test fallback when stratification constraints aren't met."""
    records: list[Record] = []

    # Create 12 records with 4 tool types (3 records each)
    tools = ["tool_a", "tool_b", "tool_c", "tool_d"]
    for i, tool in enumerate(tools):
        for j in range(3):
            records.append(
                {
                    "request": {"messages": [{"role": "user", "content": f"test_{i}_{j}"}]},
                    "response": {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "tool_calls": [{"function": {"name": tool}}],
                                }
                            }
                        ]
                    },
                }
            )

    # eval_size (2) < classes (4) - should fallback to random
    config = DataSplitConfig(eval_size=2, val_ratio=0.1, random_seed=42)
    eval_records, train_records, val_records = split_records(records, config)

    assert len(eval_records) == 2
    assert len(eval_records) + len(train_records) + len(val_records) == 12
