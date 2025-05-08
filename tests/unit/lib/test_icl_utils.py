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
from unittest.mock import MagicMock, patch

import pytest
import tiktoken

from src.config import ICLConfig
from src.lib.flywheel.util import (
    estimate_tokens,
    format_example,
    generate_icl_records,
)

# Fixtures are automatically discovered from tests/utils/fixtures.py


class TestEstimateTokens:
    encoding = tiktoken.get_encoding("cl100k_base")

    @pytest.mark.parametrize(
        "text,buffer_percent,expected_tokens",
        [
            ("This is a test message.", 20, 6),
            ("This is a test message.", 50, 7),
        ],
    )
    def test_estimate_tokens_with_tiktoken(self, text, buffer_percent, expected_tokens):
        with patch("tiktoken.get_encoding") as mock_get_encoding:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = list(range(5))  # 5 tokens
            mock_get_encoding.return_value = mock_encoding

            result = estimate_tokens(text, buffer_percent=buffer_percent)
            assert result == expected_tokens

    @pytest.mark.parametrize(
        "text,buffer_percent,expected_tokens",
        [
            ("This is a test message. It has punctuation!", 20, 12),
            ("This is a test message. It has punctuation!", 10, 11),
            ("", 0, 0),
        ],
    )
    def test_estimate_tokens_without_tiktoken(self, text, buffer_percent, expected_tokens):
        result = estimate_tokens(text, buffer_percent=buffer_percent)
        assert result == expected_tokens

    def test_estimate_tokens_empty_string(self):
        result = estimate_tokens("")
        assert result == 0


class TestFormatExample:
    def test_format_example_simple(self, get_record_by_name):
        simple_record = get_record_by_name("simple_conversation")
        example_str, token_count = format_example(simple_record)

        assert "For example, if the conversation looks like this:" in example_str
        assert "Then you'll respond with:" in example_str
        assert "user: Hello, how are you?" in example_str
        assert "assistant: I'm doing well, thank you!" in example_str
        assert "tool calls" not in example_str.lower()
        assert token_count > 0

    def test_format_example_with_tool_calls(self, get_record_by_name):
        tool_record = get_record_by_name("with_tool_calls")
        example_str, token_count = format_example(tool_record)

        assert "For example, if the conversation looks like this:" in example_str
        assert "user: What's the weather in New York?" in example_str
        assert "tool calls" in example_str.lower()
        assert "get_weather" in example_str
        assert "New York" in example_str

        simple_example_str, simple_token_count = format_example(
            get_record_by_name("simple_conversation")
        )
        assert token_count > simple_token_count


# Test generate_icl_records function
class TestGenerateICLRecords:
    def test_generate_icl_records_basic(self, sample_records):
        result = generate_icl_records(sample_records[:3])

        assert len(result) == 3
        for record in result:
            assert record["request"]["messages"][0]["role"] == "system"
            system_content = record["request"]["messages"][0]["content"]
            assert "Hello, how are you?" in system_content
            assert "get_weather" in system_content
            assert "New York" in system_content

    def test_generate_icl_records_with_existing_system(self, get_record_by_name):
        system_record = get_record_by_name("with_system_message")
        result = generate_icl_records([system_record])

        system_content = result[0]["request"]["messages"][0]["content"]
        assert "You are a helpful assistant." in system_content
        assert "For example" in system_content

    @pytest.mark.parametrize(
        "config_params,expected_examples",
        [
            ({"max_examples": 1}, 1),
            ({"max_context_length": 10000, "reserved_tokens": 50, "max_examples": 3}, 3),
        ],
    )
    def test_generate_icl_records_with_config(
        self, sample_records, config_params, expected_examples
    ):
        with patch("src.lib.flywheel.util.estimate_tokens", return_value=10):
            config = ICLConfig(**config_params)
            result = generate_icl_records(sample_records, config)

            system_content = result[0]["request"]["messages"][0]["content"]
            assert system_content.count("For example") == expected_examples

    def test_generate_icl_records_context_limit(self, get_record_by_name):
        system_record = get_record_by_name("with_system_message")
        small_config = ICLConfig(max_context_length=100, reserved_tokens=50)
        result = generate_icl_records([system_record], small_config)

        for record in result:
            if record["request"]["messages"][0]["role"] == "system":
                system_content = record["request"]["messages"][0]["content"]
                if "You are a helpful assistant." in system_content and len(system_content) < 50:
                    assert "For example" not in system_content

    def test_generate_icl_records_oversized_record(self, oversized_record):
        tiny_config = ICLConfig(max_context_length=100, reserved_tokens=50)
        result = generate_icl_records([oversized_record], tiny_config)

        assert len(result) == 1
        assert len(result[0]["request"]["messages"]) == 1
        assert result[0]["request"]["messages"][0]["role"] == "user"

    def test_generate_icl_records_min_examples(self, sample_records):
        with patch("src.lib.flywheel.util.estimate_tokens") as mock_estimate:

            def mock_token_count(text, *args, **kwargs):
                return 10 if "Hello, how are you?" in text else 1000

            mock_estimate.side_effect = mock_token_count

            config = ICLConfig(max_context_length=1100, reserved_tokens=50, min_examples=2)
            result = generate_icl_records(sample_records, config)

            system_content = result[0]["request"]["messages"][0]["content"]
            assert "For example" in system_content
            assert "Hello, how are you?" in system_content

    def test_generate_icl_records_empty_list(self):
        result = generate_icl_records([])
        assert result == []
