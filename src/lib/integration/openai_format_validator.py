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
from typing import Any

from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.openai_format_validator")


class OpenAIFormatValidator:
    """
    Minimal validator for OpenAI Dataset format.
    Currently supports Chat Completion format validation.
    """

    def validate_chat_completion_format(self, record: dict[str, Any]) -> bool:
        """
        Minimal validation for OpenAI Chat Completion format.

        Checks:
        - Has request and response fields
        - Request has non-empty messages list
        - Response has non-empty choices list
        - Each choice has a message field

        Args:
            record: The record to validate

        Returns:
            bool: True if valid format, False otherwise
        """
        try:
            # Check basic structure exists
            if "request" not in record or "response" not in record:
                return False

            # Check request has messages
            request = record["request"]
            if not isinstance(request, dict) or "messages" not in request:
                return False
            if not isinstance(request["messages"], list):
                return False
            # Check messages list is not empty
            if len(request["messages"]) == 0:
                return False

            # Check response has choices
            response = record["response"]
            if not isinstance(response, dict) or "choices" not in response:
                return False
            if not isinstance(response["choices"], list):
                return False
            # Check choices list is not empty
            if len(response["choices"]) == 0:
                return False

            # Check each choice has a message field
            for choice in response["choices"]:
                if not isinstance(choice, dict) or "message" not in choice:
                    return False

            return True

        except Exception:
            return False

    def validate_tool_calling_quality(self, record: dict[str, Any]) -> bool:
        """Quality check for tool calling workloads."""
        return self._has_tool_calls(record)

    def _has_tool_calls(self, record: dict[str, Any]) -> bool:
        """Check if record has tool calls in response."""
        try:
            choices = record.get("response", {}).get("choices", [])
            for choice in choices:
                message = choice.get("message", {})
                # Check for tool_calls or finish_reason indicating tool calls
                if message.get("tool_calls") or choice.get("finish_reason") == "tool_calls":
                    return True
            return False
        except Exception:
            return False

    def _parse_function_arguments_to_json(self, record: dict[str, Any]) -> bool:
        """Parse function arguments from strings to JSON objects in-place.

        Returns:
            bool: True if parsing succeeded or no parsing needed, False if parsing failed
        """
        try:
            choices = record.get("response", {}).get("choices", [])
            for choice in choices:
                tool_calls = choice.get("message", {}).get("tool_calls", [])
                if tool_calls:
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        arguments = function.get("arguments")
                        if isinstance(arguments, str):
                            try:
                                # Parse JSON string to object
                                function["arguments"] = json.loads(arguments)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse function arguments: {arguments}")
                                return False  # Indicate parsing failure
            return True  # Parsing succeeded or no parsing needed
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning(f"Error parsing function arguments: {e}")
            return False  # Indicate parsing failure
