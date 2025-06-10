from src.api.models import WorkloadClassification
from src.config import DataSplitConfig
from src.lib.flywheel.util import format_training_data
from src.lib.integration.dataset_creator import DatasetCreator


class TestDatasetCreator:
    def setup_method(self):
        """Set up test fixtures"""
        self.split_config = DataSplitConfig(train_split=0.7, val_split=0.2, eval_split=0.1)

        self.dataset_creator = DatasetCreator(
            records=[],
            flywheel_run_id="test_run_id",
            output_dataset_prefix="test",
            workload_id="test_workload",
            split_config=self.split_config,
        )

    def test_convert_assistant_content_none_and_empty_string_replacement(self):
        """Test that assistant messages with None and empty string content are converted to single space"""

        # Test data mimicking the real scenario from the user's example
        test_records = [
            {
                "request": {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a highly intelligent chatbot designed to assist users with queries related to their purchase history. \nUse the necessary tools to fetch relevant information and answer the query based on that information. \nConsider only the current product when processing order status. Ensure that your inputs to the tools include the name of the current product.\nIf there is valid a tool output, provide a proper response to the query based on it. \nAlways rely on tool outputs to generate responses with complete accuracy. Do not hallucinate. \nIf insufficient evidence is available, respond formally stating that there is not enough information to provide an answer.\n\nThe current user id is: 125\nThe current product is: \n",
                        },
                        {
                            "role": "user",
                            "content": "Why was my GeForce Triangulation Tee order canceled?",
                        },
                        {
                            "role": "assistant",
                            "content": None,  # This should be converted to " " to fix NeMo Customizer crash
                            "tool_calls": [
                                {
                                    "id": "chatcmpl-tool-9ba6aef98cbd430cbf4df8ad5ca4fdc5",
                                    "type": "function",
                                    "function": {
                                        "name": "ToOrderStatusAssistant",
                                        "arguments": '{"query": "Why was my GeForce Triangulation Tee order canceled?", "user_id": "125"}',
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "content": "The assistant is now the Order Status Assistant. Reflect on the above conversation between the host assistant and the user. The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are Order Status Assistant, and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool. If the user changes their mind or needs help for other tasks, let the primary host assistant take control. Do not mention who you are - just act as the proxy for the assistant.",
                            "tool_call_id": "chatcmpl-tool-9ba6aef98cbd430cbf4df8ad5ca4fdc5",
                        },
                        {
                            "role": "assistant",
                            "content": "",  # Empty string should also be converted to " " to fix NIM API validation
                        },
                    ],
                    "tools": [
                        {
                            "name": "ToOrderStatusAssistant",
                            "description": "The assistant is now the Order Status Assistant. Reflect on the above conversation between the host assistant and the user. The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are Order Status Assistant, and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool. If the user changes their mind or needs help for other tasks, let the primary host assistant take control. Do not mention who you are - just act as the proxy for the assistant.",
                        }
                    ],
                },
                "response": {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,  # For tool-calling workload, this should be converted to " " for accuracy
                                "tool_calls": [
                                    {
                                        "id": "call_456",
                                        "type": "function",
                                        "function": {
                                            "name": "get_order_status",
                                            "arguments": '{"product": "GeForce Triangulation Tee", "user_id": "125"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            }
        ]

        # Test for TOOL_CALLING workload (most comprehensive scenario)
        result = format_training_data(test_records, WorkloadClassification.TOOL_CALLING)

        # Assert that None content in request was converted to single space
        assert len(result) == 1
        assistant_message_with_none = result[0]["messages"][2]
        assert assistant_message_with_none["role"] == "assistant"
        assert assistant_message_with_none["content"] == ""
        assert "tool_calls" in assistant_message_with_none  # Ensure tool_calls preserved

        # Assert that empty string content in request was converted to single space
        assistant_message_with_empty = result[0]["messages"][4]
        assert assistant_message_with_empty["role"] == "assistant"
        assert assistant_message_with_empty["content"] == ""

        # Assert that response content was converted because tool_calls exist
        response_choice = result[0]["messages"][-1]
        assert response_choice["content"] == ""
        assert "tool_calls" in response_choice
