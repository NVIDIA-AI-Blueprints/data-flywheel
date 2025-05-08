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
import os
from time import sleep, time
from typing import Any

import requests

from src.api.models import NIMEvaluation, ToolEvalType, WorkloadClassification
from src.config import NIMConfig, settings
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.nemo.evaluator")

JUDGE_PROMPT = """
TASK / QUESTION
<<<
{{item.request.messages}}
>>>

REFERENCE ANSWER
<<<
{{item.response.choices[0].message.content}}
>>>

CANDIDATE ANSWER
<<<
{{sample.output_text}}
>>>
"""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator.
Read the task, reference answer, and candidate answer.
Score similarity on a scale of 1 to 10 using this rubric:

10 - Nearly identical meaning; wording differences only.
8-9 - Same key info, maybe one minor detail off.
6-7 - Rough overlap but missing or altering several points.
4-5 - Some relation but wrong on most core content.
2-3 - Largely irrelevant or incorrect.
1 - Completely irrelevant or empty.

Return ONLY the integer (1-10). No other text."""

TOOL_CALLING_JUDGE_SYSTEM_PROMPT = """
You are an expert evaluator for AI tool call outputs.
Compare the model's tool call to the ground truth,
checking for exact or semantically equivalent function
names and arguments.
"""

TOOL_CALLING_JUDGE_PROMPT = """
Evaluate this tool call output:
*Ground Truth*: {{item.response.choices[0].message.tool_calls}}
*Model Output*: {{sample.response.choices[0].message.tool_calls}}

Instructions:
- Return 'RATING' (1 is correct, 0 is incorrect)
- Check function name and ALL arguments
- Ignore harmless differences like whitespace/order
- Be strict with semantic mismatches
Respond ONLY with:
RATING: 0 or 1
"""

judge_model_params = {
    "model_id": "nvdev/meta/llama-3.3-70b-instruct",
    "url": "https://integrate.api.nvidia.com/v1/chat/completions",
    "api_key": os.environ.get("NVCF_API_KEY"),
}


class Evaluator:
    def __init__(
        self,
        llm_judge_config: NIMConfig | None = None,
        include_tools: bool = False,
        include_tool_choice: bool = False,
        include_nvext: bool = False,
    ):
        """
        Initialize the ModelEvaluator with configuration.
        """
        self.nemo_url = settings.nmp_config.nemo_base_url
        assert self.nemo_url, "nemo_base_url must be set in config"
        self.include_tools = include_tools
        self.include_tool_choice = include_tool_choice
        self.include_nvext = include_nvext

        judge_cfg = settings.llm_judge_config
        if llm_judge_config:
            # Local judge config provided explicitly
            self.judge_model_config = llm_judge_config.model_name
        elif judge_cfg.is_remote():
            self.judge_model_config = judge_cfg.get_remote_config()
        else:
            self.judge_model_config = judge_cfg.get_local_nim_config().model_name

    def get_judge_metrics(self) -> dict[str, Any]:
        return {
            "llm-judge": {
                "type": "llm-judge",
                "params": {
                    "model": self.judge_model_config,
                    "template": {
                        "messages": [
                            {
                                "role": "system",
                                "content": JUDGE_SYSTEM_PROMPT,
                            },
                            {
                                "role": "user",
                                "content": JUDGE_PROMPT,
                            },
                        ]
                    },
                    "scores": {
                        "similarity": {
                            "type": "int",
                            "parser": {"type": "regex", "pattern": "(\\d)"},
                        }
                    },
                },
            }
        }

    def get_tool_llm_as_judge_config(
        self, namespace: str, dataset_name: str, test_file: str, limit: int = 50
    ) -> dict[str, Any]:
        """
        Get LLM as judge evaluation configuration.

        Args:
            namespace: Namespace of the dataset
            dataset_name: Name of the dataset
            test_file: Name of the test file in the dataset
            limit: Maximum number of samples to evaluate

        Returns:
            Dict containing the evaluation configuration
        """
        return {
            "type": "custom",
            "tasks": {
                "llm-as-judge": {
                    "type": "chat-completion",
                    "dataset": {
                        "files_url": f"hf://datasets/{namespace}/{dataset_name}/{test_file}",
                        "limit": limit,
                    },
                    "params": self.get_template(tool_call=True),
                    "metrics": self.get_tool_judge_metrics(),
                }
            },
        }

    def get_tool_calling_metrics(self) -> dict[str, Any]:
        return {
            "tool-calling-accuracy": {
                "type": "tool-calling",
                "params": {
                    "model": self.judge_model_config,
                    "tool_calls_ground_truth": "{{ item.response.choices[0].message.tool_calls | tojson }}",
                },
            },
            "correctness": {
                "type": "llm-judge",
                "params": {
                    "model": self.judge_model_config,
                    "template": {
                        "messages": [
                            {"role": "system", "content": TOOL_CALLING_JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": TOOL_CALLING_JUDGE_PROMPT},
                        ]
                    },
                    "scores": {
                        "rating": {
                            "type": "int",
                            "parser": {"type": "regex", "pattern": r"RATING:\s*(\d+)"},
                        },
                    },
                },
            },
        }

    def get_template(self, tool_call: bool = False) -> dict[str, Any]:
        template = {
            "messages": "{{ item.request.messages | tojson }}",
        }

        if tool_call:
            template["tools"] = "{{ item.request.tools | tojson }}"
            template["tool_choice"] = "required"
        else:
            if self.include_tools or tool_call:
                template["tools"] = "{{ item.tools | tojson }}"
            if self.include_tool_choice:
                template["tool_choice"] = "{{ item.tool_choice }}"
            if self.include_nvext:
                template["nvext"] = "{{ item.nvext | tojson }}"

        return {"template": template}

    def get_llm_as_judge_config(
        self, namespace: str, dataset_name: str, test_file: str, limit: int = 50
    ) -> dict[str, Any]:
        """
        Get LLM as judge evaluation configuration.

        Args:
            namespace: Namespace of the dataset
            dataset_name: Name of the dataset
            test_file: Name of the test file in the dataset
            limit: Maximum number of samples to evaluate

        Returns:
            Dict containing the evaluation configuration
        """
        return {
            "type": "custom",
            "tasks": {
                "llm-as-judge": {
                    "type": "chat-completion",
                    "dataset": {
                        "files_url": f"hf://datasets/{namespace}/{dataset_name}/{test_file}",
                        "limit": limit,
                    },
                    "params": self.get_template(),
                    "metrics": self.get_judge_metrics(),
                }
            },
        }

    def get_tool_judge_metrics(self):
        return {
            "correctness": {
                "type": "llm-judge",
                "params": {
                    "model": self.judge_model_config,
                    "template": {
                        "messages": [
                            {"role": "system", "content": TOOL_CALLING_JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": TOOL_CALLING_JUDGE_PROMPT},
                        ]
                    },
                    "scores": {
                        "rating": {
                            "type": "int",
                            "parser": {"type": "regex", "pattern": r"RATING:\s*(\d+)"},
                        },
                    },
                },
            },
        }

    def get_tool_calling_config(
        self, namespace: str, dataset_name: str, test_file: str, limit: int = 50
    ) -> dict[str, Any]:
        """
        Get tool calling evaluation configuration.

        Args:
            namespace: Namespace of the dataset
            dataset_name: Name of the dataset
            test_file: Name of the test file in the dataset
            limit: Maximum number of samples to evaluate

        Returns:
            Dict containing the evaluation configuration
        """
        return {
            "type": "custom",
            "tasks": {
                "custom-tool-calling": {
                    "type": "chat-completion",
                    "dataset": {
                        "files_url": f"hf://datasets/{namespace}/{dataset_name}/{test_file}",
                        "limit": limit,
                    },
                    "params": self.get_template(tool_call=True),
                    "metrics": self.get_tool_calling_metrics(),
                },
            },
        }

    def get_job_uri(self, job_id: str) -> str:
        """
        Get the URI of an evaluation job.

        Args:
            job_id: ID of the evaluation job

        Returns:
            URI of the evaluation job
        """
        return f"{self.nemo_url}/v1/evaluation/jobs/{job_id}"

    def _get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Get the current status of an evaluation job.

        Args:
            job_id: ID of the evaluation job

        Returns:
            Dict containing job status and details
        """
        res = requests.get(self.get_job_uri(job_id))
        return res.json()

    def wait_for_evaluation(
        self,
        job_id: str,
        evaluation: NIMEvaluation,
        polling_interval: int = 10,
        timeout: int = 6000,
        progress_callback=None,
    ) -> dict[str, Any]:
        """
        Wait for an evaluation job to complete and notify progress through callback.

        Args:
            job_id: ID of the evaluation job
            evaluation: The NIMEvaluation object to update
            polling_interval: Time in seconds between status checks
            timeout: Maximum time in seconds to wait
            progress_callback: Optional callback function for progress updates

        Returns:
            Final job status response

        Raises:
            RuntimeError: If the job times out
            Exception: If job fails or encounters an error
        """
        start_time = time()

        while True:
            # Check for timeout
            if time() - start_time > timeout:
                logger.warning(f"Evaluation took more than {timeout} seconds")
                if progress_callback:
                    progress_callback({"progress": 0.0})
                return {"status": "timeout"}

            # Get current status
            job_data = self._get_job_status(job_id)
            status = job_data["status"]

            # Handle different status cases
            if status == "running":
                progress_value = job_data.get("status_details", {}).get("progress")
                progress = float(progress_value) if progress_value is not None else 0.0
                logger.info(f"Job status: {status} Progress: {progress}%")

                # Notify progress through callback - only send progress update
                if progress_callback:
                    progress_callback({"progress": progress})

            elif status == "completed":
                logger.info(f"Job status: {status} Progress: 100%")
                if progress_callback:
                    progress_callback({"progress": 100.0})
                return job_data

            elif status == "created":
                logger.info(f"Job status: {status} - Waiting for job to start...")
                if progress_callback:
                    progress_callback({"progress": 0.0})

            else:
                msg = f"Job status: {status} / {job_data.get('status_details', {})} "
                logger.error(msg)
                raise Exception(msg)

            # Sleep before next check
            sleep(polling_interval)

    def get_evaluation_status(self, job_id: str) -> dict[str, Any]:
        """
        Get the current status of an evaluation job.

        Args:
            job_id: ID of the evaluation job

        Returns:
            Dict containing job status and details
        """
        job_data = self._get_job_status(job_id)
        status = job_data["status"]

        if status == "running":
            progress_value = job_data.get("status_details", {}).get("progress")
            progress = float(progress_value) if progress_value is not None else 0.0
            logger.info(f"Job status: {status} Progress: {progress}%")
        elif status == "completed":
            logger.info("Job completed")

        return job_data

    def get_evaluation_results(self, job_id: str) -> dict[str, Any]:
        """
        Get the results of a completed evaluation job.

        Args:
            job_id: ID of the evaluation job

        Returns:
            Evaluation results
        """
        res = requests.get(self.get_job_uri(job_id) + "/results")
        assert res.status_code == 200, f"Failed to get evaluation results: {res.text}"
        return res.json()

    def run_evaluation(
        self,
        namespace: str,
        dataset_name: str,
        workload_type: WorkloadClassification,
        target_model: str | dict[str, Any],
        test_file: str,
        tool_eval_type: ToolEvalType | None = None,
        limit: int = 50,
    ) -> str:
        """
        Run a complete evaluation workflow.

        Args:
            namespace: Namespace of the dataset
            dataset_name: Name of the dataset
            workload_type: Type of workload to evaluate
            tool_eval_type: Type of tool evaluation to perform if workload is tool calling
            target_model: Model to evaluate. If not provided, uses the base model.
            test_file: Name of the test file in the dataset
            limit: Maximum number of samples to evaluate
        Returns:
            Evaluation results
        """
        if workload_type == WorkloadClassification.TOOL_CALLING:
            if tool_eval_type is None:
                raise ValueError("tool_eval_type must be provided for tool calling workload")

            if tool_eval_type == ToolEvalType.TOOL_CALLING_METRIC:
                config = self.get_tool_calling_config(
                    namespace=namespace, dataset_name=dataset_name, test_file=test_file, limit=limit
                )
            elif tool_eval_type == ToolEvalType.TOOL_CALLING_JUDGE:
                config = self.get_tool_llm_as_judge_config(
                    namespace=namespace, dataset_name=dataset_name, test_file=test_file, limit=limit
                )
        else:
            config = self.get_llm_as_judge_config(
                namespace=namespace, dataset_name=dataset_name, test_file=test_file, limit=limit
            )

        res = requests.post(
            f"{self.nemo_url}/v1/evaluation/jobs",
            json={"config": config, "target": {"type": "model", "model": target_model}},
        )

        assert res.status_code in (200, 201), f"Failed to launch evaluation job: {res.text}"
        return res.json()["id"]
