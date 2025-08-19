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

"""Unit tests for MLflow integration module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.api.models import WorkloadClassification
from src.config import MLflowConfig
from src.lib.integration.mlflow_client import MLflowClient


class TestMLflowClient:
    """Test cases for MLflowClient class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mlflow_config = MLflowConfig(enabled=True)
        self.client = MLflowClient(self.mlflow_config)

        # Minimal realistic fixture based on actual results.json structure
        self.results_fixture = {
            "custom-tool-calling": [
                {
                    "item": {
                        "request": {
                            "model": "meta/llama-3.1-70b-instruct",
                            "messages": [
                                {"role": "system", "content": "system prompt"},
                                {"role": "user", "content": "user prompt"},
                            ],
                            "tools": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "structured_rag",
                                        "parameters": {
                                            "properties": {
                                                "query": {"type": "string"},
                                                "user_id": {"type": "string"},
                                            },
                                            "required": ["query", "user_id"],
                                            "type": "object",
                                        },
                                    },
                                },
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "ProductValidation",
                                        "parameters": {
                                            "properties": {"message": {"type": "string"}},
                                            "required": ["message"],
                                            "type": "object",
                                        },
                                    },
                                },
                            ],
                        },
                        "response": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "",
                                        "tool_calls": [
                                            {
                                                "type": "function",
                                                "function": {
                                                    "name": "ProductValidation",
                                                    "arguments": {
                                                        "message": "What is the order status?"
                                                    },
                                                },
                                            }
                                        ],
                                    }
                                }
                            ]
                        },
                        "workload_id": "aiva_1",
                        "client_id": "dev",
                        "timestamp": 1746138417,
                    },
                    "sample": {
                        "output_text": "tool_call_output",
                        "response": {
                            "id": "chat-1",
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "index": 0,
                                    "logprobs": None,
                                    "message": {
                                        "content": "assistant response",
                                        "refusal": None,
                                        "role": "assistant",
                                        "audio": None,
                                        "function_call": None,
                                        "tool_calls": [],
                                    },
                                    "stop_reason": 128008,
                                }
                            ],
                            "created": 1751986339,
                            "model": "meta/llama-3.2-1b-instruct",
                            "object": "chat.completion",
                            "service_tier": None,
                            "system_fingerprint": None,
                            "usage": {
                                "completion_tokens": 27,
                                "prompt_tokens": 1285,
                                "total_tokens": 1312,
                                "completion_tokens_details": None,
                                "prompt_tokens_details": None,
                            },
                            "prompt_logprobs": None,
                        },
                    },
                    "metrics": {
                        "tool-calling-accuracy": {
                            "scores": {
                                "function_name_accuracy": {"value": 0.0, "stats": None},
                                "function_name_and_args_accuracy": {"value": 0.0, "stats": None},
                            }
                        },
                        "correctness": {"scores": {"rating": {"value": 0.0, "stats": None}}},
                    },
                    "requests": [
                        {
                            "request": {
                                "model": "meta/llama-3.2-1b-instruct",
                                "messages": [
                                    {"content": "system prompt", "role": "system"},
                                    {"content": "user prompt", "role": "user"},
                                ],
                                "tools": [
                                    {
                                        "function": {
                                            "name": "structured_rag",
                                            "parameters": {
                                                "properties": {
                                                    "query": {"type": "string"},
                                                    "user_id": {"type": "string"},
                                                },
                                                "required": ["query", "user_id"],
                                                "type": "object",
                                            },
                                        },
                                        "type": "function",
                                    },
                                    {
                                        "function": {
                                            "name": "ProductValidation",
                                            "parameters": {
                                                "properties": {"message": {"type": "string"}},
                                                "required": ["message"],
                                                "type": "object",
                                            },
                                        },
                                        "type": "function",
                                    },
                                ],
                                "tool_choice": "required",
                            },
                            "response": {
                                "id": "chat-1",
                                "choices": [
                                    {
                                        "finish_reason": "stop",
                                        "index": 0,
                                        "logprobs": None,
                                        "message": {
                                            "content": "assistant response",
                                            "role": "assistant",
                                        },
                                        "stop_reason": 128008,
                                    }
                                ],
                                "created": 1751986339,
                                "model": "meta/llama-3.2-1b-instruct",
                                "object": "chat.completion",
                                "usage": {
                                    "completion_tokens": 27,
                                    "prompt_tokens": 1285,
                                    "total_tokens": 1312,
                                },
                                "prompt_logprobs": None,
                            },
                        }
                    ],
                }
            ]
        }

    def test_extract_metrics(self):
        """Test extracting metrics from evaluation item."""
        # Use the first item from our realistic fixture
        item = self.results_fixture["custom-tool-calling"][0]

        metrics = self.client._extract_metrics(item, WorkloadClassification.TOOL_CALLING)

        assert metrics["function_name_accuracy"] == 0.0
        assert metrics["function_name_and_args_accuracy"] == 0.0
        assert metrics["correctness_rating"] == 0.0
        assert metrics["total_tokens"] == 1312
        assert metrics["prompt_tokens"] == 1285
        assert metrics["completion_tokens"] == 27

    def test_extract_metadata(self):
        """Test extracting metadata from evaluation item."""
        # Use the first item from our realistic fixture
        item = self.results_fixture["custom-tool-calling"][0]

        metadata = self.client._extract_metadata(item)

        assert metadata["model"] == "meta/llama-3.2-1b-instruct"
        assert metadata["workload_id"] == "aiva_1"
        assert metadata["client_id"] == "dev"
        assert metadata["timestamp"] == 1746138417

    def test_load_results(self):
        """Test loading results from JSON file."""
        # Create temporary JSON file with realistic data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.results_fixture, f)
            temp_file = Path(f.name)

        try:
            results = self.client._load_results(temp_file)
            assert results == self.results_fixture
            assert "custom-tool-calling" in results
            assert len(results["custom-tool-calling"]) == 1
        finally:
            temp_file.unlink()

    def test_load_results_file_not_found(self):
        """Test loading results from non-existent file."""
        with pytest.raises(FileNotFoundError):
            self.client._load_results(Path("/non/existent/file.json"))

    def test_extract_metrics_with_missing_values(self):
        """Test extracting metrics when some values are missing."""
        # Create an item with missing metrics
        item_with_missing = {
            "metrics": {
                "tool-calling-accuracy": {
                    "scores": {
                        "function_name_accuracy": {"value": 0.95},
                        # Missing function_name_and_args_accuracy
                    }
                },
                "correctness": {
                    "scores": {
                        "rating": {"value": 0.92},
                    }
                },
            },
            "sample": {
                "response": {
                    "usage": {
                        "total_tokens": 1500,
                        # Missing prompt_tokens and completion_tokens
                    }
                }
            },
        }

        metrics = self.client._extract_metrics(
            item_with_missing, WorkloadClassification.TOOL_CALLING
        )

        assert metrics["function_name_accuracy"] == 0.95
        assert metrics["function_name_and_args_accuracy"] is None
        assert metrics["correctness_rating"] == 0.92
        assert metrics["total_tokens"] == 1500
        assert metrics["prompt_tokens"] is None
        assert metrics["completion_tokens"] is None

    def test_extract_metrics_generic_workload(self):
        """Test extracting metrics for generic workload (llm-as-judge)."""
        # Create an item with llm-as-judge metrics
        item_generic = {
            "metrics": {
                "llm-judge": {
                    "scores": {
                        "similarity": {"value": 8.5},
                    }
                },
            },
            "sample": {
                "response": {
                    "usage": {
                        "total_tokens": 1200,
                        "prompt_tokens": 1000,
                        "completion_tokens": 200,
                    }
                }
            },
        }

        metrics = self.client._extract_metrics(item_generic, WorkloadClassification.GENERIC)

        assert metrics["similarity"] == 8.5
        assert metrics["total_tokens"] == 1200
        assert metrics["prompt_tokens"] == 1000
        assert metrics["completion_tokens"] == 200

    def test_extract_metadata_with_missing_values(self):
        """Test extracting metadata when some values are missing."""
        # Create an item with missing metadata
        item_with_missing = {
            "sample": {
                "response": {
                    # Missing model
                }
            },
            "item": {
                "workload_id": "workload_001",
                # Missing client_id and timestamp
            },
        }

        metadata = self.client._extract_metadata(item_with_missing)

        assert metadata["model"] is None
        assert metadata["workload_id"] == "workload_001"
        assert metadata["client_id"] is None
        assert metadata["timestamp"] is None

    @patch("src.lib.integration.mlflow_client.requests.get")
    def test_download_and_process_eval_success(self, mock_get):
        """Test successful download and processing of evaluation."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = b"fake_zip_content"
        mock_get.return_value = mock_response

        # Mock zipfile operations
        with patch("src.lib.integration.mlflow_client.zipfile.ZipFile") as mock_zip:
            mock_zip.return_value.__enter__.return_value.namelist.return_value = ["file1", "file2"]

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Create a mock results.json file with realistic data
                results_file = temp_path / "results.json"
                results_file.write_text(json.dumps(self.results_fixture))

                # Mock the glob method by patching the Path class method
                with patch("pathlib.Path.glob") as mock_glob:
                    # Make glob return our results file
                    mock_glob.return_value = iter([results_file])

                    result = self.client.download_and_process_eval(
                        eval_id="test-eval",
                        nmp_eval_uri="http://test-uri",
                        save_dir=temp_path,
                        model="test/model",
                        eval_type="base-eval",
                    )

                    assert result is not None
                    assert result.name == "test_model_base-eval.json"

    @patch("src.lib.integration.mlflow_client.requests.get")
    def test_download_and_process_eval_failure(self, mock_get):
        """Test failed download of evaluation."""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = self.client.download_and_process_eval(
                eval_id="test-eval",
                nmp_eval_uri="http://test-uri",
                save_dir=temp_path,
                model="test/model",
                eval_type="base-eval",
            )

            assert result is None

    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_upload_result_disabled(self, mock_mlflow):
        """Test MLflow upload when disabled."""
        mlflow_config = MLflowConfig(enabled=False)
        client = MLflowClient(mlflow_config)

        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            temp_file = Path(f.name)

            result = client.upload_result(
                results_path=temp_file,
                flywheel_run_id="test-run",
                model_name="test/model",
                eval_type="base-eval",
            )

            assert result is None
            mock_mlflow.set_tracking_uri.assert_not_called()

    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_upload_result_enabled(self, mock_mlflow):
        """Test MLflow upload when enabled."""
        mlflow_config = MLflowConfig(enabled=True)
        client = MLflowClient(mlflow_config)

        # Mock MLflow operations
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Mock tracking URI
        mock_mlflow.get_tracking_uri.return_value = "http://mlflow:5000"

        # Create temporary results file with realistic data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.results_fixture, f)
            temp_file = Path(f.name)

        try:
            result = client.upload_result(
                results_path=temp_file,
                flywheel_run_id="test-run",
                model_name="test/model",
                eval_type="base-eval",
                workload_type=WorkloadClassification.TOOL_CALLING,
            )

            assert result == "http://mlflow:5000/#/experiments/exp-123"
            mock_mlflow.set_tracking_uri.assert_called_once_with(mlflow_config.tracking_uri)
            mock_mlflow.log_param.assert_called()
            mock_mlflow.log_metrics.assert_called()

        finally:
            temp_file.unlink()

    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_cleanup_experiment_disabled(self, mock_mlflow):
        """Test MLflow cleanup when disabled."""
        mlflow_config = MLflowConfig(enabled=False)
        client = MLflowClient(mlflow_config)

        result = client.cleanup_experiment("test-experiment")

        assert result is True
        mock_mlflow.set_tracking_uri.assert_not_called()

    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_cleanup_experiment_enabled(self, mock_mlflow):
        """Test MLflow cleanup when enabled."""
        mlflow_config = MLflowConfig(enabled=True)
        client = MLflowClient(mlflow_config)

        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        result = client.cleanup_experiment("test-experiment")

        assert result is True
        mock_mlflow.set_tracking_uri.assert_called_once_with(mlflow_config.tracking_uri)
        mock_mlflow.delete_experiment.assert_called_once_with("exp-123")

    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_cleanup_experiment_not_found(self, mock_mlflow):
        """Test MLflow cleanup when experiment not found."""
        mlflow_config = MLflowConfig(enabled=True)
        client = MLflowClient(mlflow_config)

        # Mock experiment not found
        mock_mlflow.get_experiment_by_name.return_value = None

        result = client.cleanup_experiment("test-experiment")

        assert result is True
        mock_mlflow.delete_experiment.assert_not_called()

    @patch.object(MLflowClient, "upload_result")
    @patch.object(MLflowClient, "download_and_process_eval")
    def test_upload_evaluation_results_success(self, mock_download, mock_upload):
        """Test the full upload_evaluation_results workflow (success)."""
        mock_download.return_value = Path("/tmp/fake_results.json")
        mock_upload.return_value = "http://mlflow:5000/#/experiments/exp-123"

        result = self.client.upload_evaluation_results(
            job_id="job-1",
            nmp_eval_uri="http://nmp/uri",
            flywheel_run_id="fw-1",
            model_name="test/model",
            eval_type="base-eval",
        )

        assert result == "http://mlflow:5000/#/experiments/exp-123"
        mock_download.assert_called_once_with(
            eval_id="job-1",
            nmp_eval_uri="http://nmp/uri",
            save_dir=Path("./mlruns/temp_mlflow_results_job-1"),
            model="test/model",
            eval_type="base-eval",
        )
        mock_upload.assert_called_once_with(
            results_path=Path("/tmp/fake_results.json"),
            flywheel_run_id="fw-1",
            model_name="test/model",
            eval_type="base-eval",
            workload_type=WorkloadClassification.GENERIC,
        )

    @patch.object(MLflowClient, "upload_result")
    @patch.object(MLflowClient, "download_and_process_eval")
    def test_upload_evaluation_results_download_fail(self, mock_download, mock_upload):
        """Test upload_evaluation_results returns None if download fails."""
        mock_download.return_value = None
        result = self.client.upload_evaluation_results(
            job_id="job-1",
            nmp_eval_uri="http://nmp/uri",
            flywheel_run_id="fw-1",
            model_name="test/model",
            eval_type="base-eval",
        )
        assert result is None
        mock_download.assert_called_once()
        mock_upload.assert_not_called()

    @patch("src.lib.integration.mlflow_client.shutil.rmtree")
    @patch.object(MLflowClient, "upload_result")
    @patch.object(MLflowClient, "download_and_process_eval")
    def test_upload_evaluation_results_cleanup_on_success(
        self, mock_download, mock_upload, mock_rmtree
    ):
        """Test that temporary directory is cleaned up on successful upload."""
        mock_download.return_value = Path("/tmp/fake_results.json")
        mock_upload.return_value = "http://mlflow:5000/#/experiments/exp-123"

        result = self.client.upload_evaluation_results(
            job_id="job-1",
            nmp_eval_uri="http://nmp/uri",
            flywheel_run_id="fw-1",
            model_name="test/model",
            eval_type="base-eval",
        )

        assert result == "http://mlflow:5000/#/experiments/exp-123"
        # Verify cleanup was called
        mock_rmtree.assert_called_once_with(Path("./mlruns/temp_mlflow_results_job-1"))

    @patch("src.lib.integration.mlflow_client.shutil.rmtree")
    @patch.object(MLflowClient, "upload_result")
    @patch.object(MLflowClient, "download_and_process_eval")
    def test_upload_evaluation_results_cleanup_on_exception(
        self, mock_download, mock_upload, mock_rmtree
    ):
        """Test that temporary directory is cleaned up even when an exception occurs."""
        mock_download.side_effect = Exception("Download failed")

        result = self.client.upload_evaluation_results(
            job_id="job-1",
            nmp_eval_uri="http://nmp/uri",
            flywheel_run_id="fw-1",
            model_name="test/model",
            eval_type="base-eval",
        )

        # Method should return None when exception occurs
        assert result is None
        # Verify cleanup was called even though an exception occurred
        mock_rmtree.assert_called_once_with(Path("./mlruns/temp_mlflow_results_job-1"))

    @patch("src.lib.integration.mlflow_client.shutil.rmtree")
    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_cleanup_experiment_with_artifacts(self, mock_mlflow, mock_rmtree):
        """Test MLflow cleanup when enabled, including artifact cleanup."""
        mlflow_config = MLflowConfig(enabled=True, artifact_location="./mlruns")
        client = MLflowClient(mlflow_config)

        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Mock Path.exists to return True for artifact directory
        with patch("pathlib.Path.exists", return_value=True):
            result = client.cleanup_experiment("test-experiment")

        assert result is True
        mock_mlflow.set_tracking_uri.assert_called_once_with(mlflow_config.tracking_uri)
        mock_mlflow.delete_experiment.assert_called_once_with("exp-123")
        # Verify artifact cleanup was called
        mock_rmtree.assert_called_once_with(Path("./mlruns/exp-123"))

    @patch("src.lib.integration.mlflow_client.shutil.rmtree")
    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_cleanup_experiment_artifacts_not_found(self, mock_mlflow, mock_rmtree):
        """Test MLflow cleanup when artifacts directory doesn't exist."""
        mlflow_config = MLflowConfig(enabled=True, artifact_location="./mlruns")
        client = MLflowClient(mlflow_config)

        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Mock Path.exists to return False for artifact directory
        with patch("pathlib.Path.exists", return_value=False):
            result = client.cleanup_experiment("test-experiment")

        assert result is True
        mock_mlflow.delete_experiment.assert_called_once_with("exp-123")
        # Verify artifact cleanup was NOT called since directory doesn't exist
        mock_rmtree.assert_not_called()

    @patch("src.lib.integration.mlflow_client.shutil.rmtree")
    @patch("src.lib.integration.mlflow_client.mlflow")
    def test_cleanup_experiment_artifact_cleanup_fails(self, mock_mlflow, mock_rmtree):
        """Test MLflow cleanup continues even if artifact cleanup fails."""
        mlflow_config = MLflowConfig(enabled=True, artifact_location="./mlruns")
        client = MLflowClient(mlflow_config)

        # Mock experiment
        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        # Mock artifact cleanup to fail
        mock_rmtree.side_effect = Exception("Artifact cleanup failed")

        # Mock Path.exists to return True for artifact directory
        with patch("pathlib.Path.exists", return_value=True):
            result = client.cleanup_experiment("test-experiment")

        assert result is True
        mock_mlflow.delete_experiment.assert_called_once_with("exp-123")
        # Verify artifact cleanup was attempted
        mock_rmtree.assert_called_once_with(Path("./mlruns/exp-123"))
