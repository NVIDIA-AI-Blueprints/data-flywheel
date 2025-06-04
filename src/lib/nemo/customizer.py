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
import time
from typing import Any

import requests

import os
from src.config import TrainingConfig, settings
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.nemo.customizer")


class Customizer:
    def __init__(self):
        """
        Initialize the Customizer.
        """
        self.nemo_url = settings.nmp_config.nemo_base_url
        assert self.nemo_url, "nemo_base_url must be set in config"

    def start_training_job(
        self,
        namespace: str,
        name: str,
        base_model: str,
        output_model_name: str,
        dataset_name: str,
        training_config: TrainingConfig,
    ) -> tuple[str, str]:
        """
        Start a new training job for model customization.

        Args:
            namespace: Namespace for the model
            name: Name for training job
            base_model: Base model configuration
            output_model_name: Name for the fine-tuned model
            dataset_name: Name of the dataset to use for fine-tuning
            training_config: Training configuration parameters

        Returns:
            Tuple containing the job ID and the customized model name
        """

        if settings.wandb_config.enabled:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "wandb-api-key": settings.wandb_config.api_key
            }
        else:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

        training_params = {
            "name": name,
            "output_model": f"{namespace}/{output_model_name}",
            "config": base_model,
            "dataset": {"name": dataset_name, "namespace": namespace},
            "hyperparameters": {
                "training_type": training_config.training_type,
                "finetuning_type": training_config.finetuning_type,
                "epochs": training_config.epochs,
                "batch_size": training_config.batch_size,
                "learning_rate": training_config.learning_rate,
                "lora": {
                    "adapter_dim": training_config.lora.adapter_dim,
                    "adapter_dropout": training_config.lora.adapter_dropout,
                },
            },
        }

        response = requests.post(f"{self.nemo_url}/v1/customization/jobs", json=training_params, headers=headers)

        if response.status_code != 200:
            msg = f"Failed to start training job. Status: {response.status_code}, Response: {response.text}"
            logger.error(msg)
            raise Exception(msg)

        customization = response.json()
        job_id = customization["id"]
        customized_model = customization["output_model"]

        return job_id, customized_model

    def get_job_uri(self, job_id: str) -> str:
        """
        Get the URI of a customization job.

        Args:
            job_id: ID of the customization job

        Returns:
            URI of the customization job
        """
        return f"{self.nemo_url}/v1/customization/jobs/{job_id}"

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Get the status of the current training job.

        Args:
            job_id: ID of the training job

        Returns:
            Dictionary containing the job status details
        """

        response = requests.get(self.get_job_uri(job_id) + "/status")

        if response.status_code != 200:
            msg = f"Failed to get job status. Status: {response.status_code}, Response: {response.text}"
            logger.error(msg)
            raise Exception(msg)

        return response.json()

    def get_customized_model_info(self, customized_model: str) -> dict[str, Any]:
        """
        Get information about the customized model.

        Args:
            customized_model: Name of the customized model

        Returns:
            Dictionary containing the customized model details
        """
        response = requests.get(
            f"{self.nemo_url}/v1/models/{customized_model}",
        )

        if response.status_code != 200:
            msg = f"Failed to get model info. Status: {response.status_code}, Response: {response.text}"
            logger.error(msg)
            raise Exception(msg)

        return response.json()

    def wait_for_model_sync(
        self, customized_model: str, check_interval: int = 30, timeout: int = 3600
    ) -> dict[str, Any]:
        """
        Wait for a model to be synced to the NMP.
        """

        start_time = time.time()

        while True:
            response = requests.get(f"{settings.nmp_config.nim_base_url}/v1/models")
            if response.status_code != 200:
                msg = f"Failed to get models list. Status: {response.status_code}, Response: {response.text}"
                logger.error(msg)
                raise Exception(msg)

            models_data = response.json().get("data", [])
            if any(model.get("id") == customized_model for model in models_data):
                return {"status": "synced", "model_id": customized_model}

            if time.time() - start_time > timeout:
                msg = (
                    f"Model {customized_model} did not sync within {timeout} second: {models_data}"
                )
                logger.error(msg)
                raise TimeoutError(msg)

            time.sleep(check_interval)

    def wait_for_customization(
        self, job_id: str, check_interval: int = 30, timeout: int = 3600, progress_callback=None
    ) -> dict[str, Any]:
        """
        Wait for a customization job to complete.

        Args:
            job_id: ID of the training job
            check_interval: Time in seconds between status checks (default: 30)
            timeout: Maximum time to wait in seconds (default: 1 hour)
            progress_callback: Optional callback function for progress updates

        Returns:
            Dictionary containing the final job status

        Raises:
            TimeoutError: If job stalls for too long
            Exception: If job fails or encounters an error
        """
        start_time = time.time()
        last_progress = 0.0

        while True:
            status_response = self.get_job_status(job_id)
            current_status = status_response.get("status", "")
            status_logs = status_response.get("status_logs", [])

            # Check job status field
            if current_status == "completed":
                if progress_callback:
                    progress_callback(
                        {
                            "progress": 100.0,
                            "epochs_completed": status_response.get("epochs_completed", 0),
                            "steps_completed": status_response.get("steps_completed", 0),
                        }
                    )
                return status_response

            elif current_status == "failed":
                # Get the last status log message for error details
                last_log = status_logs[-1] if status_logs else {}
                error_detail = last_log.get("detail", "No error details available")
                error_message = f"Job {job_id} failed: {error_detail}"
                # Log the error message
                logger.error(error_message)
                if progress_callback:
                    progress_callback({"progress": 0.0, "error": error_message})
                raise Exception(error_message)

            elif current_status == "running":
                # Get progress information
                progress = float(status_response.get("percentage_done", 0))
                epochs_completed = status_response.get("epochs_completed", 0)
                steps_completed = status_response.get("steps_completed", 0)

                logger.info(
                    f"Job {job_id} is running - Progress: {progress}%, Epochs: {epochs_completed}, Steps: {steps_completed}"
                )

                # Update start time only if progress increases
                if progress > last_progress:
                    start_time = time.time()
                    last_progress = progress

                # Call progress callback
                if progress_callback:
                    progress_callback(
                        {
                            "progress": progress,
                            "epochs_completed": epochs_completed,
                            "steps_completed": steps_completed,
                        }
                    )

            elif current_status not in ["pending", "created"]:
                # Unknown status - just log it
                if progress_callback:
                    progress_callback(
                        {"progress": 0.0, "error": f"Unknown job status '{current_status}'"}
                    )
                msg = f"Warning: Unknown job status '{current_status}'"
                logger.error(msg)
                raise Exception(msg)

            if time.time() - start_time > timeout:
                error_message = f"Job {job_id} stalled for more than {timeout} seconds"
                if progress_callback:
                    progress_callback({"progress": 0.0, "error": error_message})
                logger.error(error_message)
                raise TimeoutError(error_message)

            # Wait before next check
            time.sleep(check_interval)
