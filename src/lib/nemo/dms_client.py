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
from collections.abc import Callable
from typing import Any

import requests

from src.config import NIMConfig, NMPConfig
from src.lib.flywheel.cancellation import check_cancellation
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.nemo.dms_client")


class DMSClient:
    """
    Client for the DMS API.

    Args:
        nmp_config: NMPConfig object
        nim: NIMConfig object
    """

    def __init__(self, nmp_config: NMPConfig, nim: NIMConfig):
        self.nmp_config = nmp_config
        self.nim = nim

    def deploy_model(self) -> dict[str, Any]:
        """Deploy a model using the DMS API.

        Returns:
            Dict containing the deployment response
        """
        url = f"{self.nmp_config.nemo_base_url}/v1/deployment/model-deployments"

        payload = self.nim.to_dms_config()
        response = requests.post(url, json=payload)

        # Bug in DMS API, if the model deployment already exists, it will return a 500
        if "model deployment already exists" in response.text:
            return
        else:
            response.raise_for_status()
            return response.json()

    def is_deployed(self) -> bool:
        """Check if a model is deployed.

        Returns:
            True if the model is deployed, False otherwise
        """
        try:
            return self.get_deployment_status() == "deployed"
        except Exception:
            return False

    def does_deployment_exist(self) -> bool:
        """Check if a model deployment exists.

        Returns:
            True if the model deployment exists, False otherwise
        """
        response = self._call_deployment_endpoint()
        return response.status_code == 200

    def get_deployment_status(self) -> str:
        """Get the status of a model deployment.

        Returns:
            Dict containing the deployment status response
        """

        response = self._call_deployment_endpoint()
        response.raise_for_status()
        return response.json()["status_details"]["status"]

    def wait_for_deployment(
        self,
        flywheel_run_id: str,
        progress_callback: Callable[[str], None] | None = None,
        timeout: int = 3600,
    ):
        """Wait for a deployment to complete.

        Args:
            progress_callback: Optional callback function to report progress.
            timeout: Maximum time to wait in seconds (default: 3600)

        Returns:
            Dict containing the final deployment status
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                check_cancellation(flywheel_run_id)
            except Exception as e:
                logger.info(f"Deployment wait cancelled: {e}")
                raise e

            status = self.get_deployment_status()

            if progress_callback is not None:
                progress_callback({"status": status})

            logger.info(f"[{time.time() - start_time:.2f}] Deployment status: {status}")
            if status == "ready":
                if progress_callback is not None:
                    progress_callback({"status": status})
                return
            # wait before next check
            time.sleep(5)
        error_message = f"Deployment did not complete within {timeout} seconds"
        if progress_callback is not None:
            progress_callback({"status": status, "error": error_message})
        raise TimeoutError(error_message)

    def wait_for_model_sync(
        self,
        model_name: str,
        flywheel_run_id: str,
        check_interval: int = 30,
        timeout: int = 3600,
    ) -> dict[str, Any]:
        """
        Wait for a model to be synced to the NMP.
        """

        start_time = time.time()

        while True:
            # Check for cancellation
            try:
                check_cancellation(flywheel_run_id)
            except Exception as e:
                logger.info(f"Model sync wait cancelled: {e}")
                raise e

            response = requests.get(f"{self.nmp_config.nim_base_url}/v1/models")
            if response.status_code != 200:
                msg = f"Failed to get models list. Status: {response.status_code}, Response: {response.text}"
                logger.error(msg)
                raise Exception(msg)

            models_data = response.json().get("data", [])
            if any(model.get("id") == model_name for model in models_data):
                return {"status": "synced", "model_id": model_name}

            if time.time() - start_time > timeout:
                msg = f"Model {model_name} did not sync within {timeout} second: {models_data}"
                logger.error(msg)
                raise TimeoutError(msg)
            # wait before next check
            time.sleep(check_interval)

    def shutdown_deployment(self):
        response = requests.delete(self.deployment_url())
        response.raise_for_status()
        return response.json()

    def deployment_url(self) -> str:
        return f"{self.nmp_config.nemo_base_url}/v1/deployment/model-deployments/{self.nmp_config.nmp_namespace}/{self.nim.nmp_model_name()}"

    def _call_deployment_endpoint(self) -> requests.Response:
        return requests.get(self.deployment_url())
