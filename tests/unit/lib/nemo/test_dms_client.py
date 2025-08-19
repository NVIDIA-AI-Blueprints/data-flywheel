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

from unittest.mock import Mock, patch

import pytest
import requests

from src.lib.nemo.dms_client import DMSClient

# Global sleep mocking to speed up tests
pytestmark = pytest.mark.usefixtures("mock_sleep_globally")


@pytest.fixture(autouse=True)
def mock_sleep_globally():
    """Mock time.sleep globally to speed up tests."""
    with patch("time.sleep"):
        yield


class TestDMSClient:
    """Test class for DMSClient initialization and basic functionality."""

    @pytest.fixture
    def mock_nmp_config(self):
        """Fixture to create mock NMPConfig."""
        mock_config = Mock()
        mock_config.nemo_base_url = "http://test-nemo-url"
        mock_config.nim_base_url = "http://test-nim-url"
        mock_config.nmp_namespace = "test-namespace"
        return mock_config

    @pytest.fixture
    def mock_nim_config(self):
        """Fixture to create mock NIMConfig."""
        mock_config = Mock()
        mock_config.to_dms_config.return_value = {"model": "test-model", "config": "test-config"}
        mock_config.nmp_model_name.return_value = "test-model-name"
        return mock_config

    @pytest.fixture
    def dms_client(self, mock_nmp_config, mock_nim_config):
        """Fixture to create DMSClient instance."""
        return DMSClient(mock_nmp_config, mock_nim_config)

    def test_init(self, mock_nmp_config, mock_nim_config):
        """Test DMSClient initialization."""
        client = DMSClient(mock_nmp_config, mock_nim_config)

        assert client.nmp_config == mock_nmp_config
        assert client.nim == mock_nim_config

    def test_deployment_url(self, dms_client):
        """Test deployment URL generation."""
        result = dms_client.deployment_url()

        expected_url = (
            "http://test-nemo-url/v1/deployment/model-deployments/test-namespace/test-model-name"
        )
        assert result == expected_url


class TestDMSClientModelDeployment:
    """Test class for model deployment functionality."""

    @pytest.fixture
    def mock_nmp_config(self):
        """Fixture to create mock NMPConfig."""
        mock_config = Mock()
        mock_config.nemo_base_url = "http://test-nemo-url"
        mock_config.nim_base_url = "http://test-nim-url"
        mock_config.nmp_namespace = "test-namespace"
        return mock_config

    @pytest.fixture
    def mock_nim_config(self):
        """Fixture to create mock NIMConfig."""
        mock_config = Mock()
        mock_config.to_dms_config.return_value = {"model": "test-model", "config": "test-config"}
        mock_config.nmp_model_name.return_value = "test-model-name"
        return mock_config

    @pytest.fixture
    def dms_client(self, mock_nmp_config, mock_nim_config):
        """Fixture to create DMSClient instance."""
        return DMSClient(mock_nmp_config, mock_nim_config)

    def test_deploy_model_success(self, dms_client):
        """Test successful model deployment."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "deployment_id": "test-deployment-id",
            "status": "deploying",
        }
        mock_response.raise_for_status.return_value = None
        mock_response.text = "Success"

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = dms_client.deploy_model()

            mock_post.assert_called_once_with(
                "http://test-nemo-url/v1/deployment/model-deployments",
                json={"model": "test-model", "config": "test-config"},
            )
            assert result == {"deployment_id": "test-deployment-id", "status": "deploying"}

    def test_deploy_model_already_exists(self, dms_client):
        """Test model deployment when deployment already exists."""
        mock_response = Mock()
        mock_response.text = "model deployment already exists"

        with patch("requests.post", return_value=mock_response):
            result = dms_client.deploy_model()

            assert result is None
            mock_response.raise_for_status.assert_not_called()
            mock_response.json.assert_not_called()

    def test_deploy_model_http_error(self, dms_client):
        """Test model deployment with HTTP error."""
        mock_response = Mock()
        mock_response.text = "Internal server error"
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                dms_client.deploy_model()


class TestDMSClientDeploymentStatus:
    """Test class for deployment status checking functionality."""

    @pytest.fixture
    def mock_nmp_config(self):
        """Fixture to create mock NMPConfig."""
        mock_config = Mock()
        mock_config.nemo_base_url = "http://test-nemo-url"
        mock_config.nim_base_url = "http://test-nim-url"
        mock_config.nmp_namespace = "test-namespace"
        return mock_config

    @pytest.fixture
    def mock_nim_config(self):
        """Fixture to create mock NIMConfig."""
        mock_config = Mock()
        mock_config.to_dms_config.return_value = {"model": "test-model", "config": "test-config"}
        mock_config.nmp_model_name.return_value = "test-model-name"
        return mock_config

    @pytest.fixture
    def dms_client(self, mock_nmp_config, mock_nim_config):
        """Fixture to create DMSClient instance."""
        return DMSClient(mock_nmp_config, mock_nim_config)

    def test_does_deployment_exist_true(self, dms_client):
        """Test deployment existence check when deployment exists."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch.object(dms_client, "_call_deployment_endpoint", return_value=mock_response):
            result = dms_client.does_deployment_exist()
            assert result is True

    def test_does_deployment_exist_false(self, dms_client):
        """Test deployment existence check when deployment doesn't exist."""
        mock_response = Mock()
        mock_response.status_code = 404

        with patch.object(dms_client, "_call_deployment_endpoint", return_value=mock_response):
            result = dms_client.does_deployment_exist()
            assert result is False

    def test_get_deployment_status_success(self, dms_client):
        """Test successful deployment status retrieval."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status_details": {"status": "deployed"}}

        with patch.object(dms_client, "_call_deployment_endpoint", return_value=mock_response):
            result = dms_client.get_deployment_status()
            assert result == "deployed"

    def test_get_deployment_status_http_error(self, dms_client):
        """Test deployment status retrieval with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch.object(dms_client, "_call_deployment_endpoint", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                dms_client.get_deployment_status()

    def test_is_deployed_true(self, dms_client):
        """Test is_deployed when model is deployed."""
        with patch.object(dms_client, "get_deployment_status", return_value="deployed"):
            result = dms_client.is_deployed()
            assert result is True

    def test_is_deployed_false(self, dms_client):
        """Test is_deployed when model is not deployed."""
        with patch.object(dms_client, "get_deployment_status", return_value="deploying"):
            result = dms_client.is_deployed()
            assert result is False

    def test_is_deployed_exception(self, dms_client):
        """Test is_deployed when exception occurs."""
        with patch.object(
            dms_client, "get_deployment_status", side_effect=Exception("Network error")
        ):
            result = dms_client.is_deployed()
            assert result is False

    def test_call_deployment_endpoint(self, dms_client):
        """Test _call_deployment_endpoint method."""
        mock_response = Mock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = dms_client._call_deployment_endpoint()

            expected_url = "http://test-nemo-url/v1/deployment/model-deployments/test-namespace/test-model-name"
            mock_get.assert_called_once_with(expected_url)
            assert result == mock_response


class TestDMSClientWaitForDeployment:
    """Test class for wait_for_deployment functionality."""

    @pytest.fixture
    def mock_nmp_config(self):
        """Fixture to create mock NMPConfig."""
        mock_config = Mock()
        mock_config.nemo_base_url = "http://test-nemo-url"
        mock_config.nim_base_url = "http://test-nim-url"
        mock_config.nmp_namespace = "test-namespace"
        return mock_config

    @pytest.fixture
    def mock_nim_config(self):
        """Fixture to create mock NIMConfig."""
        mock_config = Mock()
        mock_config.to_dms_config.return_value = {"model": "test-model", "config": "test-config"}
        mock_config.nmp_model_name.return_value = "test-model-name"
        return mock_config

    @pytest.fixture
    def dms_client(self, mock_nmp_config, mock_nim_config):
        """Fixture to create DMSClient instance."""
        return DMSClient(mock_nmp_config, mock_nim_config)

    def test_wait_for_deployment_success(self, dms_client):
        """Test successful deployment wait."""
        with (
            patch("src.lib.nemo.dms_client.check_cancellation") as mock_check_cancellation,
            patch.object(
                dms_client, "get_deployment_status", return_value="ready"
            ) as mock_get_status,
        ):
            mock_callback = Mock()

            dms_client.wait_for_deployment("test-run-id", mock_callback, timeout=10)

            mock_check_cancellation.assert_called_with("test-run-id")
            mock_get_status.assert_called()
            mock_callback.assert_called_with({"status": "ready"})

    def test_wait_for_deployment_progress_updates(self, dms_client):
        """Test deployment wait with progress updates."""
        status_sequence = ["deploying", "deploying", "ready"]

        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch.object(dms_client, "get_deployment_status", side_effect=status_sequence),
        ):
            mock_callback = Mock()

            dms_client.wait_for_deployment("test-run-id", mock_callback, timeout=10)

            # Should be called for each status check plus final ready status
            assert mock_callback.call_count == 4  # 3 status checks + 1 final ready

    def test_wait_for_deployment_no_callback(self, dms_client):
        """Test deployment wait without progress callback."""
        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch.object(dms_client, "get_deployment_status", return_value="ready"),
        ):
            dms_client.wait_for_deployment("test-run-id", None, timeout=10)
            # Should complete without error

    def test_wait_for_deployment_cancellation(self, dms_client):
        """Test deployment wait with cancellation."""
        with (
            patch("src.lib.nemo.dms_client.check_cancellation", side_effect=Exception("Cancelled")),
            patch.object(dms_client, "get_deployment_status", return_value="deploying"),
        ):
            with pytest.raises(Exception, match="Cancelled"):
                dms_client.wait_for_deployment("test-run-id", None, timeout=10)

    def test_wait_for_deployment_timeout(self, dms_client):
        """Test deployment wait timeout."""
        # Provide enough time.time() values for the logging system and the timeout logic
        time_values = [0] * 10 + [3700] * 10  # Start time, then timeout values

        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch.object(dms_client, "get_deployment_status", return_value="deploying"),
            patch("src.lib.nemo.dms_client.time.time", side_effect=time_values),
            patch("src.lib.nemo.dms_client.logger.info"),  # Suppress logging
        ):
            mock_callback = Mock()

            with pytest.raises(
                TimeoutError, match="Deployment did not complete within 3600 seconds"
            ):
                dms_client.wait_for_deployment("test-run-id", mock_callback, timeout=3600)

            # Should call callback with error
            mock_callback.assert_called_with(
                {"status": "deploying", "error": "Deployment did not complete within 3600 seconds"}
            )


class TestDMSClientWaitForModelSync:
    """Test class for wait_for_model_sync functionality."""

    @pytest.fixture
    def mock_nmp_config(self):
        """Fixture to create mock NMPConfig."""
        mock_config = Mock()
        mock_config.nemo_base_url = "http://test-nemo-url"
        mock_config.nim_base_url = "http://test-nim-url"
        mock_config.nmp_namespace = "test-namespace"
        return mock_config

    @pytest.fixture
    def mock_nim_config(self):
        """Fixture to create mock NIMConfig."""
        mock_config = Mock()
        mock_config.to_dms_config.return_value = {"model": "test-model", "config": "test-config"}
        mock_config.nmp_model_name.return_value = "test-model-name"
        return mock_config

    @pytest.fixture
    def dms_client(self, mock_nmp_config, mock_nim_config):
        """Fixture to create DMSClient instance."""
        return DMSClient(mock_nmp_config, mock_nim_config)

    def test_wait_for_model_sync_success(self, dms_client):
        """Test successful model sync wait."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "other-model"}, {"id": "test-model"}, {"id": "another-model"}]
        }

        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch("requests.get", return_value=mock_response) as mock_get,
        ):
            result = dms_client.wait_for_model_sync(
                "test-model", "test-run-id", check_interval=1, timeout=10
            )

            mock_get.assert_called_with("http://test-nim-url/v1/models")
            assert result == {"status": "synced", "model_id": "test-model"}

    def test_wait_for_model_sync_multiple_checks(self, dms_client):
        """Test model sync wait with multiple checks."""
        # First response without the model, second response with the model
        mock_response_1 = Mock()
        mock_response_1.status_code = 200
        mock_response_1.json.return_value = {"data": [{"id": "other-model"}]}

        mock_response_2 = Mock()
        mock_response_2.status_code = 200
        mock_response_2.json.return_value = {"data": [{"id": "test-model"}]}

        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch("requests.get", side_effect=[mock_response_1, mock_response_2]),
        ):
            result = dms_client.wait_for_model_sync(
                "test-model", "test-run-id", check_interval=1, timeout=10
            )

            assert result == {"status": "synced", "model_id": "test-model"}

    def test_wait_for_model_sync_cancellation(self, dms_client):
        """Test model sync wait with cancellation."""
        with patch(
            "src.lib.nemo.dms_client.check_cancellation", side_effect=Exception("Cancelled")
        ):
            with pytest.raises(Exception, match="Cancelled"):
                dms_client.wait_for_model_sync(
                    "test-model", "test-run-id", check_interval=1, timeout=10
                )

    def test_wait_for_model_sync_api_error(self, dms_client):
        """Test model sync wait with API error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch("requests.get", return_value=mock_response),
        ):
            with pytest.raises(Exception, match="Failed to get models list. Status: 500"):
                dms_client.wait_for_model_sync(
                    "test-model", "test-run-id", check_interval=1, timeout=10
                )

    def test_wait_for_model_sync_timeout(self, dms_client):
        """Test model sync wait timeout."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "other-model"}]}

        # Provide enough time.time() values for the logging system and the timeout logic
        time_values = [0] * 10 + [15] * 10  # Start time, then timeout values

        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch("requests.get", return_value=mock_response),
            patch("src.lib.nemo.dms_client.time.time", side_effect=time_values),
            patch("src.lib.nemo.dms_client.logger.error"),  # Suppress logging
        ):
            with pytest.raises(
                TimeoutError, match="Model test-model did not sync within 10 second"
            ):
                dms_client.wait_for_model_sync(
                    "test-model", "test-run-id", check_interval=1, timeout=10
                )

    def test_wait_for_model_sync_empty_data(self, dms_client):
        """Test model sync wait with empty data response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        # Provide enough time.time() values for the logging system and the timeout logic
        time_values = [0] * 10 + [15] * 10  # Start time, then timeout values

        with (
            patch("src.lib.nemo.dms_client.check_cancellation"),
            patch("requests.get", return_value=mock_response),
            patch("src.lib.nemo.dms_client.time.time", side_effect=time_values),
            patch("src.lib.nemo.dms_client.logger.error"),  # Suppress logging
        ):
            with pytest.raises(TimeoutError):
                dms_client.wait_for_model_sync(
                    "test-model", "test-run-id", check_interval=1, timeout=10
                )


class TestDMSClientShutdown:
    """Test class for deployment shutdown functionality."""

    @pytest.fixture
    def mock_nmp_config(self):
        """Fixture to create mock NMPConfig."""
        mock_config = Mock()
        mock_config.nemo_base_url = "http://test-nemo-url"
        mock_config.nim_base_url = "http://test-nim-url"
        mock_config.nmp_namespace = "test-namespace"
        return mock_config

    @pytest.fixture
    def mock_nim_config(self):
        """Fixture to create mock NIMConfig."""
        mock_config = Mock()
        mock_config.to_dms_config.return_value = {"model": "test-model", "config": "test-config"}
        mock_config.nmp_model_name.return_value = "test-model-name"
        return mock_config

    @pytest.fixture
    def dms_client(self, mock_nmp_config, mock_nim_config):
        """Fixture to create DMSClient instance."""
        return DMSClient(mock_nmp_config, mock_nim_config)

    def test_shutdown_deployment_success(self, dms_client):
        """Test successful deployment shutdown."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "shutdown", "message": "Deployment stopped"}
        mock_response.raise_for_status.return_value = None

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            result = dms_client.shutdown_deployment()

            expected_url = "http://test-nemo-url/v1/deployment/model-deployments/test-namespace/test-model-name"
            mock_delete.assert_called_once_with(expected_url)
            assert result == {"status": "shutdown", "message": "Deployment stopped"}

    def test_shutdown_deployment_http_error(self, dms_client):
        """Test deployment shutdown with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.delete", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                dms_client.shutdown_deployment()
