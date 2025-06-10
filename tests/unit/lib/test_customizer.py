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

from unittest.mock import MagicMock, Mock, patch

import pytest
from bson.objectid import ObjectId

from src.config import LoRAConfig, TrainingConfig
from src.lib.nemo.customizer import Customizer


@pytest.fixture
def customizer():
    """Fixture to create a Customizer instance with mocked dependencies."""
    with patch("src.lib.nemo.customizer.settings") as mock_settings:
        mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
        mock_settings.nmp_config.nmp_namespace = "test-namespace"
        with patch("src.lib.flywheel.cancellation.check_cancellation"):
            return Customizer()


@pytest.fixture
def training_config():
    return TrainingConfig(
        training_type="sft",
        finetuning_type="lora",
        epochs=2,
        batch_size=8,
        learning_rate=1e-4,
        lora=LoRAConfig(adapter_dim=32, adapter_dropout=0.1),
    )


@pytest.fixture
def sample_flywheel_run_id():
    """Fixture to provide a valid ObjectId string for tests."""
    return str(ObjectId())


class TestCustomizer:
    def test_init_without_nemo_url(self):
        """Test initialization fails without nemo_base_url."""
        with patch("src.lib.nemo.customizer.settings") as mock_settings:
            mock_settings.nmp_config.nemo_base_url = None
            with pytest.raises(AssertionError, match="nemo_base_url must be set in config"):
                Customizer()

    def test_start_training_job_success(self, customizer, training_config):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "job-123",
            "output_model": "test-namespace/model-123",
        }

        with patch("requests.post", return_value=mock_response):
            job_id, model_name = customizer.start_training_job(
                name="test-job",
                base_model="base-model-config",
                output_model_name="output-model",
                dataset_name="test-dataset",
                training_config=training_config,
            )

        assert job_id == "job-123"
        assert model_name == "test-namespace/model-123"

    def test_start_training_job_failure(self, customizer, training_config):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid request"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(Exception, match="Failed to start training job"):
                customizer.start_training_job(
                    name="test-job",
                    base_model="base-model-config",
                    output_model_name="output-model",
                    dataset_name="test-dataset",
                    training_config=training_config,
                )

    def test_get_job_uri(self, customizer):
        job_id = "test-job-123"
        expected_uri = "http://test-nemo-url/v1/customization/jobs/test-job-123"
        assert customizer.get_job_uri(job_id) == expected_uri

    def test_get_job_status_success(self, customizer):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "running", "progress": 50}

        with patch("requests.get", return_value=mock_response):
            status = customizer.get_job_status("test-job-123")
            assert status == {"status": "running", "progress": 50}

    def test_get_job_status_failure(self, customizer):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Job not found"

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(Exception, match="Failed to get job status"):
                customizer.get_job_status("test-job-123")

    def test_get_customized_model_info_success(self, customizer):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"model_id": "test-model", "status": "ready"}

        with patch("requests.get", return_value=mock_response):
            info = customizer.get_customized_model_info("test-model")
            assert info == {"model_id": "test-model", "status": "ready"}

    def test_get_customized_model_info_failure(self, customizer):
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Model not found"

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(Exception, match="Failed to get model info"):
                customizer.get_customized_model_info("test-model")

    def test_wait_for_model_sync_success(self, customizer, sample_flywheel_run_id):
        """Test successful model sync wait."""
        model_name = "test-model"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": model_name}]}

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                result = customizer.wait_for_model_sync(
                    customized_model=model_name,
                    flywheel_run_id=sample_flywheel_run_id,
                    check_interval=1,
                    timeout=1,
                )
                assert result["status"] == "synced"
                assert result["model_id"] == model_name

    def test_wait_for_model_sync_timeout(self, customizer, sample_flywheel_run_id):
        """Test model sync wait timeout."""
        model_name = "test-model"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch("requests.get", return_value=mock_response):
                with pytest.raises(TimeoutError):
                    customizer.wait_for_model_sync(
                        customized_model=model_name,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                    )

    def test_wait_for_customization_success(self, customizer, sample_flywheel_run_id):
        """Test successful customization wait."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {
                    "status": "completed",
                    "epochs_completed": 10,
                    "steps_completed": 100,
                }
                result = customizer.wait_for_customization(
                    job_id=job_id,
                    flywheel_run_id=sample_flywheel_run_id,
                    check_interval=1,
                    timeout=1,
                )
                assert result["status"] == "completed"

    def test_wait_for_customization_failure(self, customizer, sample_flywheel_run_id):
        """Test customization wait failure."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {
                    "status": "failed",
                    "status_logs": [{"detail": "Test error"}],
                }
                with pytest.raises(Exception) as exc_info:
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                    )
                assert "Test error" in str(exc_info.value)

    def test_wait_for_customization_timeout(self, customizer, sample_flywheel_run_id):
        """Test customization wait timeout."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {
                    "status": "running",
                    "percentage_done": 50,
                    "epochs_completed": 5,
                    "steps_completed": 50,
                }
                with pytest.raises(TimeoutError):
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                    )

    def test_wait_for_customization_not_enough_resources(self, customizer, sample_flywheel_run_id):
        """Test customization wait with not enough resources."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {
                    "status": "running",
                    "status_logs": [{"message": "NotEnoughResources"}],
                }
                with pytest.raises(Exception) as exc_info:
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                    )
                assert "insufficient resources" in str(exc_info.value)

    def test_delete_customized_model_success(self, customizer):
        """Test successful model deletion."""
        model_name = "test-model"
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"model": "exists"}

        mock_delete_response = MagicMock()
        mock_delete_response.status_code = 200

        with patch("requests.get", return_value=mock_get_response):
            with patch("requests.delete", return_value=mock_delete_response):
                customizer.delete_customized_model(model_name)

    def test_delete_customized_model_not_found(self, customizer):
        """Test model deletion when model not found."""
        model_name = "test-model"
        mock_get_response = MagicMock()
        mock_get_response.status_code = 404
        mock_get_response.text = "Model not found"

        with patch("requests.get", return_value=mock_get_response):
            with pytest.raises(Exception) as exc_info:
                customizer.delete_customized_model(model_name)
            assert "Model not found" in str(exc_info.value)

    def test_delete_customized_model_deletion_failure(self, customizer):
        """Test model deletion failure."""
        model_name = "test-model"
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {"model": "exists"}

        mock_delete_response = MagicMock()
        mock_delete_response.status_code = 500
        mock_delete_response.text = "Internal server error"

        with patch("requests.get", return_value=mock_get_response):
            with patch("requests.delete", return_value=mock_delete_response):
                with pytest.raises(Exception) as exc_info:
                    customizer.delete_customized_model(model_name)
                assert "Failed to delete model" in str(exc_info.value)
