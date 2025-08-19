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

from src.config import CustomizerConfig, LoRAConfig, NIMConfig, TrainingConfig
from src.lib.nemo.customizer import Customizer

# Mock time.sleep globally for all tests in this module
pytestmark = pytest.mark.usefixtures("mock_sleep_globally")


@pytest.fixture(autouse=True)
def mock_sleep_globally():
    """Automatically mock time.sleep for all tests in this module."""
    with patch("time.sleep"):
        yield


@pytest.fixture
def customizer():
    """Fixture to create a Customizer instance with mocked dependencies."""
    with (
        patch("src.lib.nemo.customizer.settings") as mock_settings,
        patch("src.lib.flywheel.cancellation.check_cancellation"),
    ):
        mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
        mock_settings.nmp_config.nmp_namespace = "test-namespace"
        mock_settings.nmp_config.nim_base_url = "http://test-nim-url"
        return Customizer()


@pytest.fixture
def customizer_config():
    """Fixture to create a CustomizerConfig instance."""
    return CustomizerConfig(
        target="test-model@2.0",
        gpus=1,
        tensor_parallel_size=1,
        data_parallel_size=1,
        use_sequence_parallel=False,
        micro_batch_size=1,
        training_precision="bf16-mixed",
        max_seq_length=4096,
    )


@pytest.fixture
def nim_config(customizer_config):
    """Fixture to create a NIMConfig instance with customizer configs."""
    return NIMConfig(
        model_name="test-model",
        context_length=2048,
        gpus=1,
        pvc_size="10Gi",
        tag="latest",
        registry_base="nvcr.io/nim",
        customization_enabled=True,
        customizer_configs=customizer_config,
    )


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

    def test_start_training_job_success(self, customizer, training_config, nim_config):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "job-123",
            "output_model": "test-namespace/model-123",
        }

        with (
            patch("requests.post", return_value=mock_response),
            patch.object(customizer, "create_customizer_config", return_value="test-config-name"),
        ):
            job_id, model_name = customizer.start_training_job(
                name="test-job",
                base_model="test-namespace/test-model",
                output_model_name="output-model",
                dataset_name="test-dataset",
                training_config=training_config,
                nim_config=nim_config,
            )

        assert job_id == "job-123"
        assert model_name == "test-namespace/model-123"

    def test_start_training_job_failure(self, customizer, training_config, nim_config):
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Invalid request"

        with (
            patch("requests.post", return_value=mock_response),
            patch.object(customizer, "create_customizer_config", return_value="test-config-name"),
        ):
            with pytest.raises(Exception, match="Failed to start training job"):
                customizer.start_training_job(
                    name="test-job",
                    base_model="test-namespace/test-model",
                    output_model_name="output-model",
                    dataset_name="test-dataset",
                    training_config=training_config,
                    nim_config=nim_config,
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
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch("requests.get", return_value=mock_response),
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
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
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch("requests.get", return_value=mock_response),
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with pytest.raises(TimeoutError):
                customizer.wait_for_model_sync(
                    customized_model=model_name,
                    flywheel_run_id=sample_flywheel_run_id,
                    check_interval=1,
                    timeout=0.001,  # Keep tiny timeout to trigger immediately
                )

    def test_wait_for_model_sync_cancellation(self, customizer, sample_flywheel_run_id):
        """Test model sync wait with cancellation."""
        model_name = "test-model"

        # Mock cancellation to raise an exception immediately
        with (
            patch("src.lib.nemo.customizer.check_cancellation") as mock_check_cancellation,
        ):
            mock_check_cancellation.side_effect = Exception("Run cancelled")
            with pytest.raises(Exception, match="Run cancelled"):
                customizer.wait_for_model_sync(
                    customized_model=model_name,
                    flywheel_run_id=sample_flywheel_run_id,
                    check_interval=1,
                    timeout=10,
                )

    def test_wait_for_model_sync_request_failure(self, customizer, sample_flywheel_run_id):
        """Test model sync wait with request failure."""
        model_name = "test-model"
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        # Mock get_db_manager to prevent actual DB calls
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch("requests.get", return_value=mock_response),
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with pytest.raises(Exception, match="Failed to get models list"):
                customizer.wait_for_model_sync(
                    customized_model=model_name,
                    flywheel_run_id=sample_flywheel_run_id,
                    check_interval=1,
                    timeout=10,
                )

    def test_wait_for_customization_success(self, customizer, sample_flywheel_run_id):
        """Test successful customization wait."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch.object(customizer, "get_job_status") as mock_get_job_status,
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
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

    def test_wait_for_customization_success_with_callback(self, customizer, sample_flywheel_run_id):
        """Test successful customization wait with progress callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        # Mock get_db_manager to prevent actual DB calls
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch.object(customizer, "get_job_status") as mock_get_job_status,
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
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
                progress_callback=progress_callback,
            )
            assert result["status"] == "completed"
            assert len(progress_updates) == 1
            assert progress_updates[0]["progress"] == 100.0

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

    def test_wait_for_customization_failure_with_callback(self, customizer, sample_flywheel_run_id):
        """Test customization wait failure with progress callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {
                    "status": "failed",
                    "status_logs": [{"detail": "Callback test error"}],
                }
                with pytest.raises(Exception) as exc_info:
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                        progress_callback=progress_callback,
                    )

                assert "Callback test error" in str(exc_info.value)
                assert len(progress_updates) == 1
                assert progress_updates[0]["progress"] == 0.0
                assert "error" in progress_updates[0]

    def test_wait_for_customization_failed_no_detail(self, customizer, sample_flywheel_run_id):
        """Test customization wait failure without error detail."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {
                    "status": "failed",
                    "status_logs": [{"message": "Some message without detail"}],
                }
                with pytest.raises(Exception) as exc_info:
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                    )

                assert "No error details available" in str(exc_info.value)

    def test_wait_for_customization_timeout(self, customizer, sample_flywheel_run_id):
        """Test customization wait timeout."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                # Always return running status to trigger timeout
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
                        timeout=0.001,  # Very short timeout to trigger immediately
                    )

    def test_wait_for_customization_timeout_with_callback(self, customizer, sample_flywheel_run_id):
        """Test customization wait timeout with progress callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {
                    "status": "running",
                    "percentage_done": 75,
                    "epochs_completed": 7,
                    "steps_completed": 75,
                }
                with pytest.raises(TimeoutError):
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=0.001,  # Very short timeout to trigger immediately
                        progress_callback=progress_callback,
                    )
                # Should have at least one progress update before timeout
                assert (
                    len(progress_updates) >= 2
                )  # At least one running update + one timeout update
                # Check that we have a progress update with 75% (not the timeout one)
                progress_values = [update["progress"] for update in progress_updates]
                assert 75.0 in progress_values
                # The last update should be the timeout callback with 0.0 progress
                assert progress_updates[-1]["progress"] == 0.0
                assert "error" in progress_updates[-1]

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

    def test_wait_for_customization_not_enough_resources_with_callback(
        self, customizer, sample_flywheel_run_id
    ):
        """Test customization wait with not enough resources and callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                # Mock job status to be running initially, then check for NotEnoughResources after timeout
                mock_get_job_status.return_value = {
                    "status": "running",
                    "status_logs": [{"message": "NotEnoughResources"}],
                }
                with pytest.raises(Exception) as exc_info:
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=0.001,
                        progress_callback=progress_callback,
                    )

                # Should get the NotEnoughResources error after timeout, not the generic timeout error
                assert "insufficient resources" in str(exc_info.value)
                assert progress_updates[-1]["progress"] == 0.0
                assert "error" in progress_updates[-1]

    def test_wait_for_customization_running_with_progress_update(
        self, customizer, sample_flywheel_run_id
    ):
        """Test customization wait running status with progress callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        call_count = 0

        def mock_get_job_status_side_effect(job_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "status": "running",
                    "percentage_done": 30,
                    "epochs_completed": 3,
                    "steps_completed": 30,
                }
            else:
                return {
                    "status": "completed",
                    "epochs_completed": 10,
                    "steps_completed": 100,
                }

        # Mock get_db_manager to prevent actual DB calls
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch.object(customizer, "get_job_status", side_effect=mock_get_job_status_side_effect),
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            result = customizer.wait_for_customization(
                job_id=job_id,
                flywheel_run_id=sample_flywheel_run_id,
                check_interval=1,
                timeout=10,
                progress_callback=progress_callback,
            )
            assert result["status"] == "completed"
            # Should have progress updates for running and completed states
            assert len(progress_updates) >= 2

    def test_wait_for_customization_pending_status(self, customizer, sample_flywheel_run_id):
        """Test customization wait with pending status."""
        job_id = "test-job"

        call_count = 0

        def mock_get_job_status_side_effect(job_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"status": "pending"}
            else:
                return {"status": "completed", "epochs_completed": 10, "steps_completed": 100}

        # Mock get_db_manager to prevent actual DB calls
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch.object(customizer, "get_job_status", side_effect=mock_get_job_status_side_effect),
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            result = customizer.wait_for_customization(
                job_id=job_id,
                flywheel_run_id=sample_flywheel_run_id,
                check_interval=1,
                timeout=10,
            )
            assert result["status"] == "completed"

    def test_wait_for_customization_created_status(self, customizer, sample_flywheel_run_id):
        """Test customization wait with created status."""
        job_id = "test-job"

        call_count = 0

        def mock_get_job_status_side_effect(job_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"status": "created"}
            else:
                return {"status": "completed", "epochs_completed": 10, "steps_completed": 100}

        # Mock get_db_manager to prevent actual DB calls
        with (
            patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager,
            patch.object(customizer, "get_job_status", side_effect=mock_get_job_status_side_effect),
        ):
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            result = customizer.wait_for_customization(
                job_id=job_id,
                flywheel_run_id=sample_flywheel_run_id,
                check_interval=1,
                timeout=10,
            )
            assert result["status"] == "completed"

    def test_wait_for_customization_unknown_status(self, customizer, sample_flywheel_run_id):
        """Test customization wait with unknown status."""
        job_id = "test-job"

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {"status": "unknown_status"}
                with pytest.raises(Exception) as exc_info:
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                    )

                assert "Unknown job status 'unknown_status'" in str(exc_info.value)

    def test_wait_for_customization_unknown_status_with_callback(
        self, customizer, sample_flywheel_run_id
    ):
        """Test customization wait with unknown status and callback."""
        job_id = "test-job"
        progress_updates = []

        def progress_callback(progress_data):
            progress_updates.append(progress_data)

        # Mock get_db_manager to prevent actual DB calls
        with patch("src.lib.flywheel.cancellation.get_db_manager") as mock_get_db_manager:
            mock_db_manager = mock_get_db_manager.return_value
            mock_db_manager.is_flywheel_run_cancelled.return_value = False
            with patch.object(customizer, "get_job_status") as mock_get_job_status:
                mock_get_job_status.return_value = {"status": "weird_status"}
                with pytest.raises(Exception) as exc_info:
                    customizer.wait_for_customization(
                        job_id=job_id,
                        flywheel_run_id=sample_flywheel_run_id,
                        check_interval=1,
                        timeout=1,
                        progress_callback=progress_callback,
                    )

                assert "Unknown job status 'weird_status'" in str(exc_info.value)
                assert len(progress_updates) == 1
                assert progress_updates[0]["progress"] == 0.0
                assert "error" in progress_updates[0]

    def test_wait_for_customization_cancellation(self, customizer, sample_flywheel_run_id):
        """Test customization wait with cancellation."""
        job_id = "test-job"

        # Mock cancellation to raise an exception immediately
        with patch("src.lib.nemo.customizer.check_cancellation") as mock_check_cancellation:
            mock_check_cancellation.side_effect = Exception("Customization cancelled")
            with pytest.raises(Exception, match="Customization cancelled"):
                customizer.wait_for_customization(
                    job_id=job_id,
                    flywheel_run_id=sample_flywheel_run_id,
                    check_interval=1,
                    timeout=10,
                )

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

    def test_cancel_job_success(self, customizer):
        """Test successful job cancellation."""
        job_id = "test-job-123"
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response) as mock_post:
            customizer.cancel_job(job_id)
            mock_post.assert_called_once_with(
                f"http://test-nemo-url/v1/customizations/{job_id}/cancel"
            )

    def test_cancel_job_success_with_204(self, customizer):
        """Test successful job cancellation with 204 status code."""
        job_id = "test-job-456"
        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("requests.post", return_value=mock_response) as mock_post:
            customizer.cancel_job(job_id)
            mock_post.assert_called_once_with(
                f"http://test-nemo-url/v1/customizations/{job_id}/cancel"
            )

    def test_cancel_job_failure(self, customizer):
        """Test job cancellation failure."""
        job_id = "test-job-789"
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(Exception) as exc_info:
                customizer.cancel_job(job_id)
            assert "Failed to cancel job" in str(exc_info.value)
            assert "test-job-789" in str(exc_info.value)

    def test_cancel_job_server_error(self, customizer):
        """Test cancel_job with server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(Exception, match="Failed to cancel job"):
                customizer.cancel_job("test-job-123")

    # Tests for generate_config_name method
    def test_generate_config_name_valid(self):
        """Test generate_config_name with valid base model format."""
        base_model = "test-namespace/test-model"
        config_name = NIMConfig.generate_config_name(base_model)
        assert config_name == "test-model@v1.0.0+dfw"

    def test_generate_config_name_invalid_single_part(self):
        """Test generate_config_name with invalid single part format."""
        base_model = "test-model"
        with pytest.raises(ValueError, match="Invalid base model format"):
            NIMConfig.generate_config_name(base_model)

    def test_generate_config_name_invalid_three_parts(self):
        """Test generate_config_name with invalid three part format."""
        base_model = "a/b/c"
        with pytest.raises(ValueError, match="Invalid base model format"):
            NIMConfig.generate_config_name(base_model)

    def test_generate_config_name_empty_string(self):
        """Test generate_config_name with empty string."""
        base_model = ""
        with pytest.raises(ValueError, match="Invalid base model format"):
            NIMConfig.generate_config_name(base_model)

    def test_generate_config_name_none(self):
        """Test generate_config_name with None."""
        base_model = None
        with pytest.raises(ValueError, match="Invalid base model format"):
            NIMConfig.generate_config_name(base_model)

    def test_generate_config_name_special_characters(self):
        """Test generate_config_name with special characters in model name."""
        base_model = "test-namespace/model-name_123"
        config_name = NIMConfig.generate_config_name(base_model)
        assert config_name == "model-name_123@v1.0.0+dfw"

    # Tests for create_customizer_config method
    def test_create_customizer_config_success(self, customizer, customizer_config, training_config):
        """Test successful customizer config creation."""
        base_model = "test-namespace/test-model"
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response):
            config_name = customizer.create_customizer_config(
                base_model, customizer_config, training_config
            )
            assert config_name == "test-model@v1.0.0+dfw"

    def test_create_customizer_config_already_exists(
        self, customizer, customizer_config, training_config
    ):
        """Test customizer config creation when config already exists (409)."""
        base_model = "test-namespace/test-model"
        mock_response = Mock()
        mock_response.status_code = 409

        with patch("requests.post", return_value=mock_response):
            config_name = customizer.create_customizer_config(
                base_model, customizer_config, training_config
            )
            assert config_name == "test-model@v1.0.0+dfw"

    def test_create_customizer_config_invalid_base_model(
        self, customizer, customizer_config, training_config
    ):
        """Test customizer config creation with invalid base model format."""
        base_model = "invalid-format"
        with pytest.raises(ValueError, match="Invalid base model format"):
            customizer.create_customizer_config(base_model, customizer_config, training_config)

    def test_create_customizer_config_server_error(
        self, customizer, customizer_config, training_config
    ):
        """Test customizer config creation with server error."""
        base_model = "test-namespace/test-model"
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(Exception, match="Failed to create customizer config"):
                customizer.create_customizer_config(base_model, customizer_config, training_config)

    def test_create_customizer_config_bad_request(
        self, customizer, customizer_config, training_config
    ):
        """Test customizer config creation with bad request."""
        base_model = "test-namespace/test-model"
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(Exception, match="Failed to create customizer config"):
                customizer.create_customizer_config(base_model, customizer_config, training_config)

    def test_create_customizer_config_payload_structure(
        self, customizer, customizer_config, training_config
    ):
        """Test that the correct payload structure is sent for config creation."""
        base_model = "test-namespace/test-model"
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.post", return_value=mock_response) as mock_post:
            customizer.create_customizer_config(base_model, customizer_config, training_config)

            # Verify the request was made
            mock_post.assert_called_once()

            # Get the call arguments
            call_args = mock_post.call_args
            url = call_args[0][0]  # First positional argument
            payload = call_args[1]["json"]  # JSON payload

            # Verify URL
            expected_url = "http://test-nemo-url/v1/customization/configs"
            assert url == expected_url

            # Verify payload structure
            assert payload["name"] == "test-model@v1.0.0+dfw"
            assert payload["namespace"] == "test-namespace"
            assert payload["target"] == customizer_config.target
            assert payload["training_precision"] == customizer_config.training_precision
            assert payload["max_seq_length"] == customizer_config.max_seq_length

            # Verify training options
            training_options = payload["training_options"]
            assert len(training_options) == 1
            option = training_options[0]
            assert option["training_type"] == training_config.training_type
            assert option["finetuning_type"] == training_config.finetuning_type
            assert option["num_gpus"] == customizer_config.gpus
            assert option["num_nodes"] == customizer_config.num_nodes
            assert option["tensor_parallel_size"] == customizer_config.tensor_parallel_size
            assert option["data_parallel_size"] == customizer_config.data_parallel_size
            assert option["use_sequence_parallel"] == customizer_config.use_sequence_parallel
            assert option["micro_batch_size"] == customizer_config.micro_batch_size

    # Enhanced tests for start_training_job method
    def test_start_training_job_no_customizer_config(self, customizer, training_config):
        """Test that NIMConfig validation prevents creating configs with customization enabled but no customizer_configs."""
        # Test that Pydantic validation catches the missing customizer_configs
        with pytest.raises(
            ValueError,
            match="customizer_configs is required when customization_enabled is set to True",
        ):
            NIMConfig(
                model_name="test-model",
                context_length=2048,
                gpus=1,
                pvc_size="10Gi",
                tag="latest",
                registry_base="nvcr.io/nim",
                customization_enabled=True,
                customizer_configs=None,  # No customizer configs
            )

    def test_start_training_job_invalid_base_model(self, customizer, training_config, nim_config):
        """Test start_training_job with invalid base model format."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "job-123",
            "output_model": "test-namespace/model-123",
        }

        with (
            patch("requests.post", return_value=mock_response),
            patch.object(
                customizer,
                "create_customizer_config",
                side_effect=ValueError("Invalid base model format"),
            ),
        ):
            with pytest.raises(ValueError, match="Invalid base model format"):
                customizer.start_training_job(
                    name="test-job",
                    base_model="invalid-format",
                    output_model_name="output-model",
                    dataset_name="test-dataset",
                    training_config=training_config,
                    nim_config=nim_config,
                )

    def test_start_training_job_verify_config_creation(
        self, customizer, training_config, nim_config
    ):
        """Test that start_training_job calls create_customizer_config correctly."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "job-123",
            "output_model": "test-namespace/model-123",
        }

        with (
            patch("requests.post", return_value=mock_response),
            patch.object(
                customizer, "create_customizer_config", return_value="test-config-name"
            ) as mock_create_config,
        ):
            customizer.start_training_job(
                name="test-job",
                base_model="test-namespace/test-model",
                output_model_name="output-model",
                dataset_name="test-dataset",
                training_config=training_config,
                nim_config=nim_config,
            )

            # Verify create_customizer_config was called with correct parameters
            mock_create_config.assert_called_once_with(
                "test-namespace/test-model",
                nim_config.customizer_configs,
                training_config,
            )

    def test_start_training_job_verify_training_params(
        self, customizer, training_config, nim_config
    ):
        """Test that start_training_job sends correct training parameters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "job-123",
            "output_model": "test-namespace/model-123",
        }

        with (
            patch("requests.post", return_value=mock_response) as mock_post,
            patch.object(customizer, "create_customizer_config", return_value="test-config-name"),
        ):
            customizer.start_training_job(
                name="test-job",
                base_model="test-namespace/test-model",
                output_model_name="output-model",
                dataset_name="test-dataset",
                training_config=training_config,
                nim_config=nim_config,
            )

            # Verify the training job request was made
            mock_post.assert_called()

            # Get the training job call (only call since create_customizer_config is mocked)
            training_call = mock_post.call_args_list[0]  # First call
            url = training_call[0][0]
            payload = training_call[1]["json"]

            # Verify URL
            expected_url = "http://test-nemo-url/v1/customization/jobs"
            assert url == expected_url

            # Verify payload structure
            assert payload["name"] == "test-job"
            assert payload["output_model"] == "test-namespace/output-model"
            assert payload["config"] == "test-namespace/test-config-name"
            assert payload["dataset"]["name"] == "test-dataset"
            assert payload["dataset"]["namespace"] == "test-namespace"

            # Verify hyperparameters
            hyperparams = payload["hyperparameters"]
            assert hyperparams["training_type"] == training_config.training_type
            assert hyperparams["finetuning_type"] == training_config.finetuning_type
            assert hyperparams["epochs"] == training_config.epochs
            assert hyperparams["batch_size"] == training_config.batch_size
            assert hyperparams["learning_rate"] == training_config.learning_rate

            # Verify LoRA config
            lora_config = hyperparams["lora"]
            assert lora_config["adapter_dim"] == training_config.lora.adapter_dim
            assert lora_config["adapter_dropout"] == training_config.lora.adapter_dropout

    def test_delete_customization_config_success(self, customizer):
        """Test successful deletion of customization config."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response):
            customizer.delete_customization_config("test-config")

        # Verify the request was made with correct URL
        with patch("requests.delete", return_value=mock_response) as mock_delete:
            customizer.delete_customization_config("test-config")
            mock_delete.assert_called_once_with(
                "http://test-nemo-url/v1/customization/configs/test-namespace/test-config"
            )

    def test_delete_customization_config_failure(self, customizer):
        """Test deletion of customization config with failure."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Config not found"

        with patch("requests.delete", return_value=mock_response):
            # Should not raise exception, just return
            customizer.delete_customization_config("test-config")

    def test_delete_customization_config_server_error(self, customizer):
        """Test deletion of customization config with server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("requests.delete", return_value=mock_response):
            # Should not raise exception, just return
            customizer.delete_customization_config("test-config")
