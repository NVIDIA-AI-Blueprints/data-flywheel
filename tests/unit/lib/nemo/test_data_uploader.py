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
import tempfile
from unittest.mock import Mock, patch

import pytest

from src.lib.nemo.data_uploader import DataUploader


class TestDataUploader:
    """Test class for DataUploader initialization and basic functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Fixture to mock settings configuration."""
        with patch("src.lib.nemo.data_uploader.settings") as mock_settings:
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"
            yield mock_settings

    @pytest.fixture
    def mock_hf_token(self):
        """Fixture to mock HF_TOKEN environment variable."""
        with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}):
            yield

    @pytest.fixture
    def data_uploader(self, mock_settings, mock_hf_token):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api:
            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            yield uploader

    def test_init_success(self, mock_settings, mock_hf_token):
        """Test successful DataUploader initialization."""
        with patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api:
            uploader = DataUploader("test-dataset")

            assert uploader.entity_host == "http://test-nemo-url"
            assert uploader.ds_host == "http://test-datastore-url"
            assert uploader.hf_token == "test-hf-token"
            assert uploader.namespace == "test-namespace"
            assert uploader.dataset_name == "test-dataset"
            mock_hf_api.assert_called_once_with(
                endpoint="http://test-datastore-url/v1/hf", token="test-hf-token"
            )

    def test_init_missing_nemo_url(self, mock_hf_token):
        """Test initialization fails without nemo_base_url."""
        with patch("src.lib.nemo.data_uploader.settings") as mock_settings:
            mock_settings.nmp_config.nemo_base_url = None
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            with pytest.raises(AssertionError, match="nemo_base_url must be set in config"):
                DataUploader("test-dataset")

    def test_init_missing_datastore_url(self, mock_hf_token):
        """Test initialization fails without datastore_base_url."""
        with patch("src.lib.nemo.data_uploader.settings") as mock_settings:
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = None
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            with pytest.raises(AssertionError, match="datastore_base_url must be set in config"):
                DataUploader("test-dataset")

    def test_init_missing_hf_token(self, mock_settings):
        """Test initialization fails without HF_TOKEN."""
        with patch.dict(os.environ, {"HF_TOKEN": ""}, clear=True):
            with pytest.raises(AssertionError, match="HF_TOKEN is not set"):
                DataUploader("test-dataset")


class TestDataUploaderNamespaces:
    """Test class for namespace creation functionality."""

    @pytest.fixture
    def data_uploader(self):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with (
            patch("src.lib.nemo.data_uploader.settings") as mock_settings,
            patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}),
            patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api,
        ):
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            yield uploader

    def test_create_namespaces_success(self, data_uploader):
        """Test successful namespace creation in both stores."""
        mock_entity_response = Mock()
        mock_entity_response.status_code = 201

        mock_datastore_response = Mock()
        mock_datastore_response.status_code = 201

        with patch("requests.post") as mock_post:
            mock_post.side_effect = [mock_entity_response, mock_datastore_response]

            data_uploader._create_namespaces()

            assert mock_post.call_count == 2
            # Check Entity Store call
            mock_post.assert_any_call(
                "http://test-nemo-url/v1/namespaces", json={"id": "test-namespace"}
            )
            # Check Data Store call
            mock_post.assert_any_call(
                "http://test-datastore-url/v1/datastore/namespaces",
                data={"namespace": "test-namespace"},
            )

    def test_create_namespaces_already_exists(self, data_uploader):
        """Test namespace creation when namespaces already exist."""
        mock_entity_response = Mock()
        mock_entity_response.status_code = 409  # Conflict - already exists

        mock_datastore_response = Mock()
        mock_datastore_response.status_code = 409  # Conflict - already exists

        with patch("requests.post") as mock_post:
            mock_post.side_effect = [mock_entity_response, mock_datastore_response]

            data_uploader._create_namespaces()  # Should not raise

    def test_create_namespaces_entity_store_failure(self, data_uploader):
        """Test namespace creation failure in Entity Store."""
        mock_entity_response = Mock()
        mock_entity_response.status_code = 500  # Server error

        with patch("requests.post") as mock_post:
            mock_post.return_value = mock_entity_response

            with pytest.raises(AssertionError, match="Unexpected response from Entity Store"):
                data_uploader._create_namespaces()

    def test_create_namespaces_datastore_failure(self, data_uploader):
        """Test namespace creation failure in Data Store."""
        mock_entity_response = Mock()
        mock_entity_response.status_code = 201

        mock_datastore_response = Mock()
        mock_datastore_response.status_code = 500  # Server error

        with patch("requests.post") as mock_post:
            mock_post.side_effect = [mock_entity_response, mock_datastore_response]

            with pytest.raises(AssertionError, match="Unexpected response from Data Store"):
                data_uploader._create_namespaces()


class TestDataUploaderRepository:
    """Test class for repository creation functionality."""

    @pytest.fixture
    def data_uploader(self):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with (
            patch("src.lib.nemo.data_uploader.settings") as mock_settings,
            patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}),
            patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api,
        ):
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            yield uploader

    def test_create_repo_success(self, data_uploader):
        """Test successful repository creation."""
        data_uploader.hf_api.create_repo.return_value = None

        repo_id = data_uploader._create_repo()

        assert repo_id == "test-namespace/test-dataset"
        data_uploader.hf_api.create_repo.assert_called_once_with(
            repo_id="test-namespace/test-dataset", repo_type="dataset"
        )

    def test_create_repo_already_exists(self, data_uploader):
        """Test repository creation when repo already exists."""
        data_uploader.hf_api.create_repo.side_effect = Exception("409 Conflict")

        repo_id = data_uploader._create_repo()

        assert repo_id == "test-namespace/test-dataset"

    def test_create_repo_other_error(self, data_uploader):
        """Test repository creation with other errors."""
        error_msg = "Some other error"
        data_uploader.hf_api.create_repo.side_effect = Exception(error_msg)

        with pytest.raises(Exception, match=error_msg):
            data_uploader._create_repo()


class TestDataUploaderFileUpload:
    """Test class for file upload functionality."""

    @pytest.fixture
    def data_uploader(self):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with (
            patch("src.lib.nemo.data_uploader.settings") as mock_settings,
            patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}),
            patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api,
        ):
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            yield uploader

    def test_upload_file_success(self, data_uploader):
        """Test successful file upload."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
            temp_file.write('{"test": "data"}')
            temp_file_path = temp_file.name

        try:
            # Mock namespace and repo creation
            with (
                patch.object(data_uploader, "_create_namespaces") as mock_create_ns,
                patch.object(data_uploader, "_create_repo") as mock_create_repo,
            ):
                mock_create_repo.return_value = "test-namespace/test-dataset"

                result = data_uploader.upload_file(temp_file_path, "training")

                mock_create_ns.assert_called_once()
                mock_create_repo.assert_called_once()

                expected_filename = os.path.basename(temp_file_path)
                expected_path = f"training/{expected_filename}"
                assert result == expected_path

                data_uploader.hf_api.upload_file.assert_called_once_with(
                    path_or_fileobj=temp_file_path,
                    path_in_repo=expected_path,
                    repo_id="test-namespace/test-dataset",
                    repo_type="dataset",
                )
        finally:
            os.unlink(temp_file_path)

    def test_upload_file_invalid_data_type(self, data_uploader):
        """Test file upload with invalid data type."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
            temp_file.write('{"test": "data"}')
            temp_file_path = temp_file.name

        try:
            with pytest.raises(AssertionError, match="data_type must be one of"):
                data_uploader.upload_file(temp_file_path, "invalid_type")
        finally:
            os.unlink(temp_file_path)

    def test_upload_file_nonexistent_file(self, data_uploader):
        """Test file upload with nonexistent file."""
        with pytest.raises(AssertionError, match="Data file at .* does not exist"):
            data_uploader.upload_file("/nonexistent/file.jsonl", "training")

    def test_upload_file_repo_already_created(self, data_uploader):
        """Test file upload when repo is already created."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as temp_file:
            temp_file.write('{"test": "data"}')
            temp_file_path = temp_file.name

        try:
            # Set repo_id to simulate already created repo
            data_uploader.repo_id = "test-namespace/test-dataset"

            result = data_uploader.upload_file(temp_file_path, "validation")

            expected_filename = os.path.basename(temp_file_path)
            expected_path = f"validation/{expected_filename}"
            assert result == expected_path
        finally:
            os.unlink(temp_file_path)


class TestDataUploaderDataUpload:
    """Test class for data string upload functionality."""

    @pytest.fixture
    def data_uploader(self):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with (
            patch("src.lib.nemo.data_uploader.settings") as mock_settings,
            patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}),
            patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api,
        ):
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            yield uploader

    def test_upload_data_success(self, data_uploader):
        """Test successful data string upload."""
        test_data = '{"test": "data"}\n{"another": "line"}'
        file_path = "test/data.jsonl"

        with (
            patch.object(data_uploader, "_create_namespaces") as mock_create_ns,
            patch.object(data_uploader, "_create_repo") as mock_create_repo,
            patch.object(data_uploader, "register_dataset") as mock_register,
        ):
            mock_create_repo.return_value = "test-namespace/test-dataset"

            result = data_uploader.upload_data(test_data, file_path)

            mock_create_ns.assert_called_once()
            mock_create_repo.assert_called_once()
            mock_register.assert_called_once()

            assert result == file_path

            # Check that upload_file was called with a BytesIO object
            data_uploader.hf_api.upload_file.assert_called_once()
            call_args = data_uploader.hf_api.upload_file.call_args
            assert call_args[1]["path_in_repo"] == file_path
            assert call_args[1]["repo_id"] == "test-namespace/test-dataset"
            assert call_args[1]["repo_type"] == "dataset"

    def test_upload_data_repo_already_created(self, data_uploader):
        """Test data upload when repo is already created."""
        test_data = '{"test": "data"}'
        file_path = "test/data.jsonl"

        # Set repo_id to simulate already created repo
        data_uploader.repo_id = "test-namespace/test-dataset"

        with patch.object(data_uploader, "register_dataset") as mock_register:
            result = data_uploader.upload_data(test_data, file_path)

            mock_register.assert_called_once()
            assert result == file_path


class TestDataUploaderDatasetManagement:
    """Test class for dataset registration and verification functionality."""

    @pytest.fixture
    def data_uploader(self):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with (
            patch("src.lib.nemo.data_uploader.settings") as mock_settings,
            patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}),
            patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api,
        ):
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            uploader.repo_id = "test-namespace/test-dataset"  # Simulate uploaded files
            yield uploader

    def test_verify_dataset_success(self, data_uploader):
        """Test successful dataset verification."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-dataset",
            "namespace": "test-namespace",
            "files_url": "hf://datasets/test-namespace/test-dataset",
        }

        with patch("requests.get", return_value=mock_response):
            result = data_uploader.verify_dataset()

            assert result["name"] == "test-dataset"
            assert result["namespace"] == "test-namespace"
            assert result["files_url"] == "hf://datasets/test-namespace/test-dataset"

    def test_verify_dataset_no_files_uploaded(self, data_uploader):
        """Test dataset verification when no files have been uploaded."""
        # Remove repo_id to simulate no files uploaded
        delattr(data_uploader, "repo_id")

        with pytest.raises(ValueError, match="No files have been uploaded yet"):
            data_uploader.verify_dataset()

    def test_verify_dataset_request_failure(self, data_uploader):
        """Test dataset verification with request failure."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Dataset not found"

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(AssertionError, match="Status Code 404 Failed to fetch dataset"):
                data_uploader.verify_dataset()

    def test_verify_dataset_url_mismatch(self, data_uploader):
        """Test dataset verification with URL mismatch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "files_url": "hf://datasets/wrong-namespace/wrong-dataset"
        }

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(AssertionError, match="Dataset files_url mismatch"):
                data_uploader.verify_dataset()

    def test_register_dataset_create_new(self, data_uploader):
        """Test registering a new dataset."""
        # Mock GET request to check if dataset exists (returns 404)
        mock_get_response = Mock()
        mock_get_response.status_code = 404

        # Mock POST request to create dataset
        mock_post_response = Mock()
        mock_post_response.status_code = 201
        mock_post_response.json.return_value = {"id": "dataset-id"}

        with patch("requests.get", return_value=mock_get_response):
            with patch("requests.post", return_value=mock_post_response) as mock_post:
                result = data_uploader.register_dataset("Test description", "test-project")

                assert result["id"] == "dataset-id"
                mock_post.assert_called_once_with(
                    url="http://test-nemo-url/v1/datasets",
                    json={
                        "name": "test-dataset",
                        "namespace": "test-namespace",
                        "description": "Test description",
                        "files_url": "hf://datasets/test-namespace/test-dataset",
                        "project": "test-project",
                    },
                )

    def test_register_dataset_update_existing(self, data_uploader):
        """Test updating an existing dataset."""
        # Mock GET request to check if dataset exists (returns 200)
        mock_get_response = Mock()
        mock_get_response.status_code = 200

        # Mock PATCH request to update dataset
        mock_patch_response = Mock()
        mock_patch_response.status_code = 200
        mock_patch_response.json.return_value = {"id": "dataset-id"}

        with patch("requests.get", return_value=mock_get_response):
            with patch("requests.patch", return_value=mock_patch_response) as mock_patch:
                result = data_uploader.register_dataset("Updated description")

                assert result["id"] == "dataset-id"
                mock_patch.assert_called_once_with(
                    url="http://test-nemo-url/v1/datasets/test-namespace/test-dataset",
                    json={
                        "name": "test-dataset",
                        "namespace": "test-namespace",
                        "description": "Updated description",
                        "files_url": "hf://datasets/test-namespace/test-dataset",
                        "project": "flywheel",
                    },
                )

    def test_register_dataset_no_files_uploaded(self, data_uploader):
        """Test dataset registration when no files have been uploaded."""
        # Remove repo_id to simulate no files uploaded
        delattr(data_uploader, "repo_id")

        with pytest.raises(ValueError, match="No files have been uploaded yet"):
            data_uploader.register_dataset()

    def test_register_dataset_failure(self, data_uploader):
        """Test dataset registration failure."""
        # Mock GET request to check if dataset exists (returns 404)
        mock_get_response = Mock()
        mock_get_response.status_code = 404

        # Mock POST request failure
        mock_post_response = Mock()
        mock_post_response.status_code = 500
        mock_post_response.text = "Internal server error"

        with patch("requests.get", return_value=mock_get_response):
            with patch("requests.post", return_value=mock_post_response):
                with pytest.raises(
                    AssertionError, match="Status Code 500 Failed to create dataset"
                ):
                    data_uploader.register_dataset()

    def test_get_file_uri_success(self, data_uploader):
        """Test getting file URI successfully."""
        mock_dataset = {"files_url": "hf://datasets/test-namespace/test-dataset"}

        with patch.object(data_uploader, "verify_dataset", return_value=mock_dataset):
            result = data_uploader.get_file_uri()
            assert result == "hf://datasets/test-namespace/test-dataset"

    def test_get_file_uri_no_files_uploaded(self, data_uploader):
        """Test getting file URI when no files have been uploaded."""
        # Remove repo_id to simulate no files uploaded
        delattr(data_uploader, "repo_id")

        with pytest.raises(ValueError, match="No files have been uploaded yet"):
            data_uploader.get_file_uri()


class TestDataUploaderFolderUpload:
    """Test class for folder upload functionality."""

    @pytest.fixture
    def data_uploader(self):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with (
            patch("src.lib.nemo.data_uploader.settings") as mock_settings,
            patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}),
            patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api,
        ):
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            yield uploader

    def test_upload_data_from_folder_nonexistent(self, data_uploader):
        """Test folder upload with nonexistent folder."""
        with pytest.raises(ValueError, match="Data folder .* does not exist"):
            data_uploader.upload_data_from_folder("/nonexistent/folder")

    def test_upload_data_from_folder_success(self, data_uploader):
        """Test successful folder upload."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test folder structure
            customization_dir = os.path.join(temp_dir, "customization")
            validation_dir = os.path.join(temp_dir, "validation")
            evaluation_dir = os.path.join(temp_dir, "evaluation")

            os.makedirs(customization_dir)
            os.makedirs(validation_dir)
            os.makedirs(evaluation_dir)

            # Create test files
            with open(os.path.join(customization_dir, "train.jsonl"), "w") as f:
                f.write('{"train": "data"}')
            with open(os.path.join(validation_dir, "val.jsonl"), "w") as f:
                f.write('{"val": "data"}')
            with open(os.path.join(evaluation_dir, "test.jsonl"), "w") as f:
                f.write('{"test": "data"}')

            with (
                patch.object(data_uploader, "upload_file") as mock_upload,
                patch.object(data_uploader, "register_dataset") as mock_register,
            ):
                data_uploader.upload_data_from_folder(temp_dir, "Test description", "test-project")

                # Should upload 3 files
                assert mock_upload.call_count == 3
                mock_register.assert_called_once_with(
                    description="Test description", project="test-project"
                )

    def test_upload_data_from_folder_missing_subfolders(self, data_uploader):
        """Test folder upload with missing subfolders."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Only create customization folder
            customization_dir = os.path.join(temp_dir, "customization")
            os.makedirs(customization_dir)

            with open(os.path.join(customization_dir, "train.jsonl"), "w") as f:
                f.write('{"train": "data"}')

            with (
                patch.object(data_uploader, "upload_file") as mock_upload,
                patch.object(data_uploader, "register_dataset") as mock_register,
            ):
                data_uploader.upload_data_from_folder(temp_dir)

                # Should only upload 1 file
                assert mock_upload.call_count == 1
                mock_register.assert_called_once()

    def test_upload_data_from_folder_no_jsonl_files(self, data_uploader):
        """Test folder upload with no JSONL files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create folders but no .jsonl files
            customization_dir = os.path.join(temp_dir, "customization")
            os.makedirs(customization_dir)

            with open(os.path.join(customization_dir, "data.txt"), "w") as f:
                f.write("not a jsonl file")

            with (
                patch.object(data_uploader, "upload_file") as mock_upload,
                patch.object(data_uploader, "register_dataset") as mock_register,
            ):
                data_uploader.upload_data_from_folder(temp_dir)

                # Should not upload any files
                mock_upload.assert_not_called()
                mock_register.assert_called_once()


class TestDataUploaderDeletion:
    """Test class for dataset deletion functionality."""

    @pytest.fixture
    def data_uploader(self):
        """Fixture to create DataUploader instance with mocked dependencies."""
        with (
            patch("src.lib.nemo.data_uploader.settings") as mock_settings,
            patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}),
            patch("src.lib.nemo.data_uploader.HfApi") as mock_hf_api,
        ):
            mock_settings.nmp_config.nemo_base_url = "http://test-nemo-url"
            mock_settings.nmp_config.datastore_base_url = "http://test-datastore-url"
            mock_settings.nmp_config.nmp_namespace = "test-namespace"

            uploader = DataUploader("test-dataset")
            uploader.hf_api = mock_hf_api.return_value
            yield uploader

    def test_delete_dataset_success(self, data_uploader):
        """Test successful dataset deletion."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            data_uploader.delete_dataset()

            mock_delete.assert_called_once_with(
                "http://test-datastore-url/v1/hf/api/repos/delete",
                json={"name": "test-dataset", "organization": "test-namespace", "type": "dataset"},
            )

    def test_delete_dataset_success_204(self, data_uploader):
        """Test successful dataset deletion with 204 status."""
        mock_response = Mock()
        mock_response.status_code = 204

        with patch("requests.delete", return_value=mock_response):
            data_uploader.delete_dataset()  # Should not raise

    def test_delete_dataset_failure(self, data_uploader):
        """Test dataset deletion failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("requests.delete", return_value=mock_response):
            with pytest.raises(AssertionError, match="Failed to delete dataset from Data Store"):
                data_uploader.delete_dataset()

    def test_unregister_dataset_success(self, data_uploader):
        """Test successful dataset unregistration."""
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            data_uploader.unregister_dataset()

            mock_delete.assert_called_once_with(
                "http://test-nemo-url/v1/datasets/test-namespace/test-dataset"
            )

    def test_unregister_dataset_success_204(self, data_uploader):
        """Test successful dataset unregistration with 204 status."""
        mock_response = Mock()
        mock_response.status_code = 204

        with patch("requests.delete", return_value=mock_response):
            data_uploader.unregister_dataset()  # Should not raise

    def test_unregister_dataset_failure(self, data_uploader):
        """Test dataset unregistration failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("requests.delete", return_value=mock_response):
            with pytest.raises(
                AssertionError, match="Failed to unregister dataset from Entity Store"
            ):
                data_uploader.unregister_dataset()
