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
import io
import os
from typing import Any

import requests
from huggingface_hub import HfApi

from src.config import settings
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.nemo.data_uploader")


class DataUploader:
    def __init__(self, namespace: str, dataset_name: str):
        """
        Initialize the DataUploader with necessary configuration.

        Args:
            namespace: Namespace for the dataset
            dataset_name: Name of the dataset
        """
        self.entity_host = settings.nmp_config.nemo_base_url
        assert self.entity_host, "nemo_base_url must be set in config"

        self.ds_host = settings.nmp_config.datastore_base_url
        assert self.ds_host, "datastore_base_url must be set in config"

        self.hf_token = os.environ.get("HF_TOKEN", "nothing")
        assert self.hf_token, "HF_TOKEN is not set"

        self.namespace = namespace
        self.dataset_name = dataset_name

        # Initialize HF API client
        self.hf_api = HfApi(endpoint=f"{self.ds_host}/v1/hf", token=self.hf_token)

    def _create_namespaces(self) -> None:
        """Create namespaces in both Entity Store and Data Store."""
        # Create namespace in Entity Store
        entity_store_url = f"{self.entity_host}/v1/namespaces"
        resp = requests.post(entity_store_url, json={"id": self.namespace})
        assert resp.status_code in (
            200,
            201,
            409,
            422,
        ), f"Unexpected response from Entity Store during namespace creation: {resp.status_code}"

        # Create namespace in Data Store
        nds_url = f"{self.ds_host}/v1/datastore/namespaces"
        resp = requests.post(nds_url, data={"namespace": self.namespace})
        assert resp.status_code in (
            200,
            201,
            409,
            422,
        ), f"Unexpected response from Data Store during namespace creation: {resp.status_code}"

    def _create_repo(self) -> str:
        """Create a new repository in the data store."""
        repo_id = f"{self.namespace}/{self.dataset_name}"
        try:
            self.hf_api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            if "409" in str(e):
                logger.info(f"Repository {repo_id} already exists. Continuing...")
            else:
                msg = f"Failed to create repository {repo_id}: {e}"
                logger.error(msg)
                raise e
        return repo_id

    def upload_file(
        self,
        data_fp: str,
        data_type: str,
    ) -> str:
        """
        Upload a single data file to the repository.

        Args:
            data_fp: Path to the data file
            data_type: Type of data (training, validation, or testing)

        Returns:
            Path of the uploaded file in the repository
        """
        # Validate data type
        valid_types = ["training", "validation", "testing"]
        assert data_type in valid_types, f"data_type must be one of {valid_types}"

        # Validate file exists
        assert os.path.exists(data_fp), f"Data file at '{data_fp}' does not exist"

        # Create namespaces if not already done
        if not hasattr(self, "repo_id"):
            self._create_namespaces()
            self.repo_id = self._create_repo()

        # Determine filename in repo
        filename = os.path.basename(data_fp)

        # Construct path in repo
        path_in_repo = f"{data_type}/{filename}"

        # Upload file
        self.hf_api.upload_file(
            path_or_fileobj=data_fp,
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

        return path_in_repo

    def upload_data(
        self,
        data: str,
        file_path: str,
    ) -> str:
        """
        Upload a string as a file in the repository.

        Args:
            data: The string to upload
            file_path: The path to the file in the repository

        Returns:
            Path of the uploaded file in the repository
        """

        # Create namespaces if not already done
        if not hasattr(self, "repo_id"):
            self._create_namespaces()
            self.repo_id = self._create_repo()

        data_io = io.BytesIO(data.encode("utf-8"))

        # Upload file
        self.hf_api.upload_file(
            path_or_fileobj=data_io,
            path_in_repo=file_path,
            repo_id=self.repo_id,
            repo_type="dataset",
        )

        # Register the dataset
        self.register_dataset()

        return file_path

    def verify_dataset(self) -> dict[str, Any]:
        """
        Verify that the dataset is properly registered in the entity store.

        Returns:
            Dict containing the dataset information from the entity store

        Raises:
            ValueError: If no files have been uploaded yet
            AssertionError: If the dataset is not properly registered or the files_url doesn't match
        """
        if not hasattr(self, "repo_id"):
            msg = "No files have been uploaded yet. Call upload_data() first."
            logger.error(msg)
            raise ValueError(msg)

        # Fetch dataset information
        res = requests.get(
            url=f"{self.entity_host}/v1/datasets/{self.namespace}/{self.dataset_name}"
        )
        assert res.status_code in (
            200,
            201,
        ), f"Status Code {res.status_code} Failed to fetch dataset {res.text}"
        dataset_obj = res.json()

        # Verify files_url matches the repository
        expected_files_url = f"hf://datasets/{self.repo_id}"
        assert (
            dataset_obj["files_url"] == expected_files_url
        ), f"Dataset files_url mismatch. Expected {expected_files_url}, got {dataset_obj['files_url']}"

        return dataset_obj

    def register_dataset(self, description: str = "", project: str = "flywheel") -> dict[str, Any]:
        """
        Register the dataset with the entity store after all files are uploaded.

        Args:
            description: Description of the dataset
            project: Project name

        Returns:
            Dict containing the dataset registration response

        Raises:
            ValueError: If no files have been uploaded yet
            AssertionError: If the dataset registration fails
        """
        if not hasattr(self, "repo_id"):
            msg = "No files have been uploaded yet. Call upload_data() first."
            logger.error(msg)
            raise ValueError(msg)
        # Check if dataset already exists
        res = requests.get(
            url=f"{self.entity_host}/v1/datasets/{self.namespace}/{self.dataset_name}"
        )

        # Prepare dataset payload
        dataset_payload = {
            "name": self.dataset_name,
            "namespace": self.namespace,
            "description": description,
            "files_url": f"hf://datasets/{self.repo_id}",
            "project": project,
        }

        # Update existing dataset or create new one
        if res.status_code == 200:
            # Dataset exists, update it
            resp = requests.patch(
                url=f"{self.entity_host}/v1/datasets/{self.namespace}/{self.dataset_name}",
                json=dataset_payload,
            )
        else:
            # Dataset doesn't exist, create it
            resp = requests.post(url=f"{self.entity_host}/v1/datasets", json=dataset_payload)
        assert resp.status_code in (
            200,
            201,
        ), f"Status Code {resp.status_code} Failed to create dataset {resp.text}"

        return resp.json()

    def get_file_uri(self) -> str:
        """
        Get the HuggingFace URI for an uploaded file.

        Raises:
            ValueError: If no files have been uploaded yet
        """
        if not hasattr(self, "repo_id"):
            msg = "No files have been uploaded yet. Call upload_data() or upload_file() first."
            logger.error(msg)
            raise ValueError(msg)
        dataset_obj = self.verify_dataset()
        return dataset_obj["files_url"]

    def upload_data_from_folder(
        self, data_folder: str, description: str = "", project: str = ""
    ) -> None:
        """
        Load data from a folder into the repository.
        """
        # Check if folder exists
        if not os.path.exists(data_folder):
            msg = f"Data folder {data_folder} does not exist"
            logger.error(msg)
            raise ValueError(msg)

        # Define folder to data_type mapping
        folder_type_map = {
            "customization": "training",
            "validation": "validation",
            "evaluation": "testing",
        }

        # Iterate through expected folders
        for folder_name, data_type in folder_type_map.items():
            folder_path = os.path.join(data_folder, folder_name)

            # Skip if folder doesn't exist
            if not os.path.exists(folder_path):
                logger.warning(f"Warning: Folder {folder_name} not found in {data_folder}")
                continue

            # Find all .jsonl files in the folder
            jsonl_files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]

            if not jsonl_files:
                logger.warning(f"Warning: No .jsonl files found in {folder_name}")
                continue

            # Upload each .jsonl file
            for jsonl_file in jsonl_files:
                file_path = os.path.join(folder_path, jsonl_file)
                self.upload_file(data_fp=file_path, data_type=data_type)
                logger.info(f"Uploaded {jsonl_file} as {data_type} data")

        # Register the dataset
        self.register_dataset(description=description, project=project)
