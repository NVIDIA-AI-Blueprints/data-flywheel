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


from bson import ObjectId

from src.api.db_manager import TaskDBManager
from src.api.models import FlywheelRun
from src.lib.nemo.customizer import Customizer
from src.lib.nemo.data_uploader import DataUploader
from src.lib.nemo.evaluator import Evaluator
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.cleanup")


class FlywheelJobManager:
    """Manager class for cleaning up flywheel job resources."""

    def __init__(self, db_manager: TaskDBManager):
        """Initialize the cleanup manager.

        Args:
            db_manager: Database manager instance for accessing MongoDB
        """
        self.db_manager = db_manager
        self.evaluator = Evaluator()
        self.customizer = Customizer()
        self.cleanup_errors: list[str] = []

    def delete_job(self, job_id: str) -> None:
        """Delete all resources associated with a flywheel job.

        This includes:
        - Customized models
        - Evaluation jobs
        - Datasets
        - MongoDB records

        Args:
            job_id: ID of the job to delete

        Raises:
            Exception: If the job deletion fails completely
        """
        job_object_id = ObjectId(job_id)
        self.cleanup_errors = []

        try:
            # Get the flywheel run
            flywheel_run = FlywheelRun.from_mongo(self.db_manager.get_flywheel_run(job_id))

            # Get all NIMs for this job
            nims = self.db_manager.find_nims_for_job(job_object_id)

            # Clean up NIM resources
            for nim in nims:
                self._cleanup_nim_resources(nim["_id"])

            # Clean up datasets
            self._cleanup_datasets(flywheel_run)

            # Delete all related MongoDB records
            self.db_manager.delete_job_records(job_object_id)

            if self.cleanup_errors:
                logger.warning(
                    f"Job {job_id} deleted with warnings: {'; '.join(self.cleanup_errors)}"
                )
            else:
                logger.info(f"Successfully deleted job with ID: {job_id}")

        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {e!s}")
            raise

    def _cleanup_nim_resources(self, nim_id: ObjectId) -> None:
        """Clean up all resources associated with a NIM.

        Args:
            nim_id: ID of the NIM to clean up
        """
        # Handle customizations
        nim_customizations = self.db_manager.find_customizations_for_nim(nim_id)
        for customization in nim_customizations:
            if customization.get("customized_model"):
                try:
                    model_name = customization["customized_model"]
                    self.customizer.delete_customized_model(model_name)
                    logger.info(f"Deleted customized model {model_name}")
                except Exception as e:
                    error_msg = f"Failed to delete model {model_name}: {e!s}"
                    logger.warning(error_msg)
                    self.cleanup_errors.append(error_msg)

        # Handle evaluations
        nim_evaluations = self.db_manager.find_evaluations_for_nim(nim_id)
        for evaluation in nim_evaluations:
            if "job_id" in evaluation:
                try:
                    self.evaluator.delete_evaluation_job(evaluation["job_id"])
                    logger.info(f"Deleted evaluation job {evaluation['job_id']}")
                except Exception as e:
                    error_msg = f"Failed to delete evaluation job {evaluation['job_id']}: {e!s}"
                    logger.warning(error_msg)
                    self.cleanup_errors.append(error_msg)

    def _cleanup_datasets(self, flywheel_run: FlywheelRun) -> None:
        """Clean up all datasets associated with a flywheel run.

        Args:
            flywheel_run: The flywheel run containing dataset information
        """
        for dataset in flywheel_run.datasets:
            try:
                data_uploader = DataUploader(dataset_name=dataset.name)
                data_uploader.delete_dataset()
                data_uploader.unregister_dataset()
                logger.info(f"Deleted dataset {dataset.name}")
            except Exception as e:
                error_msg = f"Failed to delete dataset {dataset.name}: {e!s}"
                logger.warning(error_msg)
                self.cleanup_errors.append(error_msg)
