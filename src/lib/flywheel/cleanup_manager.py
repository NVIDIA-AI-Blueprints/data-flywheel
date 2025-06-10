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

from typing import Any

from bson import ObjectId

from src.api.db_manager import TaskDBManager
from src.config import settings
from src.lib.nemo.customizer import Customizer
from src.lib.nemo.dms_client import DMSClient
from src.log_utils import setup_logging

logger = setup_logging("data_flywheel.cleanup_manager")


class CleanupManager:
    """Manager class for cleaning up all running flywheel resources."""

    def __init__(self, db_manager: TaskDBManager):
        """Initialize the cleanup manager.

        Args:
            db_manager: Database manager instance for accessing MongoDB
        """
        self.db_manager = db_manager
        self.customizer = Customizer()
        self.cleanup_errors: list[str] = []

    def find_running_flywheel_runs(self) -> list[dict[str, Any]]:
        """Find all flywheel runs that are currently running."""
        logger.info("Finding running flywheel runs...")

        running_runs = self.db_manager.find_running_flywheel_runs()

        logger.info(f"Found {len(running_runs)} running flywheel runs")
        return running_runs

    def find_running_nims(self, flywheel_run_id: ObjectId) -> list[dict[str, Any]]:
        """Find all NIMs with RUNNING or PENDING deployment status for a flywheel run."""
        running_nims = self.db_manager.find_running_nims_for_flywheel(flywheel_run_id)

        logger.info(f"Found {len(running_nims)} running NIMs for flywheel run {flywheel_run_id}")
        return running_nims

    def find_customization_jobs(self, nim_id: ObjectId) -> list[dict[str, Any]]:
        """Find all customization jobs associated with a NIM."""
        customizations = self.db_manager.find_customizations_for_nim(nim_id)

        logger.info(f"Found {len(customizations)} customizations for NIM {nim_id}")
        return customizations

    def find_evaluation_jobs(self, nim_id: ObjectId) -> list[dict[str, Any]]:
        """Find all evaluation jobs associated with a NIM."""
        evaluations = self.db_manager.find_evaluations_for_nim(nim_id)

        logger.info(f"Found {len(evaluations)} evaluations for NIM {nim_id}")
        return evaluations

    def cancel_customization_jobs(self, customizations: list[dict[str, Any]]):
        """Cancel all running customization jobs."""
        if not customizations:
            return

        for customization in customizations:
            if "job_id" in customization:
                try:
                    self.customizer.cancel_job(customization["job_id"])
                    logger.info(f"Cancelled customization job {customization['job_id']}")
                except Exception as e:
                    error_msg = f"Failed to cancel customization job {customization['job_id']}: {e}"
                    logger.warning(error_msg)
                    self.cleanup_errors.append(error_msg)

    def shutdown_nim(self, nim: dict[str, Any]):
        """Shutdown a NIM deployment."""
        model_name = nim["model_name"]

        # Find the NIM config in settings
        nim_config = None
        for config in settings.nims:
            if config.model_name == model_name:
                nim_config = config
                break

        if not nim_config:
            logger.warning(f"NIM config for {model_name} not found in settings")
            return

        try:
            dms_client = DMSClient(nmp_config=settings.nmp_config, nim=nim_config)
            dms_client.shutdown_deployment()
            logger.info(f"Shutdown NIM deployment for {model_name}")
        except Exception as e:
            error_msg = f"Failed to shutdown NIM {model_name}: {e}"
            logger.warning(error_msg)
            self.cleanup_errors.append(error_msg)

    def shutdown_llm_judge(self):
        """Shutdown LLM judge if it's running locally."""
        try:
            llm_judge_config = settings.llm_judge_config

            if llm_judge_config.is_remote:
                logger.info("LLM judge is remote, no shutdown needed")
                return

            dms_client = DMSClient(nmp_config=settings.nmp_config, nim=llm_judge_config)
            dms_client.shutdown_deployment()
            logger.info(f"Shutdown LLM judge deployment for {llm_judge_config.model_name}")
        except Exception as e:
            error_msg = f"Failed to shutdown LLM judge: {e}"
            logger.warning(error_msg)
            self.cleanup_errors.append(error_msg)

    def mark_resources_as_cancelled(self, flywheel_run_id: ObjectId):
        """Mark all resources as cancelled in the database."""
        try:
            # Mark flywheel run as cancelled
            self.db_manager.mark_flywheel_run_cancelled(
                flywheel_run_id, error_msg="Cancelled by cleanup manager"
            )

            # Mark all NIMs as cancelled
            nims = self.db_manager.find_nims_for_job(flywheel_run_id)
            for nim in nims:
                self.db_manager.mark_nim_cancelled(
                    nim["_id"], error_msg="Cancelled by cleanup manager"
                )

            # Mark LLM judge as cancelled
            self.db_manager.mark_llm_judge_cancelled(
                flywheel_run_id, error_msg="Cancelled by cleanup manager"
            )

            logger.info(f"Marked all resources as cancelled for flywheel run {flywheel_run_id}")

        except Exception as e:
            error_msg = f"Failed to mark resources as cancelled: {e}"
            logger.warning(error_msg)
            self.cleanup_errors.append(error_msg)

    def cleanup_flywheel_run(self, flywheel_run: dict[str, Any]):
        """Clean up all resources for a single flywheel run."""
        flywheel_run_id = flywheel_run["_id"]
        logger.info(f"Cleaning up flywheel run {flywheel_run_id}")

        # Find running NIMs
        running_nims = self.find_running_nims(flywheel_run_id)

        for nim in running_nims:
            nim_id = nim["_id"]
            logger.info(f"Processing NIM {nim['model_name']} (ID: {nim_id})")

            # Find and clean up customization jobs
            customizations = self.find_customization_jobs(nim_id)
            if customizations:
                self.cancel_customization_jobs(customizations)

            # TODO: Find and cancel evaluation jobs once available from NMP

            # Shutdown the NIM
            self.shutdown_nim(nim)

        # Mark all resources as cancelled in the database
        self.mark_resources_as_cancelled(flywheel_run_id)

    def cleanup_all_running_resources(self):
        """Main cleanup procedure for all running resources."""
        logger.info("Starting cleanup of all running resources...")
        self.cleanup_errors = []

        try:
            # Find all running flywheel runs
            running_flywheel_runs = self.find_running_flywheel_runs()

            if not running_flywheel_runs:
                logger.info("No running flywheel runs found. Nothing to clean up.")
            else:
                # Clean up each flywheel run
                for flywheel_run in running_flywheel_runs:
                    try:
                        self.cleanup_flywheel_run(flywheel_run)
                    except Exception as e:
                        error_msg = f"Failed to clean up flywheel run {flywheel_run['_id']}: {e}"
                        logger.error(error_msg)
                        self.cleanup_errors.append(error_msg)

            # Shutdown LLM judge
            self.shutdown_llm_judge()

            # Report results
            if self.cleanup_errors:
                logger.warning(f"Cleanup completed with {len(self.cleanup_errors)} errors:")
                for error in self.cleanup_errors:
                    logger.warning(f"  - {error}")
            else:
                logger.info("Cleanup completed successfully with no errors!")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            raise
